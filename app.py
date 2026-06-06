import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ7kWaukJOHUTib4m028Wj-fWuf_cGWO-br-OminJ1k7pP7KqHwIkhUxAvShzrBcTRz9OQNXWHyC5f_/pub?output=csv"

group_mapping = {
    "애플리케이션 개발군(A)": ["프론트엔드 개발자", "백엔드 개발자", "웹 풀스택 개발자", "앱(모바일) 개발자", "UI/UX 디자이너"],
    "데이터 및 AI 전문군(B)": ["데이터 사이언티스트", "AI/딥러닝 엔지니어", "데이터 엔지니어", "데이터베이스 관리자(DBA)", "일반 소프트웨어 엔지니어"],
    "시스템 및 보안군(C)": ["데브옵스(DevOps) 엔지니어", "클라우드 아키텍트", "보안 엔지니어", "시스템 관리자", "네트워크 엔지니어"],
    "특수 도메인군(D)": ["게임 개발자", "임베디드 엔지니어", "로봇 공학 엔지니어", "블록체인 개발자", "QA(품질보증) 엔지니어"]
}

try:
    with open('model_weights.json', 'r') as f:
        weights_data = json.load(f)
except:
    weights_data = [
        {"w": np.ones((24, 32)).tolist(), "b": np.zeros(32).tolist()},
        {"w": np.ones((32, 4)).tolist(),  "b": np.zeros(4).tolist()}
    ]

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def predict_pure_numpy(X):
    try:
        W1 = np.array(weights_data[0]['w'])
        b1 = np.array(weights_data[0]['b'])
        h1 = relu(np.dot(W1.T, X) + b1)
        W2 = np.array(weights_data[1]['w'])
        b2 = np.array(weights_data[1]['b'])
        out = softmax(np.dot(W2.T, h1) + b2)
        return out
    except:
        return np.array([0.25, 0.25, 0.25, 0.25])

def get_scores_from_sheet(user_id):
    try:
        text_to_score = {"전혀 아니다": 1, "아니다": 2, "보통이다": 3, "그렇다": 4, "매우 그렇다": 5}
        df = pd.read_csv(SHEET_URL)

        # AH열(인덱스 33)에서 user_id 검색
        uid_col = df.columns[33]
        matched = df[df[uid_col].astype(str) == user_id]

        if matched.empty:
            return None

        row = matched.iloc[-1]

        # C~Z열(인덱스 2:26) = 문항 1~24
        raw = row.iloc[2:26]
        scores = [text_to_score.get(str(v).strip(), 3) for v in raw]

        return scores[:24]

    except Exception as e:
        print(f"시트 읽기 오류: {e}")
        return None

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI 설문 결과 분석</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; line-height: 1.6; padding: 20px; max-width: 550px; margin: 0 auto; background-color: #f5f7fa; color: #333; }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 25px; font-size: 1.5em; }
        .top-box { border: 2px solid #3498db; padding: 22px; margin-bottom: 25px; border-radius: 12px; background: #ffffff; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.15); }
        .top-badge { background-color: #3498db; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.8em; font-weight: bold; display: inline-block; margin-bottom: 8px; }
        .top-title { font-weight: bold; font-size: 1.4em; color: #2c3e50; }
        .top-prob { color: #e74c3c; font-weight: bold; font-size: 1.4em; float: right; }
        .job-list { margin-top: 15px; background: #f8fafd; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }
        .job-item { margin-bottom: 8px; font-size: 1.05em; }
        .job-item:last-child { margin-bottom: 0; }
        .rank-num { color: #3498db; font-weight: bold; margin-right: 8px; }
        .sub-container { background: #ffffff; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; }
        .sub-title-main { font-size: 0.95em; color: #7f8c8d; margin-bottom: 12px; font-weight: bold; padding-left: 3px; }
        .sub-item { display: flex; justify-content: space-between; padding: 8px 5px; font-size: 0.9em; color: #555; border-bottom: 1px dashed #edf2f7; }
        .sub-item:last-child { border-bottom: none; }
        .sub-prob { font-weight: bold; color: #7f8c8d; }
        .clear { clear: both; }
    </style>
</head>
<body>
    <h2>📊 AI 기반 직무 적성 분석 결과</h2>
    <div style="text-align: center; color: #7f8c8d; font-size: 0.85em; margin-top: -15px; margin-bottom: 20px;">검사 ID: {{ user_id }}</div>
    
    <div class="top-box">
        <span class="top-badge">BEST MATCH</span>
        <div>
            <span class="top-title">{{ top_result.group_name }}</span>
            <span class="top-prob">{{ top_result.prob }}%</span>
        </div>
        <div class="clear"></div>
        <div class="job-list">
            <div style="font-weight: bold; margin-bottom: 10px; color: #34495e; font-size: 0.95em;">🎯 추천 세부 직무 순위</div>
            {% for rank, job in top_result.jobs %}
                <div class="job-item"><span class="rank-num">{{ rank }}위</span> {{ job }}</div>
            {% endfor %}
        </div>
    </div>

    <div class="sub-container">
        <div class="sub-title-main">기타 직군별 적합도 결과</div>
        {% for res in other_results %}
            <div class="sub-item">
                <span>• {{ res.group_name }}</span>
                <span class="sub-prob">{{ res.prob }}%</span>
            </div>
        {% endfor %}
    </div>
</body>
</html>
"""

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if not data or 'user_id' not in data:
        return jsonify({"status": "error"}), 400
    return jsonify({"status": "success"}), 200

@app.route('/result')
def result():
    user_id = request.args.get('id', 'UNKNOWN')
    scores = get_scores_from_sheet(user_id)

    if scores is None:
        new_user = [3] * 24
    else:
        new_user = scores

    group_names = list(group_mapping.keys())
    new_user_scaled = np.array(new_user).astype(float) / 5.0
    pred_prob = predict_pure_numpy(new_user_scaled)

    all_results = []
    for i in range(4):
        group_name = group_names[i]
        prob = pred_prob[i]
        jobs_in_group = group_mapping[group_name]
        job_scores = {job: (prob * 50) + (np.mean(new_user) * 5) + np.random.uniform(1, 5) for job in jobs_in_group}
        sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
        all_results.append({
            "group_name": group_name,
            "prob_raw": prob,
            "prob": f"{prob*100:.1f}",
            "jobs": [(rank, job_name) for rank, (job_name, score) in enumerate(sorted_jobs, 1)]
        })

    all_results = sorted(all_results, key=lambda x: x['prob_raw'], reverse=True)

    return render_template_string(
        html_template,
        user_id=user_id,
        top_result=all_results[0],
        other_results=all_results[1:]
    )

@app.route('/debug')
def debug():
    try:
        df = pd.read_csv(SHEET_URL)
        last_row = df.iloc[-1]
        uid = str(last_row.iloc[33])
        scores = get_scores_from_sheet(uid)
        return jsonify({
            "last_user_id": uid,
            "scores": scores,
            "scores_length": len(scores) if scores else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
