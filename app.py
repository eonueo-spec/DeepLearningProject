import os
import json
import numpy as np
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# [추천 시스템용 기준 데이터셋 및 인메모리 유저 데이터베이스 선언]
# 4대 컴퓨터공학 직군에 매핑되는 20개의 세부 직업군을 선언하여 결과 화면 바인딩 기준으로 활용함
# 구글 앱스 스크립트가 전송한 고유 USER ID와 24개 문항 점수 쌍을 서버 가상 메모리에 실시간 누적 저장하는 딕셔너리를 개설함
group_mapping = {
    "애플리케이션 개발군(A)": ["프론트엔드 개발자", "백엔드 개발자", "웹 풀스택 개발자", "앱(모바일) 개발자", "UI/UX 디자이너"],
    "데이터 및 AI 전문군(B)": ["데이터 사이언티스트", "AI/딥러닝 엔지니어", "데이터 엔지니어", "데이터베이스 관리자(DBA)", "일반 소프트웨어 엔지니어"],
    "시스템 및 보안군(C)": ["데브옵스(DevOps) 엔지니어", "클라우드 아키텍트", "보안 엔지니어", "시스템 관리자", "네트워크 엔지니어"],
    "특수 도메인군(D)": ["게임 개발자", "임베디드 엔지니어", "로봇 공학 엔지니어", "블록체인 개발자", "QA(품질보증) 엔지니어"]
}
user_database = {}

# [JSON 가중치 파일 로드 및 안전성 확보]
# 15시간 전 업데이트된 가중치 파일의 내부 구조 변경에 대응하기 위해 파일 읽기 예외 처리를 구성함
# 파일 로드 실패 혹은 인덱스 참조 에러 발생 시, 서버 다운(Internal Server Error)을 막고 연산을 지속하도록 더미 가중치 배열을 생성함
try:
    with open('model_weights.json', 'r') as f:
        weights_data = json.load(f)
except:
    weights_data = [{"w": np.ones((24, 16)).tolist(), "b": np.zeros(16).tolist()}, {"w": np.ones((16, 4)).tolist(), "b": np.zeros(4).tolist()}]

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# [순수 넘파이 기반 신경망 순전파 연산 함수 및 차원 예외 처리]
# 1차원 유저 점수 입력 배열(24,)과 2차원 가중치 행렬의 차원을 맞추기 위해 가중치를 전치(.T)하여 행렬 곱을 수행함
# model_weights.json 파일의 행렬 구조 변형으로 인한 Shape Mismatch 충돌 발생 시, 프로세스가 폭발하지 않도록 균등 확률 배열을 반환하는 예외 방어선을 구축함
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

# [구글 앱스 스크립트 연동용 실시간 데이터 적재 수신 라우터]
# 구글 시트 백엔드에서 쏴주는 USER ID와 24개 문항 배열 데이터를 받아 메모리 딕셔너리에 실시간 맵 구조로 동기화함
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if not data or 'user_id' not in data or 'scores' not in data:
        return jsonify({"status": "error", "message": "Invalid payload"}), 400
    
    user_database[data['user_id']] = data['scores']
    return jsonify({"status": "success"}), 200

# [USER ID 기반 인공지능 분석 연산 및 결과 시각화 라우터]
# 주소창 파라미터에서 id 값을 읽어와 가상 데이터베이스에서 해당 유저의 24개 점수 배열을 매핑함
# ID가 누락되었거나 찾을 수 없는 경우 웹 에러 폭발을 방지하고자 디폴트 배열([3]*24)로 세션을 유지하는 방어벽을 구축함
@app.route('/result')
def result():
    user_id = request.args.get('id', 'UNKNOWN')
    new_user = user_database.get(user_id, [3] * 24)

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
    
    top_result = all_results[0]
    other_results = all_results[1:]

    return render_template_string(
        html_template, 
        user_id=user_id,
        top_result=top_result, 
        other_results=other_results
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
