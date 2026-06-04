import os
import json
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

group_mapping = {
    "애플리케이션 개발군(A)": ["frontend", "backend", "web_developer", "mobile_developer", "uiux_designer"],
    "데이터 및 AI 전문군(B)": ["data_scientist", "ai_engineer", "data_engineer", "database_admin", "software_engineer"],
    "시스템 및 보안군(C)": ["devops_engineer", "cloud_engineer", "security_engineer", "system_engineer", "network_engineer"],
    "특수 도메인군(D)": ["game_developer", "embedded_engineer", "robotics_engineer", "blockchain_developer", "qa_engineer"]
}

# JSON 가중치 로드
with open('model_weights.json', 'r') as f:
    weights_data = json.load(f)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def predict_pure_numpy(X):
    W1 = np.array(weights_data[0]['w'])
    b1 = np.array(weights_data[0]['b'])
    h1 = relu(np.dot(X, W1) + b1)
    
    W2 = np.array(weights_data[1]['w'])
    b2 = np.array(weights_data[1]['b'])
    out = softmax(np.dot(h1, W2) + b2)
    return out

# 🎨 디자인이 대폭 개선된 HTML 템플릿
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI 설문 결과 분석</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Malgun Gothic', sans-serif; line-height: 1.6; padding: 20px; max-width: 550px; margin: 0 auto; background-color: #f5f7fa; color: #333; }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 25px; font-size: 1.5em; }
        
        /* 1등 직군 독점 강조 스타일 */
        .top-box { border: 2px solid #3498db; padding: 22px; margin-bottom: 25px; border-radius: 12px; background: #ffffff; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.15); }
        .top-badge { background-color: #3498db; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.8em; font-weight: bold; display: inline-block; margin-bottom: 8px; }
        .top-title { font-weight: bold; font-size: 1.4em; color: #2c3e50; }
        .top-prob { color: #e74c3c; font-weight: bold; font-size: 1.4em; float: right; }
        
        /* 1등 세부 직업 순위 스타일 */
        .job-list { margin-top: 15px; background: #f8fafd; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }
        .job-item { margin-bottom: 8px; font-size: 1.05em; }
        .job-item:last-child { margin-bottom: 0; }
        .rank-num { color: #3498db; font-weight: bold; margin-right: 8px; }

        /* 나머지 하위 직군 서브 스타일 */
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

@app.route('/result')
def result():
    try:
        new_user = [int(request.args.get(f'q{i}', 3)) for i in range(1, 25)]
    except:
        new_user = [3] * 24

    group_names = list(group_mapping.keys())
    new_user_scaled = np.array(new_user).astype(float) / 5.0
    pred_prob = predict_pure_numpy(new_user_scaled)

    # 전처리 및 결과 빌드
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

    # 1등 직군 찾기 (확률 기준 내림차순 정렬)
    all_results = sorted(all_results, key=lambda x: x['prob_raw'], reverse=True)
    
    top_result = all_results[0]      # 가장 높은 직군
    other_results = all_results[1:]  # 나머지 직군들

    return render_template_string(
        html_template, 
        top_result=top_result, 
        other_results=other_results
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
