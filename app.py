import os
import numpy as np
from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model

app = Flask(__name__)

group_mapping = {
    "애플리케이션 개발군(A)": ["frontend", "backend", "web_developer", "mobile_developer", "uiux_designer"],
    "데이터 및 AI 전문군(B)": ["data_scientist", "ai_engineer", "data_engineer", "database_admin", "software_engineer"],
    "시스템 및 보안군(C)": ["devops_engineer", "cloud_engineer", "security_engineer", "system_engineer", "network_engineer"],
    "특수 도메인군(D)": ["game_developer", "embedded_engineer", "robotics_engineer", "blockchain_developer", "qa_engineer"]
}

# 🔥 중요: 매번 학습하지 않고, 깃허브에 올린 완성된 모델 파일만 쏙 읽어옵니다!
model = load_model('survey_model.h5')

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>설문 결과 분석</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 600px; margin: 0 auto; background-color: #f9f9f9; }
        .box { border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; background: white; }
        .title { font-weight: bold; font-size: 1.2em; color: #333; }
        .rank { margin-left: 10px; margin-top: 5px; color: #666; }
    </style>
</head>
<body>
    <h2>📊 AI 기반 직무 적성 및 세부 직업 추천 결과</h2>
    <hr>
    {% for res in results %}
        <div class="box">
            <div class="title">▶ {{ res.group_name }}: {{ res.prob }}%</div>
            <hr style="border:0; height:1px; background:#eee;">
            {% for rank, job in res.jobs %}
                <div class="rank"><b>{{ rank }}위:</b> {{ job }}</div>
            {% endfor %}
        </div>
    {% endfor %}
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
    new_user_scaled = np.array([new_user]).astype(float) / 5.0
    pred_prob = model.predict(new_user_scaled)[0]

    web_results = []
    for i in range(4):
        group_name = group_names[i]
        prob = pred_prob[i]
        
        jobs_in_group = group_mapping[group_name]
        job_scores = {job: (prob * 50) + (np.mean(new_user) * 5) + np.random.uniform(1, 5) for job in jobs_in_group}
        sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
        
        web_results.append({
            "group_name": group_name,
            "prob": f"{prob*100:.1f}",
            "jobs": [(rank, job_name) for rank, (job_name, score) in enumerate(sorted_jobs, 1)]
        })

    return render_template_string(html_template, results=web_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
