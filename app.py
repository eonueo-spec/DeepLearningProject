import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)

group_mapping = {
    "애플리케이션 개발군(A)": ["frontend", "backend", "web_developer", "mobile_developer", "uiux_designer"],
    "데이터 및 AI 전문군(B)": ["data_scientist", "ai_engineer", "data_engineer", "database_admin", "software_engineer"],
    "시스템 및 보안군(C)": ["devops_engineer", "cloud_engineer", "security_engineer", "system_engineer", "network_engineer"],
    "특수 도메인군(D)": ["game_developer", "embedded_engineer", "robotics_engineer", "blockchain_developer", "qa_engineer"]
}

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ7kWaukJOHUTib4m028Wj-fWuf_cGWO-br-OminJ1k7pP7KqHwIkhUxAvShzrBcTRz9OQNXWHyC5f_/pub?output=csv"
df = pd.read_csv(SHEET_URL)

text_to_score = {"전혀 아니다": 1, "아니다": 2, "보통이다": 3, "그렇다": 4, "매우 그렇다": 5}
survey_df = df.iloc[:, 2:26].replace(text_to_score)                                            
survey_matrix = survey_df.to_numpy().astype(float)

np.random.seed(42)
virtual_data = np.random.randint(1, 6, size=(1000, 24)).astype(float)
X_data = np.concatenate((survey_matrix, virtual_data), axis=0)
X_scaled = X_data / 5.0

y_raw = np.random.randint(0, 4, size=X_scaled.shape[0])

y_real = df['25. 당신이 향후 가장 도전해보고 싶거나, 현재 가장 관심이 많은 컴퓨터 공학 분야는 무엇인가요? '].replace({
    "사용자 중심의 웹/앱 서비스를 만드는 분야": 0,
    "데이터 분석 및 인공지능 모델을 연구하는 분야": 1,
    "서버 인프라 구축 및 사이버 보안을 담당하는 분야": 2,
    "게임 개발, 로봇/장비 제어, 블록체인 등 특수 기술 분야": 3
}).to_numpy().astype(int)

y_raw[:len(y_real)] = y_real

Y_data = to_categorical(y_raw, num_classes=4)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_data, test_size=0.2, random_state=42)

model = Sequential([
    Dense(32, input_shape=(24,), activation='relu'),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

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
    # Render 환경의 가변 포트 번호를 수신하기 위해 임포트 os 대신 시스템 값 다이렉트 호출 처리
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))