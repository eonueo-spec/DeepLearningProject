import os
import json
import numpy as np
from flask import Flask, request, render_template_string
import gspread
from google.oauth2.service_account import Credentials

app = Flask(__name__)

group_mapping = {
    "애플리케이션 개발군(A)": ["프론트엔드 개발자", "백엔드 개발자", "웹 풀스택 개발자", "앱(모바일) 개발자", "UI/UX 디자이너"],
    "데이터 및 AI 전문군(B)": ["데이터 사이언티스트", "AI/딥러닝 엔지니어", "데이터 엔지니어", "데이터베이스 관리자(DBA)", "일반 소프트웨어 엔지니어"],
    "시스템 및 보안군(C)": ["데브옵스(DevOps) 엔지니어", "클라우드 아키텍트", "보안 엔지니어", "시스템 관리자", "네트워크 엔지니어"],
    "특수 도메인군(D)": ["게임 개발자", "임베디드 엔지니어", "로봇 공학 엔지니어", "블록체인 개발자", "QA(품질보증) 엔지니어"]
}

with open('model_weights.json', 'r') as f:
    weights_data = json.load(f)

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
client = gspread.authorize(creds)

sheet = client.open("Form_Responses").sheet1


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def predict_pure_numpy(X):
    W1 = np.array(weights_data[0]['w'])
    b1 = np.array(weights_data[0]['b'])
    h1 = relu(np.dot(W1.T, X) + b1)

    W2 = np.array(weights_data[1]['w'])
    b2 = np.array(weights_data[1]['b'])
    out = softmax(np.dot(W2.T, h1) + b2)

    return out


html_template = """(네 기존 HTML 그대로 넣어)"""


@app.route('/result')
def result():

    user_id = request.args.get("id")
    if not user_id:
        return "No ID provided"

    data = sheet.get_all_values()

    user_row = None
    for row in data:
        if row[-1] == user_id:
            user_row = row
            break

    if not user_row:
        return "User not found"

    new_user = []

    for x in user_row[1:-1]:
        try:
            new_user.append(int(float(x)))
        except:
            new_user.append(3)

    if len(new_user) < 24:
        new_user += [3] * (24 - len(new_user))

    new_user = new_user[:24]

    new_user_scaled = np.array(new_user) / 5.0

    try:
        pred_prob = predict_pure_numpy(new_user_scaled)
    except:
        pred_prob = np.array([0.25, 0.25, 0.25, 0.25])

    group_names = list(group_mapping.keys())

    all_results = []

    for i in range(4):
        group_name = group_names[i]
        prob = pred_prob[i]

        jobs = group_mapping[group_name]

        job_scores = {
            job: (prob * 50)
                 + (np.mean(new_user) * 5)
                 + np.random.uniform(1, 5)
            for job in jobs
        }

        sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)

        all_results.append({
            "group_name": group_name,
            "prob_raw": prob,
            "prob": f"{prob*100:.1f}",
            "jobs": [(rank, job) for rank, (job, _) in enumerate(sorted_jobs, 1)]
        })

    all_results = sorted(all_results, key=lambda x: x["prob_raw"], reverse=True)

    return render_template_string(
        html_template,
        top_result=all_results[0],
        other_results=all_results[1:]
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
