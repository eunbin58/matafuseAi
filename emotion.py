import os
import random
from openai import OpenAI
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import mysql.connector
from flask import jsonify

# 경고 메시지 무시 설정
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

client = OpenAI()  # OpenAI 클라이언트


# 저장된 모델과 토크나이저 로드
model_path = "./saved_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# label_mapping 설정
label_mapping = {"기쁨": 0, "당황": 1, "분노": 2, "불안": 3, "상처": 4, "슬픔": 5}
id_to_label = {v: k for k, v in label_mapping.items()}

# 감정 예측 함수 정의
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=-1).item()
    emotion = id_to_label[predicted_label]
    confidence = probs[0][predicted_label].item()
    return emotion, confidence

# 데이터베이스에서 최근 10개의 기록 가져오기 함수
def get_recent_entries(user_id, limit=10):
    try:
        conn = mysql.connector.connect(
            user='eunbin',
            password='eunbin',
            host='localhost',
            database='db_connect'
        )
        cursor = conn.cursor()
        query = """
            SELECT record_contents FROM Record
            WHERE member_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """
        cursor.execute(query, (user_id, limit))
        entries = [row[0] for row in cursor.fetchall()]
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        entries = []
    finally:
        cursor.close()
        conn.close()
    return entries

# 감정에 따른 GPT 응원 메시지 생성 함수
def generate_response(emotion, entry):
    prompt = f"사용자가 다음과 같은 감정을 느낀 기록을 남겼습니다: '{entry}'. 그들의 감정은 '{emotion}'입니다. 이 감정을 바탕으로 그들에게 해줄 수 있는 따뜻한 응원 메시지를 생성해 주세요. 100자 이내로 응원 메시지를 작성해 주세요."

    try:
        # ChatGPT API 호출하여 응원 메시지 생성
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 사용자를 위한 따뜻한 응원 메시지를 제공하는 친절한 챗봇입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "응원 메시지를 생성할 수 없습니다. 잠시 후 다시 시도해 주세요."

# Flask와 연결될 응원 메시지 반환 함수
def get_encouragement():
    user_id = "1"
    recent_entries = get_recent_entries(user_id)
    
    if not recent_entries:
        return jsonify({"message": "No records found"}), 404
    
    # 최근 10개의 기록 중에서 랜덤으로 하나 선택
    entry = random.choice(recent_entries)

    # 선택된 기록에 대해 감정 예측 및 응원 메시지 생성
    emotion, confidence = predict_emotion(entry)
    encouragement_message = generate_response(emotion, entry)

    # 결과 반환
    return jsonify({"message": encouragement_message, "emotion": emotion, "confidence": confidence})