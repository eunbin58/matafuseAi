import os
import random
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import mysql.connector
from flask import jsonify

# 경고 메시지 무시 설정
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

client = OpenAI()  # OpenAI 클라이언트

# KoBERT 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained("monologg/kobert", num_labels=3, trust_remote_code=True)

# 감정 분석 함수
def analyze_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    emotion = torch.argmax(probabilities).item()
    return emotion, probabilities

# 데이터베이스에서 최근 기록 가져오기 함수
def get_recent_entries(user_id, limit=10):
    conn = mysql.connector.connect(user='eunbin', password='eunbin', host='localhost', database='db_connect')
    cursor = conn.cursor()
    query = """
        SELECT record_contents FROM Record
        WHERE member_id = %s
        ORDER BY created_at DESC
        LIMIT %s
    """
    cursor.execute(query, (user_id, limit))
    entries = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return entries

# 감정에 따른 응원 메시지 생성 함수
def generate_response(emotion, entry):
    emotion_labels = {0: '긍정', 1: '부정', 2: '중립'}
    prompt = f"사용자가 '{emotion_labels[emotion]}' 감정을 느낀 기록을 남겼습니다: '{entry}'. 100자 이내의 따뜻한 응원 메시지를 작성해 주세요."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 사용자를 위한 따뜻한 응원 메시지를 제공하는 친절한 챗봇입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content  # 응원 메시지 반환

# Flask와 연결될 응원 메시지 반환 함수
def get_encouragement():
    user_id = "1"
    recent_entries = get_recent_entries(user_id)
    if not recent_entries:
        return jsonify({"message": "No records found"}), 404
    
    entry = random.choice(recent_entries)
    emotion, _ = analyze_emotion(entry)
    encouragement_message = generate_response(emotion, entry)
    return jsonify({"message": encouragement_message})
