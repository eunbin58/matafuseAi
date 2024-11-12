from flask import Flask, request, jsonify
from flask_cors import CORS
import aichatbot
import aiGame
import emotion

app = Flask(__name__)
CORS(app)

# aichatbot 모듈의 초기화 함수 호출
aichatbot.create_conversations_table()  # 데이터베이스 초기화

# 라우트 정의

# AI Chatbot의 '/chat' 라우트
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    return aichatbot.chat(data)

# AI Game의 '/game-result' 라우트
@app.route('/game-result', methods=['POST'])
def game_result():
    data = request.get_json()
    return aiGame.gameResult(data)

# Emotion Analysis의 '/encouragement' 라우트
@app.route('/encouragement', methods=['GET'])
def encouragement():
    return emotion.get_encouragement()

# 테스트 결과를 반환하는 라우트
@app.route('/test-result', methods=['GET'])
def test_result():
    username = request.args.get('username')
    return aichatbot.test_result(username)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
