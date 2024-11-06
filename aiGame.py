from flask import jsonify
import openai 

def gameResult(data):
    # 게임 결과 데이터를 받음
    game_stats = data.get('message', {})
    user_message = (
        f"나이: {game_stats['age']}세, 건강: {game_stats['health']}, 스트레스: {game_stats['stress']}, "
        f"대인 관계: {game_stats['relationships']}, 돈: {game_stats['money']}. "
        "이 사람은 끊임없는 선택의 연속 속에서 자신을 정의하고 있습니다. "
        "이 선택들이 그의 삶에 어떤 영향을 미치고, 어떤 길을 만들어 나가는지에 대한 철학적 통찰을 제공해 주세요."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "당신은 삶의 선택과 결과에 대해 문학적이고 철학적인 깊이 있는 통찰을 제공하는 전문가입니다."
            )
        },
        {"role": "user", "content": user_message}
    ]
    
    try:
        result = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.8
        )
        chatbot_reply = result['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error: {str(e)}")
        chatbot_reply = (
            "입력 데이터를 바탕으로 심오한 통찰을 생성하지 못했습니다. "
            "다시 시도해 주시기 바랍니다."
        )
    return jsonify({'reply': chatbot_reply})
