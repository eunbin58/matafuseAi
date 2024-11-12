from flask import jsonify
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader, TextLoader
import mysql.connector

db_config = {
    'user': 'eunbin',
    'password': 'eunbin',
    'host': 'localhost',
    'database': 'db_connect'
}

client = OpenAI()

# 데이터 로더 및 전처리 함수
def load_and_preprocess_documents():
    pdf_loader = DirectoryLoader('dataset', glob="*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader('dataset', glob="*.txt", loader_cls=lambda p: TextLoader(p, encoding="utf-8"))
    
    pdf_documents = pdf_loader.load()
    txt_documents = txt_loader.load()
    
    documents = pdf_documents + txt_documents
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

# 임베딩 및 벡터 저장소 초기화 함수
def initialize_vector_store(documents):
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=documents, embedding=embedding)
    return vectordb.as_retriever(search_kwargs={"k": 2})

# 문서 로드 및 벡터 저장소 초기화
documents = load_and_preprocess_documents()
retriever = initialize_vector_store(documents)

# QA 체인 초기화
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def create_conversations_table():
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS conversations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            role ENUM('user', 'bot') NOT NULL,
            message TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)
        connection.commit()

    except mysql.connector.Error as err:
        print(f"[ERROR] {err}")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# 대화 저장
def save_conversation(user_id, role, message):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        insert_query = """
        INSERT INTO conversations (user_id, role, message)
        VALUES (%s, %s, %s);
        """
        cursor.execute(insert_query, (user_id, role, message))
        connection.commit()

    except mysql.connector.Error as err:
        print(f"[ERROR] Database error: {err}")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# 과거 대화 불러오기
def get_recent_conversations(user_id, limit=6):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT role, message 
            FROM conversations
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        cursor.execute(query, (user_id, limit))
        conversations = cursor.fetchall()
        
        # 최신 대화부터 순서대로 가져왔으므로 역순으로 정렬
        return conversations[::-1]

    except mysql.connector.Error as err:
        print(f"[ERROR] Database error: {err}")
        return []

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def chat(data):
    user_id = data.get('user_id')
    user_message = data.get('message', '')

    # 사용자 메시지 저장
    save_conversation(user_id, 'user', user_message)

    # 과거 대화 불러오기
    past_conversations = get_recent_conversations(user_id, limit=5)
    context = "\n".join([f"{conv['role'].capitalize()}: {conv['message']}" for conv in past_conversations])
    
    # AI 응답 생성
    prompt = f"{context}\nUser: {user_message}\nBot:"
    result = qa_chain.invoke(prompt)
    
    chatbot_reply = result['result']
    
    # 소스 정보 JSON 배열로 포맷
    sources = [{"source": doc.metadata['source']} for doc in result['source_documents']]
    chatbot_reply += " # sources : " + ", ".join([f"[{i+1}] {src['source']}" for i, src in enumerate(sources)])

    # 봇 응답 저장
    save_conversation(user_id, 'bot', chatbot_reply)
        
    return jsonify({
        'reply': chatbot_reply,
        #'sources': sources
    })
    
def test_result(user_id):
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT test_id, score 
            FROM test_result
            WHERE member_id = %s
            ORDER BY test_id DESC
            LIMIT 1
        """
        cursor.execute(query, (user_id,))
        results = cursor.fetchall()

        return jsonify(results if results else [])

    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        return jsonify({"error": str(err)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
