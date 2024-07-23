import pytest
from fastapi.testclient import TestClient
from chatbot import app, retriever, llm, rag_chain
import time
client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "GROQ Chat" in response.text

def test_ask_question():
    response = client.post("/ask", json={"question": "What is this document about?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "response_time" in response.json()

def test_form_get():
    response = client.get("/")
    assert response.status_code == 200
    assert '<form method="post">' in response.text

def test_form_post():
    response = client.post("/", data={"question": "What is this document about?"})
    assert response.status_code == 200
    assert "Question:" in response.text
    assert "Answer:" in response.text
    assert "Response Time:" in response.text

def test_rag_chain():
    question = "What is this document about?"
    start_time = time.time()
    try:
        response = rag_chain.invoke(question)
    except Exception as e:
        response = None
    end_time = time.time()
    response_time = end_time - start_time

    assert response is not None
    assert isinstance(response, str)
    assert response_time > 0

def test_retriever():
    assert retriever is not None

def test_llm():
    assert llm is not None
    
    assert llm.model_name == 'mixtral-8x7b-32768'

def test_template_render():
    response = client.post("/", data={"question": "Test question"})
    assert response.status_code == 200
    assert "Test question" in response.text
