import os
import threading
import subprocess
import time

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pyngrok import ngrok

# Start the Ollama service
def ollama():
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "serve"])

ollama_thread = threading.Thread(target=ollama)
ollama_thread.start()

# Authenticate ngrok
auth_token = "2jdmuFZrR89bRmUPrD4EHVMFz4f_6jgvBVNAqkvwFEyZeireZ"
os.system(f"ngrok authtoken {auth_token}")

# Create ngrok tunnel
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

# Initialize FastAPI app
app = FastAPI()

# Load and process the PDF document
loader = PyPDFDirectoryLoader("data")
the_text = loader.load()
print(f"The uploaded documents contain {len(the_text)} pages.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(the_text)

vectorstore = Chroma.from_documents(
    documents=chunks,
    collection_name="ollama_embeds",
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
)

retriever = vectorstore.as_retriever()
groq_api_key = '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'
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='mixtral-8x7b-32768'
)

rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Define the request and response models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    response_time: float

# Endpoint to ask questions via API
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    start_time = time.time()

    try:
        response = rag_chain.invoke(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    end_time = time.time()
    response_time = end_time - start_time

    return AnswerResponse(answer=response, response_time=response_time)

# Endpoint to serve the HTML form and handle form submission
@app.get("/", response_class=HTMLResponse)
@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, question: str = Form(None)):
    if request.method == "POST" and question:
        start_time = time.time()
        try:
            response = rag_chain.invoke(question)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        end_time = time.time()
        response_time = end_time - start_time

        return HTMLResponse(f"""
            <html>
            <head>
                <title>GROQ Chat</title>
            </head>
            <body>
                <h1>GROQ Chat</h1>
                <form method="post">
                    <label for="question">Type your question here:</label><br>
                    <input type="text" id="question" name="question" required><br>
                    <input type="submit" value="Submit">
                </form>
                <h2>Question:</h2>
                <p>{question}</p>
                <h2>Answer:</h2>
                <p>{response}</p>
                <h2>Response Time:</h2>
                <p>{response_time} seconds</p>
            </body>
            </html>
        """)
    else:
        return HTMLResponse(f"""
            <html>
            <head>
                <title>GROQ Chat</title>
            </head>
            <body>
                <h1>GROQ Chat</h1>
                <form method="post">
                    <label for="question">Type your question here:</label><br>
                    <input type="text" id="question" name="question" required><br>
                    <input type="submit" value="Submit">
                </form>
            </body>
            </html>
        """)

# Start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
