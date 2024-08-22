from fastapi import FastAPI
from pydantic import BaseModel
from app.rag.preprocessor import Preprocessor
from app.rag.retriever import Retriever
from app.rag.generator import Generator

app = FastAPI()

class Question(BaseModel):
    question: str

preprocessor = Preprocessor()
preprocessor.load('index')
retriever = Retriever(preprocessor)
generator = Generator()

@app.post("/rag")
async def rag_endpoint(question: Question):
    relevant_chunks = retriever.retrieve(question.question)
    context = " ".join(relevant_chunks)
    answer = generator.generate(question.question, context)
    return {"answer": answer, "relevant_chunks": relevant_chunks}