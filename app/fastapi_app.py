from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware


vector_db = None
chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, chain
    try:

        vector_db = Chroma(
            embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
            collection_name="bmw-rag",
            persist_directory="../chroma_db_store",
        )
        local_model = "llama3.2"
        llm = ChatOllama(model=local_model)
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(),
            llm,
            prompt=QUERY_PROMPT
        )
        template = """
        Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
        )
        yield
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app = FastAPI(title="BMW X1 Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for your specific requirements
    allow_methods=["GET", "POST", "OPTIONS"],  # Include OPTIONS
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class ResponseModel(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ask", response_model=ResponseModel)
async def ask_question(request: QuestionRequest):
    global chain
    if not chain:
        raise HTTPException(status_code=500, detail="Chain not initialized.")

    try:
        response = chain.invoke(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    