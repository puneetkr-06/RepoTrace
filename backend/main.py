import sys
print("=== STARTING MAIN.PY ===", flush=True)

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional


from pinecone import Pinecone



load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

app = FastAPI(title="RepoTrace API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    repo_url: str

class IngestRequest(BaseModel):
    repo_url: str

# We will initialize these lazily so they don't block the server startup!
pc = None
embeddings = None
llm = None

def get_pinecone():
    global pc
    if pc is None:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    return pc

def get_embeddings():
    global embeddings
    if embeddings is None:
        import os
        from langchain_pinecone import PineconeEmbeddings
        embeddings = PineconeEmbeddings(model="multilingual-e5-large", pinecone_api_key=os.environ.get("PINECONE_API_KEY"))
    return embeddings

def get_llm():
    global llm
    if llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    return llm



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(repo_name: str):
    from langchain_pinecone import PineconeVectorStore
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    vectorstore = PineconeVectorStore(
        index_name="repotrace-pc", 
        embedding=get_embeddings(),
        namespace=repo_name
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    system_prompt = (
        "You are an expert AI coding assistant for a GitHub repository. "
        "Use the following retrieved context to answer the user's question about the codebase. "
        "Answer Concisely If User asks specific question just answer that in 2-3 lines Only give in detail when asked specifically for details or where it is necessary. Else Try to remain very short & concise with your answer"
        "If you don't know the answer, say that you don't know. Use code snippets where relevant.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    rag_chain_lcel = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    
    class RAGWrapper:
        def invoke(self, inputs):
            question = inputs["input"]
            docs = retriever.invoke(question)
            answer = rag_chain_lcel.invoke(question)
            return {"answer": answer, "context": docs}
            
    return RAGWrapper()

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "ok"}

@app.post("/api/ingest")
async def ingest_repo_endpoint(request: IngestRequest):
    # Strip https://github.com/ if present
    repo_name = request.repo_url.replace("https://github.com/", "").strip("/")
    if not repo_name or len(repo_name.split("/")) != 2:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL. Must be in format owner/repo")
        
    try:
        # Check if already ingested (but handle if index doesn't exist yet)
        try:
            index = get_pinecone().Index("repotrace-pc")
            stats = index.describe_index_stats()
            if repo_name in stats.get("namespaces", {}):
                return {"status": "success", "message": f"Already ingested {repo_name}", "repo_name": repo_name}
        except Exception:
            pass # Index probably doesn't exist yet, which is fine, we will create it!
            
        # Run synchronously for MVP (will block, but frontend will wait)
        from ingest import ingest_repository
        ingest_repository(repo_name)
        return {"status": "success", "message": f"Successfully ingested {repo_name}", "repo_name": repo_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    repo_name = request.repo_url.replace("https://github.com/", "").strip("/")
    
    try:
        rag_chain = get_rag_chain(repo_name)
        response = rag_chain.invoke({"input": request.message})
        
        # Extract citations
        citations = []
        for doc in response.get("context", []):
            source = doc.metadata.get("source", "Unknown file")
            if source not in citations:
                citations.append(source)
                
        return {
            "reply": response["answer"],
            "citations": citations
        }
    except Exception as e:
        print(f"Chat error: {e}")
        return {"reply": f"Sorry, an error occurred: {str(e)}", "citations": []}
