import os
import warnings
warnings.filterwarnings('ignore')

from main import rag_chain

if not rag_chain:
    print("Error: RAG chain failed to initialize. Check API keys.")
    exit(1)

question = "Which Backend tool is used in this project?"
print(f"Question: {question}")
print("Thinking...\n")

try:
    response = rag_chain.invoke({"input": question})

    print("--- AI Answer ---")
    print(response["answer"])
    
    print("\n--- Files Cited ---")
    citations = []
    for doc in response.get("context", []):
        source = doc.metadata.get("source", "Unknown file")
        if source not in citations:
            citations.append(source)
            print(f"- {source}")
            
except Exception as e:
    print(f"Error during retrieval: {e}")
