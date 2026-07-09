print("=== IMPORTING INGEST.PY ===", flush=True)
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

# Pinecone will be initialized lazily
pc = None

# Local HF embeddings use dimension 384
index_name = "repotrace-hf"

def setup_pinecone():
    global pc
    if pc is None:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name} (Dimension: 384)...")
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(index_name)

def ingest_repository(repo_name: str, branch: str = "main"):
    """
    repo_name: e.g. 'puneetkumar/my-repo' or any stranger's public repo
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token or github_token == "your_github_personal_access_token_here":
        raise ValueError("Please provide a valid GITHUB_TOKEN in your .env file.")

    print(f"Loading files from {repo_name}...")
    loader = GithubFileLoader(
        repo=repo_name,
        branch=branch,
        access_token=github_token,
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: (
            file_path.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".html", ".css")) and
            not any(ignored in file_path for ignored in ["node_modules", "dist", "build", "package-lock.json", ".min.js"])
        )
    )
    
    try:
        documents = loader.load()
        print(f"Loaded {len(documents)} files.")
    except Exception as e:
        print(f"Failed to load repository: {e}")
        return

    # Intelligently split the code files into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} searchable chunks.")

    # Setup Local HuggingFace Embeddings
    print("Loading local embedding model...")
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Connecting to Pinecone...")
    index = setup_pinecone()
    
    print(f"Clearing any existing data for {repo_name} to prevent duplicates...")
    try:
        index.delete(delete_all=True, namespace=repo_name)
    except Exception as e:
        pass # It's fine if the namespace doesn't exist yet
    
    print("Uploading vector embeddings to Pinecone at FULL SPEED...")
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        namespace=repo_name
    )
    
    # We no longer need the time.sleep() wait loops!
    vectorstore.add_documents(chunks)

    print(f"✅ Successfully ingested {repo_name} into Pinecone using Local Embeddings!")

if __name__ == "__main__":
    # Example usage
    repo_to_ingest = input("Enter a public GitHub repo to ingest (e.g. 'facebook/react' or 'hwchase17/langchain'): ")
    ingest_repository(repo_to_ingest)
