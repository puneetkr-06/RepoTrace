# print("=== IMPORTING INGEST.PY ===", flush=True)
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

# Pinecone will be initialized lazily
pc = None

# Pinecone API embeddings use dimension 1024
index_name = "repotrace-pc"

def setup_pinecone():
    global pc
    if pc is None:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name} (Dimension: 1024)...")
        pc.create_index(
            name=index_name,
            dimension=1024, 
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(index_name)

def ingest_repository(repo_name: str, branch: str = None):
    """
    repo_name: e.g. 'puneetkumar/my-repo' or any stranger's public repo
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token or github_token == "your_github_personal_access_token_here":
        raise ValueError("Please provide a valid GITHUB_TOKEN in your .env file.")

    if not branch:
        import requests
        try:
            headers = {"Authorization": f"Bearer {github_token}"}
            res = requests.get(f"https://api.github.com/repos/{repo_name}", headers=headers, timeout=10)
            if res.status_code == 200:
                branch = res.json().get("default_branch", "main")
            else:
                branch = "main"
        except Exception:
            branch = "main"

    print(f"Loading files from {repo_name} (branch: {branch})...")
    loader = GithubFileLoader(
        repo=repo_name,
        branch=branch,
        access_token=github_token,
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: (
            file_path.endswith((
                ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".html", ".css",
                ".cpp", ".c", ".h", ".hpp", ".cu", ".cuh", ".cs", ".java", ".rs", ".go", ".rb", ".php",
                ".swift", ".kt", ".scala", ".sol", ".ex", ".exs", ".lua", ".pl"
            )) and
            not any(ignored in file_path for ignored in ["node_modules", "dist", "build", "package-lock.json", ".min.js"])
        )
    )
    
    try:
        documents = loader.load()
        print(f"Loaded {len(documents)} files.")
    except Exception as e:
        print(f"Failed to load repository: {e}")
        raise RuntimeError(f"Failed to load repository from GitHub: {e}")

    if not documents:
        raise ValueError(f"No compatible code files found in repository '{repo_name}'.")

    # Mapping extensions to LangChain Languages for True Syntax-Aware Chunking
    extension_mapping = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".jsx": Language.JS,
        ".ts": Language.TS,
        ".tsx": Language.TS,
        ".html": Language.HTML,
        ".md": Language.MARKDOWN,
        ".cpp": Language.CPP,
        ".c": Language.C,
        ".h": Language.C,
        ".hpp": Language.CPP,
        ".cu": Language.CPP,
        ".cuh": Language.CPP,
        ".cs": Language.CSHARP,
        ".java": Language.JAVA,
        ".rs": Language.RUST,
        ".go": Language.GO,
        ".rb": Language.RUBY,
        ".php": Language.PHP,
        ".swift": Language.SWIFT,
        ".kt": Language.KOTLIN,
        ".scala": Language.SCALA,
        ".sol": Language.SOL,
        ".ex": Language.ELIXIR,
        ".exs": Language.ELIXIR,
        ".lua": Language.LUA,
        ".pl": Language.PERL
    }

    chunks = []
    generic_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for doc in documents:
        # Extract file extension
        file_path = doc.metadata.get("source", "")
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in extension_mapping:
            # Syntax-aware splitting based on the detected language
            lang = extension_mapping[ext]
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=1000, chunk_overlap=200
            )
            chunks.extend(splitter.split_documents([doc]))
        else:
            # Fallback for extensions like .css or unknown ones
            chunks.extend(generic_splitter.split_documents([doc]))

    print(f"Split into {len(chunks)} syntax-aware searchable chunks.")
    if not chunks:
        raise ValueError(f"Could not extract any chunks from repository '{repo_name}'.")

    # Setup Pinecone API Embeddings (Blazing Fast, No PyTorch)
    print("Connecting to Pinecone Inference API for embeddings...")
    from langchain_pinecone import PineconeEmbeddings
    embeddings = PineconeEmbeddings(model="multilingual-e5-large", pinecone_api_key=os.environ.get("PINECONE_API_KEY"))
    
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

    print(f"✅ Successfully ingested {repo_name} into Pinecone using API Embeddings!")

if __name__ == "__main__":
    # Example usage
    repo_to_ingest = input("Enter a public GitHub repo to ingest (e.g. 'facebook/react' or 'hwchase17/langchain'): ")
    ingest_repository(repo_to_ingest)
