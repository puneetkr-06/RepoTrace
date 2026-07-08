import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    res = embeddings.embed_query("hello")
    print(f"Success! Dimension: {len(res)}")
except Exception as e:
    print(f"Error: {e}")
