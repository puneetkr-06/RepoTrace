import os
from dotenv import load_dotenv
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
api_key = os.environ.get("GOOGLE_API_KEY", "")
genai.configure(api_key=api_key)

try:
    models = list(genai.list_models())
    print("Available generation models:")
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
