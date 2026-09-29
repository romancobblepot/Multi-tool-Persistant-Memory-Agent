import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)
os.environ.setdefault("LANGSMITH_PROJECT", "chatbot-project-1")

model = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=1000)
crag_model = ChatGroq(model=os.getenv("CRAG_MODEL", "openai/gpt-oss-20b"), max_tokens=400, temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
SUPABASE_DB_URL = os.environ["SUPABASE_DB_URL"]
