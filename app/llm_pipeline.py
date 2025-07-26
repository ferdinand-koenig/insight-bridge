import pickle
import faiss

from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

from langchain.llms import HuggingFaceHub

import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Dynamic configs
embedding_model_name = config.get("embedding_model_name", "sentence-transformers/sci-base")
top_k = config.get("top_k", 3)

# Grouped FAISS config with defaults fallback
faiss_config = config.get("FAISS", {})
FAISS_INDEX_PATH = faiss_config.get("index_path", "data/faiss_index")
FAISS_METADATA_PATH = faiss_config.get("metadata_path", "data/faiss_metadata.pkl")

# Load FAISS index and metadata
index = faiss.read_index(FAISS_INDEX_PATH)
with open(FAISS_METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

embeddings = SentenceTransformerEmbeddings(embedding_model_name)
vectorstore = FAISS(embedding_function=embeddings, index=index)

# Initialize HuggingFace LLM with deterministic output (temperature=0) to ensure consistent answers,
# and limit responses to 512 tokens to keep answers concise and efficient.
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xl",
    model_kwargs={"temperature": 0, "max_length": 512}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": top_k})
)

def answer_question(question: str) -> str:
    return  qa_chain.run(question)
