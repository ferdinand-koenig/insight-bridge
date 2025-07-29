"""
llm_inference_pipeline.py

This script loads a local language model and a FAISS vector store to create
a Retrieval-Augmented Generation (RAG) pipeline using LangChain.

It replaces remote API-based LLM inference with a local transformer model
for privacy, cost efficiency, and flexibility.

Dependencies:
- transformers
- torch
- faiss-cpu
- langchain
- langchain-community
- langchain-huggingface
- pyyaml
"""
import pickle
import faiss
import yaml

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from app.local_llm_wrapper import LocalTransformersLLM



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
# Load docstore + ID mapping
with open(FAISS_METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)
    docstore = metadata["docstore"]
    index_to_docstore_id = metadata["index_to_docstore_id"]

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Reconstruct vector store
vectorstore = FAISS(
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    index=index,
    embedding_function=embeddings
)

# # Initialize HuggingFace LLM with deterministic output (temperature=0) to ensure consistent answers,
# # and limit responses to 512 tokens to keep answers concise and efficient.
# llm = HuggingFaceHub(
#     repo_id='google/flan-t5-base', #  "google/flan-t5-xl",
#     model_kwargs={"temperature": 0, "max_length": 512}
# )
# ────────────────────────────── Initialize Local LLM ────────────────────────────── #

# Create local LLM wrapper (using flan-t5-base or compatible model)
llm = LocalTransformersLLM(  # TODO make config
    model_name="google/gemma-2b", # "google/flan-t5-large",
    max_length=1024
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": top_k})
)

def answer_question(question: str) -> str:
    """
    Run a natural language query against the vectorstore and return an answer.

    Args:
        question (str): The user’s input question.

    Returns:
        str: The generated answer from the local LLM.
    """
    return qa_chain.run(question)

if __name__ == "__main__":
    print("Running test query...")
    print(answer_question("What is your knowledge base about?"))