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
import time

import faiss
import yaml

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
llm = LocalTransformersLLM(  # Todo move to config
    model_name="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", # "Qwen/Qwen2.5-0.5B-Instruct",  #  "Qwen/Qwen3-0.6B",  # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_length=256,  # Limit output to 512 tokens to keep responses concise and efficient
    temperature=0.1,   # Set temperature to 0 for deterministic, focused output (no randomness)
    do_sample=False, # Disable sampling to ensure repeatable and stable answers
    no_repeat_ngram_size=3,  # prevent the model from repeating the same sequence of words (n-grams) during text gen.
    num_beams=1,     # Use beam search with 2 beams to improve answer quality by exploring multiple candidate
    # sequences Beam search slightly increases latency and memory but produces more accurate and coherent responses,
    # which is important for a QA system querying arXiv preprints where factual accuracy is critical
    stop=["\n"],
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        """
        You are a QA Assistant for Preprints on arXiv in cs.CL (NLP, LLMs), with knowledge of June 2025.
        When asked about your knowledge, respond by clearly stating your scope of expertise.
        Use the context below to answer the question clearly and concisely in one paragraph.
        If the context doesn't contain the answer, say 'I don't know.'
        You are warm and professional.
        In your answer, please **restate relevant information from the context** to support your response.
        Avoid repeating the same phrases or ideas.
        Answer the question thoroughly and clearly, using as much detail as needed,
        but without repeating phrases or introducing irrelevant personal background.
        Avoid bullet points or self-descriptions.
        
        Context:
        {context}
        
        Question: {question}
        Answer:
        """
    )
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT}
)

def clean_answer(raw_output: str) -> str:
    # Find the "Answer:" keyword and take everything after it
    if "Answer:" in raw_output:
        answer = raw_output.split("Answer:")[-1].strip()
        # Optionally stop at first double newline or special delimiter
        answer = answer.split("\n\n")[0].strip()
        return answer
    else:
        return raw_output.strip()


def answer_question(question: str) -> str:
    """
    Run a natural language query against the vectorstore and return an answer.

    Args:
        question (str): The user’s input question.

    Returns:
        str: The generated answer from the local LLM.
    """
    docs = vectorstore.as_retriever(search_kwargs={"k": top_k}).get_relevant_documents(question)
    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"--- Doc {i + 1} ---\n{doc.page_content}...\n")

    start_time = time.time()
    answer = qa_chain.run(question)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Time taken for inference: {elapsed:.2f} seconds")
    return clean_answer(answer)

if __name__ == "__main__":

    print("Running test query...")
    print(answer_question("What is your knowledge base about?"))