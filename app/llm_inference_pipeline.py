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
import os
import pickle
import time
import re

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

# LLM Config
llm_config = config.get("llm", {})

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
llm = LocalTransformersLLM(
    model_name=llm_config.get("model_name", "/model/phi-2.Q4_K_M.gguf"),
    max_length=llm_config.get("max_length", 256),
    temperature=llm_config.get("temperature", 0.1),
    do_sample=llm_config.get("do_sample", False),
    no_repeat_ngram_size=llm_config.get("no_repeat_ngram_size", 3),
    num_beams=llm_config.get("num_beams", 1),
    stop=llm_config.get("stop", ["\n"]),
    context_length=llm_config.get("context_length", 4096),
    n_threads=llm_config.get("n_threads", os.cpu_count()),
)

with open("prompt_template.yaml", "r") as f:
    prompts = yaml.safe_load(f)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompts["qa_prompt"]
)
# template=(
#     "You are a QA Assistant for Preprints on arXiv in cs.CL (NLP, LLMs), with knowledge of June 2025.\n"
#     "If asked about your own purpose or capabilities, respond clearly with your own role as an assistant.\n"
#     "Otherwise, use the context below to answer the question clearly and concisely in one paragraph.\n"
#     "If the context doesn't contain the answer, say 'I don't know.'\n"
#     "You are warm and professional.\n"
#     "In your answer, please **restate relevant information from the context** to support your response.\n"
#     "Avoid repeating the same phrases or ideas.\n"
#     "Answer the question thoroughly and clearly, using as much detail as needed,\n"
#     "but without repeating phrases or introducing irrelevant personal background.\n"
#     "Avoid bullet points or self-descriptions.\n\n"
#     "Context:\n{context}\n\n"
#     "Question: {question}\n"
#     "Answer:"
# )

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True
)

def clean_answer(raw_output: str) -> str:
    # Step 1: Remove all "Question:" lines
    no_questions = re.sub(r'Question:.*?\n', '', raw_output)

    # Step 2: Remove all "Output:" labels but keep the text after
    no_outputs = re.sub(r'Output:', '', no_questions)

    # Step 3: Remove extra newlines and merge text into one paragraph, replace multiple spaces/newlines with one space
    clean_text = re.sub(r'\s+', ' ', no_outputs).strip()

    # Cut off everything after last full stop to avoid trailing incomplete sentences
    last_period_pos = clean_text.rfind('.')
    if last_period_pos != -1:
        clean_text = clean_text[:last_period_pos + 1]

    return clean_text


def format_sources_with_excerpt_and_arxiv_link(docs):
    formatted_sources = []
    for doc in docs:
        # Get excerpt (e.g. first 200 chars or so)
        excerpt = doc.page_content[:300].strip().replace('\n', ' ') + " [...]"

        source_path = doc.metadata.get("source", None)
        arxiv_link = source_path  # default fallback

        if source_path:
            filename = os.path.basename(source_path)  # e.g. "2506.00783v1.pdf"
            paper_id = os.path.splitext(filename)[0]  # e.g. "2506.00783v1"

            # Simple arXiv ID pattern match
            if re.match(r"\d{4}\.\d{4,5}v\d+", paper_id) or re.match(r"[a-z\-]+/\d{7}v\d+", paper_id):
                arxiv_link = f"https://arxiv.org/abs/{paper_id}"

        # Format the output: excerpt + clickable arXiv link
        formatted_sources.append(
            f"{excerpt}<br><a href='{arxiv_link}' target='_blank'>📄 View Paper {paper_id}</a>"
        )
    return "<hr>".join(formatted_sources)


def answer_question(question: str) -> str:
    """
    Run a natural language query against the vectorstore and return an answer.

    Args:
        question (str): The user’s input question.

    Returns:
        str: The generated answer from the local LLM.
    """
    print("\n\n\nQuestion:", question)
    docs = vectorstore.as_retriever(search_kwargs={"k": top_k}).get_relevant_documents(question)
    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"--- Doc {i + 1} ---<doc>\n{doc.page_content}<\doc>\n")

    start_time = time.time()
    result = qa_chain.invoke({'query': question})
    end_time = time.time()
    elapsed = end_time - start_time
    print("\nQuestion:", question)
    print(f"Time taken for inference: {elapsed:.1f} seconds")
    # Build HTML for sources with clickable links
    sources_html = format_sources_with_excerpt_and_arxiv_link(result["source_documents"])
    html_string = (f"{clean_answer(result['result'])}<br>"
                   f"<hr style=\"border-top: 3px double #333;\">"
                   f"<h3>Sources (First 300 characters per document):</h3>"
                   f"<br>{sources_html}"
                   f"<br><hr style=\"border-top: 3px double #333;\">"
                   f"Time taken: {elapsed//60:.0f} minute(s) and {elapsed % 60:.1f} second(s)")
    return html_string


if __name__ == "__main__":
    print("Running test query...")
    print(answer_question("What is your knowledge base about?"))