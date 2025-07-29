import os
import numpy as np
import pickle
import yaml
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# --- Config loading and constants ---
def load_config(path="config.yaml"):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

embedding_model_name = config.get("embedding_model_name", "sentence-transformers/msmarco-distilbert-base-v4")

faiss_config = config.get("FAISS", {})
INDEX_SAVE_PATH = faiss_config.get("index_path", "data/faiss_index")
METADATA_SAVE_PATH = faiss_config.get("metadata_path", "data/faiss_metadata.pkl")

RAW_DOCS_PATH = config.get("raw_docs_path", "data/raw")

rag_config = config.get("RAG", {})
RAG_CHUNK_SIZE = rag_config.get("chunk_size", 1000)
RAG_CHUNK_OVERLAP = rag_config.get("chunk_overlap", 200)
RAG_BATCH_SIZE = rag_config.get("batch_size", 4096)

# --- Chunking ---
def chunk_document(doc):
    splitter = RecursiveCharacterTextSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)
    return splitter.split_documents([doc])

# --- Embedding ---
def embed_batch(chunks, model, show_progress_bar=False):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.embed_documents(texts)
    return np.array(embeddings), chunks

# --- Process a single PDF ---
def process_pdf_file(filepath, model, index, docs):
    loader = PyPDFLoader(filepath)
    doc = loader.load()[0]
    chunks = chunk_document(doc)

    for i in range(0, len(chunks), RAG_BATCH_SIZE):
        batch_chunks = chunks[i : i + RAG_BATCH_SIZE]
        embeddings, batch_chunks = embed_batch(batch_chunks, model, show_progress_bar=False)

        if index is None:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        for chunk in batch_chunks:
            chunk.metadata["source"] = chunk.metadata.get("source", os.path.basename(filepath))
            docs.append(chunk)

    return index, docs

# --- Save LangChain FAISS index ---
def save_langchain_faiss(faiss_store):
    faiss.write_index(faiss_store.index, INDEX_SAVE_PATH)
    print(f"\nFAISS index saved to {INDEX_SAVE_PATH}")

    with open(METADATA_SAVE_PATH, "wb") as f:
        pickle.dump({
            "docstore": faiss_store.docstore,
            "index_to_docstore_id": faiss_store.index_to_docstore_id
        }, f)
    print(f"Metadata saved to {METADATA_SAVE_PATH}")

# --- Main indexing pipeline ---
def main():
    model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    index = None
    docs = []

    pdf_files = [f for f in os.listdir(RAW_DOCS_PATH) if f.endswith(".pdf")]
    total_files = len(pdf_files)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("Processing PDFs", total=total_files)

        for filename in pdf_files:
            filepath = os.path.join(RAW_DOCS_PATH, filename)
            index, docs = process_pdf_file(filepath, model, index, docs)
            progress.update(task, advance=1)

    if index is not None:
        faiss_store = FAISS.from_documents(docs, embedding=model)
        save_langchain_faiss(faiss_store)
    else:
        print("No PDFs found or no data indexed.")

if __name__ == "__main__":
    main()
