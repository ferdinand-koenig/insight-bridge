import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import yaml
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Config loading and constants
def load_config(path="config.yaml"):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

embedding_model_name = config.get("embedding_model_name", "sentence-transformers/sci-base")

faiss_config = config.get("FAISS", {})
INDEX_SAVE_PATH = faiss_config.get("index_path", "data/faiss_index")
METADATA_SAVE_PATH = faiss_config.get("metadata_path", "data/faiss_metadata.pkl")

RAW_DOCS_PATH = config.get("raw_docs_path", "data/raw")

rag_config = config.get("RAG", {})
RAG_CHUNK_SIZE = rag_config.get("chunk_size", 1000)
RAG_CHUNK_OVERLAP = rag_config.get("chunk_overlap", 200)
RAG_BATCH_SIZE = rag_config.get("batch_size", 4096)

# Document chunking
def chunk_document(doc):
    """Split a single langchain Document into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)
    return splitter.split_documents([doc])

# Embedding batch of chunks
def embed_batch(chunks, model, show_progress_bar=False):
    """Generate embeddings for a batch of chunks."""
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
    return np.array(embeddings), chunks

# Load and process one PDF file into embeddings and metadata
def process_pdf_file(filepath, model, index):
    """
    Load one PDF file, chunk, embed, and add embeddings to FAISS index.
    Returns metadata list for all chunks from this PDF.
    """
    loader = PyPDFLoader(filepath)
    doc = loader.load()[0]
    chunks = chunk_document(doc)

    metadata_list = []

    for i in range(0, len(chunks), RAG_BATCH_SIZE):
        batch_chunks = chunks[i : i + RAG_BATCH_SIZE]
        embeddings, batch_chunks = embed_batch(batch_chunks, model, show_progress_bar=False)

        # Initialize FAISS index if not yet created
        if index is None:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        for chunk in batch_chunks:
            meta = {
                "source": chunk.metadata.get("source", os.path.basename(filepath)),
                "page": chunk.metadata.get("page", None),
                "text": chunk.page_content[:200]
            }
            metadata_list.append(meta)

    return index, metadata_list

# Save index to disk
def save_faiss_index(index):
    """Save FAISS index to disk."""
    faiss.write_index(index, INDEX_SAVE_PATH)
    print(f"\nFAISS index saved to {INDEX_SAVE_PATH}")

# Save metadata to disk
def save_metadata(metadata_list):
    """Save metadata list as pickle file."""
    with open(METADATA_SAVE_PATH, "wb") as f:
        pickle.dump(metadata_list, f)
    print(f"Metadata saved to {METADATA_SAVE_PATH}")

def main():
    """Main pipeline with Rich progress bar and clean function calls."""
    model = SentenceTransformer(embedding_model_name)
    index = None
    all_metadata = []

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
            index, metadata = process_pdf_file(filepath, model, index)
            all_metadata.extend(metadata)
            progress.update(task, advance=1)

    if index is not None:
        save_faiss_index(index)
        save_metadata(all_metadata)
    else:
        print("No PDFs found or no data indexed.")

if __name__ == "__main__":
    main()
