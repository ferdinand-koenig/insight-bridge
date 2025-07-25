import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Dynamic configs
embedding_model_name = config.get("embedding_model_name", "sentence-transformers/sci-base")

# Grouped FAISS config with defaults fallback
faiss_config = config.get("FAISS", {})
INDEX_SAVE_PATH = faiss_config.get("index_path", "data/faiss_index")
METADATA_SAVE_PATH = faiss_config.get("metadata_path", "data/faiss_metadata.pkl")

RAW_DOCS_PATH = config.get("raw_docs_path", "data/raw")

def load_documents(path):
    """
    Load all PDF documents from the specified directory.

    Iterates over files in the given directory, loads content from PDFs using PyPDFLoader,
    and aggregates the loaded documents into a single list.

    Args:
        path (str): Path to the directory containing PDF files.

    Returns:
        list: A list of loaded documents, each representing the content of a PDF.

    Side Effects:
        Prints the total number of documents loaded.
    """
    docs = []
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, filename))
            docs.extend(loader.load())
    print(f"Loaded {len(docs)} documents")
    return docs

def chunk_documents(docs):
    """
    Split documents into smaller overlapping chunks for better retrieval.

    Uses RecursiveCharacterTextSplitter to chunk documents into pieces of
    approximately 1000 characters with 200 characters overlap between chunks.
    This helps preserve context across chunks for downstream embedding and retrieval.

    Args:
        docs (list): List of documents to split.

    Returns:
        list: A list of document chunks suitable for embedding and indexing.

    Side Effects:
        Prints the total number of chunks created.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_documents([doc]))
    print(f"{len(chunks)} Chunks")
    return chunks

def embed_chunks(chunks, model_name=None):  # model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings for each document chunk using a sentence transformer model.

    Loads the specified sentence transformer model, extracts the text content from
    each chunk, and computes their vector embeddings.

    Args:
        chunks (list): List of document chunks with text content.
        model_name (str): HuggingFace sentence-transformer model name to use for embedding.
                          Defaults to "all-MiniLM-L6-v2".

    Returns:
        tuple: A tuple containing:
            - embeddings (np.ndarray): 2D array of vector embeddings for each chunk.
            - chunks (list): The original list of chunks (unchanged).

    Side Effects:
        Displays a progress bar during embedding generation.
    """
    if model_name is None:
        model_name = config.get("embedding_model_name", "sentence-transformers/sci-base")
    model = SentenceTransformer(model_name)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, chunks

def build_faiss_index(embeddings):
    """
    Build a FAISS index from the given embeddings and save it to disk.

    FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
    of high-dimensional vectors. In a retrieval-augmented generation (RAG) pipeline,
    embeddings represent text chunks as dense vectors. FAISS indexes these vectors
    to enable fast nearest neighbor search, allowing quick retrieval of relevant
    document sections based on similarity to a query vector.

    Creating an index is essential because it organizes the embeddings in a data
    structure optimized for rapid similarity search, making retrieval scalable even
    for large document collections.

    Args:
        embeddings (np.ndarray): A 2D numpy array of shape (num_vectors, vector_dim)
                                 containing vector embeddings of document chunks.

    Returns:
        faiss.Index: The built FAISS index object for performing similarity search.

    Side Effects:
        Saves the index to disk at the location specified by INDEX_SAVE_PATH.
        Prints a confirmation message upon successful save.
    """
    dimension = embeddings.shape[1]  # vector size
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_SAVE_PATH)
    print(f"Index saved to {INDEX_SAVE_PATH}")
    return index

# Save chunk metadata (for source tracking)
def save_metadata(chunks):
    """
    Extract and save metadata from document chunks for retrieval attribution.

    Metadata typically includes the source file name, page number, and a snippet
    of the chunk's text content. This metadata helps provide context and traceability
    when displaying retrieval results.

    Args:
        chunks (list): List of document chunks with associated metadata and content.

    Side Effects:
        Saves the extracted metadata list as a pickled file to the path specified by METADATA_SAVE_PATH.
        Prints a confirmation message upon successful save.
    """
    # Save metadata (e.g. page numbers, source file) for retrieval attribution
    metadata = []
    for chunk in chunks:
        meta = {
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page", None),
            "text": chunk.page_content[:200]  # store snippet for quick preview
        }
        metadata.append(meta)
    with open(METADATA_SAVE_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {METADATA_SAVE_PATH}")


def main():
    """
    Orchestrates the document ingestion, processing, embedding, indexing, and metadata saving.

    Executes the full pipeline to:
    1. Load PDF documents from the RAW_DOCS_PATH directory.
    2. Chunk the documents into smaller overlapping text segments.
    3. Generate embeddings for each chunk.
    4. Build and save a FAISS index for similarity search over embeddings.
    5. Save associated chunk metadata for retrieval context.

    This is the main entry point for preprocessing the document corpus before running the retrieval system.
    """
    docs = load_documents(RAW_DOCS_PATH)
    chunks = chunk_documents(docs)
    embeddings, chunks = embed_chunks(chunks)
    build_faiss_index(np.array(embeddings))
    save_metadata(chunks)

if __name__ == "__main__":
    main()
