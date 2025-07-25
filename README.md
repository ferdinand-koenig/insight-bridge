# InsightBridge
<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
  <img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc-sa.png" width="58"/>
</a>

**InsightBridge** is a semantic question-answering platform over complex documents using Retrieval-Augmented Generation (RAG), open-source Large Language Models (LLMs), and modern MLOps practices.

> **Currently under construction**

---

## Project Overview

This project enables natural language Q&A on technical PDFs by:

- Ingesting and chunking documents  
- Generating semantic embeddings using SentenceTransformers  
- Storing embeddings in a FAISS vector store for fast retrieval  
- (Planned) Integrating LLMs for answer generation  
- (Planned) Packaging with Docker and adding monitoring for production readiness

---

## Setup

### Requirements

- Python 3.10 (>=3.10, <4.0)  
- Poetry for dependency management

### Installation

1. Clone the repository

   ```bash
   git clone <your-repo-url>
   cd insight-bridge
	```
	
2. Install dependencies
	```bash
	poetry install
	```
	
3. Prepare your document folder

Place PDF documents inside the `data/raw/` directory.

## Usage

To run the document ingestion and indexing pipeline:

	```bash
	poetry run python ingest/ingest_and_index.py
	```
This will:

- Load PDFs from data/raw/

- Chunk documents into overlapping pieces

- Generate embeddings with SentenceTransformers

- Store embeddings in a FAISS index (data/faiss_index)

- Save chunk metadata for source tracking (data/faiss_metadata.pkl)

## Project Structure

	```
	insight-bridge/
	├── app/                  # API or UI (coming soon)
	├── data/
	│   └── raw/              # Place raw PDFs here
	├── ingest/               # Document loading and indexing scripts
	├── retriever/            # Retrieval pipeline (planned)
	├── generator/            # LLM integration (planned)
	├── notebooks/            # Prototyping and experiments
	├── Dockerfile            # Containerization (planned)
	├── pyproject.toml        # Poetry config
	└── README.md             # This file
	```
	
## Roadmap
- Week 1: Document ingestion, chunking, embedding, and vector indexing (current)

- Week 2: Build retrieval and LLM pipeline for answer generation

- Week 3: Containerize app, build API/UI, add CI/CD

- Week 4: Add MLOps monitoring, evaluation, and polish

## Contact
Ferdinand Koenig

ferdinand -at- koenix.de