# worker_server.py
import yaml
import asyncio
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from .llm_inference_pipeline import answer_question as sync_answer_question
from app import logger

with open("config.yaml", "r") as f:
    worker_server_config = (yaml.safe_load(f)).get("worker_server", {})

app = FastAPI(title="LLM Inference Worker")

async def answer_question(question: str):
    # Run the synchronous answer_question in a separate thread
    result = await asyncio.to_thread(sync_answer_question, question)
    return result

@app.get("/infer", response_class=PlainTextResponse)
async def infer(question: str = Query(...)):
    logger.info(f"Request for Question: {question}")
    return await answer_question(question)

@app.get("/health", tags=["health"])
async def healthcheck():
    """
    Simple healthcheck endpoint to signal that the worker is up.
    """
    return JSONResponse({"status": "ok"})

if __name__ == "__main__":
    uvicorn.run(app,
                host=worker_server_config.get("host", "127.0.0.1"),
                port=worker_server_config.get("port", 8000))
