import gradio as gr
from app.llm_inference_pipeline import answer_question
import yaml


# Load config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

server_host = config.get("gradio_server", {}).get("host", "127.0.0.1")
server_port = config.get("gradio_server", {}).get("port", 7860)

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=3, placeholder="Ask a question about STEM field preprints"),
    outputs="text",
    title="InsightBridge: Semantic Q&A with LangChain & HuggingFace",
    description="Enter a question related to your document corpus. Powered by a FAISS index and HuggingFace LLM."
)

if __name__ == "__main__":
    demo.launch(server_name=server_host, server_port=server_port)
