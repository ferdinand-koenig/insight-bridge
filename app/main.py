import gradio as gr
from app.llm_inference_pipeline import answer_question
import yaml


# Load config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

server_host = config.get("gradio_server", {}).get("host", "0.0.0.0")
server_port = config.get("gradio_server", {}).get("port", 7860)


def answer_question_with_status(question):
    # Show a spinning wheel immediately
    spinner_html = """
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            animation: spin 1s linear infinite;">
        </div>
        <span>Thinking… This usually takes ~1-2 minutes as no GPU resources are used.</span>
    </div>
    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """
    yield spinner_html

    # Then the real result
    result = answer_question(question)
    yield result



demo = gr.Interface(
    fn=answer_question_with_status,
    inputs=gr.Textbox(lines=3, placeholder="Ask a question about STEM field preprints"),
    outputs=gr.HTML(),
    title="InsightBridge: Semantic Q&A with LangChain & HuggingFace",
    description="Enter a question related to your document corpus. Powered by a FAISS index and HuggingFace LLM.",
    flagging_mode="never"  # disables the button entirely
)

if __name__ == "__main__":
    demo.launch(server_name=server_host, server_port=server_port)
