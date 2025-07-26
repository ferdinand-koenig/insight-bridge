import gradio as gr
from llm_pipeline import answer_question

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=3, placeholder="Ask a question about STEM field preprints"),
    outputs="text",
    title="InsightBridge: Semantic Q&A with LangChain & HuggingFace",
    description="Enter a question related to your document corpus. Powered by a FAISS index and HuggingFace LLM."
)

if __name__ == "__main__":
    demo.launch()
