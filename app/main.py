import gradio as gr
import yaml

USE_MOCK = True
# Make sure to include the html file in Docker when setting true. Consult README

if USE_MOCK:
    import time

    with open("./dev_data/dummy_response.html", "r") as f:
        template_html = f.read()

    def answer_question(question):
        time.sleep(3)  # simulate processing
        return template_html.replace("{{QUESTION}}", question)

else:
    from app.llm_inference_pipeline import answer_question

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

server_host = config.get("gradio_server", {}).get("host", "0.0.0.0")
server_port = config.get("gradio_server", {}).get("port", 7860)


def answer_question_with_status(question):
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
    result = answer_question(question)
    yield result

with gr.Blocks(title="InsightBridge: Semantic Q&A with LangChain & HuggingFace") as demo:
    # Title and description
    gr.HTML("<h1 style='text-align: center;'>InsightBridge: Semantic Q&A with LangChain & HuggingFace</h1>")
    gr.Markdown(
        "Enter a question related to your document corpus (Preprints of cs.CL June '25). "
        "Powered by a FAISS index and HuggingFace LLM."
    )
    with gr.Row():
        # LEFT COLUMN: input + submit + info box
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                lines=3,
                placeholder="Ask a question about LLM preprints of June"
            )
            submit_btn = gr.Button("Submit")

            user = "ferdinand.koenig"
            domain = "koenix.de"  # replace with your real domain
            obfuscated_email = f"{user} &lt;at&gt; {domain}"
            info_box = gr.HTML(
                f"""
                <b>Example Questions</b><br>
                <ul>
                    <li>What is your knowledge about?</li>
                    <li>What are KG-TRACES?</li>
                    <li>What is L3Cube-MahaEmotions?</li>
                    <li>What is KG-Traces and when are they used? Explain for beginners with no knowledge of IT</li>
                    <li>What are the dangers and risks of LLMs?</li>
                </ul>
                <br>
                
                <b>Project Info</b><br>
                <ul>
                    <li><b>GitHub:</b> <a href="https://github.com/ferdinand-koenig/insight-bridge" target="_blank">ferdinand-koenig/insight-bridge</a> &nbsp; | &nbsp; 
                        <b>LinkedIn:</b> <a href="https://www.linkedin.com/in/ferdinand-koenig" target="_blank">ferdinand-koenig</a>
                    </li>
                    <li><b>Email:</b> {obfuscated_email}</li>
                    <li><b>Tech Stack:</b> LangChain, HuggingFace, FAISS, Llama.cpp, Docker, Gradio, Python</li>
                    <li><b>Resource Efficient:</b> Runs without GPU → small model → longer responses</li>
                    <li><b>Limitations:</b> lower quality answers, fewer documents can be processed</li>
                    <li><b>Diagram:</b> <i>(Architecture diagram goes here)</i></li>
                </ul>
                """
            )

        # RIGHT COLUMN: output box spanning right half
        with gr.Column(scale=1):
            output_html = gr.HTML()

    # Connect submit button to function
    submit_btn.click(answer_question_with_status, inputs=question_input, outputs=output_html)


if __name__ == "__main__":
    demo.launch(server_name=server_host, server_port=server_port)
