import rjsmin
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # this is app/server/webpage
NOTIF_JS_PATH = BASE_DIR / "notification.js"


user = "ferdinand"
domain = "koenix.de"
obfuscated_email = f"{user} &lt;at&gt; {domain}"

info_box_html = f"""
<b>Example Questions</b><br>
<ul>
    <li>What is your knowledge about?</li>
    <li>What are KG-TRACES?</li>
    <li>What is L3Cube-MahaEmotions?</li>
    <li>What is KG-Traces and when are they used? Explain for beginners with no knowledge of IT</li>
    <li>What are the dangers and risks of LLMs?</li>
    <li>Qu’est-ce qu’un grand modèle de langage ? Réponds en français</li>
    <li>Schreibe ein Gedicht über ein LLM auf deutsch</li>
</ul>
<br>

<b>Project Info</b><br>
<ul style="list-style-position: outside; padding-left: 1.5em;">
    <li><b>GitHub:</b> <a href="https://github.com/ferdinand-koenig/insight-bridge" target="_blank">ferdinand-koenig/insight-bridge</a> &nbsp; | &nbsp; 
        <b>LinkedIn:</b> <a href="https://www.linkedin.com/in/ferdinand-koenig" target="_blank">ferdinand-koenig</a>
    </li>
    <li><b>Email:</b> {obfuscated_email}</li>
    <li><b>An MLOps Project:</b> Production-ready pipeline for LLM inference with autoscaling, caching, and secure cloud networking.
        Learn more on the <a href="https://github.com/ferdinand-koenig/insight-bridge" target="_blank">GitHub Repository</a>.
    </li>
    <li><b>Tech Stack:</b> LangChain, HuggingFace, FAISS, Llama.cpp, Custom Caching, Docker, Gradio, Python</li>
    <li><b>Model:</b> <i>mistral-7b-instruct-v0.2.Q4_K_M.</i> with 7B parameters (quantized to 4-bit for efficiency) vs.
        much larger proprietary models like GPT-4 (exact size undisclosed, estimated in the trillions)
        or open models like Llama 3.1-405B</li>
    <li><b>Resource Efficient:</b> Runs without GPU → small model → longer responses</li>
    <li><b>High Privacy:</b> Inference is local and not send to a third party.</b></li>
    <li><b>Limitations:</b> lower quality answers, fewer documents can be processed</li>
    <li><b>Diagram:</b><br>
        <img src="assets/architecture.png" alt="Architecture Diagram" width="400">
    </li>
</ul>

<br>
<b>Disclaimer</b><br>
This project is a personal demo for portfolio purposes. All input data is anonymous.
The outputs of the LLM do not represent personal statements and may contain inaccurate or misleading information;
they should not be relied upon for critical decisions.

"""




def spinner_html(message: str) -> str:
    return f"""
    <div id="insight-spinner" style="display: flex; align-items: center; gap: 8px;">
        <div style="
            flex: none; 
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            animation: spin 1s linear infinite;">
        </div>
        <span>{message}</span>
    </div>
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """


# Read the JS code
with open(NOTIF_JS_PATH, mode="r", encoding="utf-8") as f:
    js_code = f.read()

# Minify using rjsmin
notification_js = rjsmin.jsmin(js_code)