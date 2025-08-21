import asyncio
import os
import pickle
import signal
import threading
from pathlib import Path

import gradio as gr
import yaml
from .cache_utils import cache_with_hit_delay
from .html_resources import spinner_html, info_box_html
from .backend_pool_singleton import SingletonBackendPool
from app import logger

CACHE_FILE = Path("cache.pkl")

# Make sure to include the html file in Docker when setting true. Consult README
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    gradio_server_config = config.get("gradio_server", {})
    # Extract backend pool config
    pool_config = {
        "mode": config.get("backend_pool", {}).get("mode", "local"),
        "backend": {
            "max_backends": config.get("backend_pool", {}).get("max_backends", 5),
            "idle_buffer": config.get("backend_pool", {}).get("idle_buffer", 3300),
            "health_timeout": config.get("backend_pool", {}).get("health_timeout", 120),
            "inference_timeout": config.get("backend_pool", {}).get("inference_timeout", 300)
        },
        "hetzner": config.get("hetzner", {})
    }
    del config

if gradio_server_config.get("use_mock_answer", False):
    # Mock for faster UI development
    with open("./dev_data/dummy_response.html", "r") as f:
        template_html = f.read()

    async def answer_question_cached(question):
        await asyncio.sleep(5)
        return template_html.replace("{{QUESTION}}", question)

else:
    @cache_with_hit_delay(maxsize=8192, hit_delay=gradio_server_config.get("hit_delay", 20),
                          return_with_meta=True, logger=logger)
    async def answer_question_cached(question: str):
        """
        Check cache first. If miss, either:
            - MODE=="local": spin up local container
            - MODE=="hetzner": spin up Hetzner VPS
        """
        backend_pool = await SingletonBackendPool.get_instance(pool_config)

        async with backend_pool.acquire() as backend:
            response = await backend.run_on_backend(question)
            return response


server_host = gradio_server_config.get("host", "0.0.0.0")
server_port = gradio_server_config.get("port", 7860)
SSL_CERT_PATH = gradio_server_config.get("ssl_cert_path")
SSL_KEY_PATH = gradio_server_config.get("ssl_key_path")

# Only enable SSL if both cert and key exist
ssl_enabled = SSL_CERT_PATH and SSL_KEY_PATH and os.path.isfile(SSL_CERT_PATH) and os.path.isfile(SSL_KEY_PATH)
if not ssl_enabled:
    logger.warning("SSL not configured or files missing, falling back to HTTP.")
    if server_port == 443:
        logger.warning("As a result, changing HTTPS port from 443 to HTTP port 80")
        server_port = 80


async def answer_question_with_status(question):
    look_ahead_result = await answer_question_cached.look_ahead(question)
    backend_pool = await SingletonBackendPool.get_instance(pool_config)
    has_ready = backend_pool.has_ready_instance()

    logger.debug(
        f"look_ahead result: {look_ahead_result}, has ready instance: {has_ready}"
    )
    # Note: There’s a potential race condition here. In the worst case, the user might see a "1–2 minutes" wait
    # message instead of the more accurate ~10 minutes if a new worker is being launched.
    # This does not affect functionality, only the displayed waiting time.

    if not (look_ahead_result == "hit" or has_ready):
        # if not cached result and has no ready instance
        # yield spinner_html("Thinking… This usually takes ~1-2 minutes as no GPU resources are used.")
        yield spinner_html(
            "Thinking… A new worker is booting (up to 10 min) 🚀. "
            "If that feels long, try one of the suggested questions. "
            "Idle workers aren’t kept running—this saves money 💰 and is sustainable 🌱. Thanks for your patience!"
            "<br><br>Did you know? Workers get a server that is more powerful to serve the request fast. "
            "When it is economically no more justified to keep it running based on startup and processing time, "
            "It gets shut down, but restarted on demand."
        )

    else:
        yield spinner_html("Thinking… This usually takes ~1-2 minutes as no GPU resources are used.")

    loop = asyncio.get_event_loop()
    start_time = loop.time()
    if pool_config.get("backend", {}).get("max_backends", 5) == 0:
        result, meta = ("Static demo. No backend available. Try another question that is possibly in cache",
                        {"from_cache": True})
    else:
        result, meta = await answer_question_cached(question)

    logger.debug(("answer:", (result[:50] + '...' if isinstance(result, str) and len(result) > 50 else result), meta))
    if type(result) is asyncio.Future:  # TODO fix whatever is happening here
        result = await result

    elapsed = loop.time() - start_time

    logger.debug(("answer:", (result[:50] + '...' if isinstance(result, str) and len(result) > 50 else result), meta))

    if meta["from_cache"]:
        result += (f"<br><i>This response was retrieved from the custom cache with "
                   f"{gradio_server_config.get('hit_delay', 20)}s delay.</i>")
    else:
        logger.info(f"Time to respond to question '{question}': {elapsed:.1f} seconds")
        result += "<br><i>This request spawned a new compute instance (Separate container).</i>"
    yield result

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# goes from app/server/main.py → /insight-bridge
favicon_file = os.path.join(BASE_DIR, "assets", "favicon.ico")

# UI definition:
with gr.Blocks(
    title="InsightBridge: Semantic Q&A with LangChain & HuggingFace",
    favicon_path=favicon_file
) as app:
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
                placeholder="Ask a question about LLM preprints of June. " + \
                            "Question will be logged for demo and improvement purposes.\n" + \
                            "⚠️ Please do not include any personal information (names, emails, etc.)."
            )
            submit_btn = gr.Button("Submit")
            info_box = gr.HTML(info_box_html)

        # RIGHT COLUMN: output box spanning right half
        with gr.Column(scale=1):
            output_html = gr.HTML()

    # Connect submit button to function
    submit_btn.click(answer_question_with_status,
                     inputs=question_input,
                     outputs=output_html,
                     concurrency_limit=20)


stop_event = threading.Event()

def cleanup():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # ---- save cache before shutdown ----
        try:
            serializable_cache = answer_question_cached.get_serializable_cache()
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(serializable_cache, f)
            logger.info(f"Serialized cache saved to {CACHE_FILE} ({len(serializable_cache)} entries).")
        except Exception as e:
            logger.exception(f"Error serializing cache: {e}")

        # ---- shut down backends ----
        backend_pool = loop.run_until_complete(SingletonBackendPool.get_instance(pool_config))
        loop.run_until_complete(backend_pool.shutdown_all_backends())
        loop.close()

        logger.info("Cleanup complete")
    except Exception as e:
        logger.exception(f"Error during backend cleanup: {e}")
    finally:
        stop_event.set()

def launch_gradio():
    app.queue()
    app.launch(
        server_name=server_host,
        server_port=server_port,
        ssl_certfile=SSL_CERT_PATH if ssl_enabled else None,
        ssl_keyfile=SSL_KEY_PATH if ssl_enabled else None
    )


def handle_shutdown_signal(signum, frame):
    logger.warning(f"Signal {signum} ({signal.Signals(signum).name}) received, performing cleanup...")
    cleanup()


if __name__ == "__main__":
    # Register signals
    signal.signal(signal.SIGINT, handle_shutdown_signal)   # Ctrl+C
    signal.signal(signal.SIGTERM, handle_shutdown_signal)  # Docker/K8s termination

    # Start Gradio in a background thread
    gr_thread = threading.Thread(target=launch_gradio, daemon=True)
    gr_thread.start()

    # Wait for Ctrl+C
    try:
        while gr_thread.is_alive() and not stop_event.is_set():
            gr_thread.join(timeout=1)
    except KeyboardInterrupt:
        logger.warning("Ctrl+C received, performing cleanup...")
        cleanup()
