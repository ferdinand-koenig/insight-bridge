from app import logger
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

# Get the project root (insight-bridge) relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # app/server/main -> insight-bridge
CACHE_DIR = PROJECT_ROOT / "cache"

CACHE_FILE = CACHE_DIR / "cache.pkl"
logger.info(f"Cache file: {CACHE_FILE}")

# Counter file path
COUNTER_FILE = CACHE_DIR / "counters.pkl"
logger.info(f"Counter file: {COUNTER_FILE}")

# Initialize counters
counter_lock = threading.Lock()
page_load_count = 0
question_count = 0

if COUNTER_FILE.exists():
    try:
        with open(COUNTER_FILE, "rb") as f:
            counters = pickle.load(f)
            page_load_count = counters.get("page_load_count", 0)
            question_count = counters.get("question_count", 0)
            logger.info(f"Loaded counters: page_load_count={page_load_count}, question_count={question_count}")
    except Exception as e:
        logger.warning(f"Failed to load counters: {e}")
else:
    logger.warning(f"Counter file {COUNTER_FILE} not found.")

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
    logger.info("Using mock answer")
    # Mock for faster UI development
    with open("./dev_data/dummy_response.html", "r") as f:
        template_html = f.read()

    async def answer_question_cached(question):
        await asyncio.sleep(5)
        return template_html.replace("{{QUESTION}}", question)

else:
    @cache_with_hit_delay(maxsize=8192, hit_delay=gradio_server_config.get("hit_delay", 10),
                          return_with_meta=True, logger=logger)
    async def answer_question_cached(question: str):
        """
        Check cache first. If miss, either:
            - MODE=="local": spin up local container
            - MODE=="hetzner": spin up Hetzner VPS
        """
        if pool_config.get("backend", {}).get("max_backends", 5) == 0:
            return "Static demo. No backend available. Try another question that is possibly in cache"

        backend_pool = await SingletonBackendPool.get_instance(pool_config)

        async with backend_pool.acquire() as backend:
            start = asyncio.get_event_loop().time()
            response = await backend.run_on_backend(question)
            inference_duration = asyncio.get_event_loop().time() - start
            if not backend.is_small:
                await backend_pool.record_request_time(inference_sec=inference_duration)
            return response

    # inject cache
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            answer_question_cached.load_serializable_cache(data)
            logger.info(f"Restored {len(data)} cache entries from {CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")


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
    # question counter
    global question_count
    with counter_lock:
        question_count += 1
    logger.info(f"Total questions asked: {question_count}")

    if gradio_server_config.get("use_mock_answer", False):
        logger.debug("Redirecting to mock answer")
        yield spinner_html("Thinking… This usually takes ~1-2 minutes as no GPU resources are used.")
        yield await answer_question_cached(question)
    else:
        look_ahead_result = await answer_question_cached.look_ahead(question)
        backend_pool = await SingletonBackendPool.get_instance(pool_config)
        has_ready = backend_pool.has_ready_instance()
        capacity = backend_pool.get_capacity()

        logger.debug(
            f"look_ahead result: {look_ahead_result}, has ready instance: {has_ready}"
        )
        # Note: There’s a potential race condition here. In the worst case, the user might see a "1–2 minutes" wait
        # message instead of the more accurate ~10 minutes if a new worker is being launched.
        # This does not affect functionality, only the displayed waiting time.

        if not (look_ahead_result == "hit" or has_ready):
            # if not cached result and has no ready instance
            # yield spinner_html("Thinking… This usually takes ~1-2 minutes as no GPU resources are used.")
            if capacity["current_capacity"] > 0:
                yield spinner_html(
                    "Thinking… A new worker is booting (up to 15 min) 🚀. "
                    "If that feels long, try one of the suggested questions. "
                    "Idle workers aren’t kept running—this saves money 💰 and is sustainable 🌱. Thanks for your patience!"
                    "<br><br>Did you know? Workers get a server that is more powerful to serve the request fast. "
                    "When it is economically no more justified to keep it running based on startup and processing time, "
                    "it gets shut down, but restarted on demand."
                )
            else:
                yield spinner_html(
                    f"We are currently experiencing high load. All {capacity['max_size']} workers are spawned and busy."
                    f" Your request is queued at position {capacity['queued_requests']} "
                    f"and will be processed in a couple of minutes. "
                    f"(Please note that the position cannot be updated). "
                    f"<br><br>Did you know? Workers get a server that is more powerful to serve the request fast. "
                    f"When it is economically no more justified to keep it running based on startup and processing time, "
                    f"it gets shut down, but restarted on demand. "
                    f"<br><br> To not overprovision, a limit of {capacity['max_size']} workers is set."
                    f"<br><br> Tight on time? Try one of the suggested questions or come later again."
                )

        else:
            yield spinner_html("Thinking… This usually takes ~2-3 minutes as no GPU resources are used.")

        loop = asyncio.get_event_loop()
        start_time = loop.time()

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
) as app:
    # GET method counter
    with counter_lock:
        page_load_count += 1
    logger.info(f"Page loads: {page_load_count}")

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
    submit_btn.click(
        answer_question_with_status,
        inputs=question_input,
        outputs=output_html
    )
    # Allow Ctrl+Enter (or Enter) in the textbox to submit
    question_input.submit(
        answer_question_with_status,
        inputs=question_input,
        outputs=output_html
    )


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

        # save counter
        try:
            with counter_lock:
                counters = {
                    "page_load_count": page_load_count,
                    "question_count": question_count
                }
            with open(COUNTER_FILE, "wb") as f:
                pickle.dump(counters, f)
            logger.info(f"Saved counters: page_load_count={page_load_count}, question_count={question_count}")
        except Exception as e:
            logger.exception(f"Error saving counters: {e}")

        stop_event.set()

def launch_gradio():
    app.queue(default_concurrency_limit=20)
    app.launch(
        server_name=server_host,
        server_port=server_port,
        ssl_certfile=SSL_CERT_PATH if ssl_enabled else None,
        ssl_keyfile=SSL_KEY_PATH if ssl_enabled else None,
        favicon_path=favicon_file
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
