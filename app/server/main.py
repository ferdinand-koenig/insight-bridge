import json
import time
import uuid

from app import logger
import asyncio
import os
import pickle
import signal
import threading
from pathlib import Path

from fastapi import APIRouter, Request, Security, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security.api_key import APIKeyHeader
import gradio as gr
import yaml
from pywebpush import webpush, WebPushException
from .webpage import vapid_generator
from .cache_utils import cache_with_hit_delay
from .webpage.webpage_resources import spinner_html, info_box_html, notification_js
from .backend_pool_singleton import SingletonBackendPool

# Get the project root (insight-bridge) relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # app/server/main -> insight-bridge
CACHE_DIR = PROJECT_ROOT / "cache"

CACHE_FILE = CACHE_DIR / "cache.pkl"
logger.info(f"Cache file: {CACHE_FILE}")

# Counter file path
COUNTER_FILE = CACHE_DIR / "counters.pkl"
logger.info(f"Counter file: {COUNTER_FILE}")

VAPID_FILE = CACHE_DIR / "vapid_keys.json"

if VAPID_FILE.exists():
    # Load existing keys
    with open(VAPID_FILE, "r") as f:
        vapid = json.load(f)
    logger.info(f"Loaded Vapid key pair for notifications from file: {VAPID_FILE}")
else:
    # Generate new keys and save
    vapid = vapid_generator.generate_vapid_keypair()
    with open(VAPID_FILE, "w") as f:
        json.dump(vapid, f)
    logger.info(f"Generated Vapid key pair for notifications and saved to file: {VAPID_FILE}")

VAPID_PUBLIC_KEY = vapid["public_key"]
VAPID_PRIVATE_KEY = vapid["private_key"]
notification_js = notification_js.replace("<VAPID_PUBLIC_KEY_FROM_PYTHON>", VAPID_PUBLIC_KEY)

API_KEY_NAME = "LLM_IB_VM_SPINUP_KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

AUTHORIZED_KEYS = {os.environ.get("LLM_IB_VM_SPINUP_KEY")}

def verify_api_key(api_key: str):
    if api_key not in AUTHORIZED_KEYS:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return api_key

# user_id -> subscription
user_subscriptions = {}
user_subscriptions_lock = asyncio.Lock()

async def get_subscription(user_id):
    async with user_subscriptions_lock:
        return user_subscriptions.get(user_id)

async def set_subscription(user_id, sub):
    async with user_subscriptions_lock:
        user_subscriptions[user_id] = sub


# question hash -> user_id
question_registry: dict[str, list[str]] = {}
registry_lock = asyncio.Lock()

def fnv1a_hash(s: str) -> str:
    hash_ = 0x811c9dc5
    for c in s:
        hash_ ^= ord(c)
        hash_ = (hash_ + (hash_ << 1) + (hash_ << 4) + (hash_ << 7) + (hash_ << 8) + (hash_ << 24)) & 0xFFFFFFFF
    return f"{hash_:08x}"

async def register_user_for_question(qhash: str, user_id: str):
    async with registry_lock:
        question_registry.setdefault(qhash, []).append(user_id)

async def pop_user_ids_for_question(question: str) -> list[str]:
    key = fnv1a_hash(question)
    async with registry_lock:
        return question_registry.pop(key, [])

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
            "inference_timeout": config.get("backend_pool", {}).get("inference_timeout", 300),
            "warm_backend": config.get("backend_pool", {}).get("warm_backend", None),
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


async def notify_users(question):
    user_ids = await pop_user_ids_for_question(question)
    logger.info(f"Notifying {len(user_ids)} users")
    for user_id in set(user_ids):
        sub_str = await get_subscription(user_id)
        if sub_str:
            sub = json.loads(sub_str)
            try:
                webpush(
                    subscription_info=sub,
                    data=json.dumps(
                        {"title": "Answer ready", "body": f"Your question '{question}' has been answered."}),
                    vapid_private_key=VAPID_PRIVATE_KEY,
                    vapid_claims={"sub": "mailto:you@yourdomain.com"},
                    ttl=300,  # expire after 60s if not delivered
                    headers={"Urgency": "high"}  # try to deliver immediately
                )
            except WebPushException as ex:
                logger.warning(f"Failed to send push to {user_id}: {ex}")


async def answer_question_with_status(question):
    try:
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

            logger.debug(
                ("answer:", (result[:50] + '...' if isinstance(result, str) and len(result) > 50 else result), meta))
            if type(result) is asyncio.Future:  # TODO fix whatever is happening here
                result = await result

            elapsed = loop.time() - start_time

            logger.debug(
                ("answer:", (result[:50] + '...' if isinstance(result, str) and len(result) > 50 else result), meta))

            if meta["from_cache"]:
                result += (f"<br><i>This response was retrieved from the custom cache with "
                           f"{gradio_server_config.get('hit_delay', 20)}s delay.</i>")
            else:
                logger.info(f"Time to respond to question '{question}': {elapsed:.1f} seconds")
                result += "<br><i>This request spawned a new compute instance (Separate container).</i>"

            yield result

            asyncio.create_task(notify_users(question))
    except asyncio.CancelledError:
        return


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # /insight-bridge
PWA_DIR = BASE_DIR / "pwa"

logger.debug(f"PWA_DIR: {PWA_DIR}")
SERVICE_WORKER_PATH = (PWA_DIR / "service-worker.js").resolve()
DESKTOP_ICONS_PATH = PWA_DIR / "desktop_icons"

# Favicon
favicon_file = PWA_DIR / "favicon.ico"

# UI definition:
with gr.Blocks(
        title="InsightBridge: Semantic Q&A with LangChain & HuggingFace",
        js=notification_js
) as app:
    # GET method counter
    with counter_lock:
        page_load_count += 1
    logger.info(f"Page loads: {page_load_count}")

    # Title and description
    gr.HTML("<h1 style='text-align: center;'>InsightBridge: Semantic Q&A with LangChain & HuggingFace</h1>")
    gr.Markdown(
        """
        📌 **Why this exists:** Recruiters, researchers, and enthusiasts often want to quickly understand recent LLM research without reading dozens of preprints.  
        🤖 **What this does:** Ask natural language questions about curated AI preprints (June 2025) and get concise answers using a FAISS index and a language model.  
        ⚠️ **Important:** This is a QA system, not a chatbot – each question is independent.  
        📝 **How to use:** Type your question, hit 'Submit', and view the answer. Processing may take several minutes.  
        🔔 **Optional:** Enable notifications to be alerted when your answer is ready.  
        📱 **Tip:** You can install this web app on your device for quick access.
        """
    )
    with gr.Row(elem_classes="main-row"):
        # LEFT COLUMN: input + submit + info box
        with gr.Column(scale=1):
            gr.Button("🔔 Enable Notifications", elem_id="notif-btn")
            question_input = gr.Textbox(
                lines=3,
                placeholder="Ask a question about LLM preprints of June. " + \
                            "Question will be logged for demo and improvement purposes.\n" + \
                            "⚠️ Please do not include any personal information (names, emails, etc.)."
            )
            submit_btn = gr.Button("Submit", elem_id="submit-btn")
            info_box = gr.HTML(info_box_html, elem_classes="info-box desktop-only")

        # RIGHT COLUMN: output box
        with gr.Column(scale=1, elem_classes="right-col"):
            output_html = gr.HTML(elem_id="answer-html", elem_classes="answer-box")
            info_box_mobile = gr.HTML(info_box_html, elem_classes="info-box mobile-only")
    # CSS for responsive behavior
    gr.HTML(
        """
        <style>
        .desktop-only { display: block; }
        .mobile-only { display: none; }
    
        @media (max-width: 768px) {
            .main-row { flex-direction: column !important; }
            .left-col, .right-col { width: 100% !important; }
            .desktop-only { display: none; }
            .mobile-only { display: block; }
            .right-col { display: flex; flex-direction: column-reverse; }
        }
        </style>
        """
    )
    # Connect submit button to function
    submit_btn.click(
        answer_question_with_status,
        inputs=[question_input],
        outputs=output_html
    )
    # Allow Ctrl+Enter (or Enter) in the textbox to submit
    question_input.submit(
        answer_question_with_status,
        inputs=[question_input],
        outputs=output_html
    )

stop_event = threading.Event()
cleanup_running = False


def cleanup():
    global cleanup_running
    if cleanup_running:
        logger.warning("Cleanup already running, skipping...")
        return
    cleanup_running = True
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

        # ---- stop Gradio queue ----
        try:
            app.queue().close()  # Stop accepting new tasks
        except Exception:
            pass

        # ---- cancel pending asyncio tasks safely ----
        pending = [t for t in asyncio.all_tasks(loop=loop) if not t.done()]
        if pending:
            logger.info(f"Cancelling {len(pending)} pending tasks...")
            for task in pending:
                task.cancel()

            # gather with return_exceptions=True to prevent unhandled exceptions
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

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
    api_app, _, _ = app.launch(
        server_name=server_host,
        server_port=server_port,
        ssl_certfile=SSL_CERT_PATH if ssl_enabled else None,
        ssl_keyfile=SSL_KEY_PATH if ssl_enabled else None,
        favicon_path=favicon_file,
        # allowed_paths=[str(ASSETS_DIR),
        #                str(SERVICE_WORKER_PATH),
        #                str(DESKTOP_ICONS_PATH),
        #                "/service-worker.js",
        #                "/manifest.json"],
        prevent_thread_lock=True
    )

    router = APIRouter()

    @router.get("/api/get_user_id")
    async def get_user_id():
        user_id = str(uuid.uuid4())  # server-generated UUID
        return JSONResponse({"user_id": user_id})

    @router.post("/api/subscribe")
    async def subscribe(request: Request):
        data = await request.json()
        user_id = data.get("user_id")
        subscription = data.get("subscription")
        sub_str = json.dumps(subscription, sort_keys=True)
        await set_subscription(user_id, sub_str)
        logger.info(f"New subscription for request_id {user_id}. Total: {len(user_subscriptions)}")
        return JSONResponse({"status": "subscribed"})

    @router.post("/api/register")
    async def register_question(data: dict):
        qhash = data.get("question_hash")
        user_id = data.get("user_id")
        if not qhash or not user_id:
            return JSONResponse({"error": "Missing hash or user_id"}, status_code=400)
        await register_user_for_question(qhash, user_id)
        return JSONResponse({"status": "registered", "hash": qhash})

    @router.get("/pwa/desktop_icons/{icon_name}")
    async def serve_icon(icon_name: str):
        icon_path = DESKTOP_ICONS_PATH / icon_name
        if icon_path.exists() and icon_path.is_file():
            return FileResponse(icon_path)
        return {"error": "Icon not found"}, 404

    @router.get("/pwa/manifest.json")
    async def serve_manifest():
        if (PWA_DIR / "manifest.json").exists():
            return FileResponse(PWA_DIR / "manifest.json")
        return {"error": "Manifest not found"}, 404

    @router.get("/pwa/service-worker.js")
    async def serve_service_worker():
        """
        Serve the PWA service worker at the root path.
        """
        if SERVICE_WORKER_PATH.exists() and SERVICE_WORKER_PATH.is_file():
            return FileResponse(
                SERVICE_WORKER_PATH,
                media_type="application/javascript"
            )
        return {"error": "Service worker not found"}, 404

    @router.post("/ctrl/add_auxiliary_instance")
    async def add_auxiliary_instance(amount: int = 1,
                                     api_key: str = Security(api_key_header)):
        verify_api_key(api_key)
        backend_pool = await SingletonBackendPool.get_instance(pool_config)
        spawned_instances = await backend_pool.spawn_multiple(amount, use_lock=True)
        actual_amount = len(spawned_instances)
        return {"status": "success", "requested": amount, "spawned": actual_amount}

    api_app.include_router(router)

    # Keep the thread alive
    stop_event = threading.Event()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        logger.info("Shutting down Gradio thread...")


def handle_shutdown_signal(signum, frame):
    logger.warning(f"Signal {signum} ({signal.Signals(signum).name}) received, performing cleanup...")
    cleanup()


if __name__ == "__main__":
    # Register signals
    signal.signal(signal.SIGINT, handle_shutdown_signal)  # Ctrl+C
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
