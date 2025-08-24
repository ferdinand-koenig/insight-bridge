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


notification_js = r"""
function () {
    const root = (window.gradioApp && window.gradioApp()) 
              || (document.querySelector("gradio-app")?.shadowRoot) 
              || document;

    const $ = (sel, r=root) => r.querySelector(sel);
    const onReady = (sel, cb, timeoutMs=10000) => {
        const start = Date.now();
        const tick = () => {
            const el = $(sel);
            if(el) return cb(el);
            if(Date.now() - start > timeoutMs) return;
            requestAnimationFrame(tick);
        };
        tick();
    };

    // Register service worker with explicit scope
    navigator.serviceWorker.register('/service-worker.js')
    .then(reg => console.log('Service Worker registered'))
    .catch(err => console.error('SW registration failed', err));

    // Ensure we always have an active controller reference
    navigator.serviceWorker.addEventListener('controllerchange', () => {
        window.sw = navigator.serviceWorker.controller;
        console.log('Service Worker controller changed and ready');
    });

    // Ask politely on first submit click
    onReady("#submit-btn", (btn) => {
        btn.addEventListener("click", () => {
            if (!("Notification" in window)) return;
            if (Notification.permission === "default") {
                const msg = "We value your time. Since the demo uses economic resources, expected request time is ~5 minutes. We would like to notify you when your result is ready.\n\nClick OK to enable notifications.";
                if (confirm(msg)) Notification.requestPermission();
            }
        }, { once: true });
    });

    // Observe answer container and send notification when spinner disappears
    onReady("#answer-html", (answer) => {
        let awaitingAnswer = false;

        const hasSpinner = () => !!answer.querySelector("#insight-spinner");
        const hasContent = () => {
            const nonSpinner = Array.from(answer.children).filter(c => c.id !== "insight-spinner");
            return nonSpinner.length > 0 || (answer.textContent?.trim().length > 0 && !hasSpinner());
        };

        const notifyReady = () => {
            const sw = window.sw || navigator.serviceWorker.controller;
            if (Notification.permission === "granted" && sw) {
                sw.postMessage({
                    type: "show-notification",
                    title: "InsightBridge",
                    body: "Your answer is ready!",
                    icon: "/favicon.ico"
                });
            }
        };

        const obs = new MutationObserver(() => {
            if (hasSpinner()) {
                awaitingAnswer = true;
                return;
            }
            if (awaitingAnswer && hasContent()) {
                awaitingAnswer = false;
                notifyReady();
            }
        });

        obs.observe(answer, { childList: true, subtree: true, characterData: true });
      });
    }
"""




