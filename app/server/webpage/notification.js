// notification.js - NOTE: this file must define a function body (no IIFE)
function () {
    const root = (window.gradioApp && window.gradioApp()) || document.querySelector("gradio-app")?.shadowRoot || document;
    const $ = (sel, r = root) => r.querySelector(sel);

    const onReady = (sel, cb, timeoutMs = 10000) => {
        const start = Date.now();
        const tick = () => {
            const el = $(sel);
            if (el) return cb(el);
            if (Date.now() - start > timeoutMs) return;
            requestAnimationFrame(tick);
        };
        tick();
    };

    function urlBase64ToUint8Array(base64String) {
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
        const rawData = atob(base64);
        const outputArray = new Uint8Array(rawData.length);
        for (let i = 0; i < rawData.length; ++i) {
            outputArray[i] = rawData.charCodeAt(i);
        }
        return outputArray;
    }

    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/service-worker.js')
            .then(reg => console.log('Service Worker registered'))
            .catch(err => console.error('SW registration failed', err));

        navigator.serviceWorker.addEventListener('controllerchange', () => {
            window.sw = navigator.serviceWorker.controller;
            console.log('Service Worker controller changed and ready');
        });
    } else {
        console.warn('Service workers are not supported in this browser.');
    }

    try {
        if (!localStorage.getItem("gradio_user_id")) {
            if (crypto && typeof crypto.randomUUID === 'function') {
                localStorage.setItem("gradio_user_id", crypto.randomUUID());
            } else {
                const rnd = () => ((1 + Math.random()) * 0x10000 | 0).toString(16).substring(1);
                localStorage.setItem("gradio_user_id", `${rnd()}${rnd()}-${rnd()}-${rnd()}-${rnd()}-${rnd()}${rnd()}${rnd()}`);
            }
        }
    } catch (e) {
        console.warn("Could not set localStorage user id:", e);
    }
    const request_id = localStorage.getItem("gradio_user_id");

    // Placeholder to be replaced server-side by Python:
    // "<VAPID_PUBLIC_KEY_FROM_PYTHON>"

    onReady("#submit-btn", (btn) => {
        btn.addEventListener("click", async () => {
            if (!("Notification" in window)) return;
            if (Notification.permission === "default") {
                if (confirm("We would like to notify you when your result is ready. Click OK to enable notifications.")) {
                    try { await Notification.requestPermission(); } catch (err) { console.warn(err); }
                }
            }

            if (Notification.permission === "granted") {
                if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
                    console.warn("Push not supported in this browser.");
                    return;
                }
                try {
                    const registration = await navigator.serviceWorker.ready;
                    const vapidKey = "<VAPID_PUBLIC_KEY_FROM_PYTHON>";
                    if (!vapidKey || vapidKey.trim() === "") {
                        console.warn("VAPID public key not set on the page.");
                        return;
                    }
                    const applicationServerKey = urlBase64ToUint8Array(vapidKey);
                    let subscription = await registration.pushManager.getSubscription();
                    if (!subscription) {
                        subscription = await registration.pushManager.subscribe({
                            userVisibleOnly: true,
                            applicationServerKey: applicationServerKey
                        });
                    }
                    await fetch("/api/subscribe", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ request_id, subscription })
                    });
                    console.log("Subscribed for push:", request_id);
                } catch (err) {
                    console.error("Push subscription failed:", err);
                }
            }
        }, { once: true });
    });

    onReady("#answer-html", (answer) => {
        let awaitingAnswer = false;
        const hasSpinner = () => !!answer.querySelector("#insight-spinner");
        const hasContent = () => Array.from(answer.children).some(c => c.id !== "insight-spinner") || (answer.textContent?.trim().length > 0 && !hasSpinner());

        const notifyBackend = async () => {
            try {
                await fetch("/api/notify", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        request_id,
                        title: "InsightBridge",
                        body: "Your answer is ready!"
                    })
                });
            } catch (err) {
                console.error("Backend notification failed:", err);
            }
        };

        const obs = new MutationObserver(() => {
            if (hasSpinner()) { awaitingAnswer = true; return; }
            if (awaitingAnswer && hasContent()) { awaitingAnswer = false; notifyBackend(); }
        });

        obs.observe(answer, { childList: true, subtree: true, characterData: true });
    });
}
