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

    function fnv1aHash(str) {
        let hash = 0x811c9dc5;
        for (let i = 0; i < str.length; i++) {
            hash ^= str.charCodeAt(i);
            hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
            hash = hash >>> 0; // ensure 32-bit unsigned
        }
        return hash.toString(16).padStart(8, '0'); // zero-padded hex
    }

    function urlBase64ToUint8Array(base64String) {
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
        const rawData = atob(base64);
        const outputArray = new Uint8Array(rawData.length);
        for (let i = 0; i < rawData.length; ++i) outputArray[i] = rawData.charCodeAt(i);
        return outputArray;
    }

    // Register the service worker
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/service-worker.js')
            .then(reg => console.log('Service Worker registered'))
            .catch(err => console.error('SW registration failed', err));

        navigator.serviceWorker.addEventListener('controllerchange', () => {
            window.sw = navigator.serviceWorker.controller;
            console.log('Service Worker controller changed and ready');
        });
    }

    // -------------------------------
    // FETCH persistent user_id from server
    async function fetchUserId() {
        let user_id = localStorage.getItem("gradio_user_id");
        if (!user_id) {
            try {
                const res = await fetch("/api/get_user_id");
                const data = await res.json();
                user_id = data.user_id;
                localStorage.setItem("gradio_user_id", user_id);
                console.log("Fetched user_id from server:", user_id);
            } catch (err) {
                console.error("Failed to fetch user_id from server:", err);
            }
        }
        return user_id;
    }

    // -------------------------------
    // Ensure push subscription once SW is ready
    async function ensurePushSubscription() {
        const user_id = await fetchUserId();
        if (!user_id) return;

        if (!("Notification" in window)) return;
        if (Notification.permission === "default") {
            try { await Notification.requestPermission(); }
            catch (err) { console.warn("Notification permission error:", err); }
        }
        if (Notification.permission !== "granted") return;
        if (!('serviceWorker' in navigator) || !('PushManager' in window)) return;

        try {
            const registration = await navigator.serviceWorker.ready;
            const vapidKey = "<VAPID_PUBLIC_KEY_FROM_PYTHON>";
            if (!vapidKey || vapidKey.trim() === "") return;
            const applicationServerKey = urlBase64ToUint8Array(vapidKey);

            let subscription = await registration.pushManager.getSubscription();
            if (!subscription) {
                subscription = await registration.pushManager.subscribe({
                    userVisibleOnly: true,
                    applicationServerKey
                });
            }

            await fetch("/api/subscribe", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_id, subscription })
            });
            console.log("Subscribed for push:", user_id);
        } catch (err) {
            console.error("Push subscription failed:", err);
        }
    }

    if ('serviceWorker' in navigator) navigator.serviceWorker.ready.then(() => ensurePushSubscription());

    // After you have fetched user_id from server and saved it in localStorage
    async function registerQuestion(question) {
    const user_id = localStorage.getItem("gradio_user_id");
    if (!user_id || !question || question.trim().length === 0) return;

    const hashHex = fnv1aHash(question.trim()); // Use FNV-1a 32-bit

    try {
        await fetch("/api/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question_hash: hashHex, user_id })
        });
        console.log(`Registered user_id ${user_id} for question hash ${hashHex}`);
    } catch (err) {
        console.error("Failed to register question:", err);
    }
}

    // Add event listeners
    onReady("#submit-btn", (submitBtn) => {
        submitBtn.addEventListener("click", async () => {
            const questionInput = document.querySelector("textarea[placeholder*='Ask a question']");
            if (!questionInput) return;
            const question = questionInput.value;
            await registerQuestion(question);
        });
    });

    onReady("textarea[placeholder*='Ask a question']", (questionInput) => {
        questionInput.addEventListener("keydown", async (e) => {
            // Ctrl+Enter or Enter triggers registration
            if ((e.key === "Enter" && e.ctrlKey) || (e.key === "Enter" && !e.shiftKey)) {
                const question = questionInput.value;
                await registerQuestion(question);
            }
        });
    });

}
