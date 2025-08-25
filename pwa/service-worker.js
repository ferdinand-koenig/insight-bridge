// service-worker.js

self.addEventListener('install', (event) => {
    console.log('Service Worker installed.');
    self.skipWaiting(); // activate immediately
});

self.addEventListener('activate', (event) => {
    console.log('Service Worker activated.');
    event.waitUntil(self.clients.claim());
});

// Handle push events (background notifications)
self.addEventListener('push', (event) => {
    let data = {};
    try {
        if (event.data) {
            data = event.data.json();
        }
    } catch (err) {
        console.warn('Push event data is not valid JSON:', err);
        data = { title: "InsightBridge", body: event.data.text() };
    }

    const title = data.title || "InsightBridge";
    const options = {
        body: data.body || "Your answer is ready!",
        icon: data.icon || "/favicon.ico",
        badge: data.badge || "/favicon.ico",
        requireInteraction: true,
        data: data.url || "/" // store a URL to open on click
    };

    event.waitUntil(
        self.registration.showNotification(title, options)
    );
});

// Handle clicks on notifications
self.addEventListener('notificationclick', (event) => {
    event.notification.close();

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
            // If a matching client tab exists, focus it
            for (const client of clientList) {
                if ('focus' in client) return client.focus();
            }
            // Otherwise open a new one
            return clients.openWindow(event.notification.data || '/');
        })
    );
});

// Allow manual notifications from the page (optional)
self.addEventListener('message', (event) => {
    const data = event.data;
    if (data && data.type === 'show-notification') {
        self.registration.showNotification(data.title, {
            body: data.body,
            icon: data.icon || '/favicon.ico',
            requireInteraction: true
        });
    }
});
