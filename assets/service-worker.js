// service-worker.js

// Install & activate immediately
self.addEventListener('install', (event) => {
    console.log('Service Worker installed.');
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    console.log('Service Worker activated.');
    event.waitUntil(self.clients.claim());
});

// Handle push events (background notifications)
self.addEventListener('push', (event) => {
    let data = {};
    if (event.data) {
        try {
            data = event.data.json();
        } catch (err) {
            console.warn('Push event data is not JSON', err);
        }
    }

    const title = data.title || "InsightBridge";
    const options = {
        body: data.body || "Your answer is ready!",
        icon: data.icon || "/favicon.ico",
        badge: data.badge || "/favicon.ico",
        requireInteraction: true // keeps the notification until user clicks
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
            // Focus first focused client, or open a new one
            for (const client of clientList) {
                if (client.focused) return client.focus();
            }
            if (clientList.length > 0) return clientList[0].focus();
            return clients.openWindow('/');
        })
    );
});

// Optional: handle messages from page (for immediate notifications when tab is active)
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
