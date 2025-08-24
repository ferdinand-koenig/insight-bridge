// service-worker.js
// Simple service worker to handle notifications
self.addEventListener('install', (event) => {
    console.log('Service Worker installed.');
    self.skipWaiting(); // Activate immediately
});

self.addEventListener('activate', (event) => {
    console.log('Service Worker activated.');
    return self.clients.claim();
});

// Listen for messages from the page to show notifications
self.addEventListener('message', (event) => {
    const data = event.data;
    if (data && data.type === 'show-notification') {
        self.registration.showNotification(data.title, {
            body: data.body,
            icon: data.icon || '/favicon.ico',
        });
    }
});

self.addEventListener('push', (event) => {
    let data = {};
    if (event.data) {
        data = event.data.json();
    }

    const title = data.title || "InsightBridge";
    const options = {
        body: data.body || "Your answer is ready!",
        icon: data.icon || "/favicon.ico"
    };

    event.waitUntil(
        self.registration.showNotification(title, options)
    );
});


self.addEventListener('notificationclick', function(event) {
    event.notification.close();

    event.waitUntil(
        clients.matchAll({ type: 'window' }).then(function(clientList) {
            if (clientList.length > 0) {
                let client = clientList[0];
                for (let i = 0; i < clientList.length; i++) {
                    if (clientList[i].focused) {
                        client = clientList[i];
                        break;
                    }
                }
                return client.focus();
            }
            return clients.openWindow('/');
        })
    );
});
