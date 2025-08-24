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
