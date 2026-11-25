// Network-first so updates show up immediately; cached copy is used only when offline.
const CACHE = 'rl-reels-v9';
const FILES = ['./', 'index.html', 'style.css', 'app.js', 'anim.js', 'anim_caps.js', 'anim_1.js', 'anim_2.js', 'anim_3.js', 'anim_4.js', 'topics.js', 'speech.js', 'audio/index.json', 'manifest.json', 'icon.svg'];
self.addEventListener('install', e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES))); self.skipWaiting(); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))); self.clients.claim(); });
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(fetch(e.request, { cache: 'no-cache' }).then(r => { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); return r; })
    .catch(() => caches.match(e.request, { ignoreSearch: true })));
});
