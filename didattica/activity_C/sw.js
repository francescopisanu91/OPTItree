var NOME_CACHE = "attivita-c-v1";
var FILE_DA_SALVARE = [
  "./",
  "./index.html",
  "./manifest.json"
];

self.addEventListener("install", function (evento) {
  evento.waitUntil(
    caches.open(NOME_CACHE).then(function (cache) {
      return cache.addAll(FILE_DA_SALVARE);
    })
  );
  self.skipWaiting();
});

self.addEventListener("activate", function (evento) {
  evento.waitUntil(
    caches.keys().then(function (nomiCache) {
      return Promise.all(
        nomiCache
          .filter(function (nome) { return nome !== NOME_CACHE; })
          .map(function (nome) { return caches.delete(nome); })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener("fetch", function (evento) {
  evento.respondWith(
    caches.match(evento.request).then(function (rispostaCache) {
      if (rispostaCache) {
        return rispostaCache;
      }
      return fetch(evento.request).then(function (rispostaRete) {
        return caches.open(NOME_CACHE).then(function (cache) {
          cache.put(evento.request, rispostaRete.clone());
          return rispostaRete;
        });
      });
    }).catch(function () {
      return caches.match("./index.html");
    })
  );
});
