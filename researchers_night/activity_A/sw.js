// Service worker per l'attività A — funzionamento offline dopo il primo caricamento.
// Cache "cache-first": se il file è già in cache lo serve subito, altrimenti lo scarica e lo salva.

var NOME_CACHE = "attivita-a-v1";
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
      // offline e file non in cache: fallback alla pagina principale
      return caches.match("./index.html");
    })
  );
});
