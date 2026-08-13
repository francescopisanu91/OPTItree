# Le fogne di Venezia — attività F (concetto 4, BME)

Pagina indipendente, vanilla HTML/CSS/JS, PWA offline-ready.

## Deploy su GitHub Pages
1. Copia l'intera cartella (con `images/` e `icons/`) nel repo, es. in `/fogne-venezia/`.
2. Attiva GitHub Pages → l'URL sarà `https://UTENTE.github.io/REPO/fogne-venezia/`.
3. Genera lo short-link bit.ly e il QR code su quell'URL per la slide.

Il service worker (`sw.js`) usa cache-first: dopo il primo caricamento la pagina
funziona anche senza wifi. Se aggiorni i file, incrementa `CACHE` in `sw.js`
(`fogne-venezia-v2`, ecc.) per forzare l'aggiornamento.

## Percorso didattico dentro la pagina
1. Intro "detective" → i 10 punti compaiono tutti assieme, pulsanti (blu = tombini T1–T4, gialli = sbocchi S1–S6), mappa a schermo pieno con pan/zoom/pinch.
2. Tocco su un tombino → flussi tratteggiati animati verso tutti gli altri 9 punti, con gocciolina di tracciante la cui velocità è proporzionale alle ore, ed etichetta del tempo. Cambiando tombino, le vecchie info svaniscono.
3. Dopo ≥3 tombini sondati → si sblocca "Mappatura completa" (matrice 10×10 con heat-map, valori in ore).
4. "Risolvi la rete" → i pallini volano dalle posizioni sulla mappa alle foglie dell'albero non radicato (UBT); compaiono gli 8 nodi interni, i 17 archi con i pesi in ore; gli archi a peso 0 restano tratteggiati. Ogni foglia mostra il suo codice sensore.

## Dati
La matrice delle distanze è esattamente quella fornita (rimappata: T1=exT3, T2=exT5,
T3=exT10, T4=exT7, S1=exT14, S2=exT13, S3=exT8, S4=exT15, S5=exT16, S6=exT12) ed è
additiva sull'albero mostrato: ogni distanza = somma dei pesi sul cammino (verificato
programmaticamente). Coppie notevoli per la discussione: T2–S2 = 0 h (stesso incrocio,
arco tratteggiato) e T3–S4 vicinissimi sulla mappa ma a 8,4 h nella rete.

Disclaimer già incluso nella pagina: posizioni puramente didattiche, non reali.
Pianta: Lodovico Ughi, 1729 (pubblico dominio).
