/* MARScope app. The toolkit prototype's interface (structure, rendering, interactions and copy),
   driven by the local API instead of simulated scores. */
(function () {
  "use strict";
  const MS = window.MS;
  MS.addStrings(window.UF_I18N);
  /* Strings that changed from the prototype: real data, benchmark wording, no simulation. */
  MS.addStrings({
    en: {
      back: "Back",
      wasBand: "was {b}", moved: "score moved", realBand: "Where real videos usually score",
      resultH: "Result", likelihood: "Probability of fake", fakeLower: "probability of fake", pctFake: "{n}% probability of fake", pctReal: "{n}% probability of real",
      heroSub: "Open a video and press Analyze. MARS scores every clip; add models to compare. Then test what they use.",
      chooseModels: "Choose models", overall: "Overall",
      strConsistent: "The models agree", leansFake: "they lean fake", leansReal: "they lean real", strMixed: "The models disagree", strWeak: "No clear lean",
      whyMixed: "Treat this as unclear: check where the video comes from.",
      whyLeansFake: "Check where the video comes from before drawing conclusions.", whyLeansReal: "A low score can still miss a fake, so check where the video comes from.",
      whyMixed: "The result is unclear: check where the video comes from, and ask someone who can verify it.",
      mapsNote: "Highlights show the areas a model relies on for its score.", noVerdictH: "How far one score goes",
      oneView: "One model ran: a single view, nothing to compare.",
      rVery: "Very unusual for a real video", rUnusual: "Unusual for a real video", rTypical: "Typical for a real video", rLow: "Low, even for a real video",
      rFewer: "fewer than 1 in 100 real videos score this high{on}", rOneIn: "about 1 in {n} real videos score this high{on}",
      rMany: "many real videos score this high{on}", rMost: "most real videos score higher{on}", rOn: " on {m}",
      realUsually: "Real videos usually {a}–{b}%", details: "Details",
      trustLine: "{h} of {n} checks hold.", trustBreaks: "It breaks when {t}.",
      whenA1: "the face outline is hidden", whenA4: "the video is shared many times", whenA5: "fine detail is removed",
      kTrustQ: "How far to trust each model on this video.",
      posLine: "Higher than {r}% of real test videos and {f}% of fake ones",
      agreeAll: "Every model gives a probability of fake that is very unusual for a real video.",
      agreeNone: "No model gives a probability of fake that is unusual for a real video.",

      whyWeak: "The scores sit in between, with no clear lean: check where the video comes from.",
      stepQueued: "Waiting for the GPU", stepLoading: "Loading {m}", stepRunClip: "Running {m} · clip {i} of about {n}",
      stepTesting: "Testing what {m} relies on", stepNotes: "Writing the notes", failed: "The analysis stopped: {e}",
      runFrames: "Hide its top {n} frames", runAreas: "Hide its top areas",
      qNoMap: "{m} has no built-in map. An occlusion map shows which areas change its score when they are hidden.",
      runUseMap: "Compute an occlusion map", mapReady: "Occlusion map for clip {c}, shown on the frame.",
      a25Note: "A2 and A5 share one check: remove the fine pixel detail of each frame and see how much of the score goes.",
      trustWhen: "Each note asks one question per model and answers it with what was measured.",
      stHolds: "Holds", stPartly: "Partly", stBreaks: "Breaks",
      notRelied: "Not relied on", cpuNote: "Ran on the CPU",
      credFoundNote: "The file carries a Content Credentials manifest: {s}.", credUnsupported: "Not checked", credUnsupportedNote: "This file format cannot carry Content Credentials.",
      credError: "Not readable", credErrorNote: "The Content Credentials in this file could not be read.",
      posIntro: "The scores each model gave real and fake test videos (11,098 real, 12,435 fake). The same score can be normal for one model and unusual for another.",
      cardsIntro: "How well each model tells real from fake, out of 100 (a measure called AUROC): 100 means it always ranks fake videos above real ones, 50 is a coin toss. All three were trained the same way on the same videos (FF++ and GTA-V). Best in each column in bold.",
      kind_mars: "Video, full frame", kind_gend: "Frames, tuned encoder", "kind_dfd-hr": "Frames, adapted encoder",
      ev_mars: "Built-in patch (24 × 24) and frame contributions.", ev_other: "No built-in map. Compute an occlusion map in Check.",
      gpuFree: "{free} GB of GPU memory free", noGpu: "No GPU found: the detectors run on the CPU, slowly.",
      sentToLlm: "Sent to the language model",
      frameContrib: "frame contribution", expired: "This result has expired. Open the video again.",
      illustration: "", noTrust: "No trust rule fired for this video.",
      occFlat: "Hiding any one area barely changes {m}’s score on clip {c}: its score is saturated, so this map shows nothing it relies on.",
      occFlatBadge: "{m} · occlusion: no change",
      runFrames: "Hide its top {n} frames", curveCap: "Score as more of its top frames are hidden. Random is the average of 3 draws.",
      curveNeed: "Run “Hide its top frames” to draw this curve.", mapWorking: "Computing the maps…", rescoreFailed: "Scoring again failed: {e}",
      shareWorking: "Editing the clip and scoring it again with every model…", shareBig: "Scores moved by up to {n}. Sharing can wipe out the fine detail detectors look for.",
      wiDesc: "From quality 100 to 10:", wiCap: "Clip {n} · fake likelihood as JPEG quality drops, other settings as set", wiNeed: "Open this section to score the clip at each quality.",
      editFailed: "Scoring the edited clip failed: {e}",
      youHid: "{n} frames hidden. Every model’s score above is without them.", hidByOther: "These are {m}’s top frames.",
      analyzeGo: "Analyze", change: "Change",
      lvWaiting: "Waiting", lvLoading: "Loading", lvRunning: "Clip {i} of about {n}", lvTesting: "Testing what it relies on", lvDone: "Done",
      lvSoFar: "So far: {label}", lvCap: "Clip {i} · {a}–{b}", lvCombine: "Putting the results together",
      lvBlind: "Scores stay hidden until you guess.", lvMeasured: "Measured", lvQueued: "Next", lvScored: "Scored · tests next", lvSkipped: "No face found",
      lvDoA1: "Hiding the face outline", lvDoA4: "Simulating a copy shared many times", lvDoA5: "Removing fine pixel detail",
      lvStep: "{m} · {what} · clip {i} ({n} of {of})",
      lvA1: "Face outline hidden", lvA4: "Shared many times", lvA5: "Fine detail removed",
      showAll: "Show all frames", picksOf: "{m} picks", hiddenHint: "{n} frames hidden from every model’s score.",
      areasHidden: "{m} · grey squares are hidden in this frame ({n})",
      mitTotal: "{t} risks are listed in the repository ({v}).", mitDeep: "{n} of them mention deepfakes or synthetic media.",
      mitTop: "Most common among those:", mitUses: "Below are the risk types this tool’s notes link to.",
      runAreasOcc: "Hide its top 8 areas", rotatedNote: "shown upright (rotated {d}°)", startOver: "Start over",
      serverDown: "Can’t reach the MARScope server. Check that it is still running, then try again.",
      reconnecting: "Connection lost, reconnecting…",
      serverLost: "Lost the connection to the MARScope server. If it stopped, start it again, then analyze the video again.", testFailed: "The test could not run: {e}", rpRank: "Compared with real videos", rpRanges: "Real videos usually", scoresH: "Scores in context", rpTried: "What you tried",
      rpWriting: "Writing the summary with {m}…", rpFailed: "The report could not be built: {e}", rpDlFailed: "Failed",
      download: "Download", exportSession: "Session (JSON)", cardFrame: "Include the current frame (card, PDF, HTML)", frameCap: "Frame {n} (clip {c}), as the models saw it, before any edit.",
      sentNote: "The result in words, the notes and your tests. Never the video or its frames.",
      cardFooter: "MARScope is an educational tool. Scores are model outputs and can be wrong.", policyV: "policy {v}",
      chipDisagree: "Why do the models disagree?", chipAgree: "What does it mean that the models agree?", chipTests: "What do the A1–A5 results say?",
      chipCertain: "What would make this more certain?", chipVerdict: "So is it real or fake?",
      askPh: "Ask a question about this result", askOffPh: "Needs a language model",
      askOff: "Without a language model, general questions are answered from the knowledge cards; questions about this result need one (see the README).",
      askOffQ: "This question is about this result and needs a language model (set one up in .env). General questions and the suggested ones work without it.",
      chipDeepfake: "What is a deepfake?", whoCard: "From the MARScope knowledge cards, written from the cited papers.",
      askTimeout: "The language model did not answer in time. Try again.", askFailed: "The language model could not be reached.",
      askFiltered: "That answer was removed because it included a number or a conclusion about the video. Try asking another way.",
      whoFixed: "Fixed answer, built from this result.", whoLlm: "Written by {m} from this result, with numbers and conclusions about the video removed.",
      whoRefused: "MARScope leaves this decision to you. Here is what the scores do say.",
      curveDesc: "Fake likelihood for clip {c} as frames are hidden. {k} top-ranked: {a}% to {b}%. {k} random: {a}% to {r}%.",
    },
    fr: {
      back: "Retour",
      wasBand: "avant : {b}", moved: "le score a bougé", realBand: "Plage habituelle des vraies vidéos",
      resultH: "Résultat", likelihood: "Probabilité de faux", fakeLower: "probabilité de faux", pctFake: "{n} % de probabilité de faux", pctReal: "{n} % de probabilité de vrai",
      heroSub: "Ouvrez une vidéo et lancez l’analyse. MARS note chaque extrait; ajoutez des modèles pour comparer. Testez ensuite ce qu’ils utilisent.",
      chooseModels: "Choisir les modèles", overall: "En bref",
      strConsistent: "Les modèles sont d’accord", leansFake: "ils penchent vers le faux", leansReal: "ils penchent vers le vrai", strMixed: "Les modèles divergent", strWeak: "Pas de tendance nette",
      whyMixed: "Considérez ce résultat comme incertain : vérifiez d’où vient la vidéo.",
      whyLeansFake: "Vérifiez d’où vient la vidéo avant de conclure.", whyLeansReal: "Un score bas peut manquer un faux : vérifiez d’où vient la vidéo.",
      whyMixed: "Le résultat est incertain : vérifiez d’où vient la vidéo et demandez à quelqu’un qui peut la vérifier.",
      mapsNote: "Le surlignage montre les zones sur lesquelles un modèle s’appuie pour son score.", noVerdictH: "Jusqu’où va un score",
      oneView: "Un seul modèle : une seule lecture, rien à comparer.",
      rVery: "Très inhabituel pour une vraie vidéo", rUnusual: "Inhabituel pour une vraie vidéo", rTypical: "Habituel pour une vraie vidéo", rLow: "Bas, même pour une vraie vidéo",
      rFewer: "moins de 1 vraie vidéo sur 100 obtient un score aussi haut{on}", rOneIn: "environ 1 vraie vidéo sur {n} obtient un score aussi haut{on}",
      rMany: "beaucoup de vraies vidéos obtiennent un score aussi haut{on}", rMost: "la plupart des vraies vidéos obtiennent un score plus haut{on}", rOn: " avec {m}",
      realUsually: "Vraies vidéos d’habitude {a}–{b} %", details: "Détails",
      trustLine: "{h} essais sur {n} tiennent.", trustBreaks: "Le score casse quand {t}.",
      whenA1: "on cache le contour du visage", whenA4: "la vidéo est partagée de nombreuses fois", whenA5: "on retire les détails fins",
      kTrustQ: "Jusqu’où se fier à chaque modèle sur cette vidéo.",
      posLine: "Plus haut que {r} % des vraies vidéos de test et {f} % des fausses",
      agreeAll: "Chaque modèle donne une probabilité de faux très inhabituelle pour une vraie vidéo.",
      agreeNone: "Aucun modèle ne donne une probabilité de faux inhabituelle pour une vraie vidéo.",

      whyWeak: "Les scores sont entre les deux, sans tendance nette : vérifiez d’où vient la vidéo.",
      stepQueued: "En attente du GPU", stepLoading: "Chargement de {m}", stepRunClip: "{m} · extrait {i} sur environ {n}",
      stepTesting: "Tests de ce sur quoi {m} s’appuie", stepNotes: "Rédaction des notes", failed: "L’analyse s’est arrêtée : {e}",
      runFrames: "Cacher ses {n} images principales", runAreas: "Cacher ses zones principales",
      qNoMap: "{m} n’a pas de carte intégrée. Une carte d’occultation montre les zones qui changent son score quand on les cache.",
      runUseMap: "Calculer une carte d’occultation", mapReady: "Carte d’occultation de l’extrait {c}, affichée sur l’image.",
      a25Note: "A2 et A5 partagent une vérification : retirer les détails fins de chaque image et voir quelle part du score disparaît.",
      trustWhen: "Chaque note pose une question par modèle et y répond avec ce qui a été mesuré.",
      stHolds: "Tient", stPartly: "En partie", stBreaks: "Casse",
      notRelied: "Non utilisée", cpuNote: "Exécuté sur le processeur",
      credFoundNote: "Le fichier porte un manifeste Content Credentials : {s}.", credUnsupported: "Non vérifiés", credUnsupportedNote: "Ce format de fichier ne peut pas porter de Content Credentials.",
      credError: "Illisibles", credErrorNote: "Les Content Credentials de ce fichier n’ont pas pu être lus.",
      posIntro: "Les scores que chaque modèle a donnés aux vidéos de test (11 098 vraies, 12 435 fausses). Un même score peut être normal pour un modèle et inhabituel pour un autre.",
      cardsIntro: "À quel point chaque modèle distingue le vrai du faux, sur 100 (une mesure appelée AUROC) : 100, il classe toujours les fausses vidéos au-dessus des vraies ; 50, c’est pile ou face. Les trois ont été entraînés de la même façon sur les mêmes vidéos (FF++ et GTA-V). Meilleur de chaque colonne en gras.",
      kind_mars: "Vidéo, image entière", kind_gend: "Images, encodeur ajusté", "kind_dfd-hr": "Images, encodeur adapté",
      ev_mars: "Contributions intégrées par zone (24 × 24) et par image.", ev_other: "Pas de carte intégrée. Calculez une carte d’occultation dans Vérifier.",
      gpuFree: "{free} Go de mémoire GPU libres", noGpu: "Aucun GPU : les détecteurs tournent sur le processeur, lentement.",
      sentToLlm: "Envoyé au modèle de langage",
      frameContrib: "contribution de l’image", expired: "Ce résultat a expiré. Rouvrez la vidéo.",
      illustration: "", noTrust: "Aucune règle de confiance déclenchée pour cette vidéo.",
      occFlat: "Cacher une zone ne change presque pas le score de {m} sur l’extrait {c} : son score est saturé, cette carte ne montre donc rien sur quoi il s’appuie.",
      occFlatBadge: "{m} · occultation : aucun changement",
      runFrames: "Cacher ses {n} images principales", curveCap: "Score à mesure qu’on cache ses images principales. Au hasard : moyenne de 3 tirages.",
      curveNeed: "Lancez « Cacher ses images principales » pour tracer cette courbe.", mapWorking: "Calcul des cartes…", rescoreFailed: "Le nouveau calcul a échoué : {e}",
      shareWorking: "Modification de l’extrait et nouveau calcul avec chaque modèle…", shareBig: "Les scores ont bougé jusqu’à {n}. Le partage peut effacer les détails fins que cherchent les détecteurs.",
      wiDesc: "De la qualité 100 à 10 :", wiCap: "Extrait {n} · probabilité de faux selon la qualité JPEG, autres réglages tels quels", wiNeed: "Ouvrez cette section pour calculer l’extrait à chaque qualité.",
      editFailed: "Le calcul de l’extrait modifié a échoué : {e}",
      youHid: "{n} images cachées. Les scores de chaque modèle ci-dessus sont calculés sans elles.", hidByOther: "Ce sont les images principales de {m}.",
      analyzeGo: "Analyser", change: "Changer",
      lvWaiting: "En attente", lvLoading: "Chargement", lvRunning: "Extrait {i} sur environ {n}", lvTesting: "Tests de ce sur quoi il s’appuie", lvDone: "Terminé",
      lvSoFar: "Pour l’instant : {label}", lvCap: "Extrait {i} · {a}–{b}", lvCombine: "Assemblage des résultats",
      lvBlind: "Les scores restent cachés jusqu’à votre réponse.", lvMeasured: "Mesuré", lvQueued: "Ensuite", lvScored: "Évalué · tests ensuite", lvSkipped: "Aucun visage trouvé",
      lvDoA1: "On cache le contour du visage", lvDoA4: "On simule une copie partagée de nombreuses fois", lvDoA5: "On retire les détails fins",
      lvStep: "{m} · {what} · extrait {i} ({n} sur {of})",
      lvA1: "Contour du visage caché", lvA4: "Partagée de nombreuses fois", lvA5: "Détails fins retirés",
      showAll: "Afficher toutes les images", picksOf: "Choix de {m}", hiddenHint: "{n} images retirées du score de chaque modèle.",
      areasHidden: "{m} · les carrés gris sont cachés dans cette image ({n})",
      mitTotal: "Le répertoire recense {t} risques ({v}).", mitDeep: "{n} d’entre eux mentionnent les hypertrucages ou les médias synthétiques.",
      mitTop: "Les plus fréquents parmi eux :", mitUses: "Ci-dessous, les types de risque auxquels renvoient les notes de cet outil.",
      runAreasOcc: "Cacher ses 8 zones principales", rotatedNote: "affichée droite (tournée de {d}°)", startOver: "Recommencer",
      serverDown: "Impossible de joindre le serveur MARScope. Vérifiez qu’il tourne toujours, puis réessayez.",
      reconnecting: "Connexion perdue, reconnexion…",
      serverLost: "Connexion au serveur MARScope perdue. S’il s’est arrêté, relancez-le, puis analysez de nouveau la vidéo.", testFailed: "L’essai n’a pas pu tourner : {e}", rpRank: "Par rapport aux vraies vidéos", rpRanges: "Vraies vidéos, d’habitude", scoresH: "Les scores en contexte", rpTried: "Ce que vous avez essayé",
      rpWriting: "Rédaction du résumé avec {m}…", rpFailed: "Impossible de produire le rapport : {e}", rpDlFailed: "Échec",
      download: "Télécharger", exportSession: "Session (JSON)", cardFrame: "Inclure l’image en cours (fiche, PDF, HTML)", frameCap: "Image {n} (extrait {c}), telle que les modèles l’ont vue, avant toute modification.",
      sentNote: "Le résultat en mots, les notes et vos essais. Jamais la vidéo ni ses images.",
      cardFooter: "MARScope est un outil pédagogique. Les scores sont des sorties de modèles et peuvent se tromper.", policyV: "politique {v}",
      chipDisagree: "Pourquoi les modèles divergent-ils?", chipAgree: "Que veut dire l’accord des modèles?", chipTests: "Que disent les résultats A1–A5?",
      chipCertain: "Qu’est-ce qui rendrait ce résultat plus sûr?", chipVerdict: "Alors, vrai ou faux?",
      askPh: "Posez une question sur ce résultat", askOffPh: "Demande un modèle de langage",
      askOff: "Sans modèle de langage, les questions générales sont répondues par les fiches de connaissances; les questions sur ce résultat en demandent un (voir le README).",
      askOffQ: "Cette question porte sur ce résultat et demande un modèle de langage (à configurer dans .env). Les questions générales et suggérées fonctionnent sans.",
      chipDeepfake: "Qu’est-ce qu’un hypertrucage?", whoCard: "Tiré des fiches de connaissances de MARScope, rédigées à partir des articles cités.",
      askTimeout: "Le modèle de langage n’a pas répondu à temps. Réessayez.", askFailed: "Le modèle de langage est injoignable.",
      askFiltered: "Cette réponse a été retirée car elle contenait un nombre ou une conclusion sur la vidéo. Reformulez la question.",
      whoFixed: "Réponse fixe, construite à partir de ce résultat.", whoLlm: "Rédigé par {m} à partir de ce résultat, sans les nombres ni les conclusions sur la vidéo.",
      whoRefused: "MARScope vous laisse cette décision. Voici ce que disent les scores.",
      curveDesc: "Probabilité de faux de l’extrait {c} à mesure qu’on cache des images. {k} mieux classées : {a} % à {b} %. {k} au hasard : {a} % à {r} %.",
    },
  });
  const T = MS.t, esc = MS.esc;
  const LANG = () => MS.lang();
  const L = (o) => (o && typeof o === "object" ? o[LANG()] || o.en : o);
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const pct = (v) => Math.round(v * 100);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const mean = (xs) => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);
  const t2s = (sec) => Math.floor(sec / 60) + ":" + String(Math.floor(sec % 60)).padStart(2, "0");
  const fmtN = (n, d) => Number(n).toLocaleString(LANG() === "fr" ? "fr-CA" : "en-CA", { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 });

  /* --------------------------------------------------------------- content */
  const ASSUMPTIONS = [
    { code: "A1", short: { en: "Boundary", fr: "Frontière" }, plain: { en: "An edited region leaves a seam where it meets the real frame.", fr: "Une zone modifiée laisse une couture là où elle rejoint l’image réelle." }, breaks: { en: "The whole video is generated, so there is no seam.", fr: "Toute la vidéo est générée : il n’y a pas de couture." } },
    { code: "A2", short: { en: "Fingerprint", fr: "Empreinte" }, plain: { en: "Each generator leaves stable, recognizable traces.", fr: "Chaque générateur laisse des traces stables et reconnaissables." }, breaks: { en: "A new generator, or heavy processing, changes the traces.", fr: "Un nouveau générateur, ou un traitement lourd, change les traces." } },
    { code: "A3", short: { en: "Like the training data", fr: "Proche des données d’entraînement" }, plain: { en: "Test videos resemble what the model learned from.", fr: "Les vidéos testées ressemblent à ce que le modèle a appris." }, breaks: { en: "New generators, no faces, unusual quality.", fr: "Nouveaux générateurs, pas de visage, qualité inhabituelle." } },
    { code: "A4", short: { en: "Survives processing", fr: "Résiste au traitement" }, plain: { en: "Traces survive compression, resizing and re-uploads.", fr: "Les traces résistent à la compression, au redimensionnement et aux republications." }, breaks: { en: "The video was shared and re-encoded many times.", fr: "La vidéo a été partagée et réencodée de nombreuses fois." } },
    { code: "A5", short: { en: "Low-level clues", fr: "Indices de bas niveau" }, plain: { en: "Pixel-level traces alone reveal a fake.", fr: "Les traces au niveau des pixels suffisent à révéler un faux." }, breaks: { en: "The fake is pixel-clean and only meaning gives it away (lighting, physics, context).", fr: "Le faux est propre au pixel près ; seul le sens le trahit (lumière, physique, contexte)." } },
  ];
  const GEN_NAMES = {
    Sora: "Sora", MorphStudio: "Morph Studio", Gen2: "Runway Gen2", HotShot: "HotShot", Lavie: "Lavie", Show_1: "Show-1", MoonValley: "Moon Valley", Crafter: "Crafter", ModelScope: "ModelScope", WildScrape: "Wild Scrape",
    Grok_Imagine: "Grok Imagine", "Kling_3.0": "Kling 3.0", "Veo_3.1": "Veo 3.1", "Wan_2.6": "Wan 2.6", "HunyuanVideo_1.5_14b_i2v": "HV1.5 I2V", "HunyuanVideo_1.5_14b_t2v": "HV1.5 T2V",
    "Wan_2.2_14b_i2v": "Wan2.2 I2V", "Wan_2.2_14b_t2v": "Wan2.2 T2V", "LTX_2.3_distilled_i2v": "LTX2.3 I2V", "LTX_2.3_distilled_t2v": "LTX2.3 T2V",
  };
  const GEN_ORDER = { genvideo: ["Sora", "MorphStudio", "Gen2", "HotShot", "Lavie", "Show_1", "MoonValley", "Crafter", "ModelScope", "WildScrape"], df26: ["Grok_Imagine", "Kling_3.0", "Veo_3.1", "Wan_2.6", "HunyuanVideo_1.5_14b_i2v", "HunyuanVideo_1.5_14b_t2v", "Wan_2.2_14b_i2v", "Wan_2.2_14b_t2v", "LTX_2.3_distilled_i2v", "LTX_2.3_distilled_t2v"] };
  const FALLBACK_MODELS = [{ id: "mars", name: "MARS" }, { id: "dfd-hr", name: "DFD-HR" }, { id: "gend", name: "GenD" }];
  const SHOWN = 16;

  /* ----------------------------------------------------------------- state */
  const S = {
    catalog: FALLBACK_MODELS, health: null, refs: [], bench: null,
    picked: { mars: true, "dfd-hr": false, gend: false }, guessFirst: true,
    file: null, job: null, R: null, src: null, models: [], focus: 0, clip: 0, frame: 0, zoom: "video",
    overlay: true, guessing: false, guess: null, playing: null, prevalence: 1000, jur: "all",
    maps: {}, occlusion: {}, probes: {}, sweeps: {}, source: null,
    test: "frames", removed: [], rescored: null, rescoring: false, method: {}, agree: false, mapBusy: false,
    edits: { jpeg: 100, scale: 100, blur: 0, noise: 0 }, edited: null, editBusy: false, compare: true, qsweeps: {},
  };
  /* Every call to the local server. A network failure means the server stopped (or the machine is
     offline); a 404 on this job means it expired: both get a plain message instead of a broken page. */
  const api = async (path, body) => {
    const opts = body === undefined ? { cache: "no-store" } : body instanceof FormData ? { method: "POST", body } : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
    let res;
    try { res = await fetch(path, opts); } catch (_) { notice(T("serverDown")); throw new Error(T("serverDown")); }
    const data = await res.json().catch(() => ({}));
    if (res.status === 404 && S.job && path.startsWith(`jobs/${S.job}`)) { notice(T("expired")); throw new Error(T("expired")); }
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  };
  /* One banner on the result, with a way back to the start. */
  function notice(text) {
    if (!S.R || V.analyze.hidden) return;
    let el = $("#notice");
    if (!el) {
      el = document.createElement("div"); el.id = "notice"; el.className = "notice"; el.setAttribute("role", "alert");
      V.analyze.prepend(el);
    }
    el.innerHTML = `<span>${esc(text)}</span><button type="button" class="btn btn-small" data-nav>${esc(T("startOver"))}</button>`;
  }

  /* --------------------------------------------------------- video helpers */
  const stride = () => S.R.video.frame_stride || 2;
  const fps = () => S.R.video.fps || 30;
  const clipOf = (mid, c) => S.R.models[mid].clips[c];
  const refClip = (c) => S.R.models[S.models[0].id].clips[c];
  /* A clip's real frames are split into SHOWN equal runs (as the server does); each run is shown by its middle frame,
     and the × on a shown frame removes its whole run. */
  const positions = (c) => { const n = refClip(c).n_valid, k = Math.min(SHOWN, n); return Array.from({ length: k }, (_, i) => { const a = Math.floor((i * n) / k), b = Math.floor(((i + 1) * n) / k); return a + Math.floor((b - a - 1) / 2); }); };
  const N = (c) => positions(c).length;
  const srcFrame = (c, i) => refClip(c).first_frame + positions(c)[clamp(i, 0, N(c) - 1)] * stride();
  const fTime = (c, i) => srcFrame(c, i) / fps();
  const total = () => S.R.video.n_clips ? Array.from({ length: S.src.nClips }, (_, c) => N(c)).reduce((a, b) => a + b, 0) : 0;
  const gIndex = () => { let g = 0; for (let c = 0; c < S.clip; c++) g += N(c); return g + S.frame; };
  const frameUrl = (idx) => `jobs/${S.job}/frames/${idx}.jpg`;
  const images = new Map();
  function imageAt(url) {
    if (!images.has(url)) images.set(url, new Promise((ok) => { const img = new Image(); img.decoding = "async"; img.onload = () => ok(img); img.onerror = () => ok(null); img.src = url; }));
    return images.get(url);
  }
  const image = (idx) => imageAt(frameUrl(idx));
  const editedUrl = (idx) => `jobs/${S.job}/edited/${idx}.jpg?` + new URLSearchParams(S.edits).toString();

  /* ---------------------------------------------------------------- scores */
  const isEdited = () => S.edits.jpeg < 100 || S.edits.scale < 100 || S.edits.blur > 0 || S.edits.noise > 0;
  const clipNow = (m, c) => (c !== S.clip ? clipOf(m.id, c).fake : isEdited() && S.edited ? S.edited[m.id] : S.removed.length && S.rescored ? S.rescored[m.id] : clipOf(m.id, c).fake);
  const videoNow = (m) => ((S.removed.length && S.rescored) || (isEdited() && S.edited) ? mean(S.R.models[m.id].clips.map((_, c) => clipNow(m, c))) : S.R.models[m.id].fake);
  /* Rank of a score among the model's benchmark videos (as marscope.analysis.ranges.rank), so re-scored values are placed too. */
  function rankIn(v, p) {
    if (!p || !p.length) return null;
    if (v <= p[0]) return 0; if (v >= p[p.length - 1]) return 100;
    let hi = p.findIndex((x) => x > v); const lo = hi - 1, span = p[hi] - p[lo];
    return lo + (span ? (v - p[lo]) / span : 0);
  }
  function placeOf(m, v) {
    const r = S.ranges[m.id], ctx = S.R.models[m.id].context;
    if (!r || v === S.R.models[m.id].fake) return { real: Math.round(ctx.real_rank), fake: Math.round(ctx.fake_rank) };
    return { real: Math.round(rankIn(v, r.real)), fake: Math.round(rankIn(v, r.fake)) };
  }
  /* How unusual a score is for a real video, in words and odds (as marscope.analysis.ranges.rarity). */
  function rarity(r, name) {
    const on = name ? T("rOn", { m: name }) : "", above = 100 - r, n = Math.round(100 / Math.max(above, 1e-6));
    if (r >= 95) return { label: T("rVery"), odds: above < 1 ? T("rFewer", { on }) : T("rOneIn", { n, on }) };
    if (r >= 75) return { label: T("rUnusual"), odds: T("rOneIn", { n, on }) };
    if (r >= 25) return { label: T("rTypical"), odds: T("rMany", { on }) };
    return { label: T("rLow"), odds: T("rMost", { on }) };
  }
  const sameSet = (x, y) => x.length === y.length && x.slice().sort().join() === y.slice().sort().join();
  /* Cells are coloured against the model's own benchmark ranges (never raw 0.5): orange above its real
     range, blue below its real median, neutral between. */
  function cellBg(m, v) {
    const ctx = S.R.models[m.id].context, [rlo, rhi] = ctx.real_range, fhi = ctx.fake_range[1];
    if (v > rhi) return `rgba(var(--fake-rgb), ${clamp(0.15 + 0.7 * (v - rhi) / Math.max(1e-6, fhi - rhi), 0.15, 0.85).toFixed(3)})`;
    const mid = (rlo + rhi) / 2;
    if (v < mid) return `rgba(var(--real-rgb), ${clamp(0.12 + 0.45 * (mid - v) / Math.max(1e-6, mid - rlo), 0.12, 0.57).toFixed(3)})`;
    return "var(--fill)";
  }
  const signedBg = (u) => (u >= 0 ? `rgba(var(--fake-rgb), ${(0.1 + 0.75 * clamp(u, 0, 1)).toFixed(3)})` : `rgba(var(--real-rgb), ${(0.1 + 0.5 * clamp(-u, 0, 1)).toFixed(3)})`);

  /* -------------------------------------------------------------- routing */
  const V = { start: $("#view-start"), run: $("#view-run"), analyze: $("#view-analyze") };
  function show(name, focus) {
    Object.entries(V).forEach(([k, el]) => { el.hidden = k !== name; });
    document.body.dataset.view = name;
    if (focus !== false) { const h = $("h1", V[name]); if (h) { h.tabIndex = -1; h.focus({ preventScroll: true }); } }
    scrollTo({ top: 0, behavior: RM ? "auto" : "smooth" });
  }
  function leave() {
    stopPlay();
    const gone = $("#notice"); if (gone) gone.remove();
    if (S.source) S.source.close();
    if (S.job) fetch(`jobs/${S.job}`, { method: "DELETE" }).catch(() => {});
    S.job = null; S.R = null;
    show("start");
  }
  document.addEventListener("click", (e) => { if (e.target.closest("[data-nav]")) { e.preventDefault(); leave(); } });

  /* ----------------------------------------------------------------- start */
  function renderStart() {
    const check = '<svg viewBox="0 0 24 24" class="ic" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7"/></svg>';
    const order = ["mars", "dfd-hr", "gend"];
    const models = order.map((id) => S.catalog.find((m) => m.id === id) || FALLBACK_MODELS.find((m) => m.id === id));
    $("#pick").setAttribute("aria-label", T("detectors"));
    $("#pick").innerHTML = models.map((m) => `<button type="button" data-pick="${m.id}" aria-pressed="${!!S.picked[m.id]}" title="${m.peak_vram_gb ? esc(`${m.peak_vram_gb} GB · ${m.seconds_per_clip} s / clip`) : ""}">${check}${esc(m.name)}</button>`).join("");
    const h = S.health;
    $("#hw").textContent = h ? (h.gpu ? T("gpuFree", { free: fmtN(h.free_vram_gb, 1) }) : T("noGpu")) : "";
  }
  $("#pick").addEventListener("click", (e) => {
    const b = e.target.closest("[data-pick]"); if (!b) return;
    const id = b.dataset.pick, on = !S.picked[id];
    if (!on && Object.values(S.picked).filter(Boolean).length === 1) return;
    S.picked[id] = on; b.setAttribute("aria-pressed", String(on));
    if (on) fetch(`models/${id}/load`, { method: "POST" }).catch(() => {});
  });
  $("#opt-guess").addEventListener("change", (e) => { S.guessFirst = e.target.checked; });
  $("#choose").addEventListener("click", () => $("#file").click());
  const err = (m) => { $("#file-error").textContent = m; $("#file-error").hidden = !m; };
  function takeFile(f) {
    err(""); if (!f) return;
    if (!/^video\//.test(f.type) && !/\.(mp4|mov|webm)$/i.test(f.name)) return err(T("errType"));
    const url = URL.createObjectURL(f), v = document.createElement("video");
    v.preload = "metadata"; v.muted = true; v.src = url;
    v.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      if (isFinite(v.duration) && v.duration > 120.5) return err(T("errLong", { t: t2s(v.duration) }));
      S.file = { f, duration: isFinite(v.duration) ? v.duration : null }; renderPicked();
    };
    v.onerror = () => { URL.revokeObjectURL(url); S.file = { f, duration: null }; renderPicked(); }; // the server decodes more formats than the browser
  }
  /* The chosen video waits on the start screen, so the detectors can still be changed before Analyze. */
  let pickedUrl = null;
  function renderPicked() {
    const has = !!S.file;
    $("#choose").hidden = has; $("#drop-help").hidden = has; $("#picked").hidden = !has; $("#go").hidden = !has;
    $(".drop .glyph").style.display = has ? "none" : "";
    if (!has) return;
    if (pickedUrl) URL.revokeObjectURL(pickedUrl);
    pickedUrl = URL.createObjectURL(S.file.f);
    const pv = $("#picked-v"); pv.src = pickedUrl; pv.onloadedmetadata = () => { pv.currentTime = Math.min(1, (pv.duration || 2) / 2); };
    $("#picked-name").textContent = S.file.f.name;
    $("#picked-meta").textContent = [S.file.duration ? t2s(S.file.duration) : "", fmtN(S.file.f.size / 1e6, 1) + " MB"].filter(Boolean).join(" · ");
    $("#go").focus();
  }
  $("#picked-change").addEventListener("click", () => $("#file").click());
  $("#go").addEventListener("click", () => { if (S.file) start(); });
  $("#file").addEventListener("change", (e) => { takeFile(e.target.files[0]); e.target.value = ""; });
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((k) => drop.addEventListener(k, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((k) => drop.addEventListener(k, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
  drop.addEventListener("drop", (e) => takeFile(e.dataTransfer.files[0]));

  /* ------------------------------------------------------------------- run */
  async function start() {
    const picked = ["mars", "dfd-hr", "gend"].filter((id) => S.picked[id]);
    const nameOf = (id) => (S.catalog.find((m) => m.id === id) || { name: id }).name;
    $("#run-h").textContent = S.file.f.name;
    $("#run-step").textContent = T("stepCheck");
    $("#run-back").hidden = true;
    const bar = $("#run-bar"), pb = $(".progress");
    bar.style.width = "0";
    V.run.classList.remove("combining");
    const live = liveCards(picked, S.file.duration ? Math.max(1, Math.ceil((S.file.duration * 30) / 2 / 64)) : 4);
    show("run");
    const form = new FormData();
    form.append("video", S.file.f); form.append("models", picked.join(",")); form.append("jurisdiction", "all"); form.append("language", LANG());
    try { S.job = (await api("jobs", form)).id; } catch (e) { show("start"); err(e.message); return; }
    images.clear(); S.maps = {}; S.occlusion = {}; S.probes = {}; S.sweeps = {}; S.removed = []; S.rescored = null; S.method = {}; S.agree = false;
    S.edits = { jpeg: 100, scale: 100, blur: 0, noise: 0 }; S.edited = null; S.qsweeps = {}; S.compare = true;
    const est = S.file.duration ? Math.max(1, Math.ceil((S.file.duration * 30) / 2 / 64)) : 4;
    const done = Object.fromEntries(picked.map((id) => [id, 0]));
    const setBar = (p) => { bar.style.width = clamp(p, 0, 100) + "%"; pb.setAttribute("aria-valuenow", String(Math.round(p))); };
    if (S.source) S.source.close();
    const src = new EventSource(`jobs/${S.job}/events`);
    S.source = src;
    /* The browser retries a dropped stream on its own; if the server is gone (or the job with it), stop and say so. */
    let lost = null;
    const giveUp = () => { src.close(); lost = null; $("#run-step").textContent = T("serverLost"); $("#run-back").hidden = false; };
    src.onerror = () => {
      if (src.readyState === EventSource.CLOSED) { giveUp(); return; }
      $("#run-step").textContent = T("reconnecting");
      if (!lost) lost = setTimeout(giveUp, 15000);
    };
    src.onmessage = (msg) => {
      if (lost) { clearTimeout(lost); lost = null; }
      const ev = JSON.parse(msg.data);
      if (ev.type === "clip") {
        done[ev.model] = ev.done;
        const tot = Math.max(est, ev.total);
        setBar(8 + 82 * Object.values(done).reduce((a, b) => a + b, 0) / (tot * picked.length));
        $("#run-step").textContent = T("stepRunClip", { m: nameOf(ev.model), i: ev.done, n: tot });
        live.clip(ev, tot);
        return;
      }
      if (ev.type === "test") { live.test(ev); return; }
      if (ev.type === "step") { live.step(ev); return; }
      if (ev.type !== "stage") return;
      live.stage(ev);
      const step = { queued: T("stepQueued"), prechecks: T("stepCheck"), loading: T("stepLoading", { m: nameOf(ev.model) }), testing: T("stepTesting", { m: nameOf(ev.model) }), policy: T("stepNotes") }[ev.stage];
      if (step) $("#run-step").textContent = step;
      if (ev.stage === "prechecks") { setBar(4); if (ev.fps) S.file.fps = ev.fps; }
      if (ev.stage === "testing") setBar(92);
      if (ev.stage === "policy") setBar(97);
      if (ev.stage === "done") {
        src.close(); setBar(100); $("#run-step").textContent = T("lvCombine"); V.run.classList.add("combining");
        setTimeout(open, RM ? 0 : 650);
      }
      if (ev.stage === "failed") { src.close(); $("#run-step").textContent = T("failed", { e: ev.error || "?" }); $("#run-back").hidden = false; }
    };
  }
  $("#run-back").addEventListener("click", leave);

  /* Live view while the detectors run: one card per model fills in clip by clip (scores coloured
     against that model's own real range), then the assumption tests arrive, then all cards merge
     into the result. */
  function liveCards(picked, guess) {
    const cat = (id) => S.catalog.find((m) => m.id === id) || FALLBACK_MODELS.find((m) => m.id === id) || { id, name: id };
    const st = Object.fromEntries(picked.map((id) => [id, { clips: [], high: null }]));
    const blind = S.guessFirst; // "Guess first": show progress, never a score, until the guess
    $("#live-img").removeAttribute("src"); $("#live-img").hidden = true; $("#live-cap").textContent = "";
    $("#live-models").innerHTML = picked.map((id) => `<article class="lm" data-m="${id}"><div class="lm-top"><b>${esc(cat(id).name)}</b><span class="lm-state secondary">${T("lvWaiting")}</span><span class="lm-score"></span></div>`
      + `<div class="lm-lane" aria-hidden="true">${"<i></i>".repeat(guess)}</div><p class="lm-ctx secondary"></p><ul class="lm-tests"></ul></article>`).join("");
    const card = (id) => $(`.lm[data-m="${id}"]`);
    const bg = (v, hi, typ) => {
      if (hi != null && v > hi) return `rgba(var(--fake-rgb), ${clamp(0.18 + 0.67 * (v - hi) / Math.max(1e-6, 1 - hi), 0.18, 0.85).toFixed(3)})`;
      if (typ && v < typ.real_p50) return `rgba(var(--real-rgb), ${clamp(0.12 + 0.45 * (typ.real_p50 - v) / Math.max(1e-6, typ.real_p50), 0.12, 0.57).toFixed(3)})`;
      return "var(--fill)";
    };
    const setState = (id, text, cls) => { const c = card(id); if (!c) return; c.querySelector(".lm-state").textContent = text; c.dataset.state = cls; };
    let lastShown = -1;
    return {
      clip(ev, tot) {
        const s = st[ev.model], c = card(ev.model); if (!s || !c) return;
        s.high = ev.real_high; s.clips[ev.index] = ev;
        setState(ev.model, T("lvRunning", { i: ev.done, n: tot }), "run");
        const lane = c.querySelector(".lm-lane");
        lane.innerHTML = Array.from({ length: Math.max(tot, s.clips.length) }, (_, i) => { const x = s.clips[i]; return x ? (blind ? '<i class="on"></i>' : `<i class="on" style="background:${bg(x.fake, s.high, cat(ev.model).typical)}">${pct(x.fake)}</i>`) : "<i></i>"; }).join("");
        const got = s.clips.filter(Boolean), avg = mean(got.map((x) => x.fake));
        c.querySelector(".lm-score").textContent = blind ? "" : pct(avg) + "%";
        c.querySelector(".lm-ctx").textContent = blind ? T("lvBlind") : T("lvSoFar", { label: rarity(mean(got.map((x) => x.real_rank))).label.toLowerCase() });
        if (ev.index > lastShown && S.file.fps) {
          lastShown = ev.index;
          const img = $("#live-img"); img.src = frameUrl(ev.last_frame); img.hidden = false;
          $("#live-cap").textContent = T("lvCap", { i: ev.index + 1, a: t2s(ev.first_frame / S.file.fps), b: t2s(ev.last_frame / S.file.fps) });
        }
      },
      step(ev) {
        const c = card(ev.model); if (!c) return;
        c.querySelectorAll(".lm-tests li").forEach((li) => li.classList.toggle("active", li.dataset.t === ev.test));
        const li = c.querySelector(`.lm-tests li[data-t="${ev.test}"]`); if (li) li.querySelector("b").textContent = `${ev.n} / ${ev.of}`;
        const what = T("lvDo" + ev.test.toUpperCase());
        $("#run-step").textContent = T("lvStep", { m: cat(ev.model).name, what, i: ev.clip + 1, n: ev.n, of: ev.of });
        if (ev.frame != null) { const img = $("#live-img"); img.src = frameUrl(ev.frame); img.hidden = false; $("#live-cap").textContent = `${cat(ev.model).name} · ${what} · ${T("clipN", { n: ev.clip + 1 })}`; }
      },
      test(ev) {
        const c = card(ev.model); if (!c) return;
        const li = c.querySelector(`.lm-tests li[data-t="${ev.test}"]`); if (!li) return;
        li.classList.remove("active", "wait"); li.querySelector("b").textContent = blind ? T("lvMeasured") : `${pct(ev.before)} → ${pct(ev.after)}%`;
        if (ev.test === "a5") { // the last result: whatever did not run had no usable face
          c.querySelectorAll(".lm-tests li.wait").forEach((x) => { x.classList.remove("wait"); x.querySelector("b").textContent = T("lvSkipped"); });
          setState(ev.model, T("lvDone"), "done");
        }
      },
      stage(ev) {
        if (ev.stage === "loading") setState(ev.model, T("lvLoading"), "load");
        if (ev.stage === "testing") {
          setState(ev.model, T("lvTesting"), "test");
          const c = card(ev.model);
          if (c) c.querySelector(".lm-tests").innerHTML = ["a1", "a4", "a5"].map((t) => `<li class="wait" data-t="${t}"><span>${T("lv" + t.toUpperCase())}</span><b>${T("lvQueued")}</b></li>`).join("");
          // every clip is scored now: drop the cells that were only an estimate
          picked.forEach((id) => {
            const c = card(id), n = st[id].clips.filter(Boolean).length; if (!c || !n) return;
            c.querySelectorAll(".lm-lane i:not(.on)").forEach((i) => i.remove());
            if (id !== ev.model && c.dataset.state === "run") setState(id, T("lvScored"), "wait");
          });
        }
        if (ev.stage === "policy") picked.forEach((id) => setState(id, T("lvDone"), "done"));
      },
    };
  }

  async function open() {
    let body;
    try { body = await api(`jobs/${S.job}`); } catch (e) { $("#run-step").textContent = e.message; $("#run-back").hidden = false; return; }
    const R = body.result;
    S.R = R; S.ranges = body.ranges || {};
    const order = ["mars", "dfd-hr", "gend"].filter((id) => R.models[id]);
    S.models = order.map((id) => ({ id, name: R.models[id].name, native: R.models[id].frame_values_kind === "contribution" }));
    S.src = { name: R.video.name, duration: R.video.duration_s, width: R.video.width, height: R.video.height, nClips: S.R.models[order[0]].clips.length, cred: R.prechecks.credentials };
    (body.references || []).forEach((r) => { if (!S.refs.find((x) => x.key === r.key)) S.refs.push(r); });
    S.focus = 0; S.clip = S.R.models[order[0]].clips.reduce((b, c) => (c.fake > S.R.models[order[0]].clips[b].fake ? c.index : b), 0);
    S.frame = topShown(S.models[0], S.clip); S.zoom = "video"; S.overlay = true;
    S.guessing = S.guessFirst; S.guess = null; S.jur = "all"; S.report = null;
    show("analyze");
    renderAll();
    prefetch();
  }
  function topShown(m, c) {
    const vals = clipOf(m.id, c).frame_values, pos = positions(c);
    let b = 0; pos.forEach((p, i) => { if (vals[p] > vals[pos[b]]) b = i; });
    return b;
  }

  /* --------------------------------------------------------------- analyze */
  function renderAll() {
    const s = S.src;
    $("#an-h").textContent = s.name;
    $("#meta-text").textContent = `${t2s(s.duration)} · ${s.width}×${s.height}` + (S.R.video.rotation ? ` · ${T("rotatedNote", { d: S.R.video.rotation })}` : "");
    $("#cred-open").textContent = s.cred.status === "found" ? T("credFoundShort") : s.cred.status === "absent" ? T("credNoneShort") : T("credUnsupported");
    $("#guess").hidden = !S.guessing; $("#models").hidden = S.guessing; $("#reading").hidden = S.guessing; $("#agree").hidden = S.guessing; $("#tabs").hidden = S.guessing; $("#trust-sum").hidden = S.guessing;
    $("#open-report").disabled = S.guessing; $("#open-report").title = S.guessing ? T("guessFirstHint") : "";
    $$("#guess [data-guess]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.guess === S.guess)));
    selectTab(S.guessing ? null : currentTab);
    renderTimeline(); renderModels(); renderStage(); renderEvidence(); renderWhatIf(); renderNotes(); renderLearn(); renderAsk();
  }
  function refresh() { renderTimeline(); renderModels(); renderStage(); renderEvidence(); renderWhatIf(); }

  /* -------------------------------------------------------------- timeline */
  function laneBox() { const r = $("#ruler").getBoundingClientRect(), t = $("#tl").getBoundingClientRect(); return { left: r.left - t.left, width: r.width }; }
  function renderTimeline() {
    if (!S.R) return;
    const clipZ = S.zoom === "clip", n = S.src.nClips;
    $("#tl").classList.toggle("clipz", clipZ);
    $$("#zoom [data-z]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.z === S.zoom)));
    $("#zoom-clip").textContent = T("clipN", { n: S.clip + 1 });
    const c0 = refClip(S.clip);
    const span = clipZ ? [c0.first_frame / fps(), (c0.last_frame + stride()) / fps()] : [0, S.src.duration];
    const len = Math.max(0.1, span[1] - span[0]), step = clipZ ? 0.5 : len > 60 ? 10 : len > 20 ? 5 : 2;
    let ticks = "";
    for (let t = Math.ceil(span[0] / step) * step; t <= span[1] + 1e-6; t += step) {
      const major = clipZ ? Math.abs(t - Math.round(t)) < 1e-6 : true;
      ticks += `<i class="tick ${major ? "major" : ""}" style="left:${((t - span[0]) / len) * 100}%">${major ? `<span>${t2s(t)}</span>` : ""}</i>`;
    }
    $("#ruler").innerHTML = ticks;
    const film = $("#film");
    if (clipZ) {
      film.innerHTML = Array.from({ length: N(S.clip) }, (_, i) => {
        const cur = i === S.frame, off = S.removed.includes(i), fn = srcFrame(S.clip, i) + 1;
        const x = S.guessing ? "" : `<span class="x" data-x="${i}" aria-hidden="true" ${cur || off ? "" : "hidden"}><svg width="11" height="11" viewBox="0 0 24 24" class="ic" aria-hidden="true">${off ? '<path d="M4 12a8 8 0 1 0 3-6.2M4 4v4h4"/>' : '<path d="M6 6l12 12M18 6L6 18"/>'}</svg></span>`;
        return `<button type="button" class="tile ${cur ? "cur" : ""} ${off ? "off" : ""}" data-v="${i}" aria-label="${T("frameN", { n: fn })}${off ? ", " + T("removedLower") : ""}" ${cur ? 'aria-current="true"' : ""}><img alt="" loading="lazy" src="${frameUrl(srcFrame(S.clip, i))}" style="width:100%;height:100%;object-fit:cover;display:block">${x}</button>`;
      }).join("");
    } else {
      const w = Math.round(film.clientWidth || 800), h = film.clientHeight || 62, cnt = Math.max(4, Math.round(w / (h * 16 / 9)));
      const all = []; for (let c = 0; c < n; c++) for (let i = 0; i < N(c); i++) all.push(srcFrame(c, i));
      film.innerHTML = Array.from({ length: cnt }, (_, k) => `<img alt="" loading="lazy" src="${frameUrl(all[Math.min(all.length - 1, Math.floor(((k + 0.5) / cnt) * all.length))])}" style="flex:1 1 0;min-width:0;height:100%;object-fit:cover;display:block">`).join("");
    }
    $("#lanes").innerHTML = S.models.map((m, mi) => {
      let cells = "";
      const cnt = clipZ ? N(S.clip) : n;
      const clip = clipOf(m.id, S.clip), pos = positions(S.clip), maxAbs = Math.max(1e-6, ...clip.frame_values.map(Math.abs));
      for (let k = 0; k < cnt; k++) {
        const sel = mi === S.focus && (clipZ ? k === S.frame : k === S.clip);
        let bg, label, text = "";
        const off = clipZ && S.removed.includes(k);
        if (off) {
          bg = "var(--grouped)"; label = `${m.name}, ${T("frameN", { n: srcFrame(S.clip, k) + 1 })}: ${T("removedLower")}`;
        } else if (clipZ) {
          const v = clip.frame_values[pos[k]];
          bg = m.native ? signedBg(v / maxAbs) : cellBg(m, v);
          label = `${m.name}, ${T("frameN", { n: srcFrame(S.clip, k) + 1 })}` + (S.guessing ? "" : m.native ? `: ${T("frameContrib")} ${v >= 0 ? "+" : ""}${v.toFixed(3)}` : `: ${T("pctFake", { n: pct(v) })}`);
        } else {
          const v = clipNow(m, k);
          bg = cellBg(m, v);
          label = S.guessing ? `${m.name}, ${T("clipAt", { n: k + 1, t: t2s(refClip(k).first_frame / fps()) })}` : `${m.name}, ${T("clipAt", { n: k + 1, t: t2s(refClip(k).first_frame / fps()) })}: ${T("pctFake", { n: pct(v) })}`;
        }
        cells += `<button type="button" role="gridcell" class="cell ${sel ? "sel" : ""} ${clipZ && S.removed.includes(k) ? "off" : ""}" data-m="${mi}" data-k="${k}" tabindex="${sel ? 0 : -1}" aria-selected="${sel}" aria-label="${esc(label)}" style="background:${S.guessing ? "var(--fill)" : bg}">${text}</button>`;
      }
      return `<div class="tl-row lane-row" role="row"><span class="tl-label ${mi === S.focus ? "on" : ""}" role="rowheader">${esc(m.name)}</span><div class="lane">${cells}</div></div>`;
    }).join("");
    let band = $("#band"); if (!band) { band = document.createElement("span"); band.id = "band"; band.className = "band"; $("#tl").appendChild(band); }
    const lb = laneBox(); band.hidden = clipZ;
    if (!clipZ) { band.style.left = (lb.left + (S.clip / n) * lb.width - 1) + "px"; band.style.width = (lb.width / n + 2) + "px"; }
    placeHead();
    $("#tl-hint").innerHTML = S.guessing ? "" : clipZ && S.removed.length
      ? `${esc(T("hiddenHint", { n: S.removed.length }))} <button type="button" class="linkish" id="tl-restore">${esc(T("showAll"))}</button>`
      : esc(clipZ ? T("hintClip") : T("hintVideo"));
  }
  function placeHead() { if (!S.R) return; const lb = laneBox(), frac = S.zoom === "clip" ? (S.frame + 0.5) / N(S.clip) : (gIndex() + 0.5) / total(); $("#playhead").style.left = (lb.left + frac * lb.width) + "px"; }
  addEventListener("resize", () => { if (S.R && !V.analyze.hidden) { renderTimeline(); renderStage(); } });
  $("#tl-hint").addEventListener("click", (e) => { if (e.target.id !== "tl-restore") return; S.removed = []; rescore(); });
  $("#zoom").addEventListener("click", (e) => { const b = e.target.closest("[data-z]"); if (!b) return; S.zoom = b.dataset.z; renderTimeline(); });
  $("#lanes").addEventListener("click", (e) => { const b = e.target.closest(".cell"); if (b) pickCell(+b.dataset.m, +b.dataset.k, false); });
  $("#lanes").addEventListener("keydown", (e) => {
    const b = e.target.closest(".cell"); if (!b) return;
    let m = +b.dataset.m, k = +b.dataset.k; const max = (S.zoom === "clip" ? N(S.clip) : S.src.nClips) - 1;
    const d = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
    if (d) { m = clamp(m + d[0], 0, S.models.length - 1); k = clamp(k + d[1], 0, max); }
    else if (e.key === "Home") k = 0; else if (e.key === "End") k = max;
    else if (e.key === "Enter" && S.zoom === "video") { pickCell(m, k, false); return; }
    else return;
    e.preventDefault(); e.stopPropagation(); pickCell(m, k, true);
  });
  function pickCell(mi, k, keyNav) {
    if (mi !== S.focus) S.focus = mi;
    if (S.zoom === "clip") S.frame = k;
    else { if (k !== S.clip) { S.clip = k; S.removed = []; S.rescored = null; S.frame = topShown(S.models[S.focus], k); clipChanged(); } if (!keyNav) S.zoom = "clip"; }
    refresh();
    const cell = $(`.cell[data-m="${mi}"][data-k="${S.zoom === "clip" ? S.frame : k}"]`); if (cell) cell.focus();
  }
  $("#film").addEventListener("click", (e) => {
    const x = e.target.closest("[data-x]"), v = e.target.closest("[data-v]");
    if (x) { e.stopPropagation(); toggleFrame(+x.dataset.x); return; }
    if (v) { S.frame = +v.dataset.v; renderTimeline(); renderStage(); renderEvidence(); const t = $(`.tile[data-v="${S.frame}"]`); if (t) t.focus(); }
  });
  $("#film").addEventListener("keydown", (e) => { const v = e.target.closest("[data-v]"); if (v && (e.key === "Delete" || e.key === "Backspace")) { e.preventDefault(); toggleFrame(+v.dataset.v); } });
  /* Removing frames re-scores the clip for every model on the server; only the latest request is kept. */
  let rescoreTok = 0;
  async function rescore() {
    const tok = ++rescoreTok, clip = S.clip, drop = S.removed.slice();
    if (!drop.length) { S.rescored = null; S.rescoring = false; refresh(); return; }
    S.rescoring = true; refresh();
    try { const out = await api(`jobs/${S.job}/rescore`, { clip, drop }); if (tok === rescoreTok && clip === S.clip) S.rescored = out.scores; }
    catch (e) { if (tok === rescoreTok) { S.removed = []; S.rescored = null; $("#agree").textContent = T("rescoreFailed", { e: e.message }); } }
    if (tok === rescoreTok) { S.rescoring = false; refresh(); }
  }
  function toggleFrame(i) {
    if (S.guessing) return;
    if (isEdited()) { S.edits = { jpeg: 100, scale: 100, blur: 0, noise: 0 }; S.edited = null; renderWhatIf(); }
    S.removed = S.removed.includes(i) ? S.removed.filter((k) => k !== i) : S.removed.concat(i);
    rescore().then(() => { const t = $(`.tile[data-v="${i}"]`); if (t) t.focus(); });
  }
  function scrubAt(clientX) {
    const r = $("#ruler").getBoundingClientRect(), f = clamp((clientX - r.left) / r.width, 0, 0.9999);
    if (S.zoom === "clip") go(S.clip, Math.floor(f * N(S.clip)));
    else { let g = Math.floor(f * total()), c = 0; while (c < S.src.nClips - 1 && g >= N(c)) { g -= N(c); c++; } go(c, g); }
  }
  ["ruler", "film"].forEach((id) => {
    const el = $("#" + id);
    el.addEventListener("pointerdown", (e) => {
      if (id === "film" && S.zoom === "clip") return;
      e.preventDefault(); el.setPointerCapture(e.pointerId); scrubAt(e.clientX);
      const mv = (ev) => scrubAt(ev.clientX), up = () => { el.removeEventListener("pointermove", mv); el.removeEventListener("pointerup", up); };
      el.addEventListener("pointermove", mv); el.addEventListener("pointerup", up);
    });
  });

  /* ----------------------------------------------------------------- models */
  function renderModels() {
    $("#models").innerHTML = S.models.map((m, mi) => {
      const s = S.R.models[m.id], now = pct(videoNow(m)), base = pct(s.fake), ctx = s.context;
      const pos = placeOf(m, videoNow(m)), rr = rarity(pos.real, m.name), posTxt = `${rr.label}${LANG() === "fr" ? " :" : ":"} ${rr.odds}`;
      const was = rarity(ctx.real_rank, "").label;
      const small = S.rescoring || S.editBusy ? T("rescoring") : was !== rr.label ? T("wasBand", { b: was.toLowerCase() }) : now !== base ? T("moved") : "";
      const [rlo, rhi] = ctx.real_range.map((v) => clamp(v * 100, 0, 100));
      return `<button type="button" class="m" data-f="${mi}" aria-pressed="${mi === S.focus}" aria-label="${esc(m.name)}: ${T("pctFake", { n: now })}, ${T("pctReal", { n: 100 - now })}${now !== base ? ", " + T("wasPct", { n: base }) : ""}. ${esc(posTxt)}">
        <span class="m-name">${esc(m.name)}</span><span class="m-num m-word">${esc(rr.label)}${small ? `<small>${esc(small)}</small>` : ""}</span>
        <span class="m-kind">${esc(T("kind_" + m.id))}</span>
        <span class="m-range" aria-hidden="true"><span class="m-band" style="left:${rlo}%;width:${Math.max(1, rhi - rlo)}%"></span>${now !== base ? `<i class="m-dot was" style="left:${base}%"></i>` : ""}<i class="m-dot" style="left:${now}%"></i></span>
        <span class="m-legend" aria-hidden="true"><i class="m-sw"></i>${esc(T("realBand"))}</span>
        ${s.device === "cpu" ? `<span class="m-note">${T("cpuNote")}${s.device_note ? ": " + esc(s.device_note) : ""}</span>` : ""}</button>`;
    }).join("");
    /* The reading: which way the detectors lean and how confident that is (rule: marscope.explain.policy.assess). */
    const r = S.R.reading;
    $("#reading").className = "reading" + (r ? " lean-" + r.lean : "");
    $("#reading").innerHTML = r ? `<b class="rd-lean">${esc(r.headline)}</b><ul class="rd-why">${[...r.reasons, r.action].map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
    $("#agree").textContent = S.guess ? T("youSaid", { g: T({ real: "real", fake: "fake", unsure: "notSure" }[S.guess]) }) : "";
    renderTrustSum();
  }
  $("#models").addEventListener("click", (e) => { const b = e.target.closest("[data-f]"); if (!b || +b.dataset.f === S.focus) return; S.focus = +b.dataset.f; refresh(); prefetch(); $(`[data-f="${S.focus}"]`).focus(); });
  $("#guess").addEventListener("click", (e) => {
    const b = e.target.closest("[data-guess]"); if (!b) return;
    S.guess = b.dataset.guess; S.guessing = false;
    renderAll();
  });

  /* ------------------------------------------------------------------ stage */
  const main = { cv: $("#frame"), tok: 0 };
  const methodOf = (m) => S.method[m.id] || (m.native ? "native" : "occlusion");
  /* Each model's map for the clip on a common 8 x 8 grid (MARS: its positive patch contributions over the clip). */
  function coarse(m) {
    const key = m.id + ":" + S.clip;
    if (m.native && methodOf(m) === "native") {
      const mp = S.maps[key]; if (!mp) return null;
      const out = new Array(64).fill(0);
      mp.frames.forEach((g) => g.forEach((row, y) => row.forEach((v, x) => { out[Math.floor(y / 3) * 8 + Math.floor(x / 3)] += Math.max(0, v); })));
      return out;
    }
    const occ = S.occlusion[key]; return occ && occ.length === 8 ? occ.flat() : null;
  }
  const top16 = (h) => h.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 16).map((p) => p[1]);
  function agreeCells() {
    const maps = S.models.map(coarse); if (maps.some((h) => !h)) return null;
    const sets = maps.map((h) => new Set(top16(h)));
    return new Set([...sets[0]].filter((i) => sets.every((st) => st.has(i))));
  }
  async function ensureOcclusion(m, clip) {
    const key = m.id + ":" + clip;
    if (!S.occlusion[key]) S.occlusion[key] = (await api(`jobs/${S.job}/occlusion`, { model: m.id, clip })).map;
  }
  function heatFor(m) {
    const key = m.id + ":" + S.clip;
    if (S.agree) { const a = agreeCells(); return a ? { h: Array.from({ length: 64 }, (_, i) => (a.has(i) ? 1 : 0)), n: 8, kind: "agree" } : null; }
    if (methodOf(m) === "off") return null;
    if (m.native && methodOf(m) === "native") {
      const mp = S.maps[key]; if (!mp) return null;
      const grid = mp.frames[positions(S.clip)[S.frame]]; if (!grid) return null;
      return { h: grid.flat().map((v) => clamp(v / mp.scale, 0, 1)), n: 24, kind: "native" };
    }
    const occ = S.occlusion[key]; if (!occ) return null;
    const mx = Math.max(...occ.flat());
    if (mx < 0.005) return null; // flat map: nothing to show (explained in Check)
    return { h: occ.flat().map((v) => clamp(v / mx, 0, 1)), n: occ.length, kind: "occlusion" };
  }
  async function ensureMap(m) {
    const key = m.id + ":" + S.clip;
    if (!m.native || key in S.maps) return;
    S.maps[key] = null;
    try { S.maps[key] = await api(`jobs/${S.job}/maps/${m.id}/${S.clip}`); } catch (_) { delete S.maps[key]; }
  }
  async function renderStage() {
    const tok = ++main.tok, m = S.models[S.focus], idx = srcFrame(S.clip, S.frame), t = idx / fps();
    $("#time").textContent = `${t2s(t)} / ${t2s(S.src.duration)}`;
    main.cv.setAttribute("aria-label", T("frameAt", { t: t2s(t) }));
    const g = gIndex(); $("#prev").disabled = g === 0; $("#next").disabled = g === total() - 1;
    $("#map").setAttribute("aria-pressed", String(S.overlay)); $("#map").hidden = S.guessing;
    const ed = isEdited() && !S.guessing;
    $("#compare").hidden = !ed; $("#compare").setAttribute("aria-pressed", String(ed && S.compare)); $("#split-line").hidden = !(ed && S.compare);
    await ensureMap(m);
    const [img, edImg] = await Promise.all([image(idx), ed ? imageAt(editedUrl(idx)) : null]);
    if (tok !== main.tok || !S.R) return;
    const cv = main.cv, box = $("#player").getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = Math.round(box.width * dpr); cv.height = Math.round(box.height * dpr);
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    let rect = { x: 0, y: 0, w: cv.width, h: cv.height };
    if (img) {
      const k = Math.min(cv.width / img.naturalWidth, cv.height / img.naturalHeight), w = img.naturalWidth * k, h = img.naturalHeight * k;
      rect = { x: (cv.width - w) / 2, y: (cv.height - h) / 2, w, h };
      if (edImg) {
        ctx.drawImage(edImg, rect.x, rect.y, w, h);
        if (S.compare) { const sx = img.naturalWidth / 2; ctx.drawImage(img, 0, 0, sx, img.naturalHeight, rect.x, rect.y, w / 2, h); }
      } else ctx.drawImage(img, rect.x, rect.y, w, h);
      [1, 2, 3].forEach((d) => { if (S.frame + d < N(S.clip)) image(srcFrame(S.clip, S.frame + d)); });
    }
    const pr = !S.guessing && S.test === "areas" ? S.probes[`${m.id}:${S.clip}:${m.native ? "patches" : "areas"}`] : null;
    const hid = pr && pr.cells ? new Set((pr.cells[positions(S.clip)[S.frame]] || []).map((i) => turn(i, 24))) : null;
    const hm = hid ? null : S.overlay && !S.guessing ? turned(heatFor(m)) : null, grid = $("#patches");
    if (hid) {
      Object.assign(grid.style, { left: (rect.x / cv.width) * 100 + "%", top: (rect.y / cv.height) * 100 + "%", width: (rect.w / cv.width) * 100 + "%", height: (rect.h / cv.height) * 100 + "%", right: "auto", bottom: "auto", gridTemplateColumns: "repeat(24, 1fr)", gridTemplateRows: "repeat(24, 1fr)" });
      grid.innerHTML = Array.from({ length: 576 }, (_, i) => (hid.has(i) ? '<i class="hid"></i>' : "<i></i>")).join("");
      grid.hidden = false;
    } else if (hm) {
      Object.assign(grid.style, { left: (rect.x / cv.width) * 100 + "%", top: (rect.y / cv.height) * 100 + "%", width: (rect.w / cv.width) * 100 + "%", height: (rect.h / cv.height) * 100 + "%", right: "auto", bottom: "auto", gridTemplateColumns: `repeat(${hm.n}, 1fr)`, gridTemplateRows: `repeat(${hm.n}, 1fr)` });
      grid.innerHTML = hm.h.map((h) => `<i style="background:rgba(var(--fake-rgb), ${(Math.pow(h, 0.8) * 0.62).toFixed(3)})"></i>`).join("");
      grid.hidden = false;
    } else { grid.innerHTML = ""; grid.hidden = true; }
    const msgs = [];
    if (ed && !S.compare) msgs.push(T("edited"));
    if (S.mapBusy) msgs.push(T("mapWorking"));
    else if (hid) msgs.push(T("areasHidden", { m: m.name, n: hid.size }));
    else if (hm) msgs.push(hm.kind === "agree" ? T("mAgreeLong") : m.name + " · " + (hm.kind === "native" ? T("mBuiltIn") : T("mOcclusion")));
    else if (S.overlay && !S.guessing && !m.native && !S.agree && methodOf(m) !== "off") msgs.push(S.occlusion[m.id + ":" + S.clip] ? T("occFlatBadge", { m: m.name }) : T("noBuiltIn", { m: m.name }));
    main.cv.setAttribute("aria-label", T("frameAt", { t: t2s(t) }) + (ed ? ", " + T("editedLower") : ""));
    const badge = $("#badge"); badge.textContent = msgs.join(" · "); badge.hidden = !msgs.length || (ed && S.compare);
  }
  /* Maps are on the frame as the models saw it (as decoded). A rotated phone video is shown upright,
     so its overlays are turned the same way: cell (row, col) of an n x n grid moves with the picture. */
  function turn(i, n) {
    const r = (((S.R && S.R.video.rotation) || 0) % 360 + 360) % 360, y = Math.floor(i / n), x = i % n;
    if (r === 90) return (n - 1 - x) * n + y;
    if (r === 180) return (n - 1 - y) * n + (n - 1 - x);
    if (r === 270) return x * n + (n - 1 - y);
    return i;
  }
  function turned(hm) {
    if (!hm || !((S.R.video.rotation || 0) % 360)) return hm;
    const h = new Array(hm.h.length); hm.h.forEach((v, i) => { h[turn(i, hm.n)] = v; });
    return Object.assign({}, hm, { h });
  }
  function go(c, i) {
    c = clamp(c, 0, S.src.nClips - 1); i = clamp(i, 0, N(c) - 1);
    if (c !== S.clip) { S.clip = c; S.frame = i; S.removed = []; S.rescored = null; clipChanged(); renderTimeline(); renderModels(); renderEvidence(); }
    else {
      S.frame = i;
      if (S.zoom === "clip") {
        $$(".tile").forEach((t) => { const cur = +t.dataset.v === i; t.classList.toggle("cur", cur); if (cur) t.setAttribute("aria-current", "true"); else t.removeAttribute("aria-current"); });
        $$("#lanes .cell").forEach((cl) => { const s = +cl.dataset.m === S.focus && +cl.dataset.k === i; cl.classList.toggle("sel", s); cl.tabIndex = s ? 0 : -1; });
      }
      placeHead();
    }
    renderStage();
  }
  function step(d) {
    let c = S.clip, i = S.frame + d;
    if (i >= N(c)) { if (c < S.src.nClips - 1) { c++; i = 0; } else i = N(c) - 1; }
    if (i < 0) { if (c > 0) { c--; i = N(c) - 1; } else i = 0; }
    go(c, i);
    return c === S.src.nClips - 1 && i === N(c) - 1;
  }
  $("#prev").addEventListener("click", () => step(-1));
  $("#next").addEventListener("click", () => step(1));
  $("#map").addEventListener("click", () => { S.overlay = !S.overlay; renderStage(); });
  $("#compare").addEventListener("click", () => { S.compare = !S.compare; renderStage(); });
  const PLAY = '<path d="M8 5.5v13l10.5-6.5z"/>', PAUSE = '<rect x="6.5" y="5" width="4" height="14" rx="1"/><rect x="13.5" y="5" width="4" height="14" rx="1"/>';
  function stopPlay() { if (S.playing) clearInterval(S.playing); S.playing = null; $("#play").setAttribute("aria-label", T("play")); $("#play").title = T("play"); $("#play-ic").innerHTML = PLAY; }
  function togglePlay() {
    if (S.playing) return stopPlay();
    if (gIndex() >= total() - 1) go(0, 0);
    $("#play").setAttribute("aria-label", T("pause")); $("#play").title = T("pause"); $("#play-ic").innerHTML = PAUSE;
    const per = S.R ? (refClip(S.clip).last_frame - refClip(S.clip).first_frame + stride()) / Math.max(1, N(S.clip)) : 8;
    S.playing = setInterval(() => { if (step(1)) stopPlay(); }, (1000 * per) / fps());
  }
  $("#play").addEventListener("click", togglePlay);
  document.addEventListener("keydown", (e) => {
    if (V.analyze.hidden || $$("dialog[open]").length || e.target.closest("input, textarea, select, [role='tab'], .cell") || e.metaKey || e.ctrlKey) return;
    if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
    else if (e.key === "[") go(S.clip - 1, 0);
    else if (e.key === "]") go(S.clip + 1, 0);
  });

  /* ------------------------------------------------------------------- tabs */
  const TABS = ["evidence", "notes", "whatif", "learn"]; let currentTab = "evidence";
  function selectTab(k) { if (k) currentTab = k; TABS.forEach((t) => { const on = t === k, tb = $("#tab-" + t); tb.setAttribute("aria-selected", String(on)); tb.tabIndex = on ? 0 : -1; $("#p-" + t).hidden = !on; }); }
  $("#tabs").addEventListener("click", (e) => { const t = e.target.closest('[role="tab"]'); if (t) selectTab(t.id.slice(4)); });
  $("#tabs").addEventListener("keydown", (e) => {
    const vis = TABS.filter((t) => !$("#tab-" + t).hidden), i = vis.indexOf(currentTab), k = vis.length;
    let n = null; if (e.key === "ArrowRight") n = (i + 1) % k; else if (e.key === "ArrowLeft") n = (i + k - 1) % k; else if (e.key === "Home") n = 0; else if (e.key === "End") n = k - 1;
    if (n === null) return; e.preventDefault(); selectTab(vis[n]); $("#tab-" + vis[n]).focus();
  });

  /* ------------------------------------------------------- check (evidence) */
  const pts = (n) => T(n === 1 || (LANG() === "fr" && n === 0) ? "pt1" : "ptN", { n });
  function dumbbell(rows) {
    return rows.map((r) => `<div class="cmp-row${r.dim ? " dim" : ""}"><span class="cmp-l">${esc(r.l)}</span>`
      + `<span class="cmp-bar" aria-hidden="true"><i style="left:${Math.min(r.a, r.b)}%;width:${Math.abs(r.b - r.a)}%"></i><s style="left:${r.a}%"></s><b style="left:${r.b}%"></b></span>`
      + `<span class="cmp-n">${r.a}<span class="arr" aria-label="${T("to")}">→</span>${r.b}%</span></div>`).join("");
  }
  function verdict(top, rnd, what) {
    const a = Math.abs(top), b = Math.abs(rnd), w = T(what);
    if (a >= 10 && a >= 2.5 * Math.max(b, 1)) return T("sayYes", { a: pts(a), b: pts(b), w });
    if (a >= 5 && a > 1.5 * b) return T("saySome", { a: pts(a), b: pts(b), w });
    return T("sayNo", { w });
  }
  function lineChart(svgEl, series, opts) {
    const W = 300, H = opts.h, Lp = 26, Rp = 70, Tp = 8, B = 20, xs = opts.xs, yr = opts.yr || [0, 1];
    const x = (v) => Lp + ((v - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - Lp - Rp), y = (v) => Tp + (1 - (v - yr[0]) / (yr[1] - yr[0])) * (H - Tp - B);
    let s = "";
    [yr[0], (yr[0] + yr[1]) / 2, yr[1]].forEach((v) => { s += `<line x1="${Lp}" x2="${W - Rp}" y1="${y(v)}" y2="${y(v)}" style="stroke:var(--sep)"/><text x="${Lp - 6}" y="${y(v) + 3.5}" text-anchor="end">${Math.round(v * 100)}</text>`; });
    (opts.xticks || []).forEach((v) => { s += `<text x="${x(v)}" y="${H - 5}" text-anchor="middle">${v}</text>`; });
    if (opts.mark != null) s += `<line x1="${x(opts.mark)}" x2="${x(opts.mark)}" y1="${Tp}" y2="${H - B}" style="stroke:var(--secondary)" stroke-width="1"/>`;
    const labels = [];
    series.forEach((se) => {
      s += `<polyline fill="none" style="stroke:${se.color}" stroke-width="2" stroke-dasharray="${se.dash || ""}" stroke-linecap="round" stroke-linejoin="round" points="${se.pts.map(([a, b]) => x(a) + "," + y(b)).join(" ")}"/>`;
      if (opts.mark != null) { const p = se.pts.find(([a]) => a === opts.mark); if (p) s += `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="3.5" style="fill:${se.color}"/>`; }
      labels.push({ y: y(se.pts[se.pts.length - 1][1]), n: se.name, c: se.color });
    });
    labels.sort((a, b) => a.y - b.y).forEach((l, i, a) => { if (i && l.y - a[i - 1].y < 12) l.y = a[i - 1].y + 12; s += `<text x="${W - Rp + 6}" y="${l.y + 3.5}" style="fill:${l.c};font-weight:600">${esc(l.n)}</text>`; });
    svgEl.innerHTML = s;
  }
  const TOPK = 4;
  function renderEvidence() {
    if (!S.R) return;
    const m = S.models[S.focus];
    $$("[data-test]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.test === S.test)));
    $("#test-ctx").textContent = m.name + " · " + T("clipN", { n: S.clip + 1 });
    const sw = S.sweeps[m.id + ":" + S.clip], res = S.probes[`${m.id}:${S.clip}:patches`];
    let q, run, done = false, rows = [], say = "";
    if (S.test === "frames") {
      const k = sw ? Math.min(TOPK, sw.top.length - 1) : TOPK;
      q = T("qFrames"); run = T("runFrames", { n: k });
      const kOf = (x) => Math.min(TOPK, x.top.length - 1), swOf = (x) => S.sweeps[x.id + ":" + S.clip];
      const by = [m].concat(S.models.filter((x) => x !== m)).find((x) => swOf(x) && sameSet(S.removed, swOf(x).order.slice(0, kOf(swOf(x)))));
      if (by) {
        const s = swOf(by), kb = kOf(s), a = pct(s.top[0]), b = pct(s.top[kb]), r = pct(s.random[kb]);
        rows = [{ l: by === m ? T("itsPicks") : T("picksOf", { m: by.name }), a, b }, { l: T("randomPicks"), a, b: r, dim: true }];
        say = (by === m ? "" : T("hidByOther", { m: by.name }) + " ") + verdict(b - a, r - a, "wFrames"); done = true;
      } else if (S.removed.length) { say = T("youHid", { n: S.removed.length }); done = true; }
    } else if (m.native) {
      q = T("qAreas"); run = T("runAreas");
      if (res) { const a = pct(res.before), b = pct(res.after), r = pct(res.random_after); rows = [{ l: T("itsPicks"), a, b }, { l: T("randomPicks"), a, b: r, dim: true }]; say = verdict(b - a, r - a, "wAreas"); done = true; }
    } else {
      const occ = S.occlusion[m.id + ":" + S.clip], ares = S.probes[`${m.id}:${S.clip}:areas`];
      if (!occ) { q = T("qNoMap", { m: m.name }); run = T("runUseMap"); }
      else if (Math.max(...occ.flat()) < 0.005) { q = T("qNoMap", { m: m.name }); run = T("runUseMap"); done = true; say = T("occFlat", { m: m.name, c: S.clip + 1 }); }
      else { // the map exists: the same test MARS has, on the map's top cells
        q = T("qAreas"); run = T("runAreasOcc");
        if (ares) { const a = pct(ares.before), b = pct(ares.after), r = pct(ares.random_after); rows = [{ l: T("itsPicks"), a, b }, { l: T("randomPicks"), a, b: r, dim: true }]; say = verdict(b - a, r - a, "wAreas"); done = true; }
      }
    }
    $("#test-q").textContent = q; $("#test-run").textContent = run;
    $("#test-run").parentElement.hidden = done; $("#test-res").hidden = !done;
    $("#test-cmp").innerHTML = dumbbell(rows); $("#test-say").textContent = say;
    $("#test-undo").textContent = S.test === "frames" ? T("showAll") : T("undo");
    $("#test-undo").parentElement.hidden = false;

    /* More detail: highlights, the areas every detector marks, the removal curve, the current frame. */
    const opts = (m.native ? [["native", T("mBuiltIn")], ["occlusion", T("mOcclusion")]] : [["off", T("off")], ["occlusion", T("mOcclusion")]]).concat(S.models.length > 1 ? [["agree", T("mAgreeLong")]] : []);
    const sel = S.agree ? "agree" : methodOf(m);
    $("#method").innerHTML = opts.map(([k, l]) => `<option value="${k}" ${k === sel ? "selected" : ""}>${esc(l)}</option>`).join("");
    $("#method").disabled = S.mapBusy;
    const cells = S.models.length > 1 ? agreeCells() : null;
    $("#overlap").textContent = cells ? T("ofN", { a: cells.size, b: 16 }) : "—";
    const svg = $("#curve");
    if (sw) {
      const K = sw.top.length - 1, ks = Array.from({ length: K + 1 }, (_, i) => i), k = Math.min(TOPK, K);
      const top = ks.map((i) => [i, sw.top[i]]), rnd = ks.map((i) => [i, sw.random[i]]), all = sw.top.concat(sw.random);
      const lo = Math.max(0, Math.floor((Math.min(...all) - 0.04) * 20) / 20), hi = Math.min(1, Math.ceil((Math.max(...all) + 0.04) * 20) / 20);
      lineChart(svg, [{ name: T("itsPicks"), color: "var(--accent)", pts: top }, { name: T("randomPicks"), color: "var(--secondary)", dash: "5 4", pts: rnd }], { h: 110, xs: ks, xticks: [0, Math.round(K / 2), K], mark: k, yr: [lo, hi] });
      $("#curve-desc").textContent = T("curveDesc", { c: S.clip + 1, k: K, a: pct(sw.top[0]), b: pct(sw.top[K]), r: pct(sw.random[K]) });
    } else {
      svg.innerHTML = `<text x="150" y="58" text-anchor="middle">${esc(T("curveNeed"))}</text>`;
      $("#curve-desc").textContent = T("curveNeed");
    }
    const curOff = S.removed.includes(S.frame);
    $("#cur-frame").textContent = T("frameN", { n: srcFrame(S.clip, S.frame) + 1 });
    $("#d-cur").textContent = curOff && S.rescored ? pct(clipNow(m, S.clip)) + "%" : "";
    $("#rm-cur").textContent = curOff ? T("show") : T("hide");
    $("#rm-cur").disabled = S.guessing;
  }
  $$("[data-test]").forEach((b) => b.addEventListener("click", () => { if (S.test === b.dataset.test) return; S.test = b.dataset.test; renderEvidence(); renderStage(); }));
  $("#rm-cur").addEventListener("click", () => { S.zoom = "clip"; toggleFrame(S.frame); $("#rm-cur").focus(); });
  $("#method").addEventListener("change", async (e) => {
    const m = S.models[S.focus], v = e.target.value, clip = S.clip;
    S.agree = v === "agree"; if (!S.agree) S.method[m.id] = v;
    S.overlay = true;
    const need = S.agree ? S.models : v === "occlusion" ? [m] : [];
    if (need.some((x) => !S.occlusion[x.id + ":" + clip] && !(x.native && methodOf(x) === "native"))) {
      S.mapBusy = true; renderEvidence(); renderStage();
      try {
        for (const x of need) { if (x.native && methodOf(x) === "native") await ensureMap(x); else await ensureOcclusion(x, clip); }
      } catch (err) { $("#test-say").textContent = err.message; S.agree = false; }
      S.mapBusy = false;
    } else if (S.agree) await Promise.all(S.models.filter((x) => x.native).map(ensureMap));
    renderEvidence(); renderStage();
  });
  $("#test-run").addEventListener("click", async () => {
    const m = S.models[S.focus], btn = $("#test-run"), label = btn.textContent, clip = S.clip, test = S.test;
    let failed = null;
    btn.disabled = true; btn.setAttribute("aria-busy", "true"); btn.textContent = T("running");
    try {
      if (test === "frames") {
        const key = m.id + ":" + clip;
        if (!S.sweeps[key]) { await warm("sweep", m, clip); S.sweeps[key] = await api(`jobs/${S.job}/sweep`, { model: m.id, clip }); }
        const sw = S.sweeps[key];
        if (clip === S.clip) { S.removed = sw.order.slice(0, Math.min(TOPK, sw.top.length - 1)); S.zoom = "clip"; await rescore(); }
      } else if (!m.native && !S.occlusion[m.id + ":" + clip]) { await ensureOcclusion(m, clip); S.method[m.id] = "occlusion"; S.agree = false; S.overlay = true; }
      else if (!m.native) { S.probes[`${m.id}:${clip}:areas`] = await api(`jobs/${S.job}/probe`, { model: m.id, clip, kind: "areas" }); }
      else { await warm("probe", m, clip); S.probes[`${m.id}:${clip}:patches`] = await api(`jobs/${S.job}/probe`, { model: m.id, clip, kind: "patches" }); }
    } catch (e) { failed = e.message; }
    btn.disabled = false; btn.removeAttribute("aria-busy"); btn.textContent = label;
    refresh();
    if (failed) { // keep the button, and say why next to it
      $("#test-res").hidden = false; $("#test-cmp").innerHTML = ""; $("#test-say").textContent = T("testFailed", { e: failed });
      $("#test-undo").parentElement.hidden = true; btn.focus();
    } else if (!$("#test-res").hidden) $("#test-undo").focus();
  });
  $("#test-undo").addEventListener("click", () => {
    const m = S.models[S.focus];
    if (S.test === "frames") { S.removed = []; S.rescored = null; }
    else if (m.native) delete S.probes[`${m.id}:${S.clip}:patches`];
    else if (S.probes[`${m.id}:${S.clip}:areas`]) delete S.probes[`${m.id}:${S.clip}:areas`];
    else delete S.occlusion[m.id + ":" + S.clip];
    refresh(); $("#test-run").focus();
  });


  /* Pre-run what most people run anyway: the focused model's "hide its top frames" on the open clip, then
     MARS's top areas. This only fills the server's cache: nothing is shown, hidden or listed in the report
     until the user presses the button, which then answers at once (or waits for the run in progress). */
  const pending = new Map(); let preTok = 0, preTimer = null, inflight = Promise.resolve();
  const preKey = (job, kind, m, clip) => `${job}:${kind}:${m.id}:${clip}`;
  /* Waits until the view has settled (not while playing or scrubbing), and keeps at most one pre-run on
     the GPU at a time, so the user's own actions never queue behind more than one. */
  function prefetch() { clearTimeout(preTimer); preTok++; preTimer = setTimeout(runPrefetch, 1500); }
  async function runPrefetch() {
    if (!S.R || !S.job) return;
    if (S.playing) { prefetch(); return; }
    const tok = preTok, job = S.job, clip = S.clip, m = S.models[S.focus], mars = S.models.find((x) => x.native);
    const todo = [["sweep", m]].concat(mars ? [["probe", mars]] : []);
    if (S.models.length > 1) S.models.filter((x) => !x.native).forEach((x) => todo.push(["occlusion", x])); // for "areas all detectors mark"
    for (const [kind, x] of todo) {
      await inflight;
      if (tok !== preTok || job !== S.job) return;
      const key = preKey(job, kind, x, clip);
      const have = { sweep: S.sweeps[x.id + ":" + clip], probe: S.probes[`${x.id}:${clip}:patches`], occlusion: S.occlusion[x.id + ":" + clip] }[kind];
      if (pending.has(key) || warmed.has(key) || have) continue;
      const body = kind === "probe" ? { model: x.id, clip, kind: "patches", record: false } : { model: x.id, clip, record: false };
      const req = api(`jobs/${job}/${kind}`, body).then(() => { warmed.add(key); }, () => {}).finally(() => pending.delete(key));
      pending.set(key, req); inflight = req;
      await req;
    }
  }
  const warmed = new Set();
  /* Before the user's own request: wait for a pre-run of the same test, so the GPU computes it once. */
  const warm = (kind, m, clip) => pending.get(preKey(S.job, kind, m, clip)) || Promise.resolve();

  /* --------------------------------------------------------------- what if */
  /* The same steps as marscope.video.edits.SHARE_STEPS. Every change re-scores the current clip on the server. */
  const STEPS = [{ jpeg: 100, scale: 100, blur: 0, noise: 0 }, { jpeg: 75, scale: 80, blur: 0, noise: 0 }, { jpeg: 55, scale: 60, blur: 0.5, noise: 0 }, { jpeg: 35, scale: 40, blur: 1, noise: 0.02 }];
  const stepOf = () => STEPS.findIndex((st) => Object.keys(st).every((k) => st[k] === S.edits[k]));
  const SL = { jpeg: (v) => v, scale: (v) => v + "%", blur: (v) => (+v ? v + " px" : T("off")), noise: (v) => (+v ? fmtN(v, 2) : T("off")) };
  let editTok = 0;
  async function requestEdit() {
    const tok = ++editTok, clip = S.clip, edits = Object.assign({}, S.edits);
    S.edited = null;
    if (!isEdited()) { S.editBusy = false; refresh(); return; }
    S.editBusy = true; refresh();
    try { const out = await api(`jobs/${S.job}/edit`, Object.assign({ clip }, edits)); if (tok === editTok && clip === S.clip) S.edited = out.scores; }
    catch (e) { if (tok === editTok) $("#share-say").textContent = T("editFailed", { e: e.message }); }
    if (tok === editTok) { S.editBusy = false; refresh(); loadQualitySweep(); }
  }
  function setEdits(next) {
    S.edits = Object.assign({}, next); S.removed = []; S.rescored = null; S.compare = true;
    requestEdit();
  }
  function clipChanged() { if (isEdited()) requestEdit(); else S.edited = null; prefetch(); }
  const qKey = () => `${S.clip}:${S.edits.scale}:${S.edits.blur}:${S.edits.noise}`;
  async function loadQualitySweep() {
    if (!$("#wi-more").open || !S.R || S.guessing) return;
    const key = qKey(); if (key in S.qsweeps) return;
    S.qsweeps[key] = null; renderWhatIf();
    try { S.qsweeps[key] = await api(`jobs/${S.job}/quality-sweep`, Object.assign({ clip: S.clip }, S.edits, { jpeg: 100 })); }
    catch (_) { delete S.qsweeps[key]; }
    renderWhatIf();
  }
  $("#wi-more").addEventListener("toggle", loadQualitySweep);
  $("#s-share").addEventListener("input", (e) => { const v = +e.target.value; $("#o-share").textContent = T("share" + v); });
  $("#s-share").addEventListener("change", (e) => setEdits(STEPS[+e.target.value]));
  Object.keys(SL).forEach((k) => {
    const el = $("#s-" + k);
    el.addEventListener("input", () => { $("#o-" + k).textContent = SL[k](el.value); });
    el.addEventListener("change", () => setEdits(Object.assign({}, S.edits, { [k]: +el.value })));
  });
  function renderWhatIf() {
    if (!S.R) return;
    Object.keys(SL).forEach((k) => { $("#s-" + k).value = S.edits[k]; $("#o-" + k).textContent = SL[k](S.edits[k]); $("#s-" + k).setAttribute("aria-valuetext", SL[k](S.edits[k])); $("#s-" + k).disabled = S.guessing; });
    const st = stepOf(), name = st < 0 ? T("shareCustom") : T("share" + st);
    if (st >= 0) $("#s-share").value = st;
    $("#s-share").disabled = S.guessing;
    $("#o-share").textContent = name; $("#s-share").setAttribute("aria-valuetext", name);
    $("#share-ctx").textContent = T("clipN", { n: S.clip + 1 });
    if (S.editBusy) { $("#share-cmp").innerHTML = ""; $("#share-say").textContent = T("shareWorking"); $("#share-res").setAttribute("aria-busy", "true"); }
    else {
      $("#share-res").removeAttribute("aria-busy");
      const moved = isEdited() && S.edited;
      const rows = moved ? S.models.map((m) => ({ l: m.name, a: pct(clipOf(m.id, S.clip).fake), b: pct(S.edited[m.id]) })) : [];
      const big = rows.length ? Math.max(...rows.map((x) => Math.abs(x.b - x.a))) : 0;
      $("#share-cmp").innerHTML = moved ? dumbbell(rows) : "";
      $("#share-say").textContent = !moved ? T("shareHint") : big < 4 ? T("shareSmall") : T("shareBig", { n: pts(big) });
    }
    const sw = S.qsweeps[qKey()], svg = $("#wi-chart");
    $("#wi-cap").textContent = T("wiCap", { n: S.clip + 1 });
    if (sw) {
      const dash = { mars: "", "dfd-hr": "6 4", gend: "1.5 3.5" }, desc = [];
      const series = S.models.map((m) => {
        const p = sw.qualities.map((q, i) => [100 - q, sw.scores[m.id][i]]);
        desc.push(`${m.name} ${pct(p[0][1])}% → ${pct(p[p.length - 1][1])}%`);
        return { name: m.name, color: "var(--label)", dash: dash[m.id], pts: p };
      });
      lineChart(svg, series, { h: 130, xs: [0, 90], mark: 100 - S.edits.jpeg });
      svg.insertAdjacentHTML("beforeend", `<text x="26" y="125">${T("quality")} 100</text><text x="230" y="125" text-anchor="end">10</text>`);
      $("#wi-desc").textContent = T("wiDesc") + " " + desc.join("; ");
    } else {
      svg.innerHTML = `<text x="150" y="65" text-anchor="middle">${esc(sw === null ? T("rescoring") : T("wiNeed"))}</text>`;
      $("#wi-desc").textContent = "";
    }
  }

  /* ----------------------------------------------------------------- notes */
  const MIT = window.UF_MIT || null;
  const refByKey = (k) => S.refs.find((r) => r.key === k);
  const refLink = (k) => { const r = refByKey(k); return r ? `<button type="button" class="cite" data-ref="${esc(k)}" aria-label="${T("reference")}: ${esc(r.title)}">${esc(r.label)}</button>` : ""; };
  const mitName = (id) => (MIT && MIT.sub[id] ? MIT.sub[id].name : id);
  const mitTip = (id) => { const e = MIT && MIT.sub[id] && MIT.sub[id].examples[0]; return mitName(id) + (e ? ` — e.g. “${e.text.slice(0, 160)}${e.text.length > 160 ? "…" : ""}” (${e.ref})` : ""); };
  const mitChip = (id) => `<span class="rchip" title="${esc(mitTip(id))}">${T("riskTag", { id })}</span>`;
  const KIND = { trust: "trust", provenance: "prov", context: "ctx", footer: "foot" };
  const stateLabel = (s) => ({ at_risk: T("atRisk"), holds: T("stHolds"), partly: T("stPartly"), breaks: T("stBreaks") }[s] || "");
  const pillClass = (s) => (s === "at_risk" || s === "breaks" ? "risk" : s === "partly" ? "neutral" : "ok");
  function asNote(n) {
    const tagShort = n.tag.split(" + ").map((c) => { const a = ASSUMPTIONS.find((x) => x.code === c); return a ? L(a.short) : c; }).join(", ");
    return { kind: KIND[n.kind], tag: n.kind === "trust" ? n.tag : n.title, name: n.kind === "trust" ? (n.title.includes(":") ? n.title.split(":")[0].replace(n.tag, "").trim() : tagShort) : "", state: n.state, text: n.text, detail: n.detail || "", cite: n.references, risk: n.mit, model: n.model ? S.R.models[n.model].name : null };
  }
  function noteRow(n) {
    const pill = n.state && n.state !== "info" ? `<span class="pill ${pillClass(n.state)}">${stateLabel(n.state)}</span>` : "";
    const needs = n.model ? `<span class="needs">${esc(n.model)}</span>` : "";
    return `<li class="note${n.state === "at_risk" ? " is-risk" : ""}"><div class="note-top">${n.tag ? `<b class="ntag">${esc(n.tag)}</b>${n.name ? `<span class="nname">${esc(n.name)}</span>` : ""}` : ""}${pill}${needs}</div><p>${esc(n.text)}</p>${n.detail ? `<details class="nd"><summary>${T("details")}</summary><p>${esc(n.detail)}</p></details>` : ""}<div class="note-refs">${n.cite.map(refLink).join("")}${n.risk.map(mitChip).join("")}</div></li>`;
  }
  function renderNotes() {
    if (!S.R) return;
    const notes = S.R.notes.map(asNote), of = (k) => notes.filter((n) => n.kind === k);
    const block = (h, q, rows, empty, extra) => `<div class="nk"><div class="nk-head"><h4>${h}</h4>${extra || ""}</div><p class="nk-q">${q}</p>${rows.length ? `<ul class="notes">${rows.map(noteRow).join("")}</ul>` : `<p class="nk-empty">${empty}</p>`}</div>`;
    const jurSeg = `<div class="seg seg-small" role="group" aria-label="${T("jur")}" id="jur">${["all", "eu", "ca", "us"].map((j) => `<button type="button" data-jur="${j}" aria-pressed="${S.jur === j}">${T("jur" + { all: "All", eu: "EU", ca: "CA", us: "US" }[j])}</button>`).join("")}</div>`;
    const trustExtra = `<p class="nk-empty">${T("trustWhen")}</p><button type="button" class="linkish nk-def" data-goto="learn">${T("whatAreA")}</button>`;
    $("#notes").innerHTML =
      block(T("kTrust"), trustLine() || T("kTrustQ"), of("trust"), T("noTrust")).replace(/<\/div>$/, trustExtra + "</div>") +
      block(T("kProv"), T("kProvQ"), of("prov"), "") +
      block(T("kCtx"), T("kCtxQ"), of("ctx"), T("noCtx"), jurSeg);
    const f = of("foot")[0];
    $("#notes-foot").innerHTML = f ? `${esc(f.text)} ${f.cite.map(refLink).join(" ")} ${f.risk.map(mitChip).join(" ")}` : "";
  }
  /* One line on top of the trust notes: how many checks hold, and what makes a score break. */
  function trustLine() {
    const tests = S.R.notes.filter((n) => n.kind === "trust" && ["holds", "partly", "breaks"].includes(n.state));
    if (!tests.length) return "";
    const breaks = [...new Set(tests.filter((n) => n.state === "breaks").map((n) => n.tag))].map((t) => T("when" + t));
    return T("trustLine", { h: tests.filter((n) => n.state === "holds").length, n: tests.length }) + (breaks.length ? " " + T("trustBreaks", { t: breaks.join(LANG() === "fr" ? " ou " : " or ") }) : "");
  }
  async function reloadNotes() {
    if (!S.R) return;
    try {
      const out = await api(`jobs/${S.job}/notes`, { jurisdiction: S.jur, language: LANG() });
      S.R.notes = out.notes; if (out.reading) S.R.reading = out.reading;
      (out.references || []).forEach((r) => { if (!refByKey(r.key)) S.refs.push(r); });
      renderNotes(); renderModels();
    } catch (_) { /* keep the current notes */ }
  }
  document.addEventListener("click", (e) => { const b = e.target.closest("[data-jur]"); if (!b || !S.R) return; S.jur = b.dataset.jur; reloadNotes().then(() => { const x = $(`[data-jur="${S.jur}"]`); if (x) x.focus(); }); });
  const SEVERITY = { breaks: 3, at_risk: 3, partly: 2, holds: 1 };
  function renderTrustSum() {
    const el = $("#trust-sum"); if (!el || !S.R) return;
    const groups = new Map();
    S.R.notes.filter((n) => n.kind === "trust" && n.state !== "info").forEach((n) => { if (!groups.has(n.tag)) groups.set(n.tag, []); groups.get(n.tag).push(n); });
    const chips = [...groups.entries()].map(([tag, ns]) => {
      const states = [...new Set(ns.map((n) => n.state))], worst = ns.reduce((w, n) => (SEVERITY[n.state] > SEVERITY[w] ? n.state : w), ns[0].state);
      const main = stateLabel(ns[0].state);
      const odd = states.length === 1 ? "" : ", " + ns.filter((n) => n.state !== ns[0].state).map((n) => `${S.R.models[n.model] ? S.R.models[n.model].name : ""} ${stateLabel(n.state).toLowerCase()}`).join(", ");
      const title = ns.map((n) => `${n.model ? S.R.models[n.model].name + ": " : ""}${stateLabel(n.state)}`).join(" · ");
      return `<span class="pill ${pillClass(worst)}" title="${esc(title)}">${esc(tag)} · ${esc(main)}${esc(odd)}</span>`;
    });
    el.innerHTML = `<span class="ts-l">${T("trustSum")}</span>${chips.length ? chips.join("") : `<span class="secondary">${T("noFlags")}</span>`}<button type="button" class="linkish" data-goto="notes">${T("seeNotes")}</button>`;
  }
  document.addEventListener("click", (e) => { const b = e.target.closest("[data-goto]"); if (!b || S.guessing) return; selectTab(b.dataset.goto); $("#tab-" + b.dataset.goto).focus(); });

  /* ----------------------------------------------------------------- learn */
  function renderMit() {
    const el = $("#mit"); if (!el) return;
    if (!MIT) { el.innerHTML = `<p class="foot pad">${T("mitMissing")}</p>`; return; }
    const m = MIT.meta, top = Object.entries(m.deepfake_by_subdomain).filter(([k]) => MIT.sub[k]).slice(0, 3).map(([k, v]) => `<li><b>${k}</b> ${esc(mitName(k))} <span class="secondary">(${v})</span></li>`).join("");
    el.innerHTML = `<ul class="facts"><li>${T("mitTotal", { t: fmtN(m.risks_total), v: esc(m.version) })} ${refLink("mit-ai-risk")}</li><li>${T("mitDeep", { n: m.deepfake_entries_total })}</li>` +
      `<li>${T("mitTop")}<ol>${top}</ol></li></ul><p class="foot pad">${T("mitUses")}</p>` +
      Object.keys(MIT.sub).filter((id) => id !== "1.3").map((id) => { const s = MIT.sub[id], e = s.examples[0]; return `<details class="disc"><summary><b>${id}</b><span class="t">${esc(mitName(id))}</span>${s.deepfake_entries ? `<span class="s">${T("nEntries", { n: s.deepfake_entries })}</span>` : ""}</summary><p class="body">${esc(s.description)}<br><span class="secondary">${T("example")}: “${esc(e.text)}” (${esc(e.ref)}, ${e.ev})</span></p></details>`; }).join("") +
      `<p class="foot pad">${esc(m.license)} · ${esc(m.cite)}</p>`;
  }
  function renderLearn() {
    renderMit();
    $("#assume").innerHTML = ASSUMPTIONS.map((a) => `<details class="disc"><summary><b>${a.code}</b><span class="t">${esc(L(a.short))}</span></summary><p class="body">${esc(L(a.plain))} <b>${T("breaksWhen")}</b> ${esc(L(a.breaks))} ${refLink("survey")}</p></details>`).join("") + `<p class="foot pad">${T("a25Note")}</p>`;
    $("#refs").innerHTML = S.refs.map((r) => `<li id="ref-${esc(r.key)}"><span class="rid">${esc(r.label)}</span><a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title)}</a></li>`).join("");
    renderBaseRate();
  }
  document.addEventListener("click", (e) => {
    const c = e.target.closest(".cite"); if (!c || S.guessing) return;
    $$("dialog[open]").forEach((d) => d.close());
    selectTab("learn");
    const det = $("#refs").closest("details"); det.open = true;
    const li = document.getElementById("ref-" + c.dataset.ref); if (!li) return;
    li.scrollIntoView({ block: "nearest", behavior: RM ? "auto" : "smooth" }); li.classList.add("flash"); setTimeout(() => li.classList.remove("flash"), 1400);
  });
  const PREV = [10, 100, 1000, 10000];
  function renderBaseRate() {
    const p = 1 / S.prevalence, prec = (p * 0.9) / (p * 0.9 + (1 - p) * 0.01), v = prec >= 0.1 ? fmtN(prec * 100, 0) : fmtN(prec * 100, 1);
    $("#br-big").innerHTML = T("brBig", { v: `<span>${v}%</span>` });
    $("#br-text").innerHTML = T("brText", { n: fmtN(S.prevalence), v }) + " " + refLink("survey");
    $("#br-seg").innerHTML = PREV.map((n) => `<button type="button" data-p="${n}" aria-pressed="${n === S.prevalence}">${T("oneIn", { n: fmtN(n) })}</button>`).join("");
  }
  $("#br-seg").addEventListener("click", (e) => { const b = e.target.closest("[data-p]"); if (b) { S.prevalence = +b.dataset.p; renderBaseRate(); $(`[data-p="${S.prevalence}"]`).focus(); } });

  /* ------------------------------------------------- report, exports, ask */
  /* The server builds the report (template numbers, optional LLM prose, the session's tests) and keeps it
     until the tests or notes change, so the sheet and every download say the same thing. */
  const sheet = $("#sheet");
  const addRefs = (refs) => (refs || []).forEach((r) => { if (!refByKey(r.key)) S.refs.push(r); });
  /* Only references the rules attached become links; any other [key] is dropped. */
  const renderCited = (text, keys) => text.split(/\n{2,}/).map((p) => `<p>${esc(p).replace(/\s?\[([a-z0-9-]+)\]/g, (_, k) => (keys.includes(k) && refByKey(k) ? " " + refLink(k) : ""))}</p>`).join("");
  function reportHtml(r) {
    const keys = (r.references || []).map((x) => x.key);
    const head = `<tr><th scope="col">${T("detector")}</th><th scope="col">${T("likelihood")}</th><th scope="col">${T("rpRank")}</th><th scope="col" class="rng">${T("rpRanges")}</th></tr>`;
    const rows = (r.rows || []).map((x) => `<tr><td>${esc(x.model)}</td><td><b>${esc(x.fake)}</b></td><td>${esc(x.rank)}</td><td class="rng">${esc(x.ranges)}</td></tr>`).join("");
    const tried = (r.experiments || []).length ? `<h4>${T("rpTried")}</h4><ul class="tried">${r.experiments.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
    const notes = (r.notes || []).map(asNote), groups = [["trust", "kTrust"], ["prov", "kProv"], ["ctx", "kCtx"]];
    const nh = `<h4>${T("notesInReport")}</h4>` + groups.map(([k, h]) => { const rs = notes.filter((n) => n.kind === k); return rs.length ? `<p class="rn-h">${T(h)}</p><ul class="notes notes-report">${rs.map(noteRow).join("")}</ul>` : ""; }).join("");
    const refs = (r.references || []).length ? `<h4>${T("citedRefs")}</h4><ol class="rrefs">${r.references.map((x) => `<li><b>${esc(x.label)}</b> · <a href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.title)}</a></li>`).join("")}</ol>` : "";
    const vline = Object.entries(r.versions || {}).map(([k, v]) => `${k} ${v}`).join(" · ");
    const fig = $("#card-frame").checked ? `<figure class="rframe"><img alt="" src="${frameUrl(srcFrame(S.clip, S.frame))}"><figcaption class="secondary">${esc(T("frameCap", { n: srcFrame(S.clip, S.frame) + 1, c: S.clip + 1 }))}</figcaption></figure>` : "";
    return `<h3>${esc(S.src.name)}</h3>${renderCited(r.summary, keys)}<h4>${T("scoresH")}</h4><div class="tbl-wrap"><table class="rtable">${head}${rows}</table></div>${fig}${tried}${nh}${refs}<p class="src">${esc(vline)}</p><p class="disc-line">${esc(r.disclaimer || "")}</p>`;
  }
  let reportTok = 0;
  async function openReport() {
    if (S.guessing || !S.R) return;
    stopPlay();
    const tok = ++reportTok;
    $("#dl").open = false; $("#sk").hidden = false; $("#report").hidden = true; $("#payload-box").hidden = true;
    $("#report-by").textContent = S.health && S.health.llm ? T("rpWriting", { m: S.health.llm_model }) : "";
    if (!sheet.open) sheet.showModal();
    try {
      const r = await api(`jobs/${S.job}/report`, { format: "json", language: LANG() });
      if (tok !== reportTok) return;
      S.report = r; addRefs(r.references);
      $("#report").innerHTML = reportHtml(r);
      $("#report-by").textContent = r.by || "";
      $("#payload-box").hidden = r.llm_status === "off" || r.llm_status === "disabled";
      $("#payload").textContent = JSON.stringify(r.payload, null, 2);
    } catch (e) {
      if (tok !== reportTok) return;
      S.report = null; $("#report").innerHTML = `<p>${esc(T("rpFailed", { e: e.message }))}</p>`; $("#report-by").textContent = "";
    }
    $("#sk").hidden = true; $("#report").hidden = false;
  }
  $("#open-report").addEventListener("click", openReport);
  $("#card-frame").addEventListener("change", () => { if (S.report) $("#report").innerHTML = reportHtml(S.report); });
  sheet.addEventListener("close", () => { reportTok++; $("#dl").open = false; });
  const stem = () => (S.src && S.src.name ? S.src.name : "video").replace(/\.[^.]+$/, "");
  const flash = (b, txt) => { const o = b.textContent; b.textContent = txt; setTimeout(() => { b.textContent = o; }, 1600); };
  function saveBlob(name, blob) {
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  $("#copy").addEventListener("click", async (e) => {
    const b = e.currentTarget; if (!S.report) return;
    try { await navigator.clipboard.writeText(S.report.markdown); flash(b, T("copied")); } catch (_) { flash(b, T("blocked")); }
  });
  $("#dl").addEventListener("click", async (e) => {
    const b = e.target.closest("[data-fmt]"); if (!b || !S.job) return;
    const fmt = b.dataset.fmt, sum = $("#dl summary"); $("#dl").open = false;
    try {
      const frame = $("#card-frame").checked && fmt !== "markdown" ? srcFrame(S.clip, S.frame) : undefined;
      const res = await fetch(`jobs/${S.job}/report`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ format: fmt, language: LANG(), frame }) });
      if (!res.ok) throw new Error(res.statusText);
      saveBlob(`${stem()}-report.${{ pdf: "pdf", html: "html", markdown: "md" }[fmt]}`, await res.blob()); flash(sum, T("saved"));
    } catch (_) { flash(sum, T("rpDlFailed")); }
  });
  $("#dl").addEventListener("keydown", (e) => { if (e.key === "Escape" && $("#dl").open) { e.preventDefault(); e.stopPropagation(); $("#dl").open = false; $("#dl summary").focus(); } });
  /* The session file: the full result, the tests the server measured, the report text and the guess. Saved locally. */
  $("#export").addEventListener("click", () => {
    $("#dl").open = false; if (!S.R) return;
    const r = S.report, data = {
      app: "MARScope", schema: "marscope.session.v1", exported_at: new Date().toISOString(), language: LANG(), guess: S.guess, jurisdiction: S.jur,
      result: S.R, experiments: r ? r.records : [], report: r ? { summary: r.summary, by: r.by, markdown: r.markdown } : null,
      references: S.refs.map(({ key, title, url }) => ({ key, title, url })),
    };
    saveBlob(`${stem()}-session.json`, new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })); flash($("#dl summary"), T("saved"));
  });

  /* Evidence card: one PNG drawn here from the result, in the light theme. A frame only when asked. */
  function rr(x, X, Y, w, h, r) { x.beginPath(); if (x.roundRect) x.roundRect(X, Y, w, h, r); else x.rect(X, Y, w, h); }
  function fit(x, text, maxW) { if (x.measureText(text).width <= maxW) return text; let t = text; while (t.length > 1 && x.measureText(t + "…").width > maxW) t = t.slice(0, -1); return t + "…"; }
  function wrap(x, text, X, Y, maxW, lh, maxLines) {
    let line = "", n = 0;
    for (const w of text.split(" ")) {
      const t = line ? line + " " + w : w;
      if (x.measureText(t).width > maxW && line) { if (++n >= maxLines) { x.fillText(fit(x, line + "…", maxW), X, Y); return; } x.fillText(line, X, Y); Y += lh; line = w; } else line = t;
    }
    if (line) x.fillText(line, X, Y);
  }
  async function evidenceCard(withFrame) {
    const W = 1200, H = 675, c = document.createElement("canvas"); c.width = W; c.height = H;
    const x = c.getContext("2d"), F = (w, sz) => `${w} ${sz}px "Open Sans", -apple-system, "Helvetica Neue", Arial, sans-serif`;
    const INK = "#15140f", SEC = "#6e6e73";
    x.fillStyle = "#fff"; x.fillRect(0, 0, W, H);
    x.fillStyle = INK; x.font = F(700, 22); x.fillText("MARScope", 40, 52);
    x.fillStyle = SEC; x.font = F(600, 15); x.textAlign = "right"; x.fillText(T("evidenceCard") + " · " + new Date().toISOString().slice(0, 10), W - 40, 50); x.textAlign = "left";
    x.fillStyle = INK; x.font = F(700, 26); x.fillText(fit(x, S.src.name, W - 80), 40, 98);
    x.fillStyle = SEC; x.font = F(500, 15); x.fillText(`${t2s(S.src.duration)} · ${S.src.width}×${S.src.height} · ${$("#cred-open").textContent}`, 40, 124);
    let rx = 40;
    if (withFrame) {
      const img = await image(srcFrame(S.clip, S.frame)), fx = 40, fy = 150, fw = 620, fh = 349;
      x.fillStyle = "#000"; rr(x, fx, fy, fw, fh, 16); x.fill();
      if (img) { const k = Math.min(fw / img.naturalWidth, fh / img.naturalHeight), w = img.naturalWidth * k, h = img.naturalHeight * k; x.save(); rr(x, fx, fy, fw, fh, 16); x.clip(); x.drawImage(img, fx + (fw - w) / 2, fy + (fh - h) / 2, w, h); x.restore(); }
      x.fillStyle = SEC; x.font = F(500, 13); x.fillText(T("clipN", { n: S.clip + 1 }) + " · " + T("frameN", { n: srcFrame(S.clip, S.frame) + 1 }), fx, fy + fh + 22);
      rx = 700;
    }
    const rw = W - 40 - rx, rd = S.R.reading; let y = 164;
    if (rd) {  /* the reading first: what an executive reads before the numbers */
      x.fillStyle = INK; x.font = F(700, 20); x.fillText(fit(x, rd.headline, rw), rx, y);
      x.fillStyle = SEC; x.font = F(500, 13); wrap(x, rd.reasons.join(" "), rx, y + 22, rw, 18, 2);
      y += 66;
    }
    x.fillStyle = INK; x.font = F(700, 16); x.fillText(T("likelihood"), rx, y);
    S.models.forEach((m) => {
      const s = S.R.models[m.id]; y += 18;
      x.fillStyle = INK; x.font = F(700, 18); x.fillText(m.name, rx, y + 14);
      const ra = rarity(s.context.real_rank, m.name), [lo, hi] = s.context.real_range;
      x.fillStyle = INK; x.font = F(700, 13); x.fillText(fit(x, ra.label, rw - 90), rx, y + 34);
      x.fillStyle = "#ececf0"; rr(x, rx, y + 58, rw, 8, 4); x.fill();
      x.fillStyle = "rgba(0, 114, 178, 0.45)"; rr(x, rx + rw * lo, y + 58, Math.max(3, rw * (hi - lo)), 8, 4); x.fill();
      x.fillStyle = INK; x.beginPath(); x.arc(rx + rw * clamp(s.fake, 0, 1), y + 62, 6, 0, 2 * Math.PI); x.fill();
      x.fillStyle = SEC; x.font = F(500, 11); x.fillText(T("realUsually", { a: pct(lo), b: pct(hi) }), rx, y + 80);
      y += 86;
    });
    const chips = [...$("#trust-sum").querySelectorAll(".pill")].map((p) => p.textContent), ay = 572; let ax = 40;
    x.font = F(700, 13); x.fillStyle = INK; x.fillText(T("kTrust"), 40, ay - 12);
    (chips.length ? chips : [T("noFlags")]).forEach((label) => { x.font = F(600, 12); const w = x.measureText(label).width + 20; if (ax + w > W - 40) return; x.fillStyle = "#f5f5f7"; rr(x, ax, ay, w, 26, 13); x.fill(); x.fillStyle = INK; x.fillText(label, ax + 10, ay + 17); ax += w + 8; });
    const v = S.R.versions, ck = Object.entries(v.checkpoints).map(([k, h]) => `${S.R.models[k] ? S.R.models[k].name : k} ${String(h).slice(0, 7)}`).join(" · ");
    x.fillStyle = SEC; x.font = F(500, 12);
    x.fillText(fit(x, T("cardFooter"), W - 80), 40, H - 40);
    x.fillText(fit(x, `MARScope ${v.package} · ${T("policyV", { v: v.policy })} · ${ck}`, W - 80), 40, H - 20);
    return new Promise((ok) => c.toBlob(ok, "image/png"));
  }
  $("#card-png").addEventListener("click", async (e) => {
    const b = e.currentTarget; if (!S.R) return;
    const blob = await evidenceCard($("#card-frame").checked);
    if (blob) { saveBlob(`${stem()}-evidence-card.png`, blob); flash(b, T("saved")); }
  });

  /* Ask: suggested questions get fixed answers from the result (offline); free questions go to the LLM
     with the same facts and filter, and a verdict question always gets the fixed no-verdict answer. */
  function renderAsk() {
    if (!S.R) return;
    const chips = [S.models.length > 1 && ["disagree", S.R.agreement.disagree ? "chipDisagree" : "chipAgree"], ["tests", "chipTests"], ["certain", "chipCertain"], ["verdict", "chipVerdict"], ["deepfake", "chipDeepfake"]].filter(Boolean);
    $("#ask-chips").innerHTML = chips.map(([id, k]) => `<button type="button" class="chip" data-chip="${id}">${esc(T(k))}</button>`).join("");
    const on = !!(S.health && S.health.llm);
    $("#ask-q").disabled = false; $("#ask-send").disabled = false;
    $("#ask-q").placeholder = T("askPh");
    $("#ask-hint").hidden = on; $("#ask-hint").textContent = on ? "" : T("askOff");
    $("#answer").hidden = true; $("#answer").innerHTML = "";
  }
  let askTok = 0;
  async function askServer(body, label) {
    const tok = ++askTok, box = $("#answer");
    box.hidden = false; box.setAttribute("aria-busy", "true");
    box.innerHTML = `<p class="who">${esc(label)}</p><p class="thinking">${T("thinking")}</p>`;
    try {
      const a = await api(`jobs/${S.job}/ask`, Object.assign({ language: LANG() }, body));
      if (tok !== askTok) return;
      addRefs(a.references);
      let who;
      if (a.source === "card") who = T("whoCard");
      else if (a.source === "fixed") who = a.refused && body.question ? T("whoRefused") : T("whoFixed");
      else if (a.text) who = T("whoLlm", { m: a.llm.model || "" });
      else who = T({ off: "askOffQ", timeout: "askTimeout", filtered: "askFiltered" }[a.llm_status] || "askFailed");
      box.innerHTML = `<p class="who">${esc(label)}</p>${a.text ? renderCited(a.text, a.references.map((r) => r.key)) : ""}<p class="who">${esc(who)}</p>`;
    } catch (_) { if (tok === askTok) box.innerHTML = `<p class="who">${esc(label)}</p><p>${esc(T("askFailed"))}</p>`; }
    if (tok === askTok) box.removeAttribute("aria-busy");
  }
  $("#ask-chips").addEventListener("click", (e) => { const b = e.target.closest("[data-chip]"); if (b && S.job) askServer({ chip: b.dataset.chip }, b.textContent); });
  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); const q = $("#ask-q").value.trim(); if (!q || !S.job) return; askServer({ question: q }, q); });

  /* ----------------------------------------------------------- model cards */
  function renderCards() {
    const ms = ["mars", "dfd-hr", "gend"].map((id) => S.catalog.find((m) => m.id === id) || FALLBACK_MODELS.find((m) => m.id === id));
    const B = S.bench, best = (vals) => Math.max(...vals);
    const cell = (v, isBest) => v == null ? "<td>—</td>" : `<td class="${isBest ? "best" : ""}" style="--v:${clamp((v - 50) / 50, 0, 1).toFixed(2)}">${fmtN(v, 2)}</td>`;
    let h = "";
    if (ms.some((m) => m.typical)) {
      h += `<h3>${T("posHead")}</h3><p class="secondary">${T("posIntro")}</p>`;
      h += `<div class="tbl-wrap"><table class="tbl"><thead><tr><th scope="col">${T("detector")}</th><th scope="col">${T("posMid")}</th><th scope="col">${T("posHigh")}</th><th scope="col">${T("posFakeMid")}</th></tr></thead><tbody>`;
      ms.forEach((m) => { const r = m.typical; if (r) h += `<tr><th scope="row">${esc(m.name)}</th><td>${pct(r.real_p50)}%</td><td>${pct(r.real_p95)}%</td><td>${pct(r.fake_p50)}%</td></tr>`; });
      h += "</tbody></table></div>";
    }
    if (B) {
      const sets = B.sets, A = (m) => B.models[m.id];
      h += `<h3>${T("bySet")}</h3><p class="secondary">${T("cardsIntro")} ${refLink("mars")}</p><div class="tbl-wrap"><table class="tbl"><thead><tr><th scope="col">${T("detector")}</th>${sets.map((s) => `<th scope="col">${esc(s)}</th>`).join("")}<th scope="col">${T("avg")}</th></tr></thead><tbody>`;
      ms.forEach((m) => { const a = A(m); h += `<tr><th scope="row">${esc(m.name)}</th>${a.sets.map((v, j) => cell(v, v === best(ms.map((x) => A(x).sets[j])))).join("")}${cell(a.avg, a.avg === best(ms.map((x) => A(x).avg)))}</tr>`; });
      h += `</tbody></table></div><p class="secondary small">${T("setGroups")}</p>`;
      h += `<h3>${T("byGen")}</h3><div class="tbl-wrap"><table class="tbl tbl-gen"><thead><tr><th scope="col">${T("generator")}</th>${ms.map((m) => `<th scope="col">${esc(m.name)}</th>`).join("")}</tr></thead><tbody>`;
      [["genvideo", "GenVideo"], ["df26", "DF26 · " + T("gen2026")]].forEach(([set, title]) => {
        h += `<tr class="grp"><th colspan="${ms.length + 1}" scope="rowgroup">${title}</th></tr>`;
        GEN_ORDER[set].forEach((g) => { const vals = ms.map((m) => (A(m).gens[set] || {})[g]); h += `<tr><th scope="row">${esc(GEN_NAMES[g] || g)}</th>${vals.map((v) => cell(v, v === best(vals.filter((x) => x != null)))).join("")}</tr>`; });
      });
      h += "</tbody></table></div>";
    }
    h += ms.map((m) => `<div class="card"><b>${esc(m.name)}</b> <span class="secondary">${esc(T("kind_" + m.id))}</span><p>${esc(m.id === "mars" ? T("ev_mars") : T("ev_other"))}</p>${m.card ? `<p class="secondary"><a href="${esc(m.card)}" target="_blank" rel="noopener">${esc(m.repo || m.card)}</a></p>` : ""}</div>`).join("");
    $("#cards-body").innerHTML = h;
  }
  $("#cards-open").addEventListener("click", () => { renderCards(); $("#cards").showModal(); });

  /* ----------------------------------------------------------- credentials */
  function renderCred() {
    const c = S.src.cred; let h = "";
    if (c.status === "found") h += `<div class="cred-top"><span class="cred-mark ok" aria-hidden="true"><svg width="16" height="16" viewBox="0 0 24 24" class="ic"><path d="M5 12.5l4.5 4.5L19 7"/></svg></span><div><b>${T("credFound")}</b><p class="secondary">${esc(T("credFoundNote", { s: c.summary || "" }))}</p></div></div><p class="secondary">${c.declares_ai ? "" : ""}${refLink("c2pa")}</p>`;
    else if (c.status === "absent") h += `<div class="cred-top"><span class="cred-mark" aria-hidden="true">–</span><div><b>${T("credNone")}</b><p class="secondary">${T("stdCredNone")}</p></div></div><p class="secondary">${T("credWhy")} ${refLink("c2pa")} ${refLink("eu-ai-act-art50")}</p>`;
    else if (c.status === "unsupported") h += `<div class="cred-top"><span class="cred-mark" aria-hidden="true">–</span><div><b>${T("credUnsupported")}</b><p class="secondary">${T("credUnsupportedNote")}</p></div></div>`;
    else h += `<div class="cred-top"><span class="cred-mark" aria-hidden="true">?</span><div><b>${T("credError")}</b><p class="secondary">${T("credErrorNote")}</p></div></div>`;
    $("#cred-body").innerHTML = h;
  }
  $("#cred-open").addEventListener("click", () => { renderCred(); $("#cred").showModal(); });
  $$("dialog").forEach((d) => { d.addEventListener("click", (e) => { if (e.target === d || e.target.closest("[data-close]")) d.close(); }); });

  /* ------------------------------------------------------------------ init */
  main.cv = $("#frame");
  MS.onChange((what) => {
    if (what === "lang") { MS.applyStatic(); renderStart(); if (S.R && !V.analyze.hidden) { renderAll(); reloadNotes().then(() => { if (sheet.open) openReport(); }); } }
    if (what === "theme" && S.R && !V.analyze.hidden) { renderTimeline(); renderStage(); }
  });
  document.addEventListener("marscope:ready", async (e) => {
    S.health = e.detail;
    renderStart();
    MS.applyStatic($("#app-root"));
    try { S.catalog = await api("models"); } catch (_) { /* keep the fallback list */ }
    try { S.refs = await api("references"); } catch (_) { S.refs = []; }
    try { S.bench = await api("benchmark"); } catch (_) { S.bench = null; }
    renderStart();
    if (V.run.hidden && V.analyze.hidden) show("start", false); // never pull the user out of a run started meanwhile
  });
})();
