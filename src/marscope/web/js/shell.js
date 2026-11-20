/* MARScope shell: language and theme switchers, shared strings. Used by index.html and app.html.
   Language and theme are user choices remembered in the browser; light and English by default. */
(function () {
  "use strict";

  const SHELL = {
    en: {
      tagline: "Video deepfake detectors, explained", docs: "Docs", faq: "FAQ", openApp: "Open the app",
      langLabel: "Language", themeLabel: "Theme", light: "Light", dark: "Dark",
      definition: "<strong>How MARScope reads a score</strong> Each detector gives a probability that a video is fake. MARScope compares that number with the scores the same detector gave thousands of real and fake test videos, shows the evidence behind it, and tests how far to trust it. Detectors can be wrong and can disagree, so MARScope leaves the final judgement to people who can check the source.",
      legal: "MARScope is an educational tool. Its scores are model outputs and can be wrong. Legal, forensic, journalistic and employment decisions, and claims about a person, need qualified experts and other evidence. Policy notes are general information, without legal advice. Supported by the Province of Ontario, the Government of Canada through CIFAR, and the EU Horizon Europe AIXPERT project (grant 101214389). Provided as is under the Apache 2.0 licence, without warranty of any kind.",
      copyright: "© The Vector Institute for Artificial Intelligence · Apache 2.0",
    },
    fr: {
      tagline: "Les détecteurs d’hypertrucages vidéo, expliqués", docs: "Docs", faq: "FAQ", openApp: "Ouvrir l’application",
      langLabel: "Langue", themeLabel: "Thème", light: "Clair", dark: "Sombre",
      definition: "<strong>Comment MARScope lit un score</strong> Chaque détecteur donne une probabilité qu’une vidéo soit fausse. MARScope compare ce nombre aux scores que le même détecteur a donnés à des milliers de vraies et de fausses vidéos de test, montre les indices qui le sous-tendent et teste jusqu’où s’y fier. Les détecteurs peuvent se tromper et diverger : MARScope laisse le jugement final aux personnes qui peuvent vérifier la source.",
      legal: "MARScope est un outil pédagogique. Ses scores sont des sorties de modèles et peuvent se tromper. Les décisions juridiques, judiciaires, journalistiques ou d’emploi, et toute affirmation sur une personne, demandent des experts qualifiés et d’autres éléments. Les notes de politique sont de l’information générale, sans avis juridique. Avec le soutien de la province de l’Ontario, du gouvernement du Canada par l’intermédiaire du CIFAR et du projet AIXPERT d’Horizon Europe (subvention 101214389). Fourni tel quel sous licence Apache 2.0, sans garantie.",
      copyright: "© L’Institut Vecteur pour l’intelligence artificielle · Apache 2.0",
    },
  };
  const LANGS = { en: "EN", fr: "FR-CA" };
  const THEMES = ["light", "dark"];
  const listeners = [];

  function store(key, value) {
    try { if (value === undefined) return localStorage.getItem(key); localStorage.setItem(key, value); } catch (_) { /* storage blocked */ }
    return null;
  }
  let lang = store("marscope-lang") === "fr" ? "fr" : "en";
  let theme = store("marscope-theme") === "dark" ? "dark" : "light";

  // Strings: pages register their own tables; the shell table is always available.
  const tables = [SHELL];
  function t(key, vars) {
    let s = null;
    for (const tb of tables) { if (tb[lang] && tb[lang][key] != null) { s = tb[lang][key]; break; } }
    if (s == null) for (const tb of tables) { if (tb.en && tb.en[key] != null) { s = tb.en[key]; break; } }
    if (s == null) s = key;
    if (vars) Object.keys(vars).forEach((k) => { s = s.split("{" + k + "}").join(String(vars[k])); });
    return s;
  }

  function applyStatic(root) {
    const r = root || document;
    document.documentElement.lang = lang === "fr" ? "fr-CA" : "en";
    r.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    r.querySelectorAll("[data-i18n-html]").forEach((el) => { el.innerHTML = t(el.dataset.i18nHtml); });
    r.querySelectorAll("[data-i18n-aria]").forEach((el) => { const s = t(el.dataset.i18nAria); el.setAttribute("aria-label", s); if (el.dataset.tip !== "no") el.title = s; });
    r.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
    renderSwitchers();
  }

  function applyTheme() {
    if (theme === "dark") document.documentElement.dataset.theme = "dark";
    else document.documentElement.dataset.theme = "light";
  }

  function renderSwitchers() {
    document.querySelectorAll("[data-switch]").forEach((sw) => {
      const kind = sw.dataset.switch, trigger = sw.querySelector(".switch-trigger"), menu = sw.querySelector(".switch-menu");
      const current = kind === "lang" ? LANGS[lang] : t(theme);
      trigger.textContent = current;
      trigger.setAttribute("aria-label", `${t(kind === "lang" ? "langLabel" : "themeLabel")}: ${current}`);
      const options = kind === "lang" ? Object.keys(LANGS).filter((k) => k !== lang).map((k) => [k, LANGS[k]]) : THEMES.filter((k) => k !== theme).map((k) => [k, t(k)]);
      menu.innerHTML = options.map(([v, label]) => `<button type="button" class="switch-option" role="option" data-value="${v}">${label}</button>`).join("");
    });
  }

  function closeMenus(except) {
    document.querySelectorAll("[data-switch]").forEach((sw) => {
      if (sw === except) return;
      sw.querySelector(".switch-menu").hidden = true;
      sw.querySelector(".switch-trigger").setAttribute("aria-expanded", "false");
    });
  }

  function initSwitchers() {
    document.querySelectorAll("[data-switch]").forEach((sw) => {
      const trigger = sw.querySelector(".switch-trigger"), menu = sw.querySelector(".switch-menu");
      trigger.setAttribute("aria-haspopup", "listbox");
      trigger.addEventListener("click", (e) => {
        e.stopPropagation();
        const open = menu.hidden;
        closeMenus(sw);
        menu.hidden = !open;
        trigger.setAttribute("aria-expanded", String(open));
        if (open) { const first = menu.querySelector(".switch-option"); if (first) first.focus(); }
      });
      menu.addEventListener("click", (e) => {
        const b = e.target.closest("[data-value]"); if (!b) return;
        if (sw.dataset.switch === "lang") setLang(b.dataset.value); else setTheme(b.dataset.value);
        closeMenus(); trigger.focus();
      });
    });
    document.addEventListener("click", () => closeMenus());
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenus(); });
  }

  function setLang(v) { lang = v === "fr" ? "fr" : "en"; store("marscope-lang", lang); applyStatic(); listeners.forEach((fn) => fn("lang")); }
  function setTheme(v) { theme = v === "dark" ? "dark" : "light"; store("marscope-theme", theme); applyTheme(); renderSwitchers(); listeners.forEach((fn) => fn("theme")); }

  applyTheme(); // as early as possible, before first paint of the body
  document.addEventListener("DOMContentLoaded", () => { initSwitchers(); applyStatic(); });

  window.MS = {
    t, lang: () => lang, theme: () => theme, applyStatic,
    addStrings: (table) => { tables.unshift(table); },
    onChange: (fn) => listeners.push(fn),
    esc: (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])),
  };
})();
