/* SkillHub SPA — zero-dependency */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  // ---------------- state ----------------
  let LANG = localStorage.getItem("sh_lang")
    || ((navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en");
  let THEME = localStorage.getItem("sh_theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  let META = null;
  const CAT = id => (META && META.categories.find(c => c.id === id)) || null;
  const t = (k, vars) => {
    let s = (I18N[LANG] && I18N[LANG][k]) || I18N.en[k] || k;
    if (vars) for (const [kk, vv] of Object.entries(vars)) s = s.replace(`{${kk}}`, vv);
    return s;
  };
  // fmtNum — single number formatter for the whole site (contract D10):
  // en → 1.5k / 25.4M ; zh → 1.5万 / 2.5亿. Exposed as window.fmtNum for mcp.js etc.
  const fmtNum = n => {
    n = Number(n) || 0;
    const neg = n < 0 ? "-" : "", a = Math.abs(n);
    const f = (v, u) => v.toFixed(1).replace(/\.0$/, "") + u;
    if (LANG === "zh") return neg + (a >= 1e8 ? f(a / 1e8, "亿") : a >= 1e4 ? f(a / 1e4, "万") : String(a));
    return neg + (a >= 1e6 ? f(a / 1e6, "M") : a >= 1e3 ? f(a / 1e3, "k") : String(a));
  };
  const fmtN = fmtNum; // legacy alias — new code should call fmtNum
  window.fmtNum = fmtNum;
  // fmtRel — bilingual relative time for trust rows ("3d ago" / "3 天前")
  const fmtRel = d => {
    const ts = d instanceof Date ? d.getTime() : Date.parse(d);
    if (!isFinite(ts)) return "";
    const day = Math.max(0, Math.round((Date.now() - ts) / 86400000));
    if (LANG === "zh") return day === 0 ? "今天" : day < 30 ? `${day} 天前` : day < 365 ? `${Math.round(day / 30)} 个月前` : `${Math.round(day / 365)} 年前`;
    return day === 0 ? "today" : day < 30 ? `${day}d ago` : day < 365 ? `${Math.round(day / 30)}mo ago` : `${Math.round(day / 365)}y ago`;
  };
  window.fmtRel = fmtRel;
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---- monochrome line icons per category (SF-symbol flavor, render everywhere) ----
  const ICON_PATHS = {
    "security":     '<path d="M12 3l7 2.6v5.2c0 4.6-3 7.6-7 9.2-4-1.6-7-4.6-7-9.2V5.6z"/><path d="M9.2 12l2 2 3.6-3.8"/>',
    "docs-office":  '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>',
    "frontend":     '<path d="M4 5h16a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1z"/><path d="M3 9h18M9 21h6M12 17v4M6.5 7h.01"/>',
    "data-viz":     '<path d="M4 20V11M10 20V4M16 20v-7M21 20H3"/>',
    "backend":      '<ellipse cx="12" cy="5.5" rx="7.5" ry="2.8"/><path d="M4.5 5.5v12.6c0 1.6 3.4 2.9 7.5 2.9s7.5-1.3 7.5-2.9V5.5M4.5 12c0 1.6 3.4 2.9 7.5 2.9s7.5-1.3 7.5-2.9"/>',
    "ai-ml":        '<rect x="7" y="7" width="10" height="10" rx="2.2"/><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.5 5.5L8 8M18.5 5.5L16 8M5.5 18.5L8 16M18.5 18.5L16 16"/>',
    "agent-meta":   '<circle cx="12" cy="13" r="6.5"/><path d="M12 6.5V3.6M12 3.6h3M9.5 12.4h.01M14.5 12.4h.01M9.8 15.4c.7.6 1.4.9 2.2.9s1.5-.3 2.2-.9"/>',
    "testing":      '<path d="M10 3v5.2L4.8 17a2.2 2.2 0 0 0 2 3.2h10.4a2.2 2.2 0 0 0 2-3.2L14 8.2V3"/><path d="M8.5 3h7M8 14h8"/>',
    "mobile":       '<rect x="7.5" y="2.8" width="9" height="18.4" rx="2.4"/><path d="M11 18.6h2"/>',
    "cloud-devops": '<path d="M17.2 18.2a4.3 4.3 0 0 0 .3-8.6 6 6 0 0 0-11.6-1A4.6 4.6 0 0 0 6.8 18z"/>',
    "research":     '<path d="M4 19.2V5.6A2.1 2.1 0 0 1 6.1 3.5H20v14.7H6.3A2.3 2.3 0 0 0 4 20.5z"/><path d="M20 18.2v3.3H6.3A2.3 2.3 0 0 1 4 19.2M8.5 8h7"/>',
    "media":        '<rect x="3" y="4.5" width="18" height="15" rx="2.6"/><path d="M10.2 9.2l4.8 2.8-4.8 2.8z"/>',
    "game":         '<path d="M6.8 6.5h10.4A4.8 4.8 0 0 1 22 11.3v3.4a4.3 4.3 0 0 1-7.9 2.4l-.5-.8h-3.2l-.5.8A4.3 4.3 0 0 1 2 14.7v-3.4a4.8 4.8 0 0 1 4.8-4.8z"/><path d="M8 10.4v3.4M6.3 12.1h3.4M15.5 10.9h.01M17.6 13.2h.01"/>',
    "web3":         '<path d="M9.4 14.6a4 4 0 0 1 0-5.6l3-3a4 4 0 0 1 5.6 5.6L16.6 13"/><path d="M14.6 9.4a4 4 0 0 1 0 5.6l-3 3a4 4 0 0 1-5.6-5.6L7.4 11"/>',
    "productivity": '<path d="M13 2.5L4.6 13.6h5.8L11 21.5l8.4-11.1h-5.8z"/>',
    "business":     '<rect x="3" y="7.5" width="18" height="12.5" rx="2.4"/><path d="M9 7.5V6a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v1.5M3 12.8h18"/>',
    "marketing":    '<path d="M3.5 10.5v3a1.5 1.5 0 0 0 1.5 1.5h2l7.5 4.5v-15L7 9H5a1.5 1.5 0 0 0-1.5 1.5z"/><path d="M18 9.5a4 4 0 0 1 0 5"/>',
    "chinese":      '<circle cx="12" cy="12" r="9"/><path d="M6.5 9h11M12 5.5v13M8.8 9v3.6a3.2 3.2 0 0 0 6.4 0V9"/>',
    "fun":          '<path d="M12 15.5c3.3 0 6-2.9 6-6.5s-2.7-6-6-6-6 2.4-6 6 2.7 6.5 6 6.5z"/><path d="M11 15.4l-.7 1.8 1.7-.4 1.4 4.2"/>',
    "general":      '<rect x="4" y="4" width="7" height="7" rx="1.8"/><rect x="13" y="4" width="7" height="7" rx="1.8"/><rect x="4" y="13" width="7" height="7" rx="1.8"/><rect x="13" y="13" width="7" height="7" rx="1.8"/>',
  };
  const icon = (catId, size = 16) =>
    `<svg class="cat-ic" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${ICON_PATHS[catId] || ICON_PATHS.general}</svg>`;

  // ---------------- W4-2: two-level category groups (contract C1) ----------------
  const GROUPS = {
    dev:    ["frontend", "backend", "mobile", "testing"],
    secops: ["security", "cloud-devops"],
    ai:     ["ai-ml", "agent-meta"],
    data:   ["data-viz", "research", "docs-office"],
    biz:    ["productivity", "business", "marketing", "web3"],
    cn:     ["chinese", "media", "game", "fun", "general"],
  };
  const CAT_GROUP = {};
  for (const [g, cats] of Object.entries(GROUPS)) for (const c of cats) CAT_GROUP[c] = g;
  const GROUP_ICON = { dev: "frontend", secops: "security", ai: "ai-ml", data: "data-viz", biz: "productivity", cn: "chinese" };
  // capability flags — light up automatically once W4-4 ships the new data fields (contract C2)
  const DATA_FLAGS = { added_at: false, tags: false };
  const sniffFields = items => {
    for (const s of items || []) {
      if (s.added_at) DATA_FLAGS.added_at = true;
      if (s.tags && s.tags.length) DATA_FLAGS.tags = true;
    }
  };
  const catLabel = c => { const x = CAT(c); return x ? (LANG === "zh" ? x.zh : x.en) : c; };

  // ---------------- theme & lang ----------------
  function applyTheme() {
    document.documentElement.dataset.theme = THEME;
    localStorage.setItem("sh_theme", THEME);
  }
  function applyLang() {
    localStorage.setItem("sh_lang", LANG);
    document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
    $("#btnLang").textContent = LANG === "zh" ? "EN" : "中";
    // static copy swaps in place — layout stability comes from CSS width
    // reservation (:lang(zh) rules, owned by W4-1), never from animation
    $$("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
    $$("[data-i18n-ph]").forEach(el => {
      el.placeholder = t(el.dataset.i18nPh, { n: META ? fmtNum(META.stats.total) : "19k+" });
    });
    if (META) renderStats();
    return route(); // re-render current view with new language
  }
  // language switch choreography (R3 §3.9): fade the visible dynamic view out
  // (--dur-1) → swap copy & re-render → fade back in. Opacity only — any
  // translate/size animation during the switch is forbidden (WAVE1 §1.3).
  const cssMs = name => {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v.endsWith("ms") ? parseFloat(v) : v.endsWith("s") ? parseFloat(v) * 1000 : 0;
  };
  let langSwitching = false;
  async function setLang(next) {
    if (next === LANG || langSwitching) return;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dyn = $$("main.view:not([hidden])");
    if (reduced || !dyn.length) { LANG = next; await applyLang(); return; }
    langSwitching = true;
    try {
      dyn.forEach(el => el.classList.add("lang-fading"));
      await new Promise(r => setTimeout(r, cssMs("--dur-1") || 90));
      LANG = next;
      await applyLang();
    } finally {
      requestAnimationFrame(() => dyn.forEach(el => el.classList.remove("lang-fading")));
      langSwitching = false;
    }
  }
  $("#btnTheme").onclick = () => { THEME = THEME === "dark" ? "light" : "dark"; applyTheme(); };
  $("#btnLang").onclick = () => { setLang(LANG === "zh" ? "en" : "zh"); };

  // ---------------- toast ----------------
  let toastTimer = 0;
  function toast(msg) {
    const el = $("#toast");
    el.textContent = msg; el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 1800);
  }
  async function copyText(txt) {
    try { await navigator.clipboard.writeText(txt); } catch {
      const ta = document.createElement("textarea");
      ta.value = txt; document.body.appendChild(ta); ta.select();
      document.execCommand("copy"); ta.remove();
    }
    toast(t("copied"));
  }

  // ---------------- api ----------------
  const api = p => fetch(p).then(r => r.json());

  // ---------------- routing (real paths, with legacy #/ fallback) ----------------
  function parseRoute() {
    let path = location.pathname || "/";
    let search = location.search || "";
    // legacy inbound links like #/skill/x?y — translate to a real path
    if ((path === "/" || path === "") && location.hash.startsWith("#/")) {
      const h = location.hash.slice(1);
      const qi = h.indexOf("?");
      path = qi >= 0 ? h.slice(0, qi) : h;
      search = qi >= 0 ? "?" + h.slice(qi + 1) : "";
    }
    return { path: path || "/", search };
  }
  function go(url) {
    if (url !== location.pathname + location.search) history.pushState({}, "", url);
    route();
  }
  // intercept internal anchor clicks so they use pushState (SSR links + nav)
  document.addEventListener("click", e => {
    const a = e.target.closest && e.target.closest('a[href^="/"]');
    if (!a) return;
    const href = a.getAttribute("href");
    if (!href || a.target === "_blank" || href.startsWith("/download/") || href.startsWith("/api/")) return;
    // static pages live outside the SPA (W4-6's legal/docs pages) — let the browser navigate
    if (href.startsWith("/legal/") || href.startsWith("/docs/") || /\.[a-z0-9]{2,5}([?#]|$)/i.test(href.split("?")[0])) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button) return;
    e.preventDefault();
    go(href);
  });

  // ---------------- skill card ----------------
  function tierBadge(s) {
    return `<span class="badge tier-${s.tier}">${t("tier_" + s.tier)}</span>`;
  }
  const descOf = s => (LANG === "zh" && s.desc_zh) ? s.desc_zh : (s.desc || "");
  function cardHTML(s) {
    const inst = s.installs ? `<span class="sc-installs">↓ ${fmtNum(s.installs)}</span>` : "";
    const feat = s.featured ? `<span class="badge b-feat">★ ${t("badge_curated")}</span>` : "";
    const isNew = s.added_at && (Date.now() - Date.parse(s.added_at)) < 14 * 86400000;
    const fresh = isNew ? `<span class="badge b-new">${t("badge_new")}</span>` : "";
    // trust signal row — every field is optional until W4-4 ships it (contracts C2/C5)
    const trust = [];
    if (Number(s.sources) > 1) trust.push(`<span class="tr-src">${t("trust_sources", { n: s.sources })}</span>`);
    if (s.stars) trust.push(`<span class="tr-star">★ ${fmtNum(s.stars)}</span>`);
    if (s.updated_at) trust.push(`<span class="tr-upd">${t("trust_updated", { t: fmtRel(s.updated_at) })}</span>`);
    return `<div class="skill-card card-edge" data-cat="${esc(s.cat)}" data-id="${s.id}">
      <div class="sc-top">
        <span class="sc-ic">${icon(s.cat, 16)}</span>
        <span class="sc-name">${esc(s.name)}</span>
        ${fresh}${feat}
      </div>
      ${trust.length ? `<div class="sc-trust">${trust.join("")}</div>` : ""}
      <p class="sc-desc">${esc(descOf(s))}</p>
      <div class="sc-meta">
        ${tierBadge(s)}
        <span class="sc-repo">${esc(s.repo)}</span>
        ${inst}
      </div>
    </div>`;
  }
  document.addEventListener("click", e => {
    const card = e.target.closest && e.target.closest(".skill-card[data-id]");
    if (card) go("/skill/" + card.dataset.id);
  });

  // wheel scrolls horizontal rails (mouse wheel = left/right)
  function hWheel(el) {
    if (!el || el._hw) return; el._hw = true;
    el.addEventListener("wheel", e => {
      if (el.scrollWidth <= el.clientWidth) return;
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        el.scrollLeft += e.deltaY;
        e.preventDefault();
      }
    }, { passive: false });
  }
  hWheel($("#topRail"));
  hWheel($("#catChips"));

  // arrow buttons page a horizontal rail; [data-off] hides an arrow at its edge (CSS 已有样式)
  function wireRail(rail, prev, next) {
    if (!rail || !prev || !next) return;
    const step = () => Math.max(240, Math.round(rail.clientWidth * 0.85));
    prev.onclick = () => rail.scrollBy({ left: -step(), behavior: "smooth" });
    next.onclick = () => rail.scrollBy({ left: step(), behavior: "smooth" });
    const upd = () => {
      const max = rail.scrollWidth - rail.clientWidth - 2;
      prev.toggleAttribute("data-off", rail.scrollLeft <= 2);
      next.toggleAttribute("data-off", max <= 0 || rail.scrollLeft >= max);
    };
    rail.addEventListener("scroll", upd, { passive: true });
    window.addEventListener("resize", upd);
    new MutationObserver(upd).observe(rail, { childList: true }); // 卡片异步注入后刷新边界态
    upd();
  }
  wireRail($("#topRail"), $("#railPrev"), $("#railNext"));

  // ---------------- home ----------------
  function renderStats() {
    const s = META.stats;
    // D7: unique leads, then collected / repos / curated
    $("#heroStats").innerHTML = [
      [fmtNum(s.unique || s.total), t("stat_unique")],
      [fmtNum(s.total), t("stat_collected")],
      [s.repos, t("stat_repos")],
      [s.curated, t("stat_curated")],
    ].map(([b, l]) => `<div class="stat"><b>${b}</b><span>${l}</span></div>`).join("");
    // search placeholders quote the unique count (D7) — re-applied after applyLang's pass
    $$("[data-i18n-ph]").forEach(el => {
      el.placeholder = t(el.dataset.i18nPh, { n: fmtNum(s.unique || s.total) });
    });
    const fb = $("#footerBuilt");
    if (fb) fb.textContent = t("built", { t: META.built_at });
  }
  async function renderHome() {
    // six top-level group cards; child categories listed inline (contract C1, WAVE1 §3.2)
    $("#catGrid").innerHTML = Object.entries(GROUPS).map(([g, cats]) => {
      const total = cats.reduce((n, c) => n + ((CAT(c) || {}).count || 0), 0);
      return `<div class="cat-card group-card" data-group="${g}">
        <div class="ico">${icon(GROUP_ICON[g], 24)}</div>
        <div class="nm">${t("group_" + g)}</div>
        <div class="ct">${fmtNum(total)}</div>
        <div class="gc-cats">${cats.map(c => `<span class="gc-cat" data-cat="${c}">${catLabel(c)}</span>`).join("")}</div>
      </div>`;
    }).join("");
    $$("#catGrid .group-card").forEach(el => {
      el.onclick = () => go("/browse?group=" + el.dataset.group);
    });
    $$("#catGrid .gc-cat").forEach(el => {
      el.onclick = e => { e.stopPropagation(); go("/browse?cat=" + el.dataset.cat); };
    });
    const [top, feat] = await Promise.all([
      api("/api/top"),
      api("/api/search?featured=1&size=6"),
    ]);
    sniffFields(top.items); sniffFields(feat.items);
    $("#topRail").innerHTML = top.items.map(cardHTML).join("");
    $("#featuredGrid").innerHTML = feat.items.map(cardHTML).join("");
    // CTA band shows a real, copyable install command for a real top skill (D7)
    const ctaEl = $("#ctaCode");
    if (ctaEl && top.items[0]) {
      const cmd = installCmd(top.items[0], INSTALL_TARGETS[0]);
      ctaEl.textContent = cmd;
      ctaEl.title = t("d_copy_hint");
      ctaEl.onclick = () => copyText(cmd);
    }
    // New arrivals band — hidden until W4-4 ships added_at (contract C2)
    const sec = $("#newSection");
    if (sec) {
      try {
        const nw = await api("/api/search?sort=new&size=12");
        sniffFields(nw.items);
        const fresh = nw.items.filter(s => s.added_at);
        if (fresh.length) {
          sec.hidden = false;
          $("#newRail").innerHTML = fresh.map(cardHTML).join("");
          hWheel($("#newRail"));
        } else sec.hidden = true;
      } catch { sec.hidden = true; }
    }
  }

  // ---------------- browse ----------------
  const TIERS = ["", "official", "popular", "niche", "chinese"];
  const SORTS = ["", "installs", "stars", "name", "new"]; // "" = score, the server default
  const sortLabel = v =>
    v === "" ? t("sort_score") : v === "installs" ? t("sort_pop")
    : v === "stars" ? t("sort_stars") : v === "name" ? t("sort_name") : t("sort_new");
  function browseParams() {
    return new URLSearchParams(parseRoute().search);
  }
  function setBrowseParams(patch) {
    const p = browseParams();
    for (const [k, v] of Object.entries(patch)) {
      if (v === "" || v == null) p.delete(k); else p.set(k, v);
    }
    if (!("page" in patch)) p.delete("page");
    go("/browse" + ([...p].length ? "?" + p.toString() : ""));
  }
  // shareable URL for the current filter state with a patch applied (real <a href> pagination)
  function browseHref(patch) {
    const p = browseParams();
    for (const [k, v] of Object.entries(patch)) {
      if (v === "" || v == null) p.delete(k); else p.set(k, v);
    }
    return "/browse" + ([...p].length ? "?" + p.toString() : "");
  }
  async function renderEmptyState() {
    let hot = [];
    try { hot = ((await api("/api/top")).items || []).slice(0, 6); } catch { }
    $("#browseGrid").innerHTML = `<div class="empty-state">
      <div class="es-ic">${icon("general", 28)}</div>
      <p class="es-msg">${t("empty")}</p>
      <button class="btn-primary" id="esClear">${t("empty_clear")}</button>
      ${hot.length ? `<h3 class="es-hot-title">${t("empty_hot")}</h3><div class="card-grid es-hot">${hot.map(cardHTML).join("")}</div>` : ""}
    </div>`;
    const b = $("#esClear");
    if (b) b.onclick = () => go("/browse");
  }
  function renderPager(total, resSize, page) {
    const pages = Math.max(1, Math.ceil(total / resSize));
    if (pages <= 1) { $("#pager").innerHTML = ""; return; }
    const ph = n => browseHref({ page: n > 1 ? String(n) : "" });
    const nums = [...new Set([1, page - 1, page, page + 1, pages])]
      .filter(n => n >= 1 && n <= pages).sort((a, b) => a - b);
    let html = page > 1
      ? `<a class="pg-btn" rel="prev" href="${ph(page - 1)}">${t("prev")}</a>`
      : `<span class="pg-btn off">${t("prev")}</span>`;
    let last = 0;
    for (const n of nums) {
      if (n - last > 1) html += `<span class="pg-gap">…</span>`;
      html += `<a class="pg-num${n === page ? " on" : ""}"${n === page ? ' aria-current="page"' : ""} href="${ph(n)}">${n}</a>`;
      last = n;
    }
    html += page < pages
      ? `<a class="pg-btn" rel="next" href="${ph(page + 1)}">${t("next")}</a>`
      : `<span class="pg-btn off">${t("next")}</span>`;
    $("#pager").innerHTML = html;
  }
  $("#pager").addEventListener("click", e => { if (e.target.closest("a")) scrollTo({ top: 0 }); });
  let browseBusy = false;
  async function renderBrowse() {
    const p = browseParams();
    const q = p.get("q") || "", cat = p.get("cat") || "", tier = p.get("tier") || "";
    let group = p.get("group") || "";
    if (!GROUPS[group]) group = "";
    if (group && cat && CAT_GROUP[cat] !== group) group = ""; // mirror server-side validation: cat wins
    const sortRaw = p.get("sort") || "";
    const sort = SORTS.includes(sortRaw) ? sortRaw : "";
    const page = Math.max(1, Number(p.get("page") || 1));
    const size = p.get("size") === "48" ? 48 : 24;
    const featured = p.get("featured") === "1", safe = p.get("safe") === "1";

    if ($("#browseInput").value !== q) $("#browseInput").value = q;
    $("#ckSafe").checked = safe;
    $("#ckFeat").checked = featured;

    // six top-level group tabs (facet counts patched in after the fetch)
    const gt = $("#groupTabs");
    if (gt) {
      gt.innerHTML =
        `<button class="gt ${!group ? "on" : ""}" data-g="">${t("tier_all")}</button>` +
        Object.keys(GROUPS).map(g =>
          `<button class="gt ${group === g ? "on" : ""}" data-g="${g}">${t("group_" + g)} <span class="cnt"></span></button>`).join("");
      $$("#groupTabs .gt").forEach(el => el.onclick = () => {
        const g = el.dataset.g;
        setBrowseParams({ group: g, cat: (g && cat && CAT_GROUP[cat] !== g) ? "" : cat });
      });
    }

    // category chips: only the active group's children when a group is selected
    const chipCats = group ? META.categories.filter(c => GROUPS[group].includes(c.id)) : META.categories;
    $("#catChips").innerHTML =
      `<button class="chip ${!cat ? "on" : ""}" data-cat="">${t("tier_all")}</button>` +
      chipCats.map(c =>
        `<button class="chip ${cat === c.id ? "on" : ""}" data-cat="${c.id}">${icon(c.id, 13)} ${LANG === "zh" ? c.zh : c.en} <span class="cnt"></span></button>`
      ).join("");
    $$("#catChips .chip").forEach(el => el.onclick = () => setBrowseParams({ cat: el.dataset.cat }));

    // tier segmented
    $("#tierSeg").innerHTML = TIERS.map(x =>
      `<button class="${tier === x ? "on" : ""}" data-tier="${x}">${x ? t("tier_" + x) : t("tier_all")}</button>`).join("");
    $$("#tierSeg button").forEach(el => el.onclick = () => setBrowseParams({ tier: el.dataset.tier }));

    // sort segmented: score | installs | stars | A–Z | new (new only once added_at data exists)
    const sorts = SORTS.filter(v => v !== "new" || DATA_FLAGS.added_at);
    $("#sortSeg").innerHTML = sorts.map(v =>
      `<button class="${sort === v ? "on" : ""}" data-sort="${v}">${sortLabel(v)}</button>`).join("");
    $$("#sortSeg button").forEach(el => el.onclick = () => setBrowseParams({ sort: el.dataset.sort }));

    // page size 24 / 48
    const sz = $("#sizeSeg");
    if (sz) {
      sz.innerHTML = [24, 48].map(v =>
        `<button class="${size === v ? "on" : ""}" data-size="${v}">${v}</button>`).join("");
      $$("#sizeSeg button").forEach(el => el.onclick = () =>
        setBrowseParams({ size: el.dataset.size === "48" ? "48" : "" }));
    }

    if (browseBusy) return; browseBusy = true;
    try {
      const qp = new URLSearchParams({ q, cat, tier, sort, page: String(page), size: String(size) });
      if (group) qp.set("group", group);
      if (featured) qp.set("featured", "1");
      if (safe) qp.set("safe", "1");
      const res = await api("/api/search?" + qp.toString());
      sniffFields(res.items);
      $("#resultCount").textContent = q ? t("results_for", { n: fmtNum(res.total), q }) : t("results", { n: fmtNum(res.total) });
      // facet counts onto group tabs & category chips (contract C3)
      const fc = res.facets || {};
      $$("#groupTabs .gt").forEach(el => {
        const c = el.querySelector(".cnt");
        if (c && el.dataset.g) c.textContent = fmtNum((fc.group || {})[el.dataset.g] || 0);
      });
      $$("#catChips .chip").forEach(el => {
        const c = el.querySelector(".cnt");
        if (c && el.dataset.cat) c.textContent = fmtNum((fc.cat || {})[el.dataset.cat] || 0);
      });
      if (res.items.length) $("#browseGrid").innerHTML = res.items.map(cardHTML).join("");
      else await renderEmptyState();
      renderPager(res.total, res.size, page);
    } finally { browseBusy = false; }
  }
  let brTimer = 0;
  $("#browseInput").addEventListener("input", e => {
    clearTimeout(brTimer);
    brTimer = setTimeout(() => setBrowseParams({ q: e.target.value.trim() }), 280);
  });
  $("#ckSafe").onchange = e => setBrowseParams({ safe: e.target.checked ? "1" : "" });
  $("#ckFeat").onchange = e => setBrowseParams({ featured: e.target.checked ? "1" : "" });

  // ---------------- detail ----------------
  const INSTALL_TARGETS = [
    { id: "cursor", label: "Cursor", dir: "~/.cursor/skills" },
    { id: "claude", label: "Claude Code", dir: "~/.claude/skills" },
    { id: "codex", label: "Codex", dir: "~/.codex/skills" },
  ];
  function installCmd(s, target) {
    const safeName = s.name.replace(/[^\w.-]+/g, "-").toLowerCase();
    const url = `${location.origin}/download/${s.id}.zip`;
    return `curl -sL "${url}" -o /tmp/${safeName}.zip && unzip -oq /tmp/${safeName}.zip -d ${target.dir}/${safeName} && rm /tmp/${safeName}.zip`;
  }
  let zhPollTimer = 0;
  async function renderDetail(id) {
    clearInterval(zhPollTimer);
    const wrap = $("#detailWrap");
    wrap.innerHTML = `<div class="cmdk-empty">…</div>`;
    const res = await api("/api/skill/" + id);
    if (res.error) { wrap.innerHTML = ""; render404(); return; }
    const s = res.skill, c = CAT(s.cat);
    const scriptBadge = s.scripts
      ? `<span class="badge b-warn">⚠ ${t("badge_scripts")}</span>`
      : `<span class="badge b-ok">✓ ${t("badge_safe")}</span>`;
    const featBadge = s.featured ? `<span class="badge b-feat">★ ${t("badge_curated")}</span>` : "";
    const instMeta = s.installs ? `<span>↓ ${fmtNum(s.installs)} ${t("d_installs")}</span>` : "";
    const starMeta = s.stars ? `<span>★ ${fmtNum(s.stars)} ${t("d_stars")}</span>` : "";

    wrap.innerHTML = `
    <div class="d-main detail-ambient" data-cat="${esc(s.cat)}">
      <h1 class="d-title">${esc(s.name)}</h1>
      <div class="d-sub">
        <span class="d-cat">${icon(s.cat, 15)} ${c ? (LANG === "zh" ? c.zh : c.en) : ""}</span>
        ${instMeta} ${starMeta}
      </div>
      <div class="d-badges">${tierBadge(s)} ${scriptBadge} ${featBadge}</div>
      ${descOf(s) ? `<p class="d-desc">${esc(descOf(s))}</p>` : ""}
      ${s.scripts ? `<div class="warn-box" style="margin-top:16px">${t("warn_scripts")}</div>` : ""}
      ${LANG === "zh" ? `<div class="md-toggle"><div class="seg" id="mdLangSeg">
        <button data-v="zh" class="on">${t("md_zh")}</button>
        <button data-v="orig">${t("md_orig")}</button>
      </div><span class="md-status" id="mdStatus"></span></div>` : ""}
      <article class="d-md" id="mdArticle">${renderMD(res.md)}</article>
      <div class="d-related">
        <h3>${t("d_related")}</h3>
        <div class="card-grid">${res.related.map(cardHTML).join("")}</div>
      </div>
    </div>
    <div class="install-block">
      <a class="btn-primary" href="/download/${s.id}.zip">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12M6.5 9.5L12 15l5.5-5.5M4 19h16"/></svg>
        ${t("d_download")}
      </a>
      <details class="install-fold">
        <summary>
          <span>${t("d_install_toggle")}</span>
          <svg class="fold-chev" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5.5 9.5L12 16l6.5-6.5"/></svg>
        </summary>
        <div class="fold-body">
          <div class="install-tabs" id="instTabs">
            ${INSTALL_TARGETS.map((x, i) => `<button class="${i === 0 ? "on" : ""}" data-t="${x.id}">${x.label}</button>`).join("")}
          </div>
          <div class="install-cmd" id="instCmd" title="${t("d_copy_hint")}">
            <span id="instCmdText"></span>
            <svg class="copy-ic" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="9" y="9" width="11" height="11" rx="2.5"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>
          </div>
        </div>
      </details>
    </div>
    <aside class="d-side">
      <div class="meta-list">
        <div class="row"><span class="k">${t("d_meta_repo")}</span><a class="v" href="https://github.com/${esc(s.repo)}" target="_blank" rel="noopener nofollow">${esc(s.repo)} ↗</a></div>
        <div class="row"><span class="k">${t("d_meta_tier")}</span><span class="v">${t("tier_" + s.tier)}</span></div>
        <div class="row"><span class="k">${t("d_meta_cat")}</span><span class="v">${c ? (LANG === "zh" ? c.zh : c.en) : s.cat}</span></div>
        <div class="row"><span class="k">${t("d_meta_license")}</span><span class="v">${esc(s.license || t("license_none"))}</span></div>
        <div class="row"><span class="k">${t("d_meta_size")}</span><span class="v">${(s.bytes / 1024).toFixed(1)} KB</span></div>
      </div>
      ${res.files && res.files.length ? `
      <div>
        <div class="meta-list" style="margin-bottom:6px"><div class="row"><span class="k">${t("d_files")}</span></div></div>
        <div class="files-box"><code>${res.files.map(esc).join("<br>")}</code></div>
      </div>` : ""}
    </aside>`;

    // install tabs behavior
    let target = INSTALL_TARGETS[0];
    const updCmd = () => { $("#instCmdText").textContent = installCmd(s, target); };
    updCmd();
    $$("#instTabs button").forEach(b => b.onclick = () => {
      $$("#instTabs button").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      target = INSTALL_TARGETS.find(x => x.id === b.dataset.t);
      updCmd();
    });
    $("#instCmd").onclick = () => copyText(installCmd(s, target));

    // ---- zh full-text translation of SKILL.md ----
    if (LANG === "zh") {
      const st = { mdZh: null, view: "zh", userChose: false };
      const seg = $("#mdLangSeg"), status = $("#mdStatus"), art = $("#mdArticle");
      const paint = () => {
        $$("button", seg).forEach(b => b.classList.toggle("on", b.dataset.v === st.view));
        art.innerHTML = renderMD(st.view === "zh" && st.mdZh ? st.mdZh : res.md);
        status.textContent = (st.view === "zh" && !st.mdZh) ? t("md_translating") : "";
      };
      $$("button", seg).forEach(b => b.onclick = () => {
        st.userChose = true; st.view = b.dataset.v; paint();
      });
      const tryLoad = async () => {
        const r = await api(`/api/skill/${id}/zh`);
        if (r.status === "ready") {
          st.mdZh = r.md_zh;
          clearInterval(zhPollTimer);
          if (!st.userChose) st.view = "zh";
          paint();
          return true;
        }
        return false;
      };
      paint();
      tryLoad().then(ok => {
        if (!ok) {
          let tries = 0;
          zhPollTimer = setInterval(async () => {
            if (++tries > 60 || parseRoute().path !== "/skill/" + id) { clearInterval(zhPollTimer); return; }
            await tryLoad();
          }, 3000);
        }
      });
    }
    scrollTo({ top: 0 });
  }

  // ---------------- mini markdown renderer ----------------
  function renderMD(src) {
    if (!src) return "";
    src = src.replace(/^---\n[\s\S]*?\n---\n?/, ""); // strip frontmatter
    const lines = src.split("\n");
    const out = [];
    let inCode = false, codeBuf = [], para = [], listStack = [], lastBlank = false;
    const flushPara = () => {
      if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; }
    };
    const closeLists = (d = 0) => {
      while (listStack.length > d) out.push(listStack.pop() === "ol" ? "</ol>" : "</ul>");
    };
    const inline = s => {
      s = esc(s);
      s = s.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
      s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      s = s.replace(/(^|\W)\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
      s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)[^)]*\)/g, (_, a) => `<em>🖼 ${a || "image"}</em>`);
      s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, `<a href="$2" target="_blank" rel="noopener nofollow">$1</a>`);
      s = s.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");
      return s;
    };
    for (let i = 0; i < lines.length; i++) {
      const L = lines[i];
      if (/^\s*```/.test(L)) {
        if (inCode) { out.push(`<pre><code>${esc(codeBuf.join("\n"))}</code></pre>`); codeBuf = []; inCode = false; }
        else { flushPara(); closeLists(); inCode = true; }
        continue;
      }
      if (inCode) { codeBuf.push(L); continue; }
      const h = L.match(/^(#{1,6})\s+(.*)/);
      if (h) { flushPara(); closeLists(); const lv = Math.min(h[1].length, 4); out.push(`<h${lv}>${inline(h[2])}</h${lv}>`); continue; }
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(L)) { flushPara(); closeLists(); out.push("<hr>"); continue; }
      const bq = L.match(/^>\s?(.*)/);
      if (bq) { flushPara(); closeLists(); out.push(`<blockquote><p>${inline(bq[1])}</p></blockquote>`); continue; }
      // table
      if (/^\s*\|.*\|\s*$/.test(L) && i + 1 < lines.length && /^\s*\|[\s\-:|]+\|\s*$/.test(lines[i + 1])) {
        flushPara(); closeLists();
        const hdr = L.trim().slice(1, -1).split("|").map(x => x.trim());
        let rowsHtml = "";
        let j = i + 2;
        while (j < lines.length && /^\s*\|.*\|\s*$/.test(lines[j])) {
          const cells = lines[j].trim().slice(1, -1).split("|").map(x => x.trim());
          rowsHtml += `<tr>${cells.map(cx => `<td>${inline(cx)}</td>`).join("")}</tr>`;
          j++;
        }
        out.push(`<table><thead><tr>${hdr.map(hx => `<th>${inline(hx)}</th>`).join("")}</tr></thead><tbody>${rowsHtml}</tbody></table>`);
        i = j - 1; continue;
      }
      const li = L.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)/);
      if (li) {
        flushPara();
        const depth = Math.floor(li[1].length / 2) + 1;
        const type = /^\d/.test(li[2]) ? "ol" : "ul";
        while (listStack.length < depth) { out.push(type === "ol" ? "<ol>" : "<ul>"); listStack.push(type); }
        closeLists(depth);
        out.push(`<li>${inline(li[3])}</li>`);
        continue;
      }
      if (!L.trim()) { flushPara(); lastBlank = true; continue; }
      if (lastBlank && listStack.length && !li) closeLists();
      lastBlank = false;
      para.push(L.trim());
    }
    if (inCode) out.push(`<pre><code>${esc(codeBuf.join("\n"))}</code></pre>`);
    flushPara(); closeLists();
    return out.join("\n");
  }

  // ---------------- ⌘K palette ----------------
  const cmdk = $("#cmdk");
  let cmdkSel = 0, cmdkItems = [];
  function openCmdk() { cmdk.hidden = false; $("#cmdkInput").value = ""; $("#cmdkResults").innerHTML = ""; $("#cmdkInput").focus(); }
  function closeCmdk() { cmdk.hidden = true; }
  $("#btnSearchOpen").onclick = openCmdk;
  $("#cmdkBackdrop").onclick = closeCmdk;
  $("#heroSearch").onclick = () => { openCmdk(); $("#cmdkInput").value = $("#heroInput").value; doCmdkSearch(); };
  $("#heroInput").addEventListener("keydown", e => {
    if (e.key === "Enter") { go("/browse?q=" + encodeURIComponent(e.target.value.trim())); }
  });
  document.addEventListener("keydown", e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); cmdk.hidden ? openCmdk() : closeCmdk(); }
    if (e.key === "Escape" && !cmdk.hidden) closeCmdk();
    if (!cmdk.hidden && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      e.preventDefault();
      cmdkSel = Math.max(0, Math.min(cmdkItems.length - 1, cmdkSel + (e.key === "ArrowDown" ? 1 : -1)));
      $$(".cmdk-item").forEach((el, i) => el.classList.toggle("sel", i === cmdkSel));
      const sel = $(".cmdk-item.sel"); if (sel) sel.scrollIntoView({ block: "nearest" });
    }
    if (!cmdk.hidden && e.key === "Enter" && cmdkItems[cmdkSel]) {
      cmdkGo(cmdkItems[cmdkSel]);
    }
  });
  function cmdkGo(it) {
    closeCmdk();
    go(it.kind === "cat" ? "/browse?cat=" + it.id : "/skill/" + it.id);
  }
  let ckTimer = 0;
  $("#cmdkInput").addEventListener("input", () => { clearTimeout(ckTimer); ckTimer = setTimeout(doCmdkSearch, 160); });
  async function doCmdkSearch() {
    const q = $("#cmdkInput").value.trim();
    if (!q) { $("#cmdkResults").innerHTML = ""; cmdkItems = []; return; }
    // two-section results: skill matches + category jumps, via /api/suggest (contract C3);
    // falls back to the legacy /api/search call when suggest is unavailable
    let sug = null;
    try { sug = await api("/api/suggest?limit=8&q=" + encodeURIComponent(q)); } catch { }
    cmdkItems = []; cmdkSel = 0;
    let html = "";
    const skillItem = s => {
      cmdkItems.push({ kind: "skill", id: s.id });
      return `<div class="cmdk-item ${cmdkItems.length === 1 ? "sel" : ""}" data-id="${s.id}">
        <span class="ci-ico">${icon(s.cat, 17)}</span>
        <div class="ci-body">
          <div class="ci-name">${esc(s.name)}</div>
          <div class="ci-desc">${esc(s.repo ? s.repo + " · " : "")}${s.desc != null ? esc(descOf(s)) : esc(catLabel(s.cat))}</div>
        </div>
        ${s.installs ? `<span class="sc-installs">↓ ${fmtNum(s.installs)}</span>` : ""}
      </div>`;
    };
    if (sug && Array.isArray(sug.items)) {
      if (sug.items.length) {
        html += `<div class="cmdk-section">${t("cmdk_sec_skills")}</div>`;
        html += sug.items.map(skillItem).join("");
      }
      const cats = sug.cats || [];
      if (cats.length) {
        html += `<div class="cmdk-section">${t("cmdk_sec_cats")}</div>`;
        html += cats.map(c => {
          cmdkItems.push({ kind: "cat", id: c.id });
          return `<div class="cmdk-item ${cmdkItems.length === 1 ? "sel" : ""}" data-id="${c.id}">
            <span class="ci-ico">${icon(c.id, 17)}</span>
            <div class="ci-body">
              <div class="ci-name">${esc(LANG === "zh" ? c.zh : c.en)}</div>
              <div class="ci-desc">${t("results", { n: fmtNum(c.count || 0) })}</div>
            </div>
          </div>`;
        }).join("");
      }
    } else {
      const res = await api("/api/search?size=9&q=" + encodeURIComponent(q));
      html = res.items.map(skillItem).join("");
    }
    $("#cmdkResults").innerHTML = html || `<div class="cmdk-empty">${t("empty")}</div>`;
    $$(".cmdk-item").forEach((el, i) => el.onclick = () => cmdkGo(cmdkItems[i]));
  }

  // ---------------- random ----------------
  $("#btnRandom").onclick = async () => {
    const r = await api("/api/random");
    go("/skill/" + r.id);
  };

  // ---------------- router ----------------
  // mcp view arrives with W4-5's mcp.js (contract C6); 404 view is #view-404 — both may be absent, hence the null guards
  const views = { home: $("#view-home"), browse: $("#view-browse"), detail: $("#view-detail"), mcp: $("#view-mcp"), nf: $("#view-404") };
  function show(name) {
    for (const [k, el] of Object.entries(views)) if (el) el.hidden = k !== name;
  }
  function render404() {
    if (!views.nf) { show("home"); return renderHome(); }
    show("nf");
    const inp = $("#nfInput");
    if (inp && !inp._wired) {
      inp._wired = true;
      inp.addEventListener("keydown", e => {
        if (e.key === "Enter") go("/browse?q=" + encodeURIComponent(e.target.value.trim()));
      });
    }
  }
  async function route() {
    if (!META) return;
    const { path, search } = parseRoute();
    // normalize legacy #/ URLs to real paths (keeps address bar clean & shareable)
    if (location.hash.startsWith("#/")) history.replaceState({}, "", path + search);
    if (path.startsWith("/skill/")) { show("detail"); await renderDetail(decodeURIComponent(path.slice(7))); }
    else if (path.startsWith("/browse")) { show("browse"); await renderBrowse(); }
    else if (path === "/mcp" || path.startsWith("/mcp/")) {
      // delegate the MCP aggregate type to W4-5's renderer (contract C6)
      if (typeof window.renderMcp === "function" && views.mcp) { show("mcp"); await window.renderMcp(path, search); }
      else await render404();
    }
    else if (path === "/") { show("home"); await renderHome(); }
    else await render404();
  }
  addEventListener("popstate", route);
  addEventListener("hashchange", route); // legacy #/ links opened while on page

  // ---------------- boot ----------------
  (async () => {
    const urlLang = new URLSearchParams(location.search).get("lang");
    if (urlLang === "zh" || urlLang === "en") LANG = urlLang;
    const urlTheme = new URLSearchParams(location.search).get("theme");
    if (urlTheme === "dark" || urlTheme === "light") THEME = urlTheme;
    applyTheme();
    META = await api("/api/meta");
    applyLang(); // triggers route()
  })();
})();
