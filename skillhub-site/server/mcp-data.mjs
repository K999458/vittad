/**
 * mcp-data.mjs — W4-5: MCP Servers aggregate type (data + API + SSR).
 * Loaded by server.mjs via anchor A1; handleMcpRoutes() is mounted at anchor A4
 * and returns false for any path it does not own, so the main site is untouched.
 *
 * Routes handled:
 *   /api/mcp/search  q / classification / page / size   (list projection only)
 *   /api/mcp/:id     full record incl. install_config / install_claude
 *   /mcp             SSR list shell (crawler-visible cards, SPA hydrates)
 *   /mcp/:id         SSR detail shell (JSON-LD SoftwareApplication)
 */
import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url))); // site root
const CFG = existsSync(join(ROOT, "config.json")) ? JSON.parse(readFileSync(join(ROOT, "config.json"), "utf8")) : {};
const SITE_URL = (process.env.SITE_URL || CFG.siteUrl || "").replace(/\/+$/, "") || "https://skillhub.tools";

// ---------------- data ----------------
let MCP = { built_at: "", stats: { total: 0, classification: {} }, servers: [] };
try {
  MCP = JSON.parse(readFileSync(join(ROOT, "data", "mcp.json"), "utf8"));
} catch {
  console.error("mcp-data: data/mcp.json missing or invalid — /mcp routes disabled");
}
const SERVERS = MCP.servers || [];
const MCP_BY_ID = new Map(SERVERS.map(s => [s.id, s]));
for (const s of SERVERS) {
  s._hay = `${s.name} ${s.title} ${s.desc} ${s.author}`.toLowerCase();
  s._rank = (s.stars || 0) * 3 + (s.downloads || 0) / 5000
    + (s.classification === "official" ? 40 : s.classification === "reference" ? 20 : 0)
    + (s.remotes && s.remotes.length ? 5 : 0) + (s.packages && s.packages.length ? 5 : 0);
}
const MCP_SORTED = [...SERVERS].sort((a, b) => b._rank - a._rank || a.title.localeCompare(b.title));

// list projection: never ship tools / packages / install_config in list APIs
function pubList(s) {
  const o = {
    id: s.id, type: "mcp", name: s.name, title: s.title, desc: s.desc,
    author: s.author, classification: s.classification, transports: s.transports,
    updated_at: s.updated_at,
  };
  if (s.stars != null) o.stars = s.stars;
  if (s.downloads != null) o.downloads = s.downloads;
  if (s.tools && s.tools.length) o.tools_count = s.tools.length;
  else if (s.tools_count) o.tools_count = s.tools_count;
  return o;
}
function pubFull(s) {
  const { _hay, _rank, ...rest } = s;
  return rest;
}

function searchMcp(q, cls) {
  let list = MCP_SORTED;
  if (cls) list = list.filter(s => s.classification === cls);
  if (q) {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean).slice(0, 8);
    const scored = [];
    for (const s of list) {
      let sc = 0, ok = true;
      for (const t of terms) {
        if (!s._hay.includes(t)) { ok = false; break; }
        sc += s.title.toLowerCase().includes(t) ? 60 : s.name.toLowerCase().includes(t) ? 30 : 8;
      }
      if (!ok) continue;
      scored.push([sc + Math.min(s._rank / 50, 40), s]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    list = scored.map(x => x[1]);
  }
  return list;
}

function facets(q) {
  // classification counts under the current q (so chips stay truthful)
  const base = q ? searchMcp(q, "") : MCP_SORTED;
  const f = { official: 0, community: 0, reference: 0 };
  for (const s of base) f[s.classification] = (f[s.classification] || 0) + 1;
  return f;
}

// ---------------- SSR ----------------
const INDEX_HTML = readFileSync(join(ROOT, "public", "index.html"), "utf8");
const EMPTY_MCP = '<div class="mcp-wrap" id="mcpWrap"></div>'; // matches anchor B3 block
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clip = (s, n) => { s = String(s || "").replace(/\s+/g, " ").trim(); return s.length > n ? s.slice(0, n - 1).trimEnd() + "\u2026" : s; };

function mcpShell({ title, desc, path, jsonld, zh, inner }) {
  const url = SITE_URL + path;
  const sep = path.includes("?") ? "&" : "?";
  const head = [
    `<link rel="canonical" href="${esc(url)}">`,
    `<meta property="og:type" content="website">`,
    `<meta property="og:site_name" content="SkillHub">`,
    `<meta property="og:title" content="${esc(title)}">`,
    `<meta property="og:description" content="${esc(desc)}">`,
    `<meta property="og:url" content="${esc(url)}">`,
    `<meta name="twitter:card" content="summary">`,
    `<link rel="alternate" hreflang="en" href="${esc(url + sep + "lang=en")}">`,
    `<link rel="alternate" hreflang="zh-CN" href="${esc(url + sep + "lang=zh")}">`,
    `<link rel="alternate" hreflang="x-default" href="${esc(url)}">`,
    jsonld ? `<script type="application/ld+json">${JSON.stringify(jsonld).replace(/</g, "\\u003c")}</script>` : "",
  ].filter(Boolean).join("\n");
  return INDEX_HTML
    .replace('<html lang="en">', `<html lang="${zh ? "zh-CN" : "en"}">`)
    .replace(/<title>[\s\S]*?<\/title>/, `<title>${esc(title)}</title>`)
    .replace(/<meta name="description"[^>]*>/, `<meta name="description" content="${esc(desc)}">`)
    .replace("</head>", head + "\n</head>")
    .replace('<main id="view-home" class="view">', '<main id="view-home" class="view" hidden>')
    .replace('<main id="view-mcp" class="view" hidden>', '<main id="view-mcp" class="view">')
    .replace(EMPTY_MCP, EMPTY_MCP.replace("></div>", ">" + inner + "</div>"));
}

function ssrMcpCard(s) {
  return `<a class="skill-card mcp-card" href="/mcp/${s.id}">`
    + `<div class="sc-top"><span class="sc-name">${esc(s.title)}</span> <span class="badge b-mcp">MCP</span></div>`
    + `<p class="sc-desc">${esc(clip(s.desc, 160))}</p>`
    + `<div class="sc-meta"><span class="badge mc-${s.classification}">${esc(s.classification)}</span> <span class="sc-repo">${esc(s.author)}</span></div>`
    + `</a>`;
}

function sendJSON(res, obj, code = 200) {
  res.writeHead(code, { "content-type": "application/json; charset=utf-8", "cache-control": "no-cache" });
  res.end(JSON.stringify(obj));
}
function sendHTML(res, html, code = 200) {
  res.writeHead(code, { "content-type": "text/html; charset=utf-8", "cache-control": "no-cache" });
  res.end(html);
}

// URL list for the sitemap — handover to W4-6 (owner of SITEMAP_URLS)
export function mcpSitemapUrls() {
  return ["/mcp"].concat(SERVERS.map(s => `/mcp/${s.id}`));
}

// ---------------- router ----------------
export function handleMcpRoutes(req, res, url) {
  const p = url.pathname;
  if (!SERVERS.length) return false;

  if (p === "/api/mcp/search") {
    const q = (url.searchParams.get("q") || "").trim().slice(0, 120).toLowerCase();
    const cls = ["official", "community", "reference"].includes(url.searchParams.get("classification"))
      ? url.searchParams.get("classification") : "";
    const page = Math.max(1, Number(url.searchParams.get("page") || 1));
    const size = Math.min(60, Math.max(1, Number(url.searchParams.get("size") || 24)));
    const list = searchMcp(q, cls);
    sendJSON(res, {
      total: list.length, page, size,
      synced: MCP.built_at, stats: MCP.stats,
      facets: { classification: facets(q) },
      items: list.slice((page - 1) * size, (page - 1) * size + size).map(pubList),
    });
    return true;
  }
  if (p.startsWith("/api/mcp/")) {
    const s = MCP_BY_ID.get(p.slice("/api/mcp/".length));
    if (!s) { sendJSON(res, { error: "not found" }, 404); return true; }
    sendJSON(res, { server: pubFull(s) });
    return true;
  }

  if (p === "/mcp" || p === "/mcp/") {
    const zh = url.searchParams.get("lang") === "zh";
    const n = MCP.stats.total;
    const title = zh ? `MCP \u670d\u52a1\u5668 \u2014 ${n} \u4e2a\u53ef\u63a5\u5165\u7684 MCP Servers | SkillHub`
      : `MCP Servers \u2014 ${n} servers, config in one click | SkillHub`;
    const desc = clip(zh
      ? `\u6d4f\u89c8 ${n} \u4e2a MCP \u670d\u52a1\u5668\uff0c\u4e00\u952e\u590d\u5236 Cursor / Claude Code \u63a5\u5165\u914d\u7f6e\uff0c\u6570\u636e\u540c\u6b65\u81ea\u5b98\u65b9 MCP registry\u3002`
      : `Browse ${n} MCP servers synced from the official MCP registry. Copy a ready-to-use mcpServers config for Cursor or Claude Code in one click.`, 155);
    const jsonld = { "@context": "https://schema.org", "@type": "CollectionPage", name: title, url: SITE_URL + "/mcp" };
    const inner = `<div class="browse-head"><h1>${zh ? "MCP \u670d\u52a1\u5668" : "MCP Servers"}</h1></div>`
      + `<div class="card-grid">${MCP_SORTED.slice(0, 48).map(ssrMcpCard).join("")}</div>`;
    sendHTML(res, mcpShell({ title, desc, path: "/mcp", jsonld, zh, inner }));
    return true;
  }
  if (p.startsWith("/mcp/")) {
    const zh = url.searchParams.get("lang") === "zh";
    const s = MCP_BY_ID.get(decodeURIComponent(p.slice("/mcp/".length)));
    if (!s) {
      sendHTML(res, mcpShell({
        title: "Not found | SkillHub", desc: "MCP server not found.", path: "/mcp", zh,
        inner: `<div class="browse-head"><h1>404</h1></div><p><a href="/mcp">\u2190 MCP Servers</a></p>`,
      }), 404);
      return true;
    }
    const title = `${s.title} \u2014 MCP Server \u00b7 ${s.author} | SkillHub`;
    const desc = clip(s.desc || `${s.title}: an MCP server by ${s.author}.`, 155);
    const jsonld = {
      "@context": "https://schema.org", "@type": "SoftwareApplication",
      name: s.title, description: desc, applicationCategory: "DeveloperApplication",
      operatingSystem: "Cross-platform", url: SITE_URL + "/mcp/" + s.id,
      offers: { "@type": "Offer", price: 0, priceCurrency: "USD" },
      author: { "@type": "Organization", name: s.author },
      ...(s.repo_url ? { isBasedOn: s.repo_url } : {}),
    };
    const inner = `<article class="d-main">`
      + `<h1 class="d-title">${esc(s.title)}</h1>`
      + `<div class="d-sub"><span class="badge mc-${s.classification}">${esc(s.classification)}</span> `
      + s.transports.map(t => `<span class="badge b-tr">${esc(t)}</span>`).join(" ") + `</div>`
      + (s.desc ? `<p class="d-desc">${esc(s.desc)}</p>` : "")
      + (s.repo_url ? `<p><a href="${esc(s.repo_url)}" rel="noopener nofollow">${esc(s.repo_url.replace(/^https?:\/\//, ""))} \u2197</a></p>` : "")
      + (s.install_config ? `<pre class="mcp-config"><code>${esc(s.install_config)}</code></pre>` : "")
      + `<p><a href="/mcp">${zh ? "\u2190 \u5168\u90e8 MCP \u670d\u52a1\u5668" : "\u2190 All MCP servers"}</a></p>`
      + `</article>`;
    sendHTML(res, mcpShell({ title, desc, path: "/mcp/" + s.id, jsonld, zh, inner }));
    return true;
  }
  return false;
}

// ==== W6-2 ====
// Appended per WAVE6 ownership matrix §0.5 — existing code above is untouched.
// mcp-router.mjs must not re-read data/mcp.json (contract M-C2), so the in-memory
// SERVERS index built above is exposed through these two query functions.

/** mcpQuery(q, classification, page, size) → { total, items } — items are raw
 *  ranked records (caller applies its own projection; page/size pre-sliced). */
export function mcpQuery(q, classification, page = 1, size = 10) {
  const list = searchMcp(String(q || "").trim().toLowerCase(), classification || "");
  return { total: list.length, items: list.slice((page - 1) * size, (page - 1) * size + size) };
}

/** mcpById(id) → full public record (all fields incl. install_config / install_claude) or null. */
export function mcpById(id) {
  const s = MCP_BY_ID.get(String(id || ""));
  return s ? pubFull(s) : null;
}
