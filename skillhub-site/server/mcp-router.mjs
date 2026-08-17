/**
 * mcp-router.mjs — W6-2: the six retrieval tool handlers + rule-based dynamic router.
 *
 * Contract M-C1: every handler is `async (args, ctx) => object`, ctx = { deps, baseUrl, ip },
 *   deps = { DB, byId, searchSkills, groupOf, pub, HUB, SITE_URL } (injected by server.mjs anchor A8).
 *   Business failures throw Error with .userMessage — mcp.mjs turns them into isError textResults.
 * Contract M-C2: this file never reads skills.json / mcp.json itself; skills come via deps,
 *   MCP servers via the W6-2 export section at the end of ./mcp-data.mjs (shared index, no copy).
 * Contract M-C3: routing table lives in server/mcp-routes.json
 *   { keyword_cats: { "<regex>": "<cat|group>" }, boosts: { curated_words, zh_boost, curated_boost } }
 *   hot-reloaded with a 60s mtime poll (same pattern as server.mjs loadZh); a broken JSON keeps
 *   the previous table and only console.warn's.
 *
 * Exports for W6-1 (mcp.mjs): named handlers matching the tool names exactly
 *   (search_skills / get_skill / list_categories / trending / search_mcp_servers / get_mcp_server)
 *   plus the aggregate registry `routerHandlers`.
 */
import { readFileSync, statSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { mcpQuery, mcpById } from "./mcp-data.mjs";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url))); // site root
const ROUTES_PATH = join(ROOT, "server", "mcp-routes.json");
const MD_ZH_DIR = join(ROOT, "data", "md_zh");
const MD_CAP = 100 * 1024;            // get_skill skill_md cap (appendix A: ≤100KB)
const ADDED_BASELINE = "2026-08-17";  // E4: this build stamped every added_at with the baseline day

// Display names for the six groups (same table as server.mjs / app.js — contract C1).
// deps only carries groupOf(), so the label half of the table is mirrored here read-only.
const GROUPS = {
  dev:    { en: "Development",             zh: "开发",       cats: ["frontend", "backend", "mobile", "testing"] },
  secops: { en: "Security & Ops",          zh: "安全与运维", cats: ["security", "cloud-devops"] },
  ai:     { en: "AI & Agents",             zh: "AI 与智能体", cats: ["ai-ml", "agent-meta"] },
  data:   { en: "Data & Research",         zh: "数据与研究", cats: ["data-viz", "research", "docs-office"] },
  biz:    { en: "Business & Productivity", zh: "商业与效率", cats: ["productivity", "business", "marketing", "web3"] },
  cn:     { en: "Chinese & Life",          zh: "中文与生活", cats: ["chinese", "media", "game", "fun", "general"] },
};
const TIERS = ["official", "popular", "niche", "chinese"];
const SORTS = ["score", "installs", "stars", "name", "new"];

// ---------------- routing table (hot-reloaded) ----------------
let BOOSTS = { curated_words: [], zh_boost: 0, curated_boost: 0 };
let RULES = [];          // compiled [{ re, src, target }]
let CURATED_RE = null;   // strip/detect regex built from curated_words
let routesMtime = 0;

const escapeRe = t => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function compileRoutes(raw) {
  const rules = [];
  for (const [src, target] of Object.entries(raw.keyword_cats || {})) {
    try { rules.push({ re: new RegExp(src, "i"), src, target: String(target) }); }
    catch { console.warn(`mcp-router: bad regex skipped in keyword_cats: ${src}`); }
  }
  const b = raw.boosts || {};
  const words = Array.isArray(b.curated_words) ? b.curated_words.map(String).filter(Boolean) : [];
  BOOSTS = {
    curated_words: words,
    zh_boost: Number(b.zh_boost) || 0,
    curated_boost: Number(b.curated_boost) || 0,
  };
  RULES = rules;
  CURATED_RE = words.length
    ? new RegExp(`(?<![a-z0-9])(?:${words.map(escapeRe).join("|")})(?![a-z0-9])`, "gi")
    : null;
}

function loadRoutes() {
  try {
    const m = statSync(ROUTES_PATH).mtimeMs;
    if (m === routesMtime) return;
    routesMtime = m; // mark first so a broken file warns once per change, not every poll
    compileRoutes(JSON.parse(readFileSync(ROUTES_PATH, "utf8")));
  } catch (e) {
    console.warn(`mcp-router: keeping previous routing table (mcp-routes.json load failed: ${e.message})`);
  }
}
loadRoutes();
setInterval(loadRoutes, 60_000).unref();

// ---------------- helpers ----------------
function bizError(msg) { const e = new Error(msg); e.userMessage = msg; return e; }
const clip = (s, n) => { s = String(s || "").replace(/\s+/g, " ").trim(); return s.length > n ? s.slice(0, n - 1).trimEnd() + "\u2026" : s; };
const intIn = (v, lo, hi, dflt) => Math.min(hi, Math.max(lo, Math.floor(Number(v)) || dflt));

function slim(s, groupOf) { // appendix A search/trending projection (desc clipped to 200)
  return {
    id: s.id, name: s.name, desc: clip(s.desc, 200), cat: s.cat, group: groupOf(s),
    tier: s.tier, score: s.score ?? 0, stars: s.stars ?? 0, sources: s.sources ?? 1,
    installs: s.installs ?? 0,
  };
}

function catIds(deps) { return new Set(deps.DB.categories.map(c => c.id)); }

/**
 * Positional-uplift rerank (bias, never a hard filter): item at index i gets an
 * adjusted position i - boost; stable sort restores order. zh_boost/curated_boost
 * are therefore "number of positions lifted" and live in mcp-routes.json.
 */
function biasRerank(list, groupOf, zh, curated) {
  const zb = zh ? BOOSTS.zh_boost : 0;
  const cb = curated ? BOOSTS.curated_boost : 0;
  if (!zb && !cb) return list;
  const pos = new Array(list.length);
  for (let i = 0; i < list.length; i++) {
    const s = list[i];
    let p = i;
    if (zb && (s.tier === "chinese" || groupOf(s) === "cn")) p -= zb;
    if (cb && (s.featured || s.tier === "official")) p -= cb;
    pos[i] = p;
  }
  return list.map((_, i) => i).sort((a, b) => pos[a] - pos[b] || a - b).map(i => list[i]);
}

// ---------------- tool: search_skills ----------------
export async function search_skills(args = {}, ctx) {
  const { deps } = ctx;
  const notes = [];
  let q = String(args.q ?? "").trim().slice(0, 100);
  let cat = args.cat ? String(args.cat) : "";
  let group = args.group ? String(args.group) : "";
  const tier = args.tier ? String(args.tier) : "";
  const sortArg = args.sort != null ? String(args.sort) : "";
  const page = intIn(args.page, 1, 1e9, 1);
  const size = intIn(args.size, 1, 20, 10);

  const cats = catIds(deps);
  if (cat && !cats.has(cat)) throw bizError(`unknown cat "${cat}" — valid: ${[...cats].join(", ")}`);
  if (group && !GROUPS[group]) throw bizError(`unknown group "${group}" — valid: ${Object.keys(GROUPS).join(", ")}`);
  if (tier && !TIERS.includes(tier)) throw bizError(`unknown tier "${tier}" — valid: ${TIERS.join(", ")}`);
  if (sortArg && !SORTS.includes(sortArg)) throw bizError(`unknown sort "${sortArg}" — valid: ${SORTS.join(", ")}`);
  if (group && cat && !GROUPS[group].cats.includes(cat)) {
    group = ""; // same rule as /api/search: cat is more specific — it wins on mismatch
    notes.push(`cat "${cat}" not in group "${args.group}" — group dropped (cat wins)`);
  }

  // -- router preprocessing of q --
  let curated = false;
  if (q && CURATED_RE) {
    const stripped = q.replace(CURATED_RE, " ").replace(/\s+/g, " ").trim();
    if (stripped !== q) {
      curated = true;
      notes.push(`curated word hit → featured/official uplift (+${BOOSTS.curated_boost} positions); word removed from q`);
      q = stripped;
    }
  }
  const cjk = /[\u4e00-\u9fff]/.test(q);
  if (cjk) notes.push(`cjk detected → chinese-set uplift (+${BOOSTS.zh_boost} positions for tier:chinese / group:cn, no hard filter)`);

  let injected = null;
  if (q && (cat || group || args.cat || args.group)) {
    notes.push("explicit cat/group given — dictionary routing skipped (params win over router)");
  } else if (q) {
    for (const r of RULES) {
      if (!r.re.test(q)) continue;
      if (cats.has(r.target)) { cat = r.target; injected = r; notes.push(`keyword_cats /${r.src}/i → cat:${r.target} (filter injected)`); }
      else if (GROUPS[r.target]) { group = r.target; injected = r; notes.push(`keyword_cats /${r.src}/i → group:${r.target} (filter injected)`); }
      else continue; // stale target in table — ignore
      break; // first hit wins (table order is priority order)
    }
  }

  // -- search (deps.searchSkills, E5: no second index) --
  // sort semantics mirror /api/search: unset sort + q keeps relevance order; unset sort
  // without q falls back to score inside searchSkills.
  const flags = { featured: !!args.featured, safe: false, all: false };
  let { list } = deps.searchSkills(q, cat, tier, sortArg, flags, group);
  if (injected && list.length === 0) { // dictionary filter zeroed out the results — retry unrouted
    const kind = cats.has(injected.target) ? "cat" : "group";
    if (kind === "cat") cat = ""; else group = "";
    ({ list } = deps.searchSkills(q, cat, tier, sortArg, flags, group));
    notes.push(`injected ${kind}:${injected.target} returned 0 — filter dropped, unrouted results returned`);
  }

  // -- bias rerank (only under default/score ordering; explicit installs/stars/name/new wins) --
  if ((cjk || curated) && (sortArg === "" || sortArg === "score")) {
    list = biasRerank(list, deps.groupOf, cjk, curated);
  } else if ((cjk || curated) && sortArg) {
    notes.push(`explicit sort=${sortArg} — bias rerank skipped`);
  }

  const total = list.length;
  const items = list.slice((page - 1) * size, (page - 1) * size + size).map(s => slim(s, deps.groupOf));
  return {
    total, page, size,
    routing_note: notes.join("; ") || "no routing rule matched — query passed through unchanged",
    items,
  };
}

// ---------------- tool: get_skill ----------------
export async function get_skill(args = {}, ctx) {
  const { deps } = ctx;
  const id = String(args.id ?? "").trim();
  if (!id) throw bizError("id is required — use an id from search_skills results");
  const s = deps.byId.get(id);
  if (!s) throw bizError(`skill not found: "${id}" — ids come from search_skills results`);
  const lang = args.lang != null ? String(args.lang) : "en";
  if (lang !== "en" && lang !== "zh") throw bizError(`unknown lang "${lang}" — valid: en, zh`);

  let skill_md = "", lang_note;
  if (lang === "zh") {
    const f = join(MD_ZH_DIR, id + ".md");
    if (existsSync(f)) {
      skill_md = readFileSync(f, "utf8").slice(0, MD_CAP);
      lang_note = "zh translation served from cache (data/md_zh)";
    } else {
      // E6: never wait for gtx on the MCP path — fall back to the original text
      lang_note = "zh not cached for this skill — returning the English original (translation runs async; retry later or open /skill/" + id + "?lang=zh on the site)";
    }
  }
  if (!skill_md) {
    try { skill_md = readFileSync(join(deps.HUB, s.path), "utf8").slice(0, MD_CAP); }
    catch { skill_md = ""; lang_note = (lang_note ? lang_note + "; " : "") + "SKILL.md unavailable on disk"; }
  }

  const p = deps.pub(s); // adds desc_zh when the zh cache has it
  const out = {
    id: s.id, name: s.name, desc: s.desc,
    ...(p.desc_zh ? { desc_zh: p.desc_zh } : {}),
    cat: s.cat, group: deps.groupOf(s), tier: s.tier, repo: s.repo, license: s.license || "",
    stars: s.stars ?? 0, installs: s.installs ?? 0, sources: s.sources ?? 1,
    tags: s.tags || [], score: s.score ?? 0, updated_at: s.updated_at || "",
    skill_md,
  };
  if (lang_note) out.lang_note = lang_note;
  return out;
}

// ---------------- tool: list_categories ----------------
let CAT_TREE = null; // data is frozen per process (Wave6 data freeze) — compute once
export async function list_categories(_args, ctx) {
  const { deps } = ctx;
  if (!CAT_TREE) {
    const gCount = {};
    for (const s of deps.DB.skills) { // unique skills, same basis as DB.categories counts
      if (s.dup) continue;
      const g = deps.groupOf(s);
      if (g) gCount[g] = (gCount[g] || 0) + 1;
    }
    CAT_TREE = {
      groups: Object.entries(GROUPS).map(([gid, g]) => ({
        id: gid,
        name: g.en,
        name_zh: g.zh,
        count: gCount[gid] || 0,
        cats: g.cats
          .map(cid => deps.DB.categories.find(c => c.id === cid))
          .filter(Boolean)
          .map(c => ({ id: c.id, en: c.en, zh: c.zh, count: c.count })),
      })),
    };
  }
  return CAT_TREE;
}

// ---------------- tool: trending ----------------
let TREND_ALL = null; // unique skills sorted by score desc, cached (data frozen per process)
export async function trending(args = {}, ctx) {
  const { deps } = ctx;
  const limit = intIn(args.limit, 1, 48, 10);
  const window = args.window != null ? String(args.window) : "all";
  if (window !== "all" && window !== "new") throw bizError(`unknown window "${window}" — valid: all, new`);
  if (!TREND_ALL) {
    TREND_ALL = deps.DB.skills.filter(s => !s.dup).sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  }
  let pool = TREND_ALL;
  let note;
  if (window === "new") {
    pool = TREND_ALL.filter(s => String(s.added_at || "") > ADDED_BASELINE);
    if (pool.length === 0) {
      note = `no skills newer than the ${ADDED_BASELINE} baseline yet — this build stamped every added_at with the baseline day; "new" fills up after the next data build (use window:"all" meanwhile)`;
    }
  }
  const items = pool.slice(0, limit).map(s => slim(s, deps.groupOf));
  const out = { window, returned: items.length, items };
  if (note) out.note = note;
  return out;
}

// ---------------- tool: search_mcp_servers ----------------
const MCP_CLS = ["official", "community", "reference"];
export async function search_mcp_servers(args = {}, _ctx) {
  const q = String(args.q ?? "").trim().slice(0, 120);
  const cls = args.classification ? String(args.classification) : "";
  if (cls && !MCP_CLS.includes(cls)) throw bizError(`unknown classification "${cls}" — valid: ${MCP_CLS.join(", ")}`);
  const page = intIn(args.page, 1, 1e9, 1);
  const size = intIn(args.size, 1, 20, 10);
  const { total, items } = mcpQuery(q, cls, page, size);
  return {
    total, page, size,
    items: items.map(s => ({
      id: s.id, name: s.name, description: clip(s.desc, 200),
      classification: s.classification, stars: s.stars ?? 0, transports: s.transports || [],
    })),
  };
}

// ---------------- tool: get_mcp_server ----------------
export async function get_mcp_server(args = {}, _ctx) {
  const id = String(args.id ?? "").trim();
  if (!id) throw bizError("id is required — use an id from search_mcp_servers results");
  const s = mcpById(id);
  if (!s) throw bizError(`MCP server not found: "${id}" — ids come from search_mcp_servers results`);
  return s; // full record: R2 §4.2 fields + install_config + install_claude
}

// M-C1 registry — W6-1's mcp.mjs can import this object or the named handlers above
export const routerHandlers = {
  search_skills, get_skill, list_categories, trending, search_mcp_servers, get_mcp_server,
};
