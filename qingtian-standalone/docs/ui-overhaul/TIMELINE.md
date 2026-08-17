# 时间线（TIMELINE.md）

> SkillHub 站点精致化 + 三仓合并 + MCP 端到端 —— 全部 6 波 Agent 的统一操作日志。
> **每个 Agent 完成自己的任务后，必须按下面的规范在本文件末尾追加一条记录。**

---

## 一、写入规范（所有 Agent 必读）

1. **只追加，不修改**：新条目永远加在文件末尾「三、时间线」区块的最后。不许改动、删除任何已有条目；发现前人条目有误，追加一条「勘误」条目说明，而不是原地改。
2. **一个 Agent 一条主条目**：任务完成时写一条；若中途发生重大转向（改方案、发现阻塞），可以额外追加「过程」条目。
3. **时间**取写入当刻的本地时间（UTC+8），格式 `YYYY-MM-DD HH:MM`。
4. **必填字段**见下方模板，缺一不可。`产出` 一律写绝对路径；`阻塞` 没有就写「无」。
5. **写完时间线 ≠ 任务完成**：还要按各自提示词里的要求向主控（CH-1）汇报。时间线是给后续波次 Agent 看的，汇报是给主控看的，两者都要做。
6. 并发写入冲突：追加前先读一次文件尾部，若发现有比自己新的条目，把自己的条目排在其后即可；不要重排他人条目。

### 条目模板（复制后填写）

```markdown
### [W{波次}] {Agent 标识} — {一句话任务名}
- 时间：YYYY-MM-DD HH:MM (UTC+8)
- 状态：完成 / 部分完成 / 受阻
- 做了什么：
  - （3~8 条，动词开头，写实际操作而非计划）
- 产出：
  - /绝对/路径/文件1 —— 一句话说明
- 关键决定 / 发现：
  - （影响后续波次的信息才写；没有就删掉本节）
- 阻塞：无（或具体描述 + 需要谁解决）
- 交接给下一波：
  - （下一波 Agent 拿到什么、从哪个文件读起）
```

---

## 二、波次总览（供快速定位）

| 波次 | 内容 | Agent 数 | 状态 |
|---|---|---|---|
| Wave 1 | 整体设计：设计系统方向、三仓合并方案、skills 展示策略、时间线模板、Wave2 提示词 | 1（规划） | ✅ 完成 |
| Wave 2 | 3 个 UI 调研 Agent（配色 / 功能信息架构 / 动效），产出调研报告与素材 | 3 | ✅ 完成 |
| Wave 3 | 回收调研 → 消歧与优先级、Wave4 六任务拆分与提示词、Wave6 提纲 | 1（回收） | ✅ 完成 |
| Wave 4 | 6 个实现 Agent：设计系统动效 / 信息架构 / 双语 / 上游数据 / MCP 聚合类型 / 页面收敛与模块回收 | 6 | ✅ 完成 |
| Wave 5 | 复盘 + 修正（1 个回收 Agent 验收 Wave4，列返工清单，定 MCP-SPEC） | 1 | ✅ 完成 |
| Wave 6 | MCP 端到端：动态路由站点 skills，供 AI 调用且可实际安装使用（见 WAVE6-OUTLINE.md） | 6 | ⬜ 未开始 |

> 注：用户给的执行顺序是 1→3→1→6→1→6（规划→调研 3 人→回收→实现 6 人→回收→MCP 6 人）。本表原版曾把「6 实现」误写在 Wave3 行、与 WAVE1-PLAN §4 及上述顺序矛盾，Wave3 回收 Agent 已于 2026-08-17 按 WAVE1-PLAN §4 勘正（详见 [W3] 条目）。

---

## 三、时间线（按时间正序追加）

### [W1] Agent-PLAN（Wave1 规划 Agent，fable 5） — 整体设计与 Wave2 提示词
- 时间：2026-08-17 17:45 (UTC+8)
- 状态：完成
- 做了什么：
  - 勘察了 4 个相关目录（不是任务书写的 3 个）：`/store/zkyang/qingtian-standalone`（晴天通道服务）、`/store/zkyang/skillhub-site`（现役站点）、`/store/zkyang/skills-hub`（旧版聚合站，含 MCP 端点）、`/storo/zkyang/agent-skills-hub`（9.1GB 数据仓库，注意在 **/storo** 挂载点，不在 /store）。
  - 统计了 skills 规模：`skillhub-site/data/skills.json` 共 **31,704 条**（去重后 15,402 个唯一 skill），196 个仓库来源，20 个分类，四个层级 tier（popular 25,669 / niche 3,847 / official 1,738 / chinese 450），精选 348，带脚本 3,705。
  - 审读了现有 UI（Apple 风设计系统、暗/亮双主题、EN/中 i18n、⌘K 面板）与旧站 `skills-hub` 可回收模块（MCP JSON-RPC 端点、反爬中间件、HMAC 下载 token、法务页）。
  - 制定设计系统总方向、三仓合并与职责边界方案、skills 展示/分类策略、6 波路线图与验收标准，写入 WAVE1-PLAN.md。
  - 撰写 Wave2 三个 UI 调研 Agent（Agent-UI-R1/R2/R3）可直接投递的完整提示词。
  - 建立本时间线模板与写入规范。
- 产出：
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/TIMELINE.md —— 本文件（模板 + 首批条目）
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/WAVE1-PLAN.md —— 总体计划、UI 画风、合并方案、skills 策略、验收标准
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/WAVE2-AGENT-PROMPTS.md —— 3 个调研 Agent 完整提示词
- 关键决定 / 发现：
  - **任务书里的 `agent-skills-hub` 实际在 `/storo/zkyang/agent-skills-hub`**（skillhub-site/config.json 的 hubRoot 指向它），`/store/zkyang/` 下不存在同名目录；后续 Agent 引用路径时务必用 /storo。
  - 合并方向定为：**skillhub-site 为唯一站点主体**，从 skills-hub 回收 MCP / 反爬 / 法务页模块后将 skills-hub 冻结归档；agent-skills-hub 保持纯数据仓库；qingtian-standalone 不参与站点合并，仅承载文档与 agent 协同。
  - 站点已部署域名 `https://skillhub.tools`（config.json.siteUrl），端口 3199；改动需保持零 npm 依赖约定。
- 阻塞：无
- 交接给下一波：
  - Wave2 三个调研 Agent 直接取用 WAVE2-AGENT-PROMPTS.md 中各自的提示词全文；调研产出统一写到 `docs/ui-overhaul/research/`。

### [W2] Agent-UI-R3 — 动效与微交互调研
- 时间：2026-08-17 17:52 (UTC+8)
- 状态：完成
- 做了什么：
  - 读取 WAVE1-PLAN.md §1.5（动效六铁律）、skillhub-site/public/app.css 现有动效、TIMELINE 写入规范建立上下文。
  - 实抓并解析 4 个站点的全量 CSS bundle（linear.app 51 个文件 / vercel.com / stripe.com / raycast.com），提取一手动效 token、keyframes、stagger 与 reduced-motion 策略。
  - 调研 framer.com、godly.website、ui.aceternity.com、magicui.design、21st.dev、codrops、joshwcomeau.com、animations.dev 共 8 个补充对象，逐个评估「可无依赖复刻性」。
  - 撰写 R3 报告：逐站拆解、动效 token 提案（时长阶梯 + 7 条 cubic-bezier + stagger 参数）、12 个组件的动效规格（净增约 190 行原生代码）、reduced-motion 降级表、8 项不做清单、现有 app.css 5 处违规修复清单。
  - 保存 5 份关键 CSS 素材到 research/assets/r3/。
- 产出：
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/research/R3-MOTION-MICROINTERACTIONS.md —— 动效调研报告与 SkillHub 动效规格
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/research/assets/r3/ —— linear/vercel/stripe/raycast 一手动效 CSS 片段 + 本站动效基线快照
- 关键决定 / 发现：
  - 现有 app.css 有 5 处违反六铁律（sheen 动画 `left`、reveal stagger 55ms 超 40ms 上限、reveal 时长 0.55s 超 320ms 上限、pulse 加载文字、`transition: all`），已列修复清单（报告 §5），Wave4-① 随 token 落地一并处理。
  - reduced-motion 降级推荐 Stripe 的「duration token 归零」模式：一处 media query 放倒全站，优于 Linear/Vercel 的逐条覆盖。
  - awwwards.com 抓取超时未实际访问（已在报告开头声明，用 godly.website + 训练知识替代且结论降权）；组件库三站（aceternity/magicui/21st）全部依赖 framer-motion，报告给出了逐效果的无依赖复刻成本表。
- 阻塞：无
- 交接给下一波：
  - Wave3 回收 Agent 从 R3 报告 §2（token 代码块）与 §3（组件规格）直接取材进 DESIGN-FINAL.md；§5 违规修复与 §4 降级表应进 WAVE4-TASKS.md 的设计系统落地任务。

### [W2] Agent-UI-R2 — 竞品功能与信息架构调研
- 时间：2026-08-17 17:50 (UTC+8)
- 状态：完成
- 做了什么：
  - 实抓 12 个调研对象：smithery.ai、mcp.so（首页+详情页）、pulsemcp.com（列表页+公开 API）、glama.ai/mcp/servers、anthropics/skills、awesome-claude-skills 生态（ComposioHQ 清单全文）、npmjs（页面+搜索 API）、VS Code Marketplace（extensionquery API）、HuggingFace（models API）、chromewebstore 首页、theresanaiforthat 首页，并额外增补官方 MCP registry（registry.modelcontextprotocol.io）API。
  - 记录 2 个无法访问的站点（cursor.directory 被 Vercel 质询、producthunt 被 Cloudflare 质询），按规则以训练知识补充并在报告开头声明。
  - 保存 4 个公开 API 一手 JSON 样例与页面结构速记到 assets/r2/。
  - 产出功能对照矩阵（20 项功能 × 11 竞品 + SkillHub 现状）、必做 9 项/该做 8 项/不做 8 项功能清单（各带一句话验收标准）。
  - 基于官方 registry server.json schema + pulsemcp API 定义「MCP Servers」聚合类型 12 核心字段最小集与列表/详情页形态（照 mcp.so 四段式）。
  - 给出 15k+ 条目的服务端分页参数规格（/api/skills 全参数表 + facets 计数返回）与搜索/⌘K suggest 建议。
- 产出：
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/research/R2-FEATURES-IA.md —— 竞品功能与信息架构调研报告（含功能矩阵、三档功能清单、MCP 字段集、分页规格）
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/research/assets/r2/ —— mcp 官方 registry / pulsemcp / npm / HF 四个 API 样例 JSON + site-structure-notes.md 页面结构速记
- 关键决定 / 发现：
  - MCP Servers 聚合类型**不需要发明 schema**：官方 registry API（开放无鉴权、cursor 分页，可作主数据源）+ pulsemcp v0beta API 已提供全部字段（packages/transports/remotes/env 含 secret 标注）；mcp.so 已反向收录 Agent Skills，双向竞争已开始，建议 W4-⑥ 尽早落地。
  - 所有 2 万+ 体量目录站（pulsemcp 22k、glama 73k、npm、HF）全部使用编号分页或 Load More，**无一家用真无限滚动**；浏览公式收敛为「搜索优先 + 3~5 个带计数分面 + 时间窗热度排序 + 卡片数字信任信号」。
  - SkillHub 差异化领先项 = ⌘K + 双语 + 双主题 + SKILL.md 中文翻译（竞品全场缺席）；最大短板 = 详情页无多客户端安装块、卡片无数字信任信号、无 Submit/作者页/相关推荐。
- 阻塞：无
- 交接给下一波：
  - Wave3 回收 Agent 从 R2 报告 §3（功能清单，验收标准可直接进 WAVE4-TASKS.md）、§4（MCP 聚合类型规格 → W4-⑥ 任务书）、§5（分页参数规格 → 浏览页改版任务书）取材；§2 矩阵用于 DESIGN-FINAL.md 的功能范围定稿。

### [W2] Agent-UI-R1 — 配色与主题体系调研
- 时间：2026-08-17 17:55 (UTC+8)
- 状态：完成
- 做了什么：
  - 实际抓取 9 个调研对象的生产环境 CSS 并脚本提取 design token：linear.app（完整 token 文件）、vercel.com（Geist 亮暗全量）、stripe.com（渐变系统）、raycast.com（暗色灰阶+霓虹强调）、apple.com/store（sk-button/badge 实测值）、supabase.com（品牌绿色阶+oklch 计算式表面）、resend.com（Radix alpha 实证）、Radix Colors 官方包（16 个色阶文件）、ui.shadcn.com/themes（oklch 槽位样本）。9 个对象全部访问成功，无缺席。
  - 提炼横向结论：暗色表面分层最佳实践 5 条（色阶分层非阴影、避纯黑纯白、暗色灰阶独立调校、alpha 边框分级、强调色暗色提亮补偿）、单强调色共识与 hover 方向规律、渐变使用红线 5 条。
  - 设计 20 分类 hue map：每分类一个 hue 数字 + 全局 HSL 派生公式（20 个 token 替代 120 个色值），hue 语义化分配、相邻间距 ≥12°、对比度全部过 WCAG AA 校验。
  - 撰写完整 CSS 变量提案（`:root` + `html[data-theme="dark"]` 直接可粘贴，兼容全部现有 app.css 变量名）与 15 条差异清单。
  - 整理 token 摘录素材至 assets/r1/（9 组小文件共 112KB，已删除原始大 HTML/CSS）。
- 产出：
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/research/R1-COLOR-THEME.md —— 调研报告（逐站色板表 + 横向结论 + CSS 变量提案 + 差异清单）
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/research/assets/r1/ —— 9 组站点 token 摘录 + Radix 官方色阶文件
- 关键决定 / 发现：
  - **现有暗色 surface 层级颠倒**：`--surface-2:#161617` 比 `--surface:#1d1d1f` 更暗，与亮色语义相反；提案改为「数字越大越浮起越亮」四级阶梯（`#0a0a0c/#151517/#1d1d20/#26262a`），是 Wave3/W4 最大迁移点（报告 §4 含风险提示）。
  - 暗色分层业界共识是色阶而非阴影（Linear 营销页暗色 shadow 全置 transparent）；暗色 `--card-shadow` 提案降级为 1px 描边。
  - 亮色 `--accent-hover` 现值方向反了：Apple store 实测 hover 是压深（`#0071e3→#0076df`）非提亮。
  - 20 分类 hue map 采用「hue 数字 + 公式派生」方案，WAVE1 §3.2 六大类可直接取组内最大子类 hue，零额外 token。
- 阻塞：无
- 交接给下一波：
  - Wave3 回收 Agent 从 R1-COLOR-THEME.md §3.2 直接取用变量代码块；§4 差异清单第 2 条（surface 层级）实施时须全局排查 `--surface-2` 用法（凹面场景改用 `--code-bg`）。

### [W3] Agent-SYNTH（Wave3 回收/规划 Agent，fable 5） — 调研汇总消歧与 Wave4 任务分派
- 时间：2026-08-17 18:20 (UTC+8)
- 状态：完成
- 做了什么：
  - 通读 WAVE1-PLAN 与 R1/R2/R3 三份调研，并实地复勘三个工作目录（skillhub-site 全部源码、skills-hub 可回收模块、agent-skills-hub 数据仓库），发现 10 条 Wave2 未覆盖的现状事实（详见 SYNTHESIS §0）：/api/search 已服务端分页、详情页已有三客户端安装 tab、related 已实现、3199 端口当前无进程、8787 旧站在运行等。
  - 裁决 12 条消歧决定（D1~D12）：表面阶梯迁移排查规则、trending 时间窗降级（无时序数据，改积累 seen_dates 快照）、Submit 降级为 contact 页、下载 token 平移待命不启用站内校验、MCP 聚合先做独立 /mcp 路由不做混排 tab、数字格式化双口径等。
  - 按「层 × 聚合类型」拆分 Wave4 六个实现 Agent（设计系统动效 / 信息架构与浏览 / 双语 / 上游数据 / MCP 聚合类型 / 页面收敛与模块回收），设计文件所有权矩阵 + 6 个 server.mjs/index.html 锚点 + 7 组契约（分类映射、数据字段、API、CSS 类名、前端挂钩、i18n key）保证六人可并行少冲突。
  - 撰写 6 份可直接投递的完整提示词（含各自任务清单 P0/P1/P2、验收、禁止事项、专属自测端口 3301~3306）。
  - 撰写 Wave6 第三波 6 Agent「MCP 动态路由 skills 端到端」方案提纲（本轮不实现）。
  - 勘正本文件「二、波次总览」表：原表把 6 实现 Agent 误写在 Wave3 行，已按 WAVE1-PLAN §4 对齐（W3=1 回收、W4=6 实现、W5=1 回收、W6=6 MCP）并刷新 W2/W3 状态。
- 产出：
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/WAVE3-SYNTHESIS.md —— 调研汇总、12 条消歧、优先级 P0~P2、所有权矩阵与契约索引（兼任 DESIGN-FINAL）
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/WAVE4-AGENT-PROMPTS.md —— 共同协议（锚点/纪律/端口）+ 6 个实现 Agent 完整提示词 + i18n key 契约附录（兼任 WAVE4-TASKS）
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/WAVE6-OUTLINE.md —— 第三波 MCP 端到端架构图、6 Agent 拆分、工具 schema 草案、验收与开放问题
- 关键决定 / 发现：
  - **生产部署链路不明**：config 口径端口 3199 当前无进程监听（data/server.log 显示历史运行记录），skillhub.tools 如何指向本机未确认；Wave4 各 Agent 一律用专属端口自测、不碰 3199，上线动作留给主控/Wave5——这是唯一阻塞项，需主控确认部署方式。
  - R2 的 M1/M3/M9 三项在现有代码里已有基础实现，Wave4 范围据实缩小；R2-M5（时间窗 trending）与 M8（GitHub issue 提交）因数据/仓库条件不具备而降级。
  - WAVE1-M1（server.mjs 拆模块）推迟到 Wave5 执行，避免与并行 Agent 在同一文件上大规模冲突；M4（旧站数据吸收）划归 W4-4 数据 Agent。
  - 磁盘水位：/store 94% 满（余 2.8T）、/storo 92% 满（余 1.7T）；W4-4 扩容预算 ≤30GB、强制 --depth 1。
- 阻塞：生产部署链路待主控确认（skillhub.tools → 本机 3199 的反代/DNS 现状；8787 旧站进程的启动方式也需主控知情后由 W4-6 停用）。不阻塞 Wave4 开发本身。
- 交接给下一波：
  - 主控直接从 WAVE4-AGENT-PROMPTS.md 复制 6 份提示词投递（可同时并行；压保险可按附录建议分两批 3+3）。
  - Wave4 每个 Agent 开工先读 WAVE3-SYNTHESIS.md 的消歧决定与所有权矩阵；完成后按本文件规范追加条目。
  - Wave5 回收 Agent 验收基准 = WAVE3-SYNTHESIS §3 优先级表 + 各提示词验收清单 + WAVE1 §5 总验收；并据 WAVE6-OUTLINE.md 细化 MCP-SPEC.md。

### [W4] Agent-W4-6 — 页面收敛与模块回收（法务页 / docs / 反爬平移 / 旧站冻结）
- 时间：2026-08-17 18:30 (UTC+8)
- 状态：完成
- 做了什么：
  - 平移 antiscrape.js → `server/security.mjs`（ESM，UA 黑白名单 / 滑动窗限流 / 蜜罐 / 临时封禁，/mcp 与 /api/mcp/* 走独立限流通道且跳过 UA 黑名单；搜索引擎白名单额外免普通限流以防爬 sitemap 误封），阈值读 config.json 新增 security 节；挂载于 server.mjs 锚点 A3。
  - 平移 tokens.js → `server/tokens.mjs`（HMAC 签发/校验/ipHash，secret 已写入 config.json 随机值）；按消歧 D5 **未接入** /download 校验，文件头已注释「Wave6 MCP download_skill 启用」。
  - 重写法务三页 `public/legal/{about,privacy,contact}.html` 与 `public/docs/mcp.html`（合并旧站 mcp-docs 多页）：自包含静态页，引 /app.css、复用 nav/footer 结构，双语（?lang > localStorage sh_lang > navigator.language，与主站 sh_lang/sh_theme 联动）、暗亮双主题、canonical + meta description；contact 含「提交技能」小节（收录标准 + 三字段 + hello@skillhub.tools，消歧 D4）；docs 页三节 = MCP 预告（Wave6 端点 + Cursor/Claude 配置样例）+ 公开只读 API 文档（含 W4-2 suggest 与 W4-5 /api/mcp/* 的口径）+ 使用条款要点。
  - server.mjs 锚点 A2（import）、A3（guard）、A6（/docs/mcp 与 /legal/* 无扩展名路径重写）+ SITEMAP_URLS 追加 5 条（/legal/×3、/docs/mcp、/mcp）；index.html footer 改造（About/Privacy/Contact/Submit/Docs 链接 + 保留 footer_note 与 #footerBuilt）；i18n.js 追加 W4-6 区段 foot_* ×5（en/zh 成对，i18n_check parity 通过）。
  - 旧站冻结（M5）：**勘误 WAVE3 F5——8787 实为 RStudio Server（他人服务），18787 被他人 Python API 占用，skills-hub 并无任何运行进程**，亦无 crontab/systemd/rc.local 自启项，无进程可停；页面存档至 `skills-hub/docs/archive-freeze/`（临时起服因 sqlite 在 NFS 上 database is locked 失败，改为源文件存档，详见 ARCHIVE-NOTE.md）；README 顶部已加归档标注。未物理删除任何文件。
  - PORT 自测冒烟全过：新四页 200、双语双主题、主站 / /browse /skill/:id /api/* /robots /sitemap 零回归、sitemap-1.xml 含 5 条新 URL；反爬四用例实测（python-requests UA→403、Googlebot→200 放行、306 次超频→429、蜜罐→404 且后续封禁 429）；/download 无 token 200 且 zip 可解（D5）；tokens 签发/校验往返正确。
- 产出：
  - /store/zkyang/skillhub-site/server/security.mjs —— 反爬 guard（锚点 A3 已挂载）
  - /store/zkyang/skillhub-site/server/tokens.mjs —— HMAC 下载 token（待命，Wave6 启用）
  - /store/zkyang/skillhub-site/public/legal/{about,privacy,contact}.html —— 法务三页（双语双主题）
  - /store/zkyang/skillhub-site/public/docs/mcp.html —— API 与 MCP 单页文档
  - /store/zkyang/skillhub-site/config.json —— 新增 security 节（随机 secret / 限流阈值 / UA 名单）与 contactEmail
  - /store/zkyang/skillhub-site/server.mjs —— 锚点 A2/A3/A6 + SITEMAP_URLS（仅此四处）
  - /store/zkyang/skillhub-site/public/index.html —— footer 元素改造（仅此一处）
  - /store/zkyang/skillhub-site/public/i18n.js —— 末尾 W4-6 区段 foot_* key
  - /store/zkyang/skills-hub/README.md + docs/archive-freeze/ —— 归档标注与冻结存档
- 关键决定 / 发现：
  - **端口勘误（重要）**：8787 = RStudio（他人服务，勿动）；skills-hub 配置端口 18787 现被他人 Python API 占用；旧站本无进程，M5 的「停服」以「核实无进程 + 存档 + README 归档」达成。
  - 专属自测端口 3306 与本机 MySQL（127.0.0.1:3306）冲突，自测改绑 127.0.0.2:3306（保持端口号约定），Wave5 复测请沿用。
  - maxRequests 从旧站 120/分钟放宽到 300/分钟（SPA 每页多次静态资源 + API 请求，120 会误伤正常浏览）；白名单爬虫不占普通限流配额。
  - security/tokens 模块自读 config.json（不经 server.mjs 传参），保证锚点只有 import 一行 + guard 一块。
- 阻塞：无
- 交接给下一波：
  - W4-5：SITEMAP_URLS 已预登记 /mcp 一条；/mcp/:id 明细 URL 清单出来后我或 Wave5 追加（SITEMAP_URLS 唯一编辑者是 W4-6）。/docs/mcp 页第二节已写 /api/mcp/search 与 /api/mcp/:id 的文档口径，若实现有出入请在 TIMELINE 说明，Wave5 统一修订。
  - W4-1：footer 新增 .footer-links 目前用内联样式（display:flex;gap:14px），如需 hover/间距精修请在 app.css 补 `.footer-links a:hover`（法务页内部已自带同名样式）。
  - W4-3：foot_* ×5 key 已成对，待统一润色；法务/docs 页为自包含双语（不走 i18n.js），文案润色直接改页面文件（归 W4-6 所有权，可代润色，改前打招呼即可）。
  - Wave6：tokens.mjs 已就位，启用时在 MCP download_skill 链路调 issueToken/verifyToken；security.mjs 的 /mcp 独立限流通道已生效。

### [W4] Agent-W4-3 — 双语体验打磨（语言切换 / fmtNum / 翻译质量与缓存治理）
- 时间：2026-08-17 18:35 (UTC+8)
- 状态：完成
- 做了什么：
  - P0-2 重写 app.js 头部 fmtN → 全站唯一 `fmtNum(n)`（en `1.5k/25.4M`、zh `1.5万/2.5亿`，消歧 D10），挂 `window.fmtNum`；附赠 `window.fmtRel(date)` 双语相对时间（"3d ago"/"3 天前"，供 W4-2 信任行与 W4-5 P1-1 使用）；app.js 全部 fmtN / toLocaleString 调用点（8 处）机械替换完毕，保留 `const fmtN = fmtNum` 兼容别名。
  - P0-1 applyLang 无跳动改造：静态 `[data-i18n]` 原地替换；新增 `setLang()` 按 R3 §3.9 编排——可见视图加 `.lang-fading`（等 `--dur-1`，token 缺失回退 90ms）→ 换语言重绘 → rAF 摘类淡入；prefers-reduced-motion 命中时直接切换；切换全程零位移（仅 opacity，样式类由 W4-1 落地）。
  - P0-3 新建 `scripts/i18n_check.mjs`（en/zh key 集合 diff、空值、public/ 全部 data-i18n / t("…") 引用校验）；终检 85 键 parity=0、72 个静态引用全有定义；审校 W4-2/W4-6 已进场区段的中英文案（与契约 C7 附录一致，全角标点规范，无需改动）。
  - P0-4 回归验证：?lang= > localStorage > navigator 优先级代码路径未动；PORT=3303 实测 `/?lang=zh|en`、`/browse?lang=zh`、`/skill/:id?lang=zh` 的 html lang / title / hreflang×3 / SSR 中文按钮全部正确。
  - P1-1 translateMd 术语保护：新增 20 词 GLOSSARY（Cursor/Claude Code/Codex/SKILL.md/skill/prompt/agent/MCP/token/repo 等）占位符保护；恢复时容忍〔〕→【】变形、残留占位符整行回退原文、并智能补空格修复 gtx 吞空格导致的粘连（如 "Claude CodeLaTeX"→"Claude Code LaTeX"）；实测 9 个新译文：代码块逐字节一致、术语计数 orig=zh、零残留。
  - P1-2 新建 `scripts/zh_audit.py`（残留/行数失配/中文占比三项检测 + zh_cache 治理，支持 --dry-run）：实跑清除 18 条坏缓存（根因见下），md_zh 14 文件全过，报告落盘。
  - P1-3 增量补翻：19/19 成功（cache 14576→14595，todo=0）；顺手修复 translate_zh.py 两个缺陷——`sl=en`→`sl=auto`（韩语/越南语 desc 此前被 gtx 原样返回=那 18 条坏缓存的根因）、代理 SSL 证书校验失败自动降级重建 opener；server.mjs gtx() 同步改 sl=auto。
- 产出：
  - /store/zkyang/skillhub-site/public/app.js —— 仅 W4-3 区段：fmtNum/fmtRel、applyLang/setLang、btnLang 绑定 + 8 处机械替换
  - /store/zkyang/skillhub-site/public/i18n.js —— 头部所有权/追加协议注释（词条本体未动，W4-2/6 区段审校通过）
  - /store/zkyang/skillhub-site/server.mjs —— 仅翻译区段：GLOSSARY + 术语保护 + 恢复兜底/补空格 + sl=auto
  - /store/zkyang/skillhub-site/scripts/translate_zh.py —— sl=auto + SSL 降级重试
  - /store/zkyang/skillhub-site/scripts/i18n_check.mjs —— 新建，key 一致性检查器（交接给全体：加 key 后必跑）
  - /store/zkyang/skillhub-site/scripts/zh_audit.py —— 新建，坏译检测与缓存治理
  - /store/zkyang/skillhub-site/data/zh_audit_report.txt —— 审计报告（当前全绿）
  - /store/zkyang/skillhub-site/data/zh_cache.json、data/md_zh/ —— 治理后 14595 条 / 14 文件
- 关键决定 / 发现：
  - **gtx 需走代理**：本机直连 translate.googleapis.com 超时；`~/.bashrc` 里有被注释的 `http://172.245.154.193:10899`，实测可用。**生产 3199 起服务时必须带 `https_proxy` 环境变量**，否则全文翻译静默失败（curl 会读该 env）——请主控/Wave5 上线时确认 serve.sh 环境。
  - 18 条坏缓存根因不是网络而是 `sl=en` 写死：非英文源（韩/越）被原样返回；已改 sl=auto 并回填正确中文。
  - Python urllib 走该代理会 CERTIFICATE_VERIFY_FAILED（curl 正常），translate_zh.py 已内置降级；zh_audit.py 不出网不受影响。
  - fmtNum 语义变化：浏览页结果计数从 `31,704` 千分位变为 `31.7k / 3.2万`（D10 全站唯一口径的直接结果，Wave5 验收时不要当回归）。
- 阻塞：无。两项待后续联动复验：① 切换零跳动的像素级截图 diff 依赖 W4-1 的 `:lang(zh)` 宽度预留与 `.lang-fading`/`--dur-*` token 落地（我侧类名/时序钩子已就位，token 缺失时回退 90ms 纯 JS 时序，功能不受影响）；② W4-4 重建 skills.json 后需再跑一轮 `translate_zh.py`（带 https_proxy）+ `zh_audit.py`。
- 交接给下一波：
  - W4-2/5/6：新增 i18n key 后跑 `node scripts/i18n_check.mjs`（退出码非 0 即 parity 破损）；数字一律 `window.fmtNum`、相对时间可用 `window.fmtRel`（均随 LANG 自动切换口径）。
  - W4-5：mcp_* 区段进场后我未再审校（本条目时点尚未出现），请 Wave5 复跑 i18n_check 并抽查 zh 文案。
  - W4-1：`.lang-fading` 请按 R3 §3.9 提供 opacity 过渡（--dur-1 淡出 / --dur-2 淡入）；`setLang` 通过读取 `--dur-1` 计算等待时长，token 就位即自动对齐。
  - Wave5：验收「切换无跳动」用两语四页（首页/浏览/详情/⌘K）前后截图 diff；本轮环境无 headless 浏览器，我以 DOM/SSR 层验证为准。

### [W4] Agent-W4-2 — 信息架构与浏览体验（两级分类 / 信任信号 / facets / 空态 404 / 分页真链接 / ⌘K suggest）
- 时间：2026-08-17 18:32 (UTC+8)
- 状态：完成
- 做了什么：
  - server.mjs（自有区段）：新增 GROUPS/CAT_GROUP 常量（契约 C1）与 groupOf()（优先读 W4-4 的 group 字段，缺失时按 cat 查表）；重构 searchSkills → 返回 {list, facets}，facets:{group,cat,tier} 在 q/featured/safe 过滤态后、group/cat/tier 收窄前计数；sort 扩为 score(默认，s.score 缺失回退 _pop)|installs|stars|name|new(added_at 缺失回退 _pop)；/api/search 新增 group 参数（非法值忽略；group×cat 冲突时 cat 胜出）并在响应追加 facets（字段只增不减）；锚点 A5 前插入 /api/suggest（name/id 前缀 + 词首二次匹配 + 分类中英名匹配，实测 p95≈1.8ms）；browse SSR 块支持 ?group= 的 title/desc/canonical/卡片池（en/zh 双语）。
  - index.html（除锚点/footer）：浏览页新增 #groupTabs（.group-tabs）与 #sizeSeg（24/48）；首页新增 New arrivals 区（#newSection，默认 hidden）；CTA code 加 id=ctaCode；新增 #view-404（.page-404 结构 + 搜索框 + Browse 出路）。
  - app.js（除 W4-3 区段）：GROUPS/CAT_GROUP/GROUP_ICON 常量 + DATA_FLAGS 字段嗅探（added_at/tags 出现即自动亮起对应功能）；cardHTML 加 data-cat + .card-edge + .sc-trust 信任行（×N sources / ★stars / updated 相对时间，字段缺失省略）+ .badge.b-new（added_at ≤14 天）；renderStats 四格改 unique(主位)/collected/repos/curated（D7），搜索框占位数字改用 unique；首页分类网格改 6 大类卡片（.group-card，子类小字可直点）；CTA 假命令换真实 top skill 安装命令（可点击复制）；New arrivals 横滑带（无 added_at 自动隐藏）；renderBrowse 重写：group tabs 与子类 chips 联动（选大类只显组内 chips）、chips/tabs 带 .cnt facets 计数、sort 五项（new 无数据时隐藏）、size 24/48、URL group×cat 校验与服务端一致；pager 渲染为真 <a href>（页码组 1 … n-1 n n+1 … max，右键可复制，点击仍走 pushState 拦截）；0 结果渲染 .empty-state（清空筛选按钮 + /api/top 热门 6 条）；renderDetail 未知 id → 404 视图，d-main 加 .detail-ambient[data-cat]，侧栏安装区包 .install-block（D8 移动端置顶由 W4-1 CSS 实现）；route() 增加 /mcp 分支委托 window.renderMcp?.(path,search)（契约 C6，view-mcp 缺失判空回退 404）与未知路径 404；⌘K 接入 suggest（失败回退旧 /api/search）、结果分两段（.cmdk-section：技能 / 分类跳转，中英分类名直达 /browse?cat=）、↑↓/Enter/Esc 跨段可用；点击拦截器排除 /legal/ /docs/ 与带扩展名路径（W4-6 静态页正常整页跳转）。
  - i18n.js：末尾 W4-2 区段追加 en/zh 成对 key ×21（group_*×6、sort_score/stars/new、trust_*、stat_unique/collected、empty_clear/hot、nf_*、new_arrivals、badge_new、cmdk_sec_*）；node scripts/i18n_check.mjs parity 通过（85 keys，diff=0）。
  - 自测（PORT=3302）：改前基线 curl 留档对比 —— 老参数（q+cat+tier+sort+page+size、featured+safe）total/ids/响应字段完全一致，仅新增 facets；group=dev 计数 6536 = 四子类之和；sort 五项排序正确；SSR /browse?group=* 双语标题、canonical、/、/skill/:id、robots、sitemap 零回归；无浏览器环境，另写最小 DOM shim（/tmp/w42_dom_smoke.mjs）连真实 API 跑 7 条路由（首页/浏览组合/空结果/详情/坏 id/未知路径/mcp 缺失回退）全部零运行时错误；W4-6 反爬 guard 落地后再跑一轮合并态回归通过。
- 产出：
  - /store/zkyang/skillhub-site/server.mjs —— GROUPS 常量、searchSkills、/api/search、/api/suggest（A5）、browse SSR 块（仅自有区段）
  - /store/zkyang/skillhub-site/public/app.js —— 浏览/首页/详情/⌘K/路由改造（W4-3 区段未动）
  - /store/zkyang/skillhub-site/public/index.html —— groupTabs/sizeSeg/newSection/view-404/ctaCode（锚点 B1~B4 与 footer 未动）
  - /store/zkyang/skillhub-site/public/i18n.js —— 末尾 W4-2 key 区段
- 关键决定 / 发现：
  - facets 口径 = 「q/featured/safe 过滤后、group/cat/tier 收窄前」的全量计数，保证切换 tab/chip 时能预判各选项结果规模（学 HF）；group×cat 冲突裁决为 cat 胜出（更具体），前后端同一规则。
  - 与 W4-3 并行零冲突：他们先落地了 fmtNum/fmtRel，我直接复用（fmtRel 做信任行相对时间），删掉了自己计划中的重复 relTime/rel_* key。
  - 旧 skills.json 无新字段，一切新功能按 DATA_FLAGS 嗅探自动降级；W4-4 数据落地后无需改代码，信任行 updated/New arrivals/sort=new 自动亮起（b-new 阈值 = added_at 距今 ≤14 天）。
  - 首页 SSR 的 catGrid 注入仍是 20 分类真链接（利于 SEO 长尾），水合后前端替换为 6 大类卡片——语义不冲突，保留现状。
- 阻塞：无
- 交接给下一波：
  - W4-1（样式缺口，现为裸结构可用）：.group-card/.gc-cats/.gc-cat（首页 6 大类卡片内的子类小字行）、.empty-state 建议 grid-column:1/-1 撑满网格、.pg-btn/.pg-num/.pg-gap（新版分页真链接的类名，旧 #pgPrev/#pgNext 按钮已移除）、#sizeSeg 复用 .seg 即可；.sc-trust/.group-tabs(.gt/.cnt)/.chip .cnt/.page-404/.cmdk-section/.badge.b-new/.card-edge[data-cat]/.detail-ambient/.install-block 均已按契约 C5 类名落在 DOM 上等样式。
  - W4-3：applyLang 内占位符仍用 stats.total，我在 renderStats 里以 unique 二次覆盖（避免动你的区段）；如愿可把 applyLang 里 {n} 直接改 stats.unique 后删我的覆盖段（renderStats 内有注释标记）。zh 的 trust_updated("更新于 {t}") + fmtRel("N 天前") 组合为「更新于 N 天前」，如需去重语感请统一润色。
  - W4-5：route() 的 /mcp 委托与 views.mcp 判空已就位（契约 C6），只需 mcp.js 暴露 window.renderMcp 并靠 B3 插入 #view-mcp；/api/suggest 响应形状 {items:[{id,name,cat}],cats:[{id,en,zh,count}]} 与 /docs/mcp 文档口径一致。
  - W4-4：前端消费契约 C2 已全部就绪，重建 skills.json 后请在时间线发「数据已更新」条目，我方无需改码；sort=new 选项、信任行 updated、New arrivals、b-new 徽章将自动出现，建议 Wave5 复测这四处。
  - Wave5：group 维度的 sitemap URL（/browse?group=× 6 条）未登记（SITEMAP_URLS 唯一编辑者是 W4-6），如需收录请转交 W4-6 追加。

### [W4] Agent-W4-1 — 设计系统与动效落地（R1 变量 / R3 token / 新组件样式 / motion.js）
- 时间：2026-08-17 18:55 (UTC+8)
- 状态：完成
- 做了什么：
  - 重构 public/app.css（全文件所有权，955 行）：R1 §3.2 变量提案 v2 整体落地（:root + dark 两块替换，旧变量名全保留；暗色表面四级阶梯 #0a0a0c/#151517/#1d1d20/#26262a、暗色阴影降描边级、亮色 accent-hover 压深 #0076df、补 accent-active/tint/focus-ring/hairline-soft/pop-shadow/danger/info/grad-soft）；20 分类 hue map + [data-cat] 派生公式 + 六大类 [data-group] hue 别名。
  - R3 §2 动效 token 全量进 :root（--dur-1~5、7 条缓动、stagger/hero-cascade/detail-lag）+ Stripe 归零式 reduced-motion 第一道闸；全文件 27 处裸时长全部替换为 token（含 aurora/gradFlow/breathe/sheen/skeleton 等装饰长时长的 token 化），脚本校验 transition/animation 声明零裸数字。
  - R3 §5 五处违规修复：sheen 改 transform:translateX(410%)、reveal stagger 35ms + min() 封顶（--rv 按行取模的注释已留给 W4-2）、reveal 320ms、.md-status 去 pulse（骨架屏 .skeleton 就位）、全部 transition:all 收窄为具体属性。
  - 契约 C5 新组件样式全量产出并与 W4-2 实际 DOM 对齐：.sc-trust、.group-tabs(.gt/.cnt)、.chip .cnt、.empty-state(.es-ic/.es-msg/.es-hot)、.page-404(.nf-*)、.skeleton(transform 扫光)、.pager 真链接(.pg-btn/.pg-num/.pg-gap)、.cmdk-section、.badge.b-mcp/.b-new、.install-block（含 :has() 移动端侧栏置顶，消歧 D8）、.card-edge（适配 W4-2 的「卡片修饰类」用法，改 inset box-shadow 实现零布局位移）、.detail-ambient、.gc-cats/.gc-cat、.skill-card.is-official、.btn-primary 全局化、.footer-links hover（应 W4-6 交接）。
  - P1 动效编排：Hero 四段级联（html:not(.hero-played) 门控只播首次）、⌘K 开合（cmdk-in 换 --ease-swift + .closing 退场协议）、toast .leaving 退场、详情页两段入场（view-in 对 #view-detail 豁免）、语言切换 .lang-fading/.view 基态 opacity 过渡（W4-3 的 setLang 已按此钩子实现）、reduced-motion 第二道闸逐项兜底（R3 §4 表）。
  - 新建 public/motion.js（~110 行，零依赖）：nav 滚动 sentinel（IntersectionObserver）、[data-countup] 数字 count-up（rAF+easeOutExpo 800ms，格式化优先 window.fmtNum）、#cmdk.closing 与 #toast.leaving 的 animationend 收尾协议、主题切换 View Transitions 圆形揭示（capture 拦截 #btnTheme 后调原 onclick 保持 app.js 状态权威，Firefox/reduced-motion 自动回退原逻辑）；index.html 仅锚点 B1 插入一行。
  - P2-1 Inter self-host 完成：4 字重 latin woff2（~96KB）下载至 public/fonts/，@font-face font-display:swap + 收窄 unicode-range（避免中文标点被 Inter 抢渲染），字栈按 WAVE1 §1.3 加 Inter 与 JetBrains Mono。
  - PORT=3301 自测：/、/browse、/skill/:id、app.css/motion.js/fonts 全 200；Googlebot UA 的 /browse?lang=zh SSR 标题正常；测毕已停服释放端口。类名保留校验：app.js/index.html 引用的 142 个类全部存在（as-btn/footer-note 为历史无样式类，tr-*/group-card 经父选择器与元素选择器覆盖）。
- 产出：
  - /store/zkyang/skillhub-site/public/app.css —— 设计系统 v2 全量（token/组件/动效/zh 排印/降级）
  - /store/zkyang/skillhub-site/public/motion.js —— 动效行为层（新建）
  - /store/zkyang/skillhub-site/public/fonts/inter-latin-{400,500,600,700}-normal.woff2 —— 自托管字体（新建）
  - /store/zkyang/skillhub-site/public/index.html —— 仅锚点 B1 一行（<script src="/motion.js">）
- 关键决定 / 发现：
  - **P0-2 surface-2 排查结论（消歧 D1）**：全站 grep `--surface-2` 仅 3 处——:root 与 dark 的变量声明 + `.d-md th` 一个用例；th 底为「表头略亮于卡面」的浮起语义，维持 --surface-2（暗色新值 #1d1d20 亮于卡面 #151517，方向正确）；无凹面误用需要改 --code-bg 的场景。app.js/server.mjs/i18n.js 均无引用。
  - **R1 §3.3 对比度表勘误**：黄绿段 8 个 hue（docs-office 42/web3 55/fun 95/game 120/productivity 145/business 160/testing 172/cloud-devops 185）亮色 L=34% 实测不足 AA（web3 仅 3.67:1，R1 表称 ≥5.2 有误），已逐类压深至 4.6:1 以上（29%~33%，见 app.css 勘误区段）；暗色 20 类全部 ≥6.7:1 未动。数值抽查：暗 text-2 7.09:1、text-3 5.03:1、正文双主题 16.8:1。
  - [data-cat] 通配上色会波及带 data-cat 的卡片（W4-2 给 .skill-card 加了 data-cat），已按 WAVE1 §1.2「hue 只用于 chip/缘条/氛围」原则对 .skill-card/.cat-card 做中性化覆盖（卡面保持 surface，图标着 hue 色）；.card-edge 因 W4-2 用作卡片修饰类而非独立条元素，改用 inset box-shadow 实现（零布局位移、随圆角裁切）。
  - 双语零跳动宽度预留策略：导航三链接/#btnLang/.seg button(76px，#sizeSeg 豁免)/分页 prev-next(96px)/.check 标签/六大类 tab（按 data-g 每槽 min-width，en 恒最长）；**chips 的 20 分类名与 W4-2 未来的 sort_score("Best match"≈88px>76px 预留) 无法纯 CSS 全预留**，已留给 W4-3 验收时按需加宽或缩短文案。
  - motion.js 主题 VT 拦截采用「capture 阶段 stopPropagation + 直接调 btn.onclick()」方案：app.js 闭包内 THEME 状态仍是唯一权威，不产生双份状态。
- 阻塞：无
- 交接给下一波：
  - W4-2：count-up 需在 renderStats 的 <b> 上加 data-countup="原始数字"（当前是纯文本，motion.js 已就绪待属性）；⌘K/toast 退场如需动画，关闭时改加 .closing/.leaving 类（motion.js 负责 animationend 后置 hidden），当前直接置 hidden 也不报错；卡片入场 .reveal/.in 与 --rv（i % perRow）钩子在 CSS 就位，JS 侧未接。
  - W4-3：.lang-fading（--dur-1 淡出 / --dur-2 淡入）与 :lang(zh) 排印、min-width 预留已全部落地，可开始逐页截图验收；chips/长 sort 文案两处像素级预留缺口见上，验收不过再找我加。
  - W4-5：mcp.css 可直接引用全部 token（--dur-*/--ease-*/--accent-tint/--pop-shadow/[data-cat] 公式/.badge.b-mcp/.skeleton/.cmdk-section）；数字用 window.fmtNum。
  - Wave5：本轮环境无 headless 浏览器，双主题截图对照与 reduced-motion 实机验收（DevTools Rendering 面板）需人工/有浏览器环境复核；对比度已用脚本数值化验证（结论见上）。
- 遗留：验收项「双主题截图对照」「reduced-motion 实机全站检查」以静态分析+数值计算替代完成，待 Wave5 有浏览器环境复核。

### [W4] Agent-W4-5 — MCP Servers 聚合类型（build_mcp / mcp-data / /mcp 列表与详情）
- 时间：2026-08-17 19:05 (UTC+8)
- 状态：完成
- 做了什么：
  - 新建 scripts/build_mcp.py（纯 stdlib）：官方 registry `/v0/servers?version=latest` cursor 分页全量同步（222 页，限速 0.35s/页、重试 3 次、样例兜底、<100 条地板保护 + 原子写）；pulsemcp v0beta 尽力补充（该 API 处于日落期，官方设计为随机失败，已按 5 次重试容忍）；字段映射照 R2 §4.2 十二字段，id=md5(name)[:10] 与 skills 同规则；install_config（{"mcpServers":{…}} 完整 JSON）与 install_claude（claude mcp add 命令）均构建期生成，npm/pypi/docker/nuget/remote 五形态覆盖。
  - 构建 data/mcp.json：**21,878 个 MCP servers**（official 6,655 / community 15,223 / reference 0），10,099 条带 stars/downloads，21,112 条带 install_config，19.6MB。
  - 新建 server/mcp-data.mjs：内存索引 + handleMcpRoutes(req,res,url)——/api/mcp/search（q/classification/page/size，返回 facets.classification 联动计数 + synced/stats 信任带；列表投影不含 tools/packages/install_config）、/api/mcp/:id（全字段）、/mcp 与 /mcp/:id 的 SSR（meta/OG/hreflang/JSON-LD CollectionPage 与 SoftwareApplication、真 <a> 卡片、复用 index.html 壳）；未命中返回 false 由主站继续。
  - server.mjs 仅锚点 A1（import 一行）+ A4（委托两行）；index.html 仅锚点 B2（mcp.css/mcp.js）/B3（#view-mcp 空容器）/B4（导航 nav_mcp 链接）。
  - 新建 public/mcp.js：window.renderMcp(path,search)（契约 C6，W4-2 的 route() 委托已就位并联通）；列表页=搜索框+classification 分面 chips（带 .cnt 计数）+卡片（title/desc/b-mcp/transports 徽章/N tools/★stars）+编号真链接分页（aria-current 当前页）；详情页照 mcp.so 四段式：徽章带 → Config 块首屏一键 copy（Cursor ~/.cursor/mcp.json / Claude Code / 原始 JSON 三 tab）→ Tools >20 折叠 → 侧栏（license/transports/author/stars/downloads/updated/远程端点/env 变量表含 required+secret 标注）；数字走 window.fmtNum（缺失本地双语回退）；附带自愈式导航垫片（pushState 包装+popstate+hidden 观察器，全部在自有文件内，C6 委托在场时自动去重为 no-op）。
  - 新建 public/mcp.css：仅 MCP 专属类（mc-*/mcp-*），布局/卡片/chips/分页/b-mcp 全部复用 W4-1 设计系统 token 与契约 C5 类。
  - i18n.js 末尾 en/zh 各追加 `// ==== W4-5 ====` 区段 28 对 key；`node scripts/i18n_check.mjs` 通过（111 keys，diff=0）。
  - PORT=3305 全流程自测后停服：/api/mcp/search（q/classification/分页 disjoint）✓、/api/mcp/:id（playwright npx 配置 JSON parse ✓ 与官方文档格式一致；remote 型 url 配置 ✓）、/mcp 与 /mcp/:id SSR（en/zh 双语 title、视图切换、JSON-LD）✓、404 ✓；主站零回归：/、/browse、/skill/:id、/api/meta、/api/search（结构 [facets,items,page,size,total] 不变）、/api/top、/api/random、robots、sitemap 全 200。
- 产出：
  - /store/zkyang/skillhub-site/scripts/build_mcp.py —— MCP 同步管线（新建）
  - /store/zkyang/skillhub-site/data/mcp.json —— 21,878 条构建产物（19.6MB）
  - /store/zkyang/skillhub-site/server/mcp-data.mjs —— 数据/API/SSR 模块（新建，另导出 mcpSitemapUrls()）
  - /store/zkyang/skillhub-site/public/mcp.js、public/mcp.css —— /mcp 前端（新建）
  - /store/zkyang/skillhub-site/server.mjs —— 仅锚点 A1/A4
  - /store/zkyang/skillhub-site/public/index.html —— 仅锚点 B2/B3/B4
  - /store/zkyang/skillhub-site/public/i18n.js —— 末尾 W4-5 区段（en/zh 成对）
- 关键决定 / 发现：
  - classification 推导规则（官方 registry 无此字段）：io.modelcontextprotocol* → reference、io.github.* → community、其余（DNS 域名验证发布者）→ official。**reference=0 是注册表现状**（官方参考实现已归档、不在 registry 收录），非映射 bug；前端 reference chip 显示 0 属实。
  - pulsemcp v0beta 已进入日落（2026-06 起 50% 请求按设计随机失败、v0.1 需 API key），本轮仍拿到 22,045 行、匹配 10,099 条；**Wave5 后重跑 build_mcp.py 时 enrichment 命中率会持续下降，2026-09 全灭**，届时需换 GitHub API 直取 stars 或申请 pulsemcp key。
  - registry 同步全程约 21 分钟（222 页限速），重跑幂等（映射确定性 + 原子写 + 地板保护：新结果 <100 条且不优于现存文件时拒绝覆盖）。
  - install_config 的 mcpServers key 对通用尾段（mcp/server）回退用命名空间（ac.inference.sh/mcp → inference-sh），修正已进脚本；现存 mcp.json 中少量此类 key 为 "mcp" 等通用名——配置照常可用，下次重跑自动更名。
- 阻塞：无
- 交接给下一波：
  - W4-6（sitemap，P2-1）：server/mcp-data.mjs 已导出 `mcpSitemapUrls()`（/mcp + 21,878 条 /mcp/:id）；SITEMAP_URLS 若只想登记入口页，加 "/mcp" 一条即可；若要全量收录可 `import { mcpSitemapUrls } from "./server/mcp-data.mjs"` 后 concat（现有 5000/片分片机制可承接）。
  - W4-3：mcp_* 系列 28 对 key 待你统一润色；数字已走 window.fmtNum（你落地后自动接管，mcp.js 内有本地回退不冲突）。
  - Wave5：①本轮环境无 headless 浏览器，/mcp 前端交互（copy/tab/折叠/分面）以 node --check + SSR + API 冒烟替代，需实机浏览器复核；②tools 列表官方 registry 不提供（字段与 >20 折叠 UI 已就位，数据为空则该节自动隐藏），Wave6 MCP 动态通道可考虑运行时 tools/list 补采；③mcp.json 19.6MB 使 server 常驻内存增加约 60MB，3199 上线前请确认宿主内存水位。
- 遗留：pulsemcp 日落风险与 tools 数据缺位见上；均不阻塞上线。

### [W4] Agent-W4-4 — 上游数据扩容与索引升级（196→1,732 仓库、五新字段、M4 吸收）
- 时间：2026-08-17 20:55 (UTC+8)
- 状态：完成
- 做了什么：
  - 勘明基线：8月3日已跑过一轮 wave4 发现+克隆（81 仓库）且已计入当前索引，实际基线即 stats.repos=196 / 31,704 条；本轮为第二轮扩容（wave4b）。
  - 写 `_research/discover_wave4b.py` 做第二轮发现：盘内 14 个 awesome 清单提链（1,230 候选）+ GitHub topic×14 / 查询×6（含中文生态专项 4 条）+ claudeskills.info 三榜，共 **2,413 个新候选仓库**；网络层因本机 python urllib 走 IPv6 逐条 30s 超时，改用 curl 子进程后恢复。
  - 写 `_research/clone_wave4b.sh` 抓取：git clone 并发被 GitHub 限流后**改用 codeload tarball 通道**（更快且无 .git 体积）；一律浅拉、>250MB 丢弃、**无 SKILL.md 立即剔除**（CLEANING.md）、每 ~20 个检查一次 /storo 红线（<1.2T 即停，未触发）。结果：**OK 1,535 / 无技能剔除 702 / 过大 9 / 永久失败 366（私有/已删）**；含 FAIL 复试一轮。
  - M4 旧站吸收（`_research/absorb_legacy.py`）：对比 skills-hub/data/skills 10,563 目录与 31.7k 集合（name+desc80 键，与 build_data 去重口径一致），**吸收 1,219 条**（71 条建真实来源伪仓库目录、1,148 条进已克隆仓库的 `_legacy/` 子目录、9,344 条重复跳过、0 错误）；只增不删，日志 `_research/absorb_log.tsv`。
  - 重建索引：SKILLS_INDEX.csv 109,556 行 / 1,733 仓库。
  - 升级 `scripts/build_data.py`（契约 C2 五字段，只增不改老字段）：group（C1 查表）、tags（60 词表 ≤6 个 + tier:/has:scripts/lang:zh 前缀）、added_at（`_research/seen_dates.json` 首见日期，本轮 2026-08-17 为基线日）、updated_at（skill 目录内最新文件 mtime；tarball 保留上游 mtime，日期分布真实）、score（log(stars+1)*0.4+log(installs+1)*0.35+featured*2+has_scripts*0.3+(sources-1)*0.2，权重进脚本常量 SCORE_W 带注释；top15 抽看排序合理未再调）；输出改为临时文件+原子替换。
  - 重建 skills.json 并 PORT=3304 冒烟：/api/meta、/api/search（含中文查询）、/api/skill/:id、/api/top、/api/random、/browse SSR 全 200；搜索 110k 条约 95ms/次；server RSS ~295MB（主机 251GB，可承受）；测毕停服。
  - curated 半自动扩容（`_research/curate_wave4.py`）：按 score 排序+每仓 ≤12+desc≥60 字符补 51 个 picks（description 列带 `[w4-pick]` 一句话理由），featured 348→**527**。
  - 修复 `_research/clone.sh` 隐患：存在性检查 `-d $dest/.git` 改 `-d $dest`，防止未来重跑对吸收目录先撞后 `rm -rf`。
- 产出：
  - /store/zkyang/skillhub-site/data/skills.json —— 109,556 条 / 唯一 56,237 / 仓库 1,732 / curated 527 / with_scripts 11,707，**70MB**
  - /store/zkyang/skillhub-site/scripts/build_data.py —— 五字段升级版
  - /storo/zkyang/agent-skills-hub/SKILLS_INDEX.csv、CURATED_PICKS.csv（+51 行）
  - /storo/zkyang/agent-skills-hub/_research/{discover_wave4b.py,clone_wave4b.sh,absorb_legacy.py,curate_wave4.py} —— 本轮管线脚本
  - /storo/zkyang/agent-skills-hub/_research/{wave4b_clone.txt,wave4b_clone_log.tsv,absorb_log.tsv,seen_dates.json,wave4_build.log} —— 清单/日志/首见日期
- 关键决定 / 发现：
  - **数据已更新**（R-2）：其他 Agent 请重启自测服务重新加载 skills.json；新字段（group/tags/added_at/updated_at/score）已通过 /api/search 下发,W4-2 信任行可点亮。
  - added_at 本轮全部为基线日 2026-08-17（D3 起点），「New arrivals」要等下一轮构建才有区分度；seen_dates.json 已持久化。
  - skills.json 70MB（远超预计 20MB），server 常驻 ~295MB、单次搜索 ~95ms；3199 上线可承受，但 Wave5 若做性能优化可考虑列表字段瘦身或索引预分桶。
  - 磁盘：/storo 剩余 1.7T（红线 1.2T 未触）；sources 9.1G→27G（增 ~18G ≤30G 预算）。
  - 大体量清单仓（sickn33 双清单各 6,317 条、aiskillstore/marketplace 5,587 条、antigravity 镜像 ×4）贡献了主要增量，镜像重复已被 name+desc80 去重折叠进 sources 多来源背书。
  - P1-3 分类失衡观察（唯一条目口径）：frontend 22.8% / chinese 17.9% / backend 14.7% / security 12.8% / agent-meta 7.4% / **general 6.4%（<15%，无需补关键词表）**；六大类 dev 22,563 / cn 14,159 / secops 7,808 / ai 5,958 / data 3,846 / biz 1,903。长尾类 game 76 / fun 127 / mobile 248 依旧小，维持折叠进大类展示即可。
  - 中文 tier：450→9,465（唯一 8,354），远超 800+ 目标（chinese 专项查询 + libukai 清单 + 旧站 zh 条目吸收共同贡献）。
- 阻塞：无
- 交接给下一波：
  - W4-2：group/tags/added_at/updated_at/score 已就绪；facets 若沿用全量线性扫描请注意 110k 条的耗时（现搜索 ~95ms/次）。
  - W4-3：请对新增条目跑一轮 zh_cache 增量补翻（条目 31,704→109,556，增量巨大，建议按浏览热度分批）；desc_zh 机制未动。
  - Wave5：①366 个永久失败仓库清单在 wave4b_clone_log.tsv（FAIL 行）可复核；②未做 CLEANING.md 的逐条内容审查（体量不允许），沿用「无 SKILL.md 剔除+250MB 上限」的机器清洗口径，恶意内容审查建议抽样进行；③score 权重可在有真实点击数据后再调。

### [W5] Agent-REVIEW（Wave5 回收/规划 Agent，fable 5） — Wave4 验收汇总与 Wave6 提示词定稿
- 时间：2026-08-17 21:10 (UTC+8)
- 状态：完成
- 做了什么：
  - 通读 W4 六条目并实勘 skillhub-site 现状核验：六路全部「完成」零受阻；逐一核对了 server.mjs 锚点 A1~A6、security.mjs /mcp 独立限流通道、tokens.mjs 接口（issueToken/verifyToken/ipHash，secret 已入 config）、mcp-data.mjs 导出、skills.json 五新字段与 /docs/mcp.html 占位内容，全部与各条目声明一致。
  - 消歧裁决 E1~E9（全文见 WAVE6-AGENT-PROMPTS §0.3）：POST/GET /mcp 按 method 分流（GET 已被 W4-5 SSR 占用，OUTLINE 架构图漏了这点）、MCP 下载走新路径 /dl/:id.zip?t= 强制 token（/download/ 保持公开，D5 不破坏）、sha256 懒计算缓存、trending 只开 all/new（added_at 全为基线日）、p95 验收放宽至 search/get <150ms（110k 全扫实测 ~95ms 均值）、get_skill zh 不同步等翻译、E2E 无公网时 127.0.0.1 直连降级、无状态 JSON 单响应不做 SSE、Wave4 遗留 8 项全部不进 Wave6（列 PROMPTS 附录 B 维护清单）。
  - 撰写 WAVE6-AGENT-PROMPTS.md：共同协议（Wave4 事实基线快照 + E1~E9 + 所有权矩阵与锚点 A7~A10 + 契约 M-C1~M-C6 + 端口 3401~3406）+ 6 份可直接投递的完整提示词 + 附录 A 工具 schema 定稿（8 工具，兼任 MCP-SPEC，按 D9 先例不单开文件）+ 附录 B 遗留清单。
  - 勘误 WAVE6-OUTLINE.md：文头加定稿指针，文末追加 §7 九条勘误（数据规模 109,556/56,237/1,732、method 分流、/dl/ 路径、schema 定稿口径、p95 修订、tools 数据缺位、限流/token 已就位无需接线等）。
  - 刷新本文件「二、波次总览」W4/W5 状态为完成。
- 产出：
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/WAVE6-AGENT-PROMPTS.md —— 共同协议 + 6 提示词 + MCP-SPEC 定稿（附录 A）+ 遗留清单（附录 B）
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/WAVE6-OUTLINE.md —— 文头指针 + §7 Wave5 勘误（原文未动）
  - /store/zkyang/qingtian-standalone/docs/ui-overhaul/TIMELINE.md —— 本条目 + 总览表状态
- 关键决定 / 发现：
  - **GET /mcp 路由冲突是 Wave6 最大的隐藏地雷**：server.mjs 全程不检查 req.method，W4-5 的 SSR 列表页占用了 GET /mcp；W6-1 必须在 guard 之后、A4 委托之前做 method 分流，否则 JSON-RPC 请求会拿到 HTML。已写死进 W6-1 提示词与锚点 A9。
  - 数据基线已冻结口径：Wave6 期间禁止重跑 build_data/build_mcp（数据冻结红线），server RSS ~360MB、启动数秒属正常，写进 §0.7 防误判。
  - get_install_plan 的三客户端口径锚定 public/app.js 的 INSTALL_TARGETS（站内站外一致），避免两套安装命令漂移。
  - Wave4 唯一系统性阻塞仍是生产部署链路（3199 无进程、skillhub.tools 指向未确认，W3 起悬而未决）；另生产起服必须带 https_proxy（W4-3 发现）。两项均归主控，不阻塞 Wave6 开发与降级验收。
- 阻塞：生产部署链路待主控确认（同 W3/W4 遗留；E2E 已内置 127.0.0.1 降级预案，不阻塞投递）。
- 交接给下一波：
  - 主控从 WAVE6-AGENT-PROMPTS.md 复制 6 份提示词投递：W6-1/W6-2 先行并行，W6-3/4 随后，W6-5 占位稿可同步写，W6-6 最后串行收口（§0.8）。
  - Wave6 各 Agent 开工先读 PROMPTS §0.2 事实基线与 §0.3 消歧 E1~E9；schema 一律以附录 A 为准。
  - Wave7/维护：附录 B 遗留清单 L1~L8（zh 增量补翻、i18n 润色、sitemap 登记、motion 接线、pulsemcp 日落、浏览器实机复核、生产上线、宽度预留）。

### [W6] Agent-W6-3 — 安装工具链（download token 接线 + /dl/ + get_install_plan）
- 时间：2026-08-17 21:30 (UTC+8)
- 状态：完成
- 做了什么：
  - 新建 server/mcp-install.mjs：downloadSkill / getInstallPlan 两个 MCP handler（M-C1 签名，业务失败 throw Error 带 userMessage）+ handleDl 路由函数 + sha256/size 懒计算缓存（E3：同款 zip 命令流经 createHash，Map 上限 2048 FIFO、并发去重、15s 超时兜底 null 不阻塞下载）。
  - server.mjs 锚点 A7（`import { handleDl }` 一行，位于 W6-1 createMcpHandler import 之后，顺序符合 §0.5）+ A10（`/dl/` 路由块，插在 `// ---- zip download ----` 之前）。契约二选一已注明：**handleDl 自带 deps 第 4 参 `{ byId, HUB }`，不走 A8 注入**。
  - GET /dl/:id.zip?t=<token>：verifyToken 失败一律 403，文案区分 expired（exp 已过）/ invalid（其余）；缺 token 403 提示调 download_skill；通过后按 /download/ 同款 spawn zip 流式返回。/download/ 既有路由一字未动（D5）。
  - get_install_plan：三客户端命令与目录逐字段照抄 public/app.js INSTALL_TARGETS（cursor `~/.cursor/skills` / claude `~/.claude/skills` / codex `~/.codex/skills`，safeName 规则同 installCmd），下载一步换成顺手签发的 /dl/ 带 token 直链（M-C5+E3）；post_install_checks 三条（SKILL.md 位于 target_dir 顶层、frontmatter name 对应、客户端重载后可见，另附 403 时重取链接提示）。
  - 自测脚本 scripts/dl_negative.mjs（P1-1）：负例用 config secret 伪造「真签名但 exp 已过」token（无需临时改 TTL）；单轮 5 次下载请求，60s 内连跑两轮会撞 maxDownloadsPerWindow=10（W6-4 已把 /dl/ 计入下载通道），重跑请间隔 1 分钟。
- 产出：
  - /store/zkyang/skillhub-site/server/mcp-install.mjs（新建，~180 行）
  - /store/zkyang/skillhub-site/scripts/dl_negative.mjs（新建，15 用例自测脚本）
  - /store/zkyang/skillhub-site/server.mjs —— 仅锚点 A7 一行 + A10 三行
- 验收（PORT=3403 实测，测毕已停服）：
  - [x] download_skill → /dl/ 直链 curl 200 → unzip 可解含 SKILL.md
  - [x] 无 token / 篡改 token / 过期 token 三负例全 403（invalid/expired 文案区分）；/download/:id.zip 无 token 仍 200（D5 不破坏）
  - [x] sha256 与实际下载文件一致（d331857f…，且跨进程重启稳定=zip 输出确定性成立）；二次调用命中缓存（172ms→1ms）
  - [x] get_install_plan 三客户端与 INSTALL_TARGETS 逐字段同口径（P4 三用例 + 未知 client/id 业务错误负例）
  - [x] 主站冒烟零回归：/、/browse、/skill/:id、/api/search、/api/mcp/search、GET /mcp、/mcp/:id、robots、sitemap 全 200
  - [x] 与 W6-1 真接线复验：POST /mcp tools/call download_skill / get_install_plan 均 isError:false 且直链 200（W6-1 mcp.mjs 的导出名探测 downloadSkill/getInstallPlan 与本文件导出精确匹配，无需 stub 过渡）
- 关键决定 / 发现：
  - expires_in 从 config security.downloadTokenTtlSeconds 自读（当前 600），不改 tokens.mjs（其 TTL 常量不导出，遵守「只读为主」）。
  - expired/invalid 区分实现：verifyToken 只回布尔，403 文案按 token 前缀 exp 是否已过判定——伪造的过期 exp 也会得到 expired 文案，属提示性区分、无信息泄露（两者同 403）。
  - safeSkillDir 在 server.mjs 不导出（M-C2 入口副作用），mcp-install.mjs 同逻辑重实现 skillDirOf（路径逃逸防护保留）。
  - 自测期间 W6-4 已把 /dl/ 并入 dlCount 下载限流通道且跳过 UA 黑名单（security.mjs 实勘），本文件无需任何限流代码，口径一致。
- 阻塞：无（生产公网链路仍是主控遗留，与本任务无关）。
- 交接给下一波：
  - W6-6：负例复演直接跑 `BASE=http://127.0.0.1:34XX node scripts/dl_negative.mjs`（15 用例含 sha 一致性与缓存命中断言）；注意单轮 5 次下载、连跑需隔 60s；过期负例是真签名 token，覆盖比篡改更严。
  - W6-5：文档口径——/dl/ 仅接受 download_skill / get_install_plan 签发的链接，600s 过期后重调工具即可；/download/ 公开直链口径不变。
  - W6-4：/dl/ 的 403（token 拒绝）发生在 guard 放行之后，即被拒请求也计一次 dlCount，属可接受现状（攻击者刷 /dl/ 无 token 也会被下载限流封禁，反而有利）。

### [W6] Agent-W6-1 — MCP 协议与端点核心（server/mcp.mjs + POST /mcp 分流）
- 时间：2026-08-17 21:13 (UTC+8)
- 状态：完成
- 做了什么：
  - 新建 server/mcp.mjs：createMcpHandler(deps) → async (req,res)。readBody 上限 256KB（超限 413）、坏 JSON -32700、单条/批量数组均支持（批量进批量出，全通知回 202 空体、空批量 -32600）、无状态单响应（E8：不发 Mcp-Session-Id、忽略会话头、不做 SSE）。
  - dispatch 实现 initialize（协议协商集 {2025-06-18, 2025-03-26, 2024-11-05}，集外回 2025-03-26；serverInfo skillhub-mcp/1.0.0；capabilities {tools:{}}；instructions 含三层模型导语与典型流程）/ ping / notifications/*（忽略）/ tools/list / tools/call；错误码按 M-C6（-32601 未知方法、-32602 工具名缺失或未知工具、-32603 internal）；handler 抛错一律转 isError textResult（业务错误不走 JSON-RPC error）。
  - TOOLS 数组按附录 A 定稿逐字实现 8 工具的 name/description/inputSchema（enum 口径与 GROUPS/DB.categories 一致：6 group / 20 cat / 4 tier / 5 sort；id 参数描述均注明「来自 search_skills / search_mcp_servers」，E2）。
  - 工具接线（M-C1）：注册表先装内联 stub（isError「pending W6-2/3」），启动时动态 import ./mcp-router.mjs 与 ./mcp-install.mjs，按 camelCase/snake_case 导出名探测换装真 handler（文件缺失=并行期正常走 stub 并 console.warn；就位后重启进程自动接真）。自测时 W6-2/3 文件已落盘，八工具全部真 handler 生效。
  - 新建 server/mcp-audit.mjs no-op 骨架（audit({ts,ip_hash,tool,args_summary,ok,ms}) 签名注释），已按矩阵移交 W6-4（其间 W6-4 已填充落盘实现且签名兼容，链路验证 OK）；mcp.mjs 每次 tools/call 后调 audit，ip_hash 用 tokens.mjs 的 ipHash，args_summary 由 mcp.mjs 提炼短字段截 80 字符。
  - server.mjs 插三锚点：A7（import createMcpHandler）、A8（const mcpHandler = createMcpHandler({DB, byId, searchSkills, groupOf, pub, HUB, SITE_URL})，M-C2）、A9（guard 后 try 前：POST /mcp → mcpHandler；GET /mcp 且 Accept 含 text/event-stream 不含 text/html → 405；DELETE /mcp → 405；其余 GET /mcp 落 A4 handleMcpRoutes 零回归）。
  - 新建 scripts/mcp_smoke.mjs 六用例并 PORT=3401 实测：initialize 三版本协商 / tools/list 8 工具 / tools/call search_skills（stub 与真 handler 双兼容断言）/ 批量+纯通知 202+空批量 / 坏 JSON -32700 + 超限 413 / 未知方法 -32601 + 未知工具与缺名 -32602 + 裸数字 -32600，共 23 断言全过。
  - 主站冒烟零回归：/、/browse、/skill/:id、/api/search、/api/mcp/search、/api/meta、/sitemap.xml、/robots.txt、/docs/mcp、/download/:id.zip 全 200；GET /mcp 浏览器形态 200 HTML；SSE-only GET 与 DELETE 405。测毕已停服（3401 释放）。
- 产出：
  - /store/zkyang/skillhub-site/server/mcp.mjs —— MCP 协议核心（8 工具 schema + dispatch + stub/真 handler 动态接线 + 审计钩子）
  - /store/zkyang/skillhub-site/server/mcp-audit.mjs —— no-op 骨架（已移交 W6-4，其已填充实现）
  - /store/zkyang/skillhub-site/server.mjs —— 仅锚点 A7/A8/A9 三处插入
  - /store/zkyang/skillhub-site/scripts/mcp_smoke.mjs —— 六用例冒烟脚本（W6-6 直接复用：node scripts/mcp_smoke.mjs http://127.0.0.1:<port>/mcp）
- 关键决定 / 发现：
  - stub→真 handler 换装用「existsSync + 动态 import + 导出名探测（searchSkills/search_skills/…/handlers 表）」而非静态 import：W6-2/3 文件落盘后重启即自动接真，无需二次改 mcp.mjs（并行期协议的具体落法）；若导出名不在探测表内会 console.warn 并维持 stub，W6-2/3 若用了其他导出名请知会。
  - baseUrl 计算：请求 Host 与 SITE_URL 同域时用 SITE_URL（保 https），否则按 Host + x-forwarded-proto 拼——本机 34XX 自测与 E7 降级直连拿到的 /dl/ 直链均可直接 GET，生产反代下自动回 https://skillhub.tools。
  - GET /mcp 无 Accept 头（curl 裸访问）按浏览器形态给 HTML（分流条件是「含 event-stream 且不含 html」，严格按 E1 字面）。
  - 413 响应体也带 JSON-RPC -32600 错误对象（HTTP 层与协议层双保险），头加 connection: close。
  - 我起服期间 W6-4 更新了 mcp-audit.mjs：ESM 模块缓存使旧进程仍用 no-op，重启后自动用落盘版——W6-6 起服顺序天然规避，无需处理。
- 阻塞：无
- 交接给下一波：
  - W6-6：scripts/mcp_smoke.mjs 六用例可直接对任意端口复跑（参数传 URL）；负例矩阵中 413/405/-32700/-32601/-32602 本条目已有实测口径。
  - W6-5：initialize instructions 三层模型文案定稿在 server/mcp.mjs 的 INSTRUCTIONS 常量（英文），docs 页照抄对齐。
  - W6-2/3：动态接线已验证你们的导出名可被探测（自测时八工具全真 handler）；若后续改导出名，同步改 mcp.mjs 顶部 ROUTER_TOOLS/INSTALL_TOOLS 探测表。

### [W6] Agent-W6-4 — 安全与配额（审计落盘轮转 + /dl/ 限流接入 + 可选 API key）
- 时间：2026-08-17 21:25 (UTC+8)
- 状态：完成
- 做了什么：
  - 实现 server/mcp-audit.mjs（接管 W6-1 骨架）：JSONL 异步追加写 data/mcp_access.log（串行队列、写失败静默不崩服）；轮转超 auditMaxBytes（默认 10MB，config 可配，env AUDIT_MAX_BYTES 可覆盖供测试）时 rename → .1/.2 两代；args_summary 兼容字符串（mcp.mjs 已提炼口径，截 80）与对象（白名单 q/id/client/lang/cat/group/tier/sort/window/classification/featured/page/size/limit，逐字段截 80），不落全参与明文 IP。
  - 改 server/security.mjs（P0-2）：新增 /dl/ 分支——计入与 /download/ 同一 dlCount 下载限流通道，且同 /mcp 跳过 UA 黑名单（agent 下载器 UA 不可控，token 已是强门槛）；蜜罐/封禁照常生效。
  - 改 server/security.mjs（P0-4）：可选 API key——config security 节新增 mcpApiKeys:[]（默认空=匿名放行现状）；非空时仅 POST /mcp 要求 Authorization: Bearer <key> 或 x-api-key 命中其一，否则 401（401 判定在限流计数之前）；GET /mcp SSR 页与 /api/mcp/*（站内前端在用）不要求 key。
  - config.json security 节新增 mcpApiKeys: [] 与 auditMaxBytes: 10485760；限流阈值评估后维持 maxMcpRequestsPerWindow=300/分钟（一次 search→get→download→install_plan 闭环约 4~6 个 POST，支撑 ~60 闭环/分钟/IP），全部保持 config 可配。
  - 写 scripts/sec_probe.mjs（P0-3，6 用例）：每用例独立随机 X-Forwarded-For IP 互不污染、可反复重跑；阈值现读 config；unban 用例在 banMinutes>0.2 时自动 SKIP 并提示调法；api-key 用例按 config 现状自动选正/反向断言。
  - 写 scripts/audit_tail.mjs（P1-1）：读 mcp_access.log(+.1) 按 tool 聚合 count/err/err%/均值 ms，供主控看板。
  - PORT=3404 全量自测后停服：审计单测（JSONL 形状/白名单/截断/字符串摘要）✓、轮转实测（AUDIT_MAX_BYTES=2000 触发 .1/.2 两代、三代均可整行解析）✓、真实 POST /mcp tools/call 端到端逐条落日志（W6-1/2/3 已就位，成功 ok:true 与业务错误 ok:false 都记、ip_hash 16 位十六进制）✓、默认态 sec_probe 5 PASS/1 SKIP、临时调 banMinutes=0.05 + mcpApiKeys=["w6-4-test-key"] 后全 6 PASS（mcp 超频 429+retry-after=60、蜜罐封禁、3s 自动解封、/dl/ 第 12 次 429、无头/坏 key 401 + 对 key 200、googlebot 200 / python-requests 403）、config 还原并重启复核默认匿名 200 ✓；主站冒烟 /、/browse、/skill/:id、/api/search、/api/mcp/search、/download/（无 token 200，D5 不破坏）、/docs/mcp、robots、sitemap、GET /mcp SSR 全 200 零回归。
  - 测毕清理：测试产生的 mcp_access.log(.1/.2) 已删除（E2E 统计从零开始），config.json 从备份原样还原（banMinutes=30、mcpApiKeys=[]）。
- 产出：
  - /store/zkyang/skillhub-site/server/mcp-audit.mjs —— 审计落盘 + 轮转实现（W6-1 骨架之上填充，签名不变）
  - /store/zkyang/skillhub-site/server/security.mjs —— /dl/ 限流分支 + 可选 API key（W6-4 区段注释标注）
  - /store/zkyang/skillhub-site/config.json —— security 节新增 mcpApiKeys / auditMaxBytes 两键（其余未动）
  - /store/zkyang/skillhub-site/scripts/sec_probe.mjs —— 6 用例负例复现脚本（可复跑）
  - /store/zkyang/skillhub-site/scripts/audit_tail.mjs —— 审计聚合观测脚本
- 关键决定 / 发现：
  - 与 W6-1 发生过一次文件竞态：我先写了 mcp-audit.mjs 实现，W6-1 随后用 no-op 骨架覆盖（其骨架头注明所有权移交），我已在骨架口径（ts=Date.now() 毫秒、args_summary 字符串）之上重落实现并兼容对象形态；按所有权矩阵 W6-1 之后不再改此文件，现版本为终稿。
  - API key 的 401 判定在 mcp 限流计数之前：无 key 扫描不消耗配额也不触发封禁（当前默认关，无实际影响；启用后若要「坏 key 也计数封禁」再调整顺序）。
  - 429 的 retry-after: 60 头由 server.mjs A3 出口统一携带（W4-6 已有），401 无 WWW-Authenticate 头（A3 出口不支持额外头且 server.mjs 冻结），文档写明用 Bearer/x-api-key 即可。
  - /dl/ 无 token 的 403（W6-3 handleDl 拒绝）发生在 guard 计数之后，即被拒请求也计入 dlCount——与 W6-3 条目的观察一致，属有利现状（刷无 token /dl/ 也会触发下载限流封禁）。
- 阻塞：无
- 交接给下一波：
  - W6-5（文档口径）：API key 默认关；config security.mcpApiKeys 非空时 POST /mcp 需 `Authorization: Bearer <key>` 或 `x-api-key: <key>`，无效 401（文案 "Unauthorized: MCP API key required…"）；mcp 通道 300 req/min/IP、429 带 retry-after: 60；下载限流 10 次/分钟/IP 含 /dl/；审计日志在 data/mcp_access.log（JSONL，10MB 轮转两代）。
  - W6-6（复用脚本）：scripts/sec_probe.mjs 覆盖超频/蜜罐/解封/dl 限流/API key/UA 回归六负例（解封与 API key 启用态需临时改 config 重启，脚本会提示）；scripts/audit_tail.mjs 可做压测后审计核对；审计日志已清零，E2E 期间的记录即真实流量。

### [W6] Agent-W6-5 — 文档与接入（/docs/mcp 终稿 + 三客户端样例 + INSTALL.md MCP 节）
- 时间：2026-08-17 21:55 (UTC+8)
- 状态：完成
- 做了什么：
  - 重写 public/docs/mcp.html 第一节（en/zh 双份同步）：去掉 coming soon 徽章；「三层模型」導語与 W6-1 mcp.mjs INSTRUCTIONS 常量对齐（skills teach methods / MCP servers give capabilities，实录里引用其真实开头）；8 工具表（一句话 + 关键参数，含全部 enum 与默认值，与附录 A 及 tools/list 实测 inputSchema 逐字段核对零出入）；1.2 三客户端接入（Cursor mcp.json / Claude Code 一行命令 / Codex config.toml + examples 三文件下载链）；1.3 典型流程 curl 实录四步（initialize→tools/list→search_skills→download_skill+下载解压，全部从 PORT=3405 真实端点抓取后仅替换域名为 skillhub.tools，含真实 total=6、routing_note、sha256、size=2685）；1.4 行为/限流/token/API key 六条（无状态单响应+批量+202、协议版本协商集、256KB/413 与 M-C6 错误码、300 req/min + 429 retry-after、/dl/ 600s 过期重取、mcpApiKeys 非空时 Bearer/x-api-key 无效 401——口径照 W6-4 交接）。
  - 增补锚点目录：文首 TOC 行 + h2/h3 id（en 用 mcp/mcp-tools/…、zh 用 zh- 前缀防重复）；第二节既有 /api/* 文档与第三节条款只增 id 未删改；meta description 同步去掉 upcoming。
  - 新建 public/docs/examples/{cursor-mcp.json, claude-add.sh, codex-config.toml} 三个可直接下载的接入样例（静态通道自动服务，实测 200）。
  - INSTALL.md 增补「MCP endpoint（站点运维视角）」一节：method 分流与模块分工、config security 节依赖（token secret/TTL、maxMcpRequestsPerWindow、mcpApiKeys）、审计日志位置与轮转、样例文件与文档页路径。
  - 盲测预演（P1-2）：开工时 W6-1/2/3 未落盘、先按附录 A 写契约稿；写作期间三者落盘，随即起服 3405 用文档里的原样命令从零跑通接入与闭环（initialize/tools/list/search/download/get_install_plan/sha256 校验/unzip 含 SKILL.md），文档样例全部回填实测值——终校完成，非占位稿。
  - 行为声明逐条实测核对：SSE-only GET /mcp 405、DELETE 405、纯通知 202、坏 JSON -32700、未知方法 -32601、未知工具 -32602、批量进出、Mcp-Session-Id 忽略、get_skill lang=zh 未缓存回退 + lang_note、list_categories 6/20、search_mcp_servers 瘦身投影与 get_mcp_server 全字段含 install_config/install_claude——与文档所写一一吻合。
  - 主站冒烟零回归：/、/browse、/skill/:id、/api/search、/api/mcp/search、GET /mcp（SSR HTML）、/download/:id.zip（无 token 200，D5）、/sitemap.xml、/robots.txt 全 200；渲染后 HTML 标签平衡校验通过（en/zh 各 16 pre / 3 表格 / 3 样例链）。测毕停服 3405、临时文件已清理。
- 产出：
  - /store/zkyang/skillhub-site/public/docs/mcp.html —— MCP 节双语终稿（工具表/接入/curl 实录/行为说明/TOC）
  - /store/zkyang/skillhub-site/public/docs/examples/cursor-mcp.json —— Cursor 接入样例（新建）
  - /store/zkyang/skillhub-site/public/docs/examples/claude-add.sh —— Claude Code 接入样例（新建）
  - /store/zkyang/skillhub-site/public/docs/examples/codex-config.toml —— Codex 接入样例（新建）
  - /store/zkyang/skillhub-site/INSTALL.md —— 增补 MCP endpoint 运维节
- 关键决定 / 发现：
  - curl 实录采用「真实抓取 + 缩略标注」：长文本（instructions/note/inputSchema）以 … 省略并注明，数值与字段（total/routing_note/sha256/size/token 形态）保留实测原值；样例 token 已过期无泄露风险。
  - 文档 URL 一律写 https://skillhub.tools；W6-1 的 baseUrl 逻辑（Host 同域用 SITE_URL，否则按 Host 拼）保证公网切换后 download_url 自动变 https 域名，文档无需二次改。
  - Codex 样例采用 `[mcp_servers.skillhub]` + url 的 streamable-HTTP 形态并注明版本要求（老版 Codex 仅支持 stdio）；W6-6 实测若发现盘内 codex CLI 不支持 url 形态，按其报告微调该样例即可（单文件、不影响其他内容）。
  - get_install_plan 的 target_dir 实为 `~/.cursor/skills/<safeName>`（含技能子目录），文档措辞已按实测改为「安装落在 ~/.cursor/skills/ 下，一技能一目录」，避免与顶层目录混淆。
- 阻塞：无（生产公网链路仍归主控；文档已按 skillhub.tools 终态书写，切换无需改稿）。
- 交接给下一波：
  - W6-6：文档盲测（P0-6）请只按 /docs/mcp 页面操作；examples 三文件路径 /docs/examples/*；curl 实录四步把域名换成 http://127.0.0.1:3406 即可原样复演（步骤④的 token 需现签，页内已写明过期重取口径）。
  - 主控：上线后建议把 /docs/examples/ 三文件加进 sitemap 与 docs 页 og 校验一并复核（非必须，静态通道已可直接下载）。

### [W6] Agent-W6-2 — 检索与动态路由（mcp-router.mjs + mcp-routes.json + mcp-data 导出区段）
- 时间：2026-08-17 21:32 (UTC+8)
- 状态：完成
- 做了什么：
  - 新建 server/mcp-router.mjs：六个检索工具真 handler（search_skills / get_skill / list_categories / trending / search_mcp_servers / get_mcp_server），全部按 M-C1 签名 `async (args, ctx) => object`，业务错误抛 Error 带 userMessage；导出名 = 工具名（snake_case）+ 聚合注册表 routerHandlers（与 W6-1 mcp.mjs 的 ROUTER_TOOLS 探测表已实测匹配）。
  - 路由器（纯规则，零依赖）：CJK 检测（/[\u4e00-\u9fff]/）→ tier:chinese / group:cn 位置上浮 +30（不硬过滤）；curated_words（official/curated/精选/官方/认证）命中 → featured/official 上浮 +40 且从 q 中剔除该词；keyword_cats 词典命中（45 条正则→cat/group，首中即停）→ 未显式传 cat/group 时注入过滤，注入后 0 结果自动撤销过滤回退全量（routing_note 说明）；显式参数 > 词典推断；显式 sort=installs/stars/name/new 时跳过偏置重排（routing_note 说明）。全部决策写进 routing_note。
  - 新建 server/mcp-routes.json（M-C3 结构）：keyword_cats 45 条（19 类中英正则，英文用 (?<![a-z0-9])…(?![a-z0-9]) 词界防误伤；含 group 目标 secops 一条）+ boosts{curated_words×5, zh_boost:30, curated_boost:40}；60s mtime 轮询热更新（loadZh 同款），坏 JSON 保留旧表 console.warn 一次不崩服，坏正则单条跳过。
  - server/mcp-data.mjs 末尾追加 `// ==== W6-2 ====` 导出区段（既有代码 1~212 行零改动）：mcpQuery(q, classification, page, size) 与 mcpById(id)，复用既有 SERVERS 内存索引（M-C2 不二次读 mcp.json）。
  - get_skill：deps.byId + pub 投影（附录 A 字段全集，group 经 groupOf），skill_md 读 HUB 原文截 100KB；lang=zh 按 E6：md_zh/:id.md 命中给译文、未命中回退英文原文 + lang_note，全程零同步翻译等待（实测 zh-miss 路径 <1ms）。
  - list_categories：6 group（en/zh 名 + groupOf 唯一条目聚合计数）→ 20 cat（DB.categories 的 en/zh/count），进程内缓存（数据冻结）。trending：window all（score 降序，预排序缓存）/ new（added_at > 2026-08-17，当前 0 条且 note 说明，E4）；limit 1..48 默认 10。
  - 自测（/tmp 脚手架直调，searchSkills 逐字复制自 server.mjs 保证口径）：功能 47 断言全过（用例见下）；热更新实测改表 65s 生效、坏 JSON 65s 后旧表仍在服务且仅告警；性能 100 次采样 search_skills p50=51.8ms / p95=117ms、get_skill p50=0.6ms / p95=1.0ms、trending p95=4.1ms、search_mcp p95=11.4ms（E5 线 150ms 内）。
  - 集成复验（W6-1 接线后）：PORT=3402 起服跑 scripts/mcp_smoke.mjs 23 断言全过；POST /mcp tools/call 实测 search_skills（中文查询 routing_note 正确）、list_categories、trending new、search_mcp_servers→get_mcp_server 回环（install_config/install_claude 在场）、get_skill 业务错误 isError:true。主站冒烟零回归：/ /browse /browse?group=dev /skill/:id /api/search /api/suggest /api/meta /api/mcp/search /api/mcp/:id GET /mcp /mcp/:id /docs/mcp /download/:id.zip(可解) /robots /sitemap 全 200。测毕停服（3402 已释放）。
- 产出：
  - /store/zkyang/skillhub-site/server/mcp-router.mjs —— 六检索 handler + 规则路由器 + 热更新（新建）
  - /store/zkyang/skillhub-site/server/mcp-routes.json —— 路由表 45 条 + boosts（新建，运维可直接改，60s 生效）
  - /store/zkyang/skillhub-site/server/mcp-data.mjs —— 仅末尾 `// ==== W6-2 ====` 区段（mcpQuery/mcpById 两导出）
- 关键决定 / 发现：
  - 偏置实现为「位置上浮」（第 i 位得 i−boost 后稳定重排）：zh_boost/curated_boost 的语义即上浮位数，全部可在 mcp-routes.json 调；不改 searchSkills、不建第二索引（E5 红线）。
  - 词典注入过滤 + 0 结果自动回退：AND 语义下窄类过滤易清空结果（如 "solidity smart contract"→web3 过滤后 0 条），回退保证可用性，routing_note 记录 `filter dropped, unrouted results returned` 供 E2E 断言。
  - list_categories 返回在附录 A 形状上补了一个加性字段 name_zh（group 级中文名；cats 级 en/zh 本就在契约内）——LLM 中文路由需要，W6-5/W6-6 若做逐字段核对请知悉。
  - GROUPS 显示名表（C1 同款）在 mcp-router.mjs 只读镜像了一份（deps 只带 groupOf 函数）；若未来改组名需同步三处（server.mjs / app.js / mcp-router.mjs）。
  - 路由用例表（W6-6 逐条复测口径，routing_note 关键词→预期）：R1 "security audit"→cat:security(480 条,全 security)；R2 "react hooks"→cat:frontend(49)；R3 "kubernetes deployment"→cat:cloud-devops(1)；R4 "微信"→cjk+cat:chinese(64,全 chinese)；R5 "official git skills"→curated word hit(1150,top=official tier)；R6 "官方 git 技能"→curated+cjk 双偏置(102)；R7 "unit testing pytest"→cat:testing(1)；R8 "翻译润色"→仅 cjk、无词典命中(0 条属语料现状)；R9 "chart dashboard"→cat:data-viz(10)；R10 q="security audit"+cat=frontend→dictionary routing skipped(params win,0 条)；R11 "llm fine-tuning"→cat:ai-ml(6)；R12 q 空+featured→passed through(341)；R13 "nft"→cat:web3(41,全 web3)；R14 "observability monitoring"→group:secops(19,全 secops)；R15 "zettelkasten"→passed through(1)；R16 "solidity smart contract"→web3 注入后 0 → filter dropped 回退(16)。
- 阻塞：无
- 交接给下一波：
  - W6-6：上面 16 条用例直调或经 POST /mcp 复跑均可（routing_note 断言用关键词匹配，勿逐字符比对——正则原文会随路由表热更新变化）；p95 实测值 search 117ms / get 1.0ms 供延迟基线对照。
  - W6-5：search_skills 的 routing_note、注入回退行为、mcp-routes.json 可配置项（zh_boost/curated_boost/keyword_cats）如需写进文档，以本条目口径为准。
  - 主控/Wave7：keyword_cats 45 条是起步词表，误路由案例出现后直接改 mcp-routes.json（60s 生效免重启）；trending window=new 会在下一轮数据构建（added_at 有区分度）后自动有货。

<!-- 新条目追加在此行上方之后（保持正序，本注释始终在文件末尾） -->


