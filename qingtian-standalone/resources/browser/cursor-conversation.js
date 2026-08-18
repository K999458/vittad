(function (global) {
    'use strict';

    /**
     * Cursor 会话流渲染层。
     *
     * 输入是现有 WS 协议里的 agent_stream 事件（kind: thinking / tool_call /
     * assistant_narration / reply_done），输出是与 cursor.com agent 视图同构的 DOM。
     * 不碰传输层：环形缓冲、断线续传、通道路由全部沿用 client.js 里已有的实现。
     *
     * 与 Cursor 对齐的几个行为，都是从其前端 bundle 反出来的，改动前先确认：
     *   - 折叠标题流式期间是 Thinking，收尾变 Thought
     *   - 耗时不足 500ms 显示 briefly，不显示 0.3s
     *   - 思考正文首行若是 markdown 标题，用该标题替代 Thought 作为摘要
     */

    const verbs = global.CursorToolVerbs;

    /** 工具名 → 图标字形。用的是从 cursor-icons 字体里提取的 382 个图标。 */
    const TOOL_ICONS = {
        shellToolCall: 'terminal',
        readToolCall: 'file',
        editToolCall: 'edit',
        deleteToolCall: 'trash',
        grepToolCall: 'search',
        globToolCall: 'search',
        lsToolCall: 'folder',
        semSearchToolCall: 'search',
        webSearchToolCall: 'globe',
        webFetchToolCall: 'globe',
        fetchToolCall: 'globe',
        mcpToolCall: 'plug',
        getMcpToolsToolCall: 'plug',
        createPlanToolCall: 'checklist',
        updateTodosToolCall: 'checklist',
        readTodosToolCall: 'checklist',
        taskToolCall: 'rocket',
        generateImageToolCall: 'file-media'
    };

    const DEFAULT_TOOL_ICON = 'tools';

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text != null) node.textContent = text;
        return node;
    }

    function icon(name) {
        const i = el('i', 'qt-icon qt-icon-' + name);
        i.setAttribute('aria-hidden', 'true');
        return i;
    }

    /**
     * 耗时文案。Cursor 对 500ms 以内的思考不报数字，直接说 briefly，
     * 因为 "Thought for 0.2s" 读起来比没有还糟。
     */
    function formatDuration(durationMs) {
        if (durationMs == null || durationMs <= 0) return '';
        if (durationMs < 500) return 'briefly';
        const seconds = Math.round(durationMs / 1000);
        if (seconds > 0) return seconds + 's';
        return (durationMs / 1000).toFixed(1) + 's';
    }

    /** 思考正文首行若是 markdown 标题，拿它当摘要，比千篇一律的 Thought 有信息量 */
    function parseLeadingHeader(text) {
        const lines = String(text || '').split('\n');
        let i = 0;
        while (i < lines.length && !lines[i].trim()) i++;
        if (i >= lines.length) return null;
        const m = lines[i].trim().match(/^#{1,6}\s+(.+)$/);
        if (!m) return null;
        return { title: m[1].trim(), body: lines.slice(i + 1).join('\n').trimStart() };
    }

    /**
     * 折叠块。Thought 和工具行共用，差别只在传入的 action/details/body。
     * 高度过渡走 CSS grid 的 0fr→1fr，所以不需要在这里量元素高度。
     */
    function createCollapsible(options) {
        const opts = options || {};
        const root = el('div', 'cx-collapsible');
        root.dataset.open = 'false';

        const header = el('button', 'cx-collapsible-header');
        header.type = 'button';

        const chevron = el('span', 'cx-collapsible-chevron cx-collapsible-chevron-slot');
        chevron.appendChild(icon('chevron-right'));

        const action = el('span', 'cx-collapsible-action');
        const details = el('span', 'cx-collapsible-details');

        header.appendChild(chevron);
        if (opts.leading) header.appendChild(opts.leading);
        header.appendChild(action);
        header.appendChild(details);

        const content = el('div', 'cx-collapsible-content');
        const body = el('div', 'cx-collapsible-body');
        content.appendChild(body);

        root.appendChild(header);
        root.appendChild(content);

        let expandable = false;

        function setExpandable(value) {
            expandable = !!value;
            header.dataset.expandable = String(expandable);
            chevron.style.visibility = expandable ? '' : 'hidden';
            if (!expandable) root.dataset.open = 'false';
        }

        header.addEventListener('click', () => {
            if (!expandable) return;
            root.dataset.open = root.dataset.open === 'true' ? 'false' : 'true';
        });

        setExpandable(false);

        return {
            root,
            body,
            setAction(text, { shimmer = false } = {}) {
                action.textContent = text;
                action.classList.toggle('cx-shine', !!shimmer);
            },
            setDetails(text, { tabular = false } = {}) {
                details.textContent = text || '';
                details.dataset.tabular = String(!!tabular);
            },
            setExpandable,
            setOpen(open) {
                if (!expandable) return;
                root.dataset.open = open ? 'true' : 'false';
            }
        };
    }

    /**
     * Thought 块。流式期间标题是 Thinking 且带流光，收尾后换成 Thought 并附耗时。
     */
    function createThought(renderMarkdown) {
        const collapsible = createCollapsible();
        let raw = '';
        let done = false;
        let durationMs = null;

        function refresh() {
            const header = parseLeadingHeader(raw);
            const hasBody = raw.trim().length > 0;

            if (!done) {
                collapsible.setAction('Thinking', { shimmer: true });
                collapsible.setDetails('');
                collapsible.setExpandable(hasBody);
                renderMarkdown(collapsible.body, raw, { streaming: true });
                return;
            }

            const title = header && !header.body.trim() ? header.title : 'Thought';
            collapsible.setAction(title);
            collapsible.setDetails(formatDuration(durationMs), { tabular: true });
            collapsible.setExpandable(hasBody);
            renderMarkdown(collapsible.body, header ? header.body || raw : raw, { streaming: false });
        }

        refresh();

        return {
            root: collapsible.root,
            append(delta) {
                raw += String(delta || '');
                refresh();
            },
            complete(ms) {
                done = true;
                durationMs = ms == null ? null : Number(ms);
                collapsible.setOpen(false);
                refresh();
            }
        };
    }

    /** 把 unified diff 渲染成带底色的行，标记行单独着色 */
    function renderDiff(target, diffString) {
        const pre = el('pre', 'cx-code cx-diff');
        for (const line of String(diffString || '').split('\n')) {
            let kind = 'ctx';
            if (line.startsWith('+++') || line.startsWith('---')) kind = 'meta';
            else if (line.startsWith('@@')) kind = 'hunk';
            else if (line.startsWith('+')) kind = 'add';
            else if (line.startsWith('-')) kind = 'del';
            const row = el('span', 'cx-diff-line', line || ' ');
            row.dataset.kind = kind;
            pre.appendChild(row);
        }
        target.appendChild(pre);
    }

    function renderDiffStat(target, added, removed) {
        if (!added && !removed) return;
        const stat = el('span', 'cx-diff-stat');
        if (added) stat.appendChild(el('span', 'cx-add', '+' + added));
        if (removed) stat.appendChild(el('span', 'cx-del', '-' + removed));
        target.appendChild(stat);
    }

    /**
     * 工具调用行。
     *
     * @param {object} spec
     * @param {string} spec.kind      形如 shellToolCall；决定文案与图标
     * @param {string} spec.summary   副文本（命令 / 路径 / 检索词）
     * @param {object} [spec.detail]  展开内容
     */
    function createToolCall(spec) {
        const kind = spec.kind || 'mcpToolCall';
        const status = el('span', 'cx-tool-status');
        status.dataset.state = 'loading';

        const leading = el('span', 'cx-collapsible-leading');
        leading.appendChild(icon(TOOL_ICONS[kind] || DEFAULT_TOOL_ICON));

        const collapsible = createCollapsible({ leading });
        collapsible.root.classList.add('cx-tool');
        collapsible.root.querySelector('.cx-collapsible-header').appendChild(status);

        function apply(state, detail) {
            collapsible.setAction(verbs.toolVerb(kind, state), { shimmer: state === 'loading' });
            collapsible.setDetails(spec.summary || '');
            status.dataset.state = state;

            collapsible.body.textContent = '';
            const d = detail || spec.detail;
            if (!d) {
                collapsible.setExpandable(false);
                return;
            }
            if (d.diffString) {
                renderDiffStat(collapsible.body, d.linesAdded, d.linesRemoved);
                renderDiff(collapsible.body, d.diffString);
            } else if (d.output) {
                const pre = el('pre', 'cx-code cx-terminal', d.output);
                collapsible.body.appendChild(pre);
            } else if (d.text) {
                collapsible.body.appendChild(el('pre', 'cx-code', d.text));
            }
            collapsible.setExpandable(collapsible.body.childNodes.length > 0);
        }

        apply('loading');

        return {
            root: collapsible.root,
            complete(detail) { apply('completed', detail); },
            fail(detail) { apply('error', detail); }
        };
    }

    global.CursorConversation = {
        createCollapsible,
        createThought,
        createToolCall,
        formatDuration,
        parseLeadingHeader,
        renderDiff,
        TOOL_ICONS
    };
})(typeof window !== 'undefined' ? window : globalThis);
