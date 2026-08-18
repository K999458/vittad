'use strict';
/**
 * Cursor 会话组件的行为验收。
 *
 * 不起浏览器：用一个够用的 DOM 桩跑渲染层，断言那几条从 Cursor 前端反推出来的
 * 规则没被改坏（briefly 阈值、Thinking→Thought 切换、标题提升、工具文案三态）。
 *
 *   node test/cursor-conversation.js
 */

const path = require('path');
const vm = require('vm');
const fs = require('fs');
const assert = require('assert');

const BROWSER_DIR = path.join(__dirname, '..', 'resources', 'browser');

// ── 最小 DOM 桩 ────────────────────────────────────────────
function makeElement(tag) {
    return {
        tagName: String(tag).toUpperCase(),
        className: '',
        childNodes: [],
        dataset: {},
        style: {},
        type: '',
        _text: '',
        _listeners: {},
        get textContent() {
            if (this._text) return this._text;
            return this.childNodes.map((c) => c.textContent).join('');
        },
        set textContent(v) {
            this._text = String(v);
            this.childNodes = [];
        },
        get classList() {
            const self = this;
            return {
                add(c) {
                    if (!self.className.split(' ').includes(c)) self.className = (self.className + ' ' + c).trim();
                },
                remove(c) {
                    self.className = self.className.split(' ').filter((x) => x && x !== c).join(' ');
                },
                toggle(c, on) { on ? this.add(c) : this.remove(c); },
                contains(c) { return self.className.split(' ').includes(c); }
            };
        },
        appendChild(child) { this.childNodes.push(child); return child; },
        setAttribute() {},
        addEventListener(type, fn) { (this._listeners[type] = this._listeners[type] || []).push(fn); },
        click() { for (const fn of this._listeners.click || []) fn(); },
        querySelector(sel) {
            const want = sel.replace(/^\./, '');
            const walk = (node) => {
                for (const c of node.childNodes) {
                    if (c.className && c.className.split(' ').includes(want)) return c;
                    const found = walk(c);
                    if (found) return found;
                }
                return null;
            };
            return walk(this);
        }
    };
}

const sandbox = { document: { createElement: makeElement }, console };
sandbox.window = sandbox;
vm.createContext(sandbox);

for (const f of ['cursor-tool-verbs.js', 'cursor-conversation.js']) {
    const src = fs.readFileSync(path.join(BROWSER_DIR, f), 'utf8');
    vm.runInContext(src, sandbox, { filename: f });
}

const { CursorToolVerbs, CursorConversation } = sandbox;
const noopMarkdown = (target, text) => { target.textContent = text; };

let passed = 0;
function check(name, fn) {
    fn();
    passed++;
    console.log('  ok  ' + name);
}

// ── 工具文案 ───────────────────────────────────────────────
console.log('工具文案');
check('三态措辞与 Cursor 一致', () => {
    assert.strictEqual(CursorToolVerbs.toolVerb('shellToolCall', 'loading'), 'Running');
    assert.strictEqual(CursorToolVerbs.toolVerb('shellToolCall', 'completed'), 'Ran');
    assert.strictEqual(CursorToolVerbs.toolVerb('grepToolCall', 'completed'), 'Grepped');
    assert.strictEqual(CursorToolVerbs.toolVerb('editToolCall', 'loading'), 'Editing');
});
check('未收录的工具回落而不是崩', () => {
    assert.strictEqual(CursorToolVerbs.toolVerb('brandNewToolCall', 'loading'), 'Working');
});
check('提取到的条目数没有缩水', () => {
    assert.ok(Object.keys(CursorToolVerbs.TOOL_VERBS).length >= 67);
});

// ── 耗时文案 ───────────────────────────────────────────────
console.log('耗时文案');
check('500ms 以内说 briefly 而不是报数字', () => {
    assert.strictEqual(CursorConversation.formatDuration(200), 'briefly');
    assert.strictEqual(CursorConversation.formatDuration(499), 'briefly');
});
check('500ms 以上按秒取整', () => {
    assert.strictEqual(CursorConversation.formatDuration(2448), '2s');
    assert.strictEqual(CursorConversation.formatDuration(11274), '11s');
});
check('无耗时不显示', () => {
    assert.strictEqual(CursorConversation.formatDuration(null), '');
    assert.strictEqual(CursorConversation.formatDuration(0), '');
});

// ── 标题提升 ───────────────────────────────────────────────
console.log('标题提升');
check('首行 markdown 标题被识别为摘要', () => {
    const r = CursorConversation.parseLeadingHeader('## 先看 HAR\n然后再说别的');
    assert.strictEqual(r.title, '先看 HAR');
    assert.strictEqual(r.body, '然后再说别的');
});
check('首行是普通文字则不提升', () => {
    assert.strictEqual(CursorConversation.parseLeadingHeader('用户要求进入持续对话模式。'), null);
});
check('前导空行不影响识别', () => {
    const r = CursorConversation.parseLeadingHeader('\n\n# 标题\n正文');
    assert.strictEqual(r.title, '标题');
});

// ── Thought 生命周期 ───────────────────────────────────────
console.log('Thought 生命周期');
check('流式期间是 Thinking 且带流光', () => {
    const t = CursorConversation.createThought(noopMarkdown);
    t.append('正在分析请求');
    const action = t.root.querySelector('cx-collapsible-action');
    assert.strictEqual(action.textContent, 'Thinking');
    assert.ok(action.classList.contains('cx-shine'), '流式期间必须挂 cx-shine');
});
check('收尾后变 Thought、去流光、带耗时', () => {
    const t = CursorConversation.createThought(noopMarkdown);
    t.append('正在分析请求');
    t.complete(2448);
    const action = t.root.querySelector('cx-collapsible-action');
    const details = t.root.querySelector('cx-collapsible-details');
    assert.strictEqual(action.textContent, 'Thought');
    assert.ok(!action.classList.contains('cx-shine'), '收尾后不该再有流光');
    assert.strictEqual(details.textContent, '2s');
});
check('空思考不可展开', () => {
    const t = CursorConversation.createThought(noopMarkdown);
    t.complete(100);
    assert.strictEqual(t.root.querySelector('cx-collapsible-header').dataset.expandable, 'false');
});
check('点击标题切换展开态', () => {
    const t = CursorConversation.createThought(noopMarkdown);
    t.append('有内容');
    t.complete(1000);
    assert.strictEqual(t.root.dataset.open, 'false');
    t.root.querySelector('cx-collapsible-header').click();
    assert.strictEqual(t.root.dataset.open, 'true');
});

// ── 工具行 ─────────────────────────────────────────────────
console.log('工具行');
check('进行中显示 loading 文案与状态点', () => {
    const tool = CursorConversation.createToolCall({ kind: 'readToolCall', summary: '/tmp/a.txt' });
    assert.strictEqual(tool.root.querySelector('cx-collapsible-action').textContent, 'Reading');
    assert.strictEqual(tool.root.querySelector('cx-tool-status').dataset.state, 'loading');
});
check('完成后切到 completed 文案', () => {
    const tool = CursorConversation.createToolCall({ kind: 'readToolCall', summary: '/tmp/a.txt' });
    tool.complete();
    assert.strictEqual(tool.root.querySelector('cx-collapsible-action').textContent, 'Read');
    assert.strictEqual(tool.root.querySelector('cx-tool-status').dataset.state, 'completed');
});
check('带 diff 的编辑可展开且逐行标注增删', () => {
    const tool = CursorConversation.createToolCall({ kind: 'editToolCall', summary: '/tmp/b.txt' });
    tool.complete({
        linesAdded: 1,
        linesRemoved: 1,
        diffString: '--- a/b.txt\n+++ b/b.txt\n@@ -1,3 +1,3 @@\n line-one\n-line-two\n+line-two-CHANGED\n line-three'
    });
    assert.strictEqual(tool.root.querySelector('cx-collapsible-action').textContent, 'Edited');
    assert.strictEqual(tool.root.querySelector('cx-collapsible-header').dataset.expandable, 'true');
    const pre = tool.root.querySelector('cx-diff');
    const kinds = pre.childNodes.map((n) => n.dataset.kind);
    assert.ok(kinds.includes('add'), 'diff 里应有新增行');
    assert.ok(kinds.includes('del'), 'diff 里应有删除行');
    assert.ok(kinds.includes('hunk'), 'diff 里应有 hunk 头');
});
check('失败态用 error 文案', () => {
    const tool = CursorConversation.createToolCall({ kind: 'shellToolCall', summary: 'ls' });
    tool.fail();
    assert.strictEqual(tool.root.querySelector('cx-collapsible-action').textContent, 'Run');
    assert.strictEqual(tool.root.querySelector('cx-tool-status').dataset.state, 'error');
});

console.log('\n全部通过：' + passed + ' 项');
