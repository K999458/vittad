'use strict';
/**
 * 指令台排队区（outbox）的行为验收。
 *
 * 用 jsdom 跑真实的 index.html + client.js，断言三条规则：
 *   1. 上一轮还在跑的时候发消息 → 不进主对话流，暂存在指令台上方
 *   2. 排队中的消息可以就地编辑
 *   3. 上一条收到最终回复后，队首消息自动弹上主对话流并真的发给服务端
 *
 *   node test/outbox-queue.js
 */

const path = require('path');
const fs = require('fs');
const assert = require('assert');

let JSDOM;
try {
    ({ JSDOM } = require('jsdom'));
} catch {
    console.log('⚠ 未安装 jsdom，跳过 outbox 前端验收（npm i -D jsdom 后可运行）');
    process.exit(0);
}

const BROWSER_DIR = path.join(__dirname, '..', 'resources', 'browser');
const sent = [];

function makeFakeWebSocket(win) {
    class FakeWebSocket {
        constructor() {
            this.readyState = 1;
            FakeWebSocket.last = this;
            this._listeners = {};
            setTimeout(() => this._emit('open', {}), 0);
        }
        addEventListener(type, fn) { (this._listeners[type] = this._listeners[type] || []).push(fn); }
        removeEventListener() {}
        send(raw) { sent.push(JSON.parse(raw)); }
        close() { this.readyState = 3; }
        _emit(type, ev) { for (const fn of this._listeners[type] || []) fn(ev); }
        /** 模拟服务端推消息 */
        push(obj) { this._emit('message', { data: JSON.stringify(obj) }); }
    }
    FakeWebSocket.OPEN = 1;
    win.WebSocket = FakeWebSocket;
    return FakeWebSocket;
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function main() {
    const html = fs.readFileSync(path.join(BROWSER_DIR, 'index.html'), 'utf8');
    const dom = new JSDOM(html, {
        runScripts: 'outside-only',
        url: 'http://localhost:9990/',
        pretendToBeVisual: true
    });
    const win = dom.window;
    const FakeWebSocket = makeFakeWebSocket(win);
    win.confirm = () => true;
    win.fetch = async () => ({ ok: true, json: async () => ({ ok: true, paths: [] }) });
    win.localStorage.clear();

    for (const file of ['cursor-tool-verbs.js', 'cursor-conversation.js', 'client.js']) {
        win.eval(fs.readFileSync(path.join(BROWSER_DIR, file), 'utf8'));
    }
    await sleep(30);

    const $ = (id) => win.document.getElementById(id);
    const ws = FakeWebSocket.last;
    assert.ok(ws, 'client.js 应已建立 WebSocket 连接');

    const outboxRow = $('outbox-row');
    const outboxList = $('outbox-list');
    const messages = $('messages');
    const input = $('input');
    const btnSend = $('btn-send');
    assert.ok(outboxRow && outboxList, 'index.html 应包含排队区');
    assert.ok(outboxRow.classList.contains('hidden'), '初始状态排队区应隐藏');

    // 第一条：通道空闲 → 直接上主对话流
    input.value = '第一条任务';
    btnSend.click();
    await sleep(30);
    assert.ok(
        sent.some(m => m.type === 'submit' && m.text === '第一条任务'),
        '空闲时的消息应立即发给服务端'
    );
    assert.ok(
        messages.textContent.includes('第一条任务'),
        '空闲时的消息应直接出现在主对话流'
    );

    // 第二、三条：上一轮还没回复 → 只进排队区，不进主对话流，也不发给服务端
    input.value = '第二条任务';
    btnSend.click();
    await sleep(10);
    input.value = '第三条任务';
    btnSend.click();
    await sleep(10);

    assert.ok(!outboxRow.classList.contains('hidden'), '有排队消息时排队区应显示');
    assert.strictEqual($('outbox-count').textContent, '2', '排队计数应为 2');
    assert.strictEqual(outboxList.querySelectorAll('.outbox-item').length, 2, '排队区应有两条');
    assert.ok(outboxList.textContent.includes('第二条任务'), '排队区应显示第二条');
    assert.ok(!messages.textContent.includes('第二条任务'), '排队消息不应进入主对话流');
    assert.ok(
        !sent.some(m => m.type === 'submit' && m.text === '第二条任务'),
        '排队消息不应提前发给服务端'
    );
    assert.strictEqual(input.value, '', '入队后输入框应清空');

    // 排队中的消息可编辑
    const secondRow = outboxList.querySelector('.outbox-item');
    const itemId = secondRow.dataset.id;
    secondRow.querySelector('button[data-act="edit"]').click();
    await sleep(10);
    const editing = outboxList.querySelector('.outbox-item.editing');
    assert.ok(editing, '点编辑后该条应进入编辑态');
    const textarea = editing.querySelector('.outbox-item-edit');
    assert.strictEqual(textarea.value, '第二条任务', '编辑框应带出原文');
    textarea.value = '第二条任务（改过了）';
    editing.querySelector('button[data-act="save"]').click();
    await sleep(10);
    assert.ok(
        outboxList.textContent.includes('第二条任务（改过了）'),
        '保存后排队区应显示改后的文本'
    );
    assert.strictEqual(
        outboxList.querySelector('.outbox-item').dataset.id,
        itemId,
        '编辑不应改变队列顺序'
    );

    // 排序：把第三条提到队首再放回去
    const rowsBefore = [...outboxList.querySelectorAll('.outbox-item')].map(r => r.dataset.id);
    outboxList.querySelectorAll('.outbox-item')[1].querySelector('button[data-act="up"]').click();
    await sleep(10);
    const rowsAfter = [...outboxList.querySelectorAll('.outbox-item')].map(r => r.dataset.id);
    assert.deepStrictEqual(rowsAfter, [rowsBefore[1], rowsBefore[0]], '上移应交换两条顺序');
    outboxList.querySelectorAll('.outbox-item')[1].querySelector('button[data-act="up"]').click();
    await sleep(10);

    // 上一条执行完成 → 队首自动弹上去
    ws.push({ type: 'reply', channelId: '1', reply: '第一条做完了', timestamp: new Date().toISOString() });
    await sleep(1200);

    assert.ok(
        sent.some(m => m.type === 'submit' && m.text === '第二条任务（改过了）'),
        '上一条完成后队首应自动发给服务端'
    );
    assert.ok(
        messages.textContent.includes('第二条任务（改过了）'),
        '自动发出的消息应上主对话流'
    );
    assert.strictEqual($('outbox-count').textContent, '1', '弹出一条后排队应只剩 1 条');
    assert.ok(
        !sent.some(m => m.type === 'submit' && m.text === '第三条任务'),
        '新一轮开始后，剩下的排队消息应继续等待'
    );

    // 手动「立即发送」不等上一条完成
    $('btn-outbox-send-now').click();
    await sleep(50);
    assert.ok(
        sent.some(m => m.type === 'submit' && m.text === '第三条任务'),
        '「立即发送」应绕过等待直接发出队首'
    );
    assert.ok(outboxRow.classList.contains('hidden'), '队列清空后排队区应隐藏');

    // 刷新恢复：排队内容写进了 localStorage
    input.value = '刷新前排队的任务';
    btnSend.click();
    await sleep(20);
    const saved = JSON.parse(win.localStorage.getItem('qingtian-mcp-browser-outbox-v1') || '{}');
    assert.ok(
        Array.isArray(saved['1']) && saved['1'].some(it => it.text === '刷新前排队的任务'),
        '排队消息应持久化到 localStorage'
    );

    dom.window.close();
    console.log('✓ outbox：忙碌入队 / 不进主对话流 / 可编辑 / 可排序 / 完成后自动弹出 / 立即发送 / 持久化');
}

main().then(
    () => process.exit(0),
    (err) => {
        console.error('✗ outbox 验收失败：', err && err.message ? err.message : err);
        if (err && err.stack) console.error(err.stack);
        process.exit(1);
    }
);
