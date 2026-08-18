"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.setLogHandler = setLogHandler;
exports.startWebServer = startWebServer;
exports.stopWebServer = stopWebServer;
exports.getWebServerInfo = getWebServerInfo;
exports.broadcastReply = broadcastReply;
exports.broadcastAgentTeamReplyStreamChunk = broadcastAgentTeamReplyStreamChunk;
exports.broadcastStatus = broadcastStatus;
exports.broadcastAgentStream = broadcastAgentStream;
exports.setAgentStreamRingSize = setAgentStreamRingSize;
exports.getAgentStreamSince = getAgentStreamSince;
exports.getAgentStreamSeqSnapshot = getAgentStreamSeqSnapshot;
/**
 * 浏览器端 Web Server
 *
 * 提供一个本地 HTTP + WebSocket 服务，让用户可以在浏览器访问 http://localhost:<port>
 * 使用和侧栏 webview 一致的核心 MCP 操作（发消息、看通道状态、收 AI 回复）。
 *
 * MVP 范围（核心 MCP 操作）：
 * - 选通道、发送文本消息
 * - 查看通道在线状态
 * - 接收 AI 回复广播
 *
 * 不做：附件上传、剪贴板、账号管理、无感切号、激活、桥接、保活、协同等
 *
 * 安全：
 * - 默认绑定 127.0.0.1，仅本机访问
 * - 无 token；如需局域网/鉴权后续迭代
 *
 * 数据同步：
 * - 所有消息走文件队列，和侧栏 webview 共享
 * - AI reply 由主进程的 replyPollTimer 统一读取，通过 broadcastReply() 推给浏览器
 */
const http = __importStar(require("http"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const os = __importStar(require("os"));
const ws_1 = require("ws");
const vscode = __importStar(require("../shim/vscode"));
const mcpServer_1 = require("./mcpServer");
const quickCommands_1 = require("./quickCommands");
const activation_1 = require("../shim/activation");
const agentSkills_1 = require("./agentSkills");
const seamlessSwitch_1 = require("../shim/seamlessSwitch");
// ── Module state ───────────────────────────────────────────
let httpServer = null;
let wss = null;
let clients = new Set();
let clientIdSeq = 0;
let currentInfo = null;
let logHandler = null;
let _quickCommandsUnsubscribe = null;
// 附件上传限制
const MAX_UPLOAD_FILES = 10;
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024; // 单次请求总大小上限 25 MB
function log(msg) {
    const line = `[web-server] ${msg}`;
    try {
        console.log(line);
    }
    catch { }
    if (logHandler) {
        try {
            logHandler(line);
        }
        catch { }
    }
}
function setLogHandler(fn) {
    logHandler = fn;
}
// ── Static file helpers ────────────────────────────────────
const MIME_MAP = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.ico': 'image/x-icon',
    '.woff2': 'font/woff2',
    '.woff': 'font/woff'
};
function serveStatic(req, res, browserDir) {
    const urlPath = (req.url || '/').split('?')[0];
    const rel = urlPath === '/' ? 'index.html' : urlPath.replace(/^\//, '');
    const absPath = path.normalize(path.join(browserDir, rel));
    // 防止路径越权
    if (!absPath.startsWith(browserDir)) {
        res.statusCode = 403;
        res.end('Forbidden');
        return;
    }
    fs.readFile(absPath, (err, data) => {
        if (err) {
            res.statusCode = 404;
            res.setHeader('Content-Type', 'text/plain; charset=utf-8');
            res.end('Not Found: ' + rel);
            return;
        }
        const ext = path.extname(absPath).toLowerCase();
        res.setHeader('Content-Type', MIME_MAP[ext] || 'application/octet-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.end(data);
    });
}
// ── Port binding with auto-increment ───────────────────────
function listenWithRetry(server, host, startPort, maxRetry = 10) {
    return new Promise((resolve, reject) => {
        let attempted = 0;
        const tryPort = (port) => {
            const onError = (e) => {
                server.off('listening', onListening);
                if (e.code === 'EADDRINUSE' && attempted < maxRetry) {
                    attempted++;
                    tryPort(port + 1);
                }
                else {
                    reject(e);
                }
            };
            const onListening = () => {
                server.off('error', onError);
                resolve(port);
            };
            server.once('error', onError);
            server.once('listening', onListening);
            server.listen(port, host);
        };
        tryPort(startPort);
    });
}
// ── Server lifecycle ───────────────────────────────────────
async function startWebServer(opts) {
    if (currentInfo?.running) {
        log(`已在运行中，端口 ${currentInfo.port}`);
        return currentInfo;
    }
    const host = opts.host || '127.0.0.1';
    const startPort = opts.port || 3180;
    const browserDir = path.join(opts.extensionPath, 'resources', 'browser');
    if (!fs.existsSync(browserDir)) {
        throw new Error(`Browser static directory not found: ${browserDir}`);
    }
    httpServer = http.createServer((req, res) => {
        const urlPath = (req.url || '/').split('?')[0];
        // API 路由优先
        if (urlPath === '/api/upload' && req.method === 'POST') {
            handleUploadRequest(req, res);
            return;
        }
        if (urlPath === '/api/file-preview' && (req.method === 'GET' || req.method === 'HEAD')) {
            handleFilePreviewRequest(req, res);
            return;
        }
        serveStatic(req, res, browserDir);
    });
    const port = await listenWithRetry(httpServer, host, startPort, 10);
    wss = new ws_1.WebSocketServer({ server: httpServer });
    wss.on('connection', (ws) => {
        const client = {
            ws,
            id: ++clientIdSeq,
            subscribedAt: Date.now()
        };
        clients.add(client);
        log(`WS 客户端已连接 #${client.id}（当前 ${clients.size} 个）`);
        // 连接时推送一次初始状态
        sendTo(client, {
            type: 'hello',
            clientId: client.id,
            streamSeqByChannel: getAgentStreamSeqSnapshot()
        });
        sendTo(client, { type: 'status', data: buildStatusSnapshot() });
        sendTo(client, { type: 'teamSnapshot', data: (0, mcpServer_1.getAgentTeamWorkbenchSnapshot)() });
        ws.on('message', (raw) => {
            let data;
            try {
                data = JSON.parse(raw.toString());
            }
            catch {
                sendTo(client, { type: 'error', message: 'Invalid JSON' });
                return;
            }
            handleWsMessage(client, data).catch((e) => {
                log(`WS 处理异常: ${e.message}`);
                sendTo(client, { type: 'error', message: e.message });
            });
        });
        ws.on('close', () => {
            clients.delete(client);
            log(`WS 客户端已断开 #${client.id}（剩余 ${clients.size} 个）`);
        });
        ws.on('error', (err) => {
            log(`WS 错误 #${client.id}: ${err.message}`);
        });
    });
    currentInfo = {
        running: true,
        host,
        port,
        url: `http://${host}:${port}`,
        clientCount: 0,
        startedAt: Date.now()
    };
    // 订阅快捷指令变更（插件侧改动时自动推送给浏览器）
    if (_quickCommandsUnsubscribe) {
        _quickCommandsUnsubscribe();
    }
    _quickCommandsUnsubscribe = (0, quickCommands_1.subscribeQuickCommands)((list) => {
        broadcastQuickCommandsChange(list);
    });
    log(`服务已启动: ${currentInfo.url}`);
    return currentInfo;
}
async function stopWebServer() {
    if (!currentInfo?.running)
        return;
    log('正在停止服务...');
    // 取消订阅
    if (_quickCommandsUnsubscribe) {
        try {
            _quickCommandsUnsubscribe();
        }
        catch { }
        _quickCommandsUnsubscribe = null;
    }
    // 关闭所有 WS 连接
    for (const client of clients) {
        try {
            client.ws.close();
        }
        catch { }
    }
    clients.clear();
    // 关闭 WS server
    if (wss) {
        await new Promise((resolve) => {
            wss.close(() => resolve());
        });
        wss = null;
    }
    // 关闭 HTTP server
    if (httpServer) {
        await new Promise((resolve) => {
            httpServer.close(() => resolve());
        });
        httpServer = null;
    }
    currentInfo = null;
    log('服务已停止');
}
function getWebServerInfo() {
    if (!currentInfo)
        return null;
    return { ...currentInfo, clientCount: clients.size };
}
// ── WS message protocol ────────────────────────────────────
async function handleWsMessage(client, msg) {
    const type = String(msg.type || '');
    switch (type) {
        case 'ping': {
            sendTo(client, { type: 'pong', ts: Date.now() });
            break;
        }
        case 'agent_stream_resume': {
            const channelId = String(msg.channelId || '1');
            const sinceSeq = Number(msg.sinceSeq) || 0;
            const events = getAgentStreamSince(channelId, sinceSeq);
            sendTo(client, {
                type: 'agent_stream_resume_result',
                channelId,
                sinceSeq,
                events
            });
            break;
        }
        case 'filePreview': {
            const requestId = String(msg.requestId || '');
            const result = buildFilePreviewResult(msg.path, msg.offset, msg.limit);
            sendTo(client, {
                type: 'filePreviewResult',
                requestId,
                ...result
            });
            break;
        }
        case 'getStatus': {
            sendTo(client, { type: 'status', data: buildStatusSnapshot() });
            break;
        }
        case 'submit': {
            const channelId = String(msg.channelId || '1');
            const text = String(msg.text || '').trim();
            const filePaths = toStringArray(msg.file_paths);
            const imagePaths = toStringArray(msg.image_paths);
            if (!text && filePaths.length === 0 && imagePaths.length === 0) {
                sendTo(client, { type: 'submitResult', ok: false, channelId, message: '消息内容为空' });
                return;
            }
            const result = (0, mcpServer_1.sendUserMessage)(channelId, {
                user_input: text,
                file_paths: filePaths,
                image_paths: imagePaths
            });
            sendTo(client, {
                type: 'submitResult',
                ok: result.ok,
                channelId,
                message: result.ok ? '已发送' : (result.error || '发送失败'),
                echo: {
                    channelId,
                    text,
                    file_paths: filePaths,
                    image_paths: imagePaths,
                    timestamp: Date.now()
                }
            });
            // 广播给所有客户端（让多个浏览器窗口保持同步）
            broadcastExcept(client, {
                type: 'remoteSubmit',
                channelId,
                text,
                file_paths: filePaths,
                image_paths: imagePaths,
                timestamp: Date.now(),
                fromClientId: client.id
            });
            // 状态推一次让通道 queueLength 刷新
            broadcastStatus();
            break;
        }
        case 'getStartPrompt': {
            const channelId = String(msg.channelId || '1');
            try {
                const text = (0, mcpServer_1.prepareStartPrompt)(channelId).prompt;
                sendTo(client, { type: 'startPrompt', channelId, text });
            }
            catch (e) {
                sendTo(client, { type: 'startPrompt', channelId, text: '', error: e.message });
            }
            break;
        }
        case 'sendStartPrompt': {
            const channelId = String(msg.channelId || '1');
            try {
                const prepared = (0, mcpServer_1.prepareStartPrompt)(channelId);
                const result = await (0, seamlessSwitch_1.sendStartPromptToCursor)(prepared.prompt, prepared.channelId);
                sendTo(client, { type: 'startPromptSendResult', channelId, data: result });
            }
            catch (e) {
                sendTo(client, {
                    type: 'startPromptSendResult',
                    channelId,
                    data: { ok: false, message: e.message, copied: false }
                });
            }
            break;
        }
        case 'getRecoveryPacket': {
            try {
                const channelId = String(msg.channelId || '1');
                const packet = (0, mcpServer_1.getRecoveryPacket)({
                    scope: msg.scope === 'workspace' || msg.scope === 'group' ? msg.scope : 'channel',
                    channelId,
                    groupId: String(msg.groupId || ''),
                    depth: msg.depth === 'fast' || msg.depth === 'deep' ? msg.depth : 'standard',
                    maxChars: Number(msg.maxChars || 12000)
                });
                sendTo(client, { type: 'recoveryPacket', channelId, data: packet });
            }
            catch (e) {
                sendTo(client, {
                    type: 'recoveryPacket',
                    channelId: String(msg.channelId || '1'),
                    data: { ok: false, message: e.message }
                });
            }
            break;
        }
        case 'restoreRecoveryContext': {
            const sourceChannelId = String(msg.sourceChannelId || msg.channelId || '1');
            const targetChannelId = String(msg.targetChannelId || sourceChannelId);
            const result = (0, mcpServer_1.enqueueRecoveryContext)(targetChannelId, {
                sourceChannelId,
                targetChannelId,
                scope: msg.scope === 'workspace' || msg.scope === 'group' ? msg.scope : 'channel',
                groupId: String(msg.groupId || ''),
                depth: msg.depth === 'fast' || msg.depth === 'deep' ? msg.depth : 'standard',
                maxChars: Number(msg.maxChars || 12000)
            });
            sendTo(client, {
                type: 'restoreRecoveryResult',
                sourceChannelId,
                targetChannelId,
                ok: result.ok,
                entryCount: result.packet?.entryCount || 0,
                message: result.error || ''
            });
            broadcastStatus();
            break;
        }
        case 'getLatestReply': {
            const channelId = String(msg.channelId || '1');
            const archived = (0, mcpServer_1.readLatestAssistantReply)(channelId);
            const r = archived
                ? { reply: archived.content, timestamp: archived.createdAt }
                : (0, mcpServer_1.readReply)(channelId);
            if (r) {
                sendTo(client, {
                    type: 'latestReply',
                    channelId,
                    reply: r.reply,
                    timestamp: r.timestamp
                });
            }
            else {
                sendTo(client, { type: 'latestReply', channelId, reply: null });
            }
            break;
        }
        case 'clearChannel': {
            const channelId = String(msg.channelId || '1');
            try {
                (0, mcpServer_1.clearQueue)(channelId);
                (0, mcpServer_1.clearReply)(channelId);
                sendTo(client, { type: 'clearResult', channelId, ok: true });
                broadcastStatus();
            }
            catch (e) {
                sendTo(client, { type: 'clearResult', channelId, ok: false, message: e.message });
            }
            break;
        }
        case 'addChannel': {
            try {
                const result = await vscode.commands.executeCommand('qingtian.addChannel');
                sendTo(client, {
                    type: 'channelActionResult',
                    action: 'add',
                    ok: result?.ok !== false,
                    ...result
                });
                broadcastStatus();
            }
            catch (e) {
                sendTo(client, {
                    type: 'channelActionResult',
                    action: 'add',
                    ok: false,
                    message: e.message
                });
            }
            break;
        }
        case 'removeChannel': {
            try {
                const result = await vscode.commands.executeCommand('qingtian.removeChannel');
                sendTo(client, {
                    type: 'channelActionResult',
                    action: 'remove',
                    ok: result?.ok !== false,
                    ...result
                });
                broadcastStatus();
            }
            catch (e) {
                sendTo(client, {
                    type: 'channelActionResult',
                    action: 'remove',
                    ok: false,
                    message: e.message
                });
            }
            break;
        }
        case 'getQuickCommands': {
            sendTo(client, { type: 'quickCommands', list: (0, quickCommands_1.getQuickCommands)() });
            break;
        }
        case 'addQuickCommand': {
            const label = String(msg.label || '').trim();
            const text = String(msg.text || '');
            const r = await (0, quickCommands_1.addQuickCommand)(label, text);
            sendTo(client, {
                type: 'quickCommandsResult',
                action: 'add',
                ok: r.ok,
                message: r.message,
                list: r.list,
                added: r.added
            });
            if (r.ok)
                broadcastQuickCommandsChange(r.list);
            break;
        }
        case 'removeQuickCommand': {
            const id = String(msg.id || '');
            const r = await (0, quickCommands_1.removeQuickCommand)(id);
            sendTo(client, {
                type: 'quickCommandsResult',
                action: 'remove',
                ok: r.ok,
                message: r.message,
                list: r.list,
                removedId: id
            });
            if (r.ok)
                broadcastQuickCommandsChange(r.list);
            break;
        }
        case 'updateQuickCommand': {
            const id = String(msg.id || '');
            const patch = {};
            if (typeof msg.label === 'string')
                patch.label = msg.label;
            if (typeof msg.text === 'string')
                patch.text = msg.text;
            const r = await (0, quickCommands_1.updateQuickCommand)(id, patch);
            sendTo(client, {
                type: 'quickCommandsResult',
                action: 'update',
                ok: r.ok,
                message: r.message,
                list: r.list,
                updatedId: id
            });
            if (r.ok)
                broadcastQuickCommandsChange(r.list);
            break;
        }
        case 'resetQuickCommands': {
            const list = await (0, quickCommands_1.resetQuickCommands)();
            sendTo(client, {
                type: 'quickCommandsResult',
                action: 'reset',
                ok: true,
                list
            });
            broadcastQuickCommandsChange(list);
            break;
        }
        case 'getConfig': {
            sendTo(client, { type: 'config', data: readRuntimeConfigSnapshot() });
            break;
        }
        case 'setConfig': {
            try {
                const cfg = vscode.workspace.getConfiguration('qingtian');
                const patch = msg;
                if (Object.prototype.hasOwnProperty.call(patch, 'keepaliveEnabled')) {
                    await cfg.update('keepaliveEnabled', Boolean(patch.keepaliveEnabled), vscode.ConfigurationTarget.Global);
                }
                if (Object.prototype.hasOwnProperty.call(patch, 'keepaliveMinutes')) {
                    const m = Math.max(1, Math.min(120, Number(patch.keepaliveMinutes) || 45));
                    await cfg.update('keepaliveMinutes', m, vscode.ConfigurationTarget.Global);
                }
                if (Object.prototype.hasOwnProperty.call(patch, 'notifyOnReply')) {
                    await cfg.update('notifyOnReply', Boolean(patch.notifyOnReply), vscode.ConfigurationTarget.Global);
                }
                const snap = readRuntimeConfigSnapshot();
                sendTo(client, { type: 'configResult', ok: true, data: snap });
                broadcastAll({ type: 'config', data: snap });
            }
            catch (e) {
                sendTo(client, { type: 'configResult', ok: false, message: e.message });
            }
            break;
        }
        case 'getLicense': {
            sendTo(client, { type: 'license', data: (0, activation_1.getLicenseCountdownStatus)() });
            break;
        }
        case 'getMCPStatus': {
            try {
                const status = (0, mcpServer_1.getMCPStatus)();
                sendTo(client, { type: 'mcpStatus', data: status });
            }
            catch (e) {
                sendTo(client, { type: 'mcpStatus', data: { ok: false, reason: e.message } });
            }
            break;
        }
        case 'resolveDroppedPaths': {
            const result = await (0, mcpServer_1.resolveDroppedPathRefs)(msg.refs, { promptAmbiguous: false });
            sendTo(client, {
                type: 'droppedPathRefsResolved',
                requestId: String(msg.requestId || ''),
                data: result
            });
            break;
        }
        case 'getTeamSnapshot': {
            sendTo(client, {
                type: 'teamSnapshot',
                data: (0, mcpServer_1.getAgentTeamWorkbenchSnapshot)(String(msg.groupId || ''))
            });
            break;
        }
        case 'getAgentSkills': {
            sendTo(client, { type: 'agentSkills', data: (0, agentSkills_1.getAgentSkillCatalog)() });
            break;
        }
        case 'installAgentSkill': {
            const result = await (0, agentSkills_1.installAgentSkill)(String(msg.skillId || ''), { confirm: false });
            sendTo(client, { type: 'agentSkillInstallResult', data: result });
            broadcastAll({ type: 'agentSkills', data: (0, agentSkills_1.getAgentSkillCatalog)() });
            break;
        }
        case 'createTeamGroup': {
            const result = (0, mcpServer_1.createAgentTeamGroup)({
                name: String(msg.name || ''),
                goal: String(msg.goal || ''),
                channelIds: toStringArray(msg.channelIds),
                members: Array.isArray(msg.members) ? msg.members : []
            });
            sendTo(client, { type: 'teamCreateResult', data: result });
            if (result.ok && result.group) {
                broadcastAll({
                    type: 'teamSnapshot',
                    data: (0, mcpServer_1.getAgentTeamWorkbenchSnapshot)(result.group.groupId)
                });
            }
            else {
                sendTo(client, {
                    type: 'teamSnapshot',
                    data: (0, mcpServer_1.getAgentTeamWorkbenchSnapshot)(String(msg.groupId || ''))
                });
            }
            break;
        }
        case 'inviteTeamMembers': {
            const result = (0, mcpServer_1.inviteAgentsToGroup)({
                groupId: String(msg.groupId || ''),
                channelIds: toStringArray(msg.channelIds),
                members: Array.isArray(msg.members) ? msg.members : []
            });
            sendTo(client, { type: 'teamInviteResult', data: result });
            if (result.ok && result.group) {
                broadcastAll({
                    type: 'teamSnapshot',
                    data: (0, mcpServer_1.getAgentTeamWorkbenchSnapshot)(result.group.groupId)
                });
            }
            else {
                sendTo(client, {
                    type: 'teamSnapshot',
                    data: (0, mcpServer_1.getAgentTeamWorkbenchSnapshot)(String(msg.groupId || ''))
                });
            }
            break;
        }
        case 'deleteTeamGroup': {
            const result = (0, mcpServer_1.deleteAgentTeamGroup)(String(msg.groupId || ''));
            sendTo(client, { type: 'teamDeleteResult', data: result });
            broadcastAll({
                type: 'teamSnapshot',
                data: (0, mcpServer_1.getAgentTeamWorkbenchSnapshot)(result.activeGroupId || String(msg.groupId || ''))
            });
            break;
        }
        case 'sendTeamMessage': {
            const result = (0, mcpServer_1.sendAgentTeamGroupMessage)({
                groupId: String(msg.groupId || ''),
                text: String(msg.text || ''),
                author: '用户'
            });
            sendTo(client, { type: 'teamSendResult', data: result });
            broadcastAll({
                type: 'teamSnapshot',
                data: (0, mcpServer_1.getAgentTeamWorkbenchSnapshot)(String(msg.groupId || ''))
            });
            break;
        }
        case 'restoreTeamGroupContext': {
            const result = (0, mcpServer_1.restoreAgentTeamGroupContext)({
                groupId: String(msg.groupId || ''),
                moderatorChannelId: String(msg.moderatorChannelId || ''),
                depth: msg.depth === 'fast' || msg.depth === 'deep' ? msg.depth : 'standard',
                maxChars: Number(msg.maxChars || 16000)
            });
            sendTo(client, { type: 'teamRecoveryResult', data: result });
            broadcastAll({
                type: 'teamSnapshot',
                data: (0, mcpServer_1.getAgentTeamWorkbenchSnapshot)(String(msg.groupId || ''))
            });
            break;
        }
        case 'takeoverTeamGroupMember': {
            const result = (0, mcpServer_1.takeoverAgentTeamGroupMember)({
                groupId: String(msg.groupId || ''),
                sourceChannelId: String(msg.sourceChannelId || ''),
                targetChannelId: String(msg.targetChannelId || ''),
                depth: msg.depth === 'fast' || msg.depth === 'deep' ? msg.depth : 'standard',
                maxChars: Number(msg.maxChars || 18000)
            });
            sendTo(client, { type: 'teamRecoveryResult', data: result });
            broadcastAll({
                type: 'teamSnapshot',
                data: (0, mcpServer_1.getAgentTeamWorkbenchSnapshot)(result.groupId || String(msg.groupId || ''))
            });
            break;
        }
        case 'updateAgentRole': {
            const result = (0, mcpServer_1.updateAgentRole)(String(msg.channelId || ''), String(msg.roleId || ''), String(msg.groupId || ''));
            sendTo(client, { type: 'teamRoleResult', data: result });
            broadcastAll({
                type: 'teamSnapshot',
                data: (0, mcpServer_1.getAgentTeamWorkbenchSnapshot)(String(msg.groupId || ''))
            });
            break;
        }
        default: {
            sendTo(client, { type: 'error', message: `Unknown message type: ${type}` });
        }
    }
}
function toStringArray(v) {
    if (!Array.isArray(v))
        return [];
    const out = [];
    for (const item of v) {
        if (typeof item === 'string' && item.trim())
            out.push(item);
    }
    return out;
}
function readRuntimeConfigSnapshot() {
    const cfg = vscode.workspace.getConfiguration('qingtian');
    return {
        keepaliveEnabled: cfg.get('keepaliveEnabled', true),
        keepaliveMinutes: Math.max(1, Math.min(120, cfg.get('keepaliveMinutes', 45))),
        notifyOnReply: cfg.get('notifyOnReply', false)
    };
}
function broadcastQuickCommandsChange(list) {
    if (!currentInfo?.running)
        return;
    broadcastAll({ type: 'quickCommands', list });
}
function buildStatusSnapshot() {
    const count = (0, mcpServer_1.getChannelCount)();
    const channels = [];
    for (let i = 1; i <= count; i++) {
        const id = String(i);
        const hb = (0, mcpServer_1.readChannelHeartbeat)(id);
        channels.push({
            channelId: id,
            online: (0, mcpServer_1.isChannelOnline)(id),
            lastSeen: hb ? hb.lastSeen : null,
            queueLength: (0, mcpServer_1.getQueueLength)(id)
        });
    }
    return {
        channelCount: count,
        channels,
        streamSeqByChannel: getAgentStreamSeqSnapshot(),
        timestamp: Date.now()
    };
}
function sendTo(client, data) {
    if (client.ws.readyState !== ws_1.WebSocket.OPEN)
        return;
    try {
        client.ws.send(JSON.stringify(data));
    }
    catch (e) {
        log(`sendTo #${client.id} 失败: ${e.message}`);
    }
}
function broadcastAll(data) {
    const str = JSON.stringify(data);
    for (const client of clients) {
        if (client.ws.readyState !== ws_1.WebSocket.OPEN)
            continue;
        try {
            client.ws.send(str);
        }
        catch { }
    }
}
function broadcastExcept(except, data) {
    const str = JSON.stringify(data);
    for (const client of clients) {
        if (client === except)
            continue;
        if (client.ws.readyState !== ws_1.WebSocket.OPEN)
            continue;
        try {
            client.ws.send(str);
        }
        catch { }
    }
}
// ── 对外广播接口（供 extension.ts 的 replyPollTimer / statusCheckTimer 调用）──
function broadcastReply(channelId, reply, timestamp) {
    if (!currentInfo?.running)
        return;
    broadcastAll({
        type: 'reply',
        channelId,
        reply,
        timestamp
    });
}
/** 每通道 agent_stream 环形缓冲（阶段 2/4） */
const agentStreamRings = new Map();
/** 每通道统一递增 seq，避免 thought_stream 用 Date.now 把 transcript 工具事件挤掉 */
const agentStreamSeq = new Map();
let agentStreamRingSize = 500;
function setAgentStreamRingSize(n) {
    const v = Number(n);
    if (Number.isFinite(v) && v > 0)
        agentStreamRingSize = Math.floor(v);
}
function nextAgentStreamSeq(channelId) {
    const key = String(channelId || '');
    const prev = Number(agentStreamSeq.get(key) || 0);
    const next = prev + 1;
    agentStreamSeq.set(key, next);
    return next;
}
function getAgentStreamSeqSnapshot() {
    const out = {};
    for (const [ch, seq] of agentStreamSeq.entries()) {
        out[String(ch)] = Number(seq) || 0;
    }
    // 环形缓冲里也可能有更高 seq（防御）
    for (const [ch, ring] of agentStreamRings.entries()) {
        const key = String(ch);
        let max = Number(out[key] || 0);
        if (Array.isArray(ring)) {
            for (const ev of ring) {
                const s = Number(ev && ev.seq || 0);
                if (s > max) max = s;
            }
        }
        out[key] = max;
    }
    return out;
}
function getAgentStreamSince(channelId, sinceSeq) {
    const ring = agentStreamRings.get(String(channelId)) || [];
    const since = Number(sinceSeq) || 0;
    return ring.filter((ev) => Number(ev.seq) > since);
}
function broadcastAgentStream(payload) {
    if (!payload || typeof payload !== 'object')
        return;
    const channelId = String(payload.channelId || '');
    // 入口统一编号：覆盖各来源自带的互不兼容 seq
    const unified = Object.assign({}, payload, {
        channelId,
        seq: nextAgentStreamSeq(channelId),
        ts: payload.ts || Date.now()
    });
    if (!agentStreamRings.has(channelId))
        agentStreamRings.set(channelId, []);
    const ring = agentStreamRings.get(channelId);
    ring.push(unified);
    while (ring.length > agentStreamRingSize)
        ring.shift();
    if (!currentInfo?.running)
        return;
    broadcastAll(unified);
}
function broadcastAgentTeamReplyStreamChunk(chunk) {
    if (!currentInfo?.running)
        return;
    broadcastAll({
        type: 'teamReplyStreamChunk',
        data: chunk
    });
}
function broadcastStatus() {
    if (!currentInfo?.running)
        return;
    broadcastAll({
        type: 'status',
        data: buildStatusSnapshot()
    });
}
// ── HTTP: 附件上传 ─────────────────────────────────────────
/**
 * GET /api/file-preview?path=&offset=&limit=
 * 安全读取工作区内源码片段，供时间线展开行号预览。
 */
function tryPreviewFile(abs, folders) {
    const normalized = path.normalize(abs);
    const under = folders.some((root) => normalized === root || normalized.startsWith(root + path.sep));
    if (!under)
        return null;
    try {
        if (fs.existsSync(normalized) && fs.statSync(normalized).isFile())
            return normalized;
    }
    catch {
        /* ignore */
    }
    return null;
}
function findFileBySuffix(root, suffixRel, maxDepth = 12, maxVisits = 20000) {
    const needle = String(suffixRel || '').replace(/\\/g, '/').replace(/^\/+/, '');
    if (!needle)
        return null;
    const parts = needle.split('/').filter(Boolean);
    if (!parts.length)
        return null;
    const fileName = parts[parts.length - 1];
    const parentName = parts.length > 1 ? parts[parts.length - 2] : '';
    let visits = 0;
    let found = null;
    const skip = new Set(['node_modules', '.git', 'dist', 'build', '.cursor', 'outputs', 'DINO']);
    const matchesNeedle = (full) => {
        const norm = full.replace(/\\/g, '/');
        return norm.endsWith('/' + needle) || norm.endsWith(needle);
    };
    // 优先：按末级目录名定位（如 browser/client.js → 找名为 browser 的目录）
    if (parentName) {
        const queue = [{ dir: root, depth: 0 }];
        while (queue.length && !found && visits < maxVisits) {
            const { dir, depth } = queue.shift();
            if (depth > maxDepth)
                continue;
            let entries;
            try {
                entries = fs.readdirSync(dir, { withFileTypes: true });
            }
            catch {
                continue;
            }
            for (const ent of entries) {
                if (found)
                    break;
                visits += 1;
                if (skip.has(ent.name))
                    continue;
                const full = path.join(dir, ent.name);
                if (ent.isDirectory()) {
                    if (ent.name === parentName) {
                        const candidate = path.join(full, fileName);
                        try {
                            if (fs.existsSync(candidate) && fs.statSync(candidate).isFile() && matchesNeedle(candidate)) {
                                found = candidate;
                                break;
                            }
                        }
                        catch { /* ignore */ }
                    }
                    queue.push({ dir: full, depth: depth + 1 });
                }
            }
        }
    }
    // 回退：按文件名 BFS
    if (!found) {
        visits = 0;
        const queue = [{ dir: root, depth: 0 }];
        while (queue.length && !found && visits < maxVisits) {
            const { dir, depth } = queue.shift();
            if (depth > maxDepth)
                continue;
            let entries;
            try {
                entries = fs.readdirSync(dir, { withFileTypes: true });
            }
            catch {
                continue;
            }
            for (const ent of entries) {
                if (found)
                    break;
                visits += 1;
                if (skip.has(ent.name))
                    continue;
                const full = path.join(dir, ent.name);
                if (ent.isDirectory()) {
                    queue.push({ dir: full, depth: depth + 1 });
                }
                else if (ent.isFile() && ent.name === fileName && matchesNeedle(full)) {
                    found = full;
                }
            }
        }
    }
    return found ? path.normalize(found) : null;
}

function resolveWorkspacePreviewPath(reqPath) {
    const raw = String(reqPath || '').trim();
    if (!raw)
        return null;
    const folders = (vscode.workspace.workspaceFolders || [])
        .map((f) => path.normalize(f.uri.fsPath))
        .filter(Boolean);
    if (!folders.length)
        return null;
    const unix = raw.replace(/\\/g, '/');
    if (path.isAbsolute(raw) || /^[A-Za-z]:[\\/]/.test(raw)) {
        const hit = tryPreviewFile(raw, folders);
        if (hit)
            return hit;
    }
    const rel = unix.replace(/^\.\//, '').replace(/^\/+/, '');
    for (const root of folders) {
        const direct = tryPreviewFile(path.join(root, rel), folders);
        if (direct)
            return direct;
        const parts = rel.split('/').filter(Boolean);
        for (let i = 1; i < parts.length; i++) {
            const sub = parts.slice(i).join('/');
            const hit = tryPreviewFile(path.join(root, sub), folders);
            if (hit)
                return hit;
        }
    }
    // 短路径如 browser/client.js：在工作区下按后缀查找
    if (rel && rel.includes('/')) {
        for (const root of folders) {
            const found = findFileBySuffix(root, rel);
            if (found && tryPreviewFile(found, folders))
                return found;
        }
    }
    else if (rel) {
        for (const root of folders) {
            const found = findFileBySuffix(root, rel);
            if (found && tryPreviewFile(found, folders))
                return found;
        }
    }
    return null;
}
function buildFilePreviewResult(filePath, offsetRaw, limitRaw) {
    try {
        const abs = resolveWorkspacePreviewPath(filePath);
        if (!abs) {
            return { ok: false, message: '路径不在工作区内或不存在: ' + String(filePath || '') };
        }
        const startLine = Number.isFinite(Number(offsetRaw)) && Number(offsetRaw) > 0 ? Math.floor(Number(offsetRaw)) : 1;
        const limit = Math.min(400, Math.max(1, Number.isFinite(Number(limitRaw)) && Number(limitRaw) > 0 ? Math.floor(Number(limitRaw)) : 80));
        const text = fs.readFileSync(abs, 'utf8');
        const allLines = text.split(/\r?\n/);
        const slice = allLines.slice(startLine - 1, startLine - 1 + limit);
        const lines = slice.map((content, i) => ({
            n: startLine + i,
            text: content.length > 500 ? content.slice(0, 500) + '…' : content
        }));
        return {
            ok: true,
            path: abs,
            startLine,
            totalLines: allLines.length,
            truncated: startLine - 1 + limit < allLines.length,
            lines
        };
    }
    catch (e) {
        return { ok: false, message: e && e.message ? e.message : String(e) };
    }
}
function handleFilePreviewRequest(req, res) {
    try {
        const u = new URL(req.url || '/', 'http://127.0.0.1');
        const result = buildFilePreviewResult(u.searchParams.get('path') || '', u.searchParams.get('offset'), u.searchParams.get('limit'));
        res.statusCode = result.ok ? 200 : 403;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.setHeader('Cache-Control', 'no-cache');
        res.end(JSON.stringify(result));
    }
    catch (e) {
        res.statusCode = 500;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end(JSON.stringify({ ok: false, message: e && e.message ? e.message : String(e) }));
    }
}
/**
 * POST /api/upload
 * Body: JSON { files: [{ name: string, type?: string, data: <base64> }] }
 * Response: { ok, paths?: string[], message? }
 *
 * 收到后把文件落到 <queueRoot>/uploads/<YYYYMMDD>/<ts-rand-safeName>
 * 返回绝对路径数组，前端随即把 paths 作为 file_paths / image_paths 跟 submit 一起发出。
 */
function handleUploadRequest(req, res) {
    const chunks = [];
    let receivedBytes = 0;
    let aborted = false;
    const reject = (code, message) => {
        aborted = true;
        try {
            res.statusCode = code;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ ok: false, message }));
        }
        catch { }
    };
    req.on('data', (chunk) => {
        if (aborted)
            return;
        receivedBytes += chunk.length;
        if (receivedBytes > MAX_UPLOAD_BYTES) {
            reject(413, `上传体积超过限制（${Math.round(MAX_UPLOAD_BYTES / 1024 / 1024)} MB）`);
            req.destroy();
            return;
        }
        chunks.push(chunk);
    });
    req.on('error', (e) => {
        if (!aborted)
            reject(500, 'Request stream error: ' + e.message);
    });
    req.on('end', () => {
        if (aborted)
            return;
        let body;
        try {
            body = JSON.parse(Buffer.concat(chunks).toString('utf-8'));
        }
        catch {
            reject(400, 'Invalid JSON body');
            return;
        }
        const files = Array.isArray(body.files) ? body.files : [];
        if (files.length === 0) {
            reject(400, 'files 字段为空');
            return;
        }
        if (files.length > MAX_UPLOAD_FILES) {
            reject(400, `单次最多上传 ${MAX_UPLOAD_FILES} 个文件`);
            return;
        }
        try {
            const uploadRoot = ensureUploadDir();
            const savedPaths = [];
            for (const raw of files) {
                if (!raw || typeof raw !== 'object')
                    continue;
                const f = raw;
                const name = sanitizeFileName(String(f.name || 'upload.bin'));
                const data = typeof f.data === 'string' ? f.data : '';
                if (!data)
                    throw new Error('文件 ' + name + ' 内容为空');
                // 兼容 data URL 形式: "data:image/png;base64,xxx"
                const b64 = data.includes(',') ? data.slice(data.indexOf(',') + 1) : data;
                const buf = Buffer.from(b64, 'base64');
                const safeName = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}-${name}`;
                const abs = path.join(uploadRoot, safeName);
                fs.writeFileSync(abs, buf);
                savedPaths.push(abs);
            }
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ ok: true, paths: savedPaths }));
        }
        catch (e) {
            reject(500, 'Save failed: ' + e.message);
        }
    });
}
function ensureUploadDir() {
    const base = (0, mcpServer_1.getQueueRoot)() || path.join(os.homedir(), '.cursor', 'qingtian-runtime', 'messages');
    const now = new Date();
    const yyyy = String(now.getFullYear());
    const mm = String(now.getMonth() + 1).padStart(2, '0');
    const dd = String(now.getDate()).padStart(2, '0');
    const dir = path.join(base, 'uploads', `${yyyy}${mm}${dd}`);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    return dir;
}
function sanitizeFileName(name) {
    // 去掉路径成分 + 危险字符，保留扩展名
    const base = path.basename(String(name || ''));
    return base.replace(/[\\/:*?"<>|\x00-\x1f]/g, '_').slice(0, 80) || 'upload.bin';
}
//# sourceMappingURL=externalWebServer.js.map