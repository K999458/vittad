/**
 * 晴天无限 MCP · 浏览器端 WebSocket 客户端 (v2 — 聊天气泡界面)
 *
 * 负责：
 *  - 与 localhost:port 的 WebSocket server 建立连接
 *  - 维护通道列表、当前选中通道、连接状态、主题偏好
 *  - 发送消息、接收 AI 回复（带 markdown 渲染）、显示历史
 *  - localStorage 持久化浏览器本地历史 + 重连时补齐最近 AI 回复
 *
 * 消息协议（对齐 externalWebServer.ts）：
 *   客户端 → 服务端：ping / getStatus / submit / getStartPrompt
 *                    / getLatestReply / clearChannel / addChannel / removeChannel
 *   服务端 → 客户端：hello / status / submitResult / reply / remoteSubmit
 *                    / startPrompt / latestReply / clearResult / channelActionResult
 *                    / error / pong
 */
(function() {
    'use strict';

    const STORAGE_KEY = 'qingtian-mcp-browser-history-v1';
    const STORAGE_TIMELINES = 'qingtian-mcp-browser-timelines-v1';
    const STORAGE_STREAM_SEQ = 'qingtian-mcp-browser-stream-seq-v1';
    const STORAGE_LAST_CHANNEL = 'qingtian-mcp-browser-last-channel-v1';
    const STORAGE_THEME = 'qingtian-mcp-browser-theme-v1';
    const STORAGE_SIDEBAR = 'qingtian-mcp-browser-sidebar-v1';
    const STORAGE_SETTINGS = 'qingtian-mcp-browser-settings-v1';
    const STORAGE_VIEW_MODE = 'qingtian-mcp-browser-view-mode-v1';
    const STORAGE_OUTBOX = 'qingtian-mcp-browser-outbox-v1';
    const MAX_HISTORY = 200;
    /** 每通道最多暂存的排队消息条数 */
    const MAX_OUTBOX_PER_CHANNEL = 20;
    /** 上一条完成后，等待界面稳定再自动送出下一条 */
    const OUTBOX_FLUSH_DELAY_MS = 700;
    /** 超过这个时长仍等不到回复的轮次视为已结束，排队消息照常放行 */
    const OUTBOX_BUSY_STALE_MS = 10 * 60 * 1000;
    const RECONNECT_DELAY_MS = 2000;
    const THINKING_DEGRADE_MS = 120000; // 思考气泡超时降级阈值
    /** 最终回复后：Thinking → Thought for Xs，再停留若干秒后自动折叠正文 */
    const THOUGHT_DONE_AUTO_COLLAPSE_MS = 5000;
    const DOC_TITLE_ORIGINAL = document.title;

    // ── DOM ──
    const $ = (id) => document.getElementById(id);
    const elApp = document.querySelector('.app');
    const elMessages = $('messages');
    const elInput = $('input');
    const elMentionDropdown = $('mention-dropdown');
    const elSend = $('btn-send');
    const elChannelList = $('channel-list');
    const elStatus = $('ws-status');
    const elStatusLabel = elStatus.querySelector('.label');
    const elAiPresence = $('ai-presence');
    const elAiPresenceText = $('ai-presence-text');
    const elTopbarChannelName = $('topbar-channel-name');
    const elTopbarChannelSub = $('topbar-channel-sub');
    const elComposerHint = $('composer-hint');
    const elBtnViewChannel = $('btn-view-channel');
    const elBtnViewTeam = $('btn-view-team');
    const elTeamToolbar = $('team-toolbar');
    const elTeamGroupSelect = $('team-group-select');
    const elTeamToolbarStatus = $('team-toolbar-status');
    const elBtnTeamRefresh = $('btn-team-refresh');
    const elBtnTeamCreate = $('btn-team-create');
    const elBtnTeamDelete = $('btn-team-delete');
    const elBtnAdd = $('btn-add-channel');
    const elBtnRemove = $('btn-remove-channel');
    const elBtnCopyPrompt = $('btn-copy-prompt');
    const elBtnSendStartPrompt = $('btn-send-start-prompt');
    const elBtnCopyRecovery = $('btn-copy-recovery');
    const elRecoveryTransferModal = $('recovery-transfer-modal');
    const elRecoveryTransferOverlay = $('recovery-transfer-overlay');
    const elBtnRecoveryTransferClose = $('btn-recovery-transfer-close');
    const elBtnRecoveryTransferCancel = $('btn-recovery-transfer-cancel');
    const elBtnRecoveryTransferConfirm = $('btn-recovery-transfer-confirm');
    const elRecoveryTransferSource = $('recovery-transfer-source');
    const elRecoveryTransferTargetSection = $('recovery-transfer-target-section');
    const elRecoveryTransferTargetList = $('recovery-transfer-target-list');
    const elTeamRecoveryGroup = $('team-recovery-group');
    const elTeamRecoveryStatus = $('team-recovery-status');
    const elBtnTeamRecovery = $('btn-team-recovery');
    const elTeamRecoveryModal = $('team-recovery-modal');
    const elTeamRecoveryOverlay = $('team-recovery-overlay');
    const elBtnTeamRecoveryClose = $('btn-team-recovery-close');
    const elBtnTeamRecoveryCancel = $('btn-team-recovery-cancel');
    const elBtnTeamRecoveryConfirm = $('btn-team-recovery-confirm');
    const elTeamRecoveryModalSubtitle = $('team-recovery-modal-subtitle');
    const elTeamRecoveryRestoreDesc = $('team-recovery-restore-desc');
    const elTeamRecoveryModeratorPanel = $('team-recovery-moderator-panel');
    const elTeamRecoveryModeratorList = $('team-recovery-moderator-list');
    const elTeamRecoveryTakeoverPanel = $('team-recovery-takeover-panel');
    const elTeamRecoveryTakeoverGrid = $('team-recovery-takeover-grid');
    const elTeamRecoverySource = $('team-recovery-source');
    const elTeamRecoveryTarget = $('team-recovery-target');
    const elTeamGroupModal = $('team-group-modal');
    const elTeamGroupOverlay = $('team-group-overlay');
    const elBtnTeamGroupClose = $('btn-team-group-close');
    const elBtnTeamGroupCancel = $('btn-team-group-cancel');
    const elBtnTeamGroupConfirm = $('btn-team-group-confirm');
    const elTeamGroupName = $('team-group-name');
    const elTeamGroupGoal = $('team-group-goal');
    const elTeamGroupMemberList = $('team-group-member-list');
    const elBtnClearChannel = $('btn-clear-channel');
    const elBtnClearHistory = $('btn-clear-history');
    const elBtnThemeToggle = $('btn-theme-toggle');
    const elBtnToggleSidebar = $('btn-toggle-sidebar');

    // 新增：设置 / 历史抽屉 / 搜索条 / MCP 状态 / 快捷指令 / 附件
    const elDrawerBackdrop = $('drawer-backdrop');
    const elSettingsDrawer = $('settings-drawer');
    const elHistoryDrawer = $('history-drawer');
    const elBtnOpenSettings = $('btn-open-settings');
    const elBtnCloseSettings = $('btn-close-settings');
    const elBtnOpenHistory = $('btn-open-history');
    const elBtnCloseHistory = $('btn-close-history');
    const elBtnToggleSearch = $('btn-toggle-search');
    const elBtnCloseSearch = $('btn-close-search');
    const elSearchBar = $('search-bar');
    const elSearchInput = $('search-input');
    const elSearchCount = $('search-count');
    const elSettingSendMode = $('setting-send-mode');
    const elSettingFontSize = $('setting-font-size');
    const elSettingThemeMode = $('setting-theme-mode');
    const elSettingNotifyDesktop = $('setting-notify-desktop');
    const elSettingKeepaliveEnabled = $('setting-keepalive-enabled');
    const elSettingKeepaliveMinutes = $('setting-keepalive-minutes');
    const elSettingKeepaliveMinutesRow = $('setting-keepalive-minutes-row');
    const elLicenseDesc = $('license-desc');
    const elMcpMode = $('mcp-mode');
    const elMcpClientCount = $('mcp-client-count');
    const elLicenseRemaining = $('license-remaining');
    const elQuickCommands = $('quick-commands');
    const elQuickCmdForm = $('quick-commands-form');
    const elQuickCmdLabelInput = $('quick-cmd-label');
    const elQuickCmdTextInput = $('quick-cmd-text');
    const elBtnQuickAdd = $('btn-quick-add');
    const elBtnQuickReset = $('btn-quick-reset');
    const elBtnQuickSave = $('btn-quick-save');
    const elBtnQuickCancel = $('btn-quick-cancel');
    const elAttachmentsRow = $('attachments-row');
    const elAttachmentsList = $('attachments-list');
    const elOutboxRow = $('outbox-row');
    const elOutboxList = $('outbox-list');
    const elOutboxCount = $('outbox-count');
    const elOutboxHint = $('outbox-hint');
    const elBtnOutboxSendNow = $('btn-outbox-send-now');
    const elBtnOutboxClear = $('btn-outbox-clear');
    const elBtnAttachFile = $('btn-attach-file');
    const elBtnAttachImage = $('btn-attach-image');
    const elHiddenFilePicker = $('hidden-file-picker');
    const elHiddenImagePicker = $('hidden-image-picker');
    const elComposerDrop = $('composer-drop');
    const elHistoryListDrawer = $('history-list');
    const elHistorySearch = $('history-search');
    const elHistoryTypeFilter = $('history-type-filter');
    const elBtnExportMd = $('btn-export-md');
    const elBtnExportJson = $('btn-export-json');
    const elHistorySummary = $('history-summary');
    const elThinkingTemplate = $('thinking-bubble-template');
    const elTimelineTemplate = $('agent-timeline-template');
    const MAX_TIMELINE_ITEMS = 300;
    const SHELL_TOOLS = new Set(['Shell', 'sh_run', 'term_exec', 'Bash']);
    const READ_TOOLS = new Set([
        'Read', 'Grep', 'Glob', 'fs_read', 'fs_list', 'fs_glob', 'fs_grep', 'fs_stat',
        'SemanticSearch', 'rg', 'search_replace_read'
    ]);
    const EDIT_TOOLS = new Set([
        'Write', 'StrReplace', 'ApplyPatch', 'apply_patch', 'fs_write', 'fs_edit', 'fs_append',
        'nb_edit', 'EditNotebook', 'Delete', 'fs_delete', 'fs_mkdir', 'fs_move'
    ]);
    const pendingFilePreviews = new Map(); // requestId -> { resolve, reject, timer }

    // ── State ──
    let ws = null;
    let wsState = 'connecting';
    let reconnectTimer = null;
    let currentChannel = localStorage.getItem(STORAGE_LAST_CHANNEL) || '1';
    let currentViewMode = localStorage.getItem(STORAGE_VIEW_MODE) === 'team' ? 'team' : 'channel';
    let history = loadHistory();
    let clientId = null;
    let renderedHistoryFingerprint = '';
    let channelInfoMap = {};
    let channelCount = 1;
    // 记录已见过的 reply 指纹，避免重连时补回的回复重复渲染
    const seenReplies = new Set();
    for (const h of history) {
        if (h.type === 'reply' && h.fingerprint) seenReplies.add(h.fingerprint);
    }

    // 新增状态
    let quickCommands = [];
    let localSettings = {
        sendMode: 'enter', // 'enter' | 'ctrl-enter'
        fontSize: 'medium', // 'small' | 'medium' | 'large'
        themeMode: 'system', // 'system' | 'dark' | 'light'
        notifyDesktop: false
    };
    let serverConfig = {
        keepaliveEnabled: true,
        keepaliveMinutes: 45,
        notifyOnReply: false
    };
    let mcpStatusInfo = { connected: false, channelCount: 0, clientCount: 0 };
    let licenseInfo = { activated: false, permanent: false, expiresAt: null, remainingMs: null };
    let thinkingMap = {}; // { channelId: { node, startedAt, degradeTimer } }
    /** @type {Record<string, { node: HTMLElement, itemsEl: HTMLElement, startedAt: number, stepCount: number, lastSeq: number, collapsed: boolean, typing: any, omitCount: number }>} */
    let streamMap = {};
    // 每通道已完成的历史时间线：每轮收尾后归档为 { node, afterId }，渲染时紧跟对应轮次消息
    let doneTimelines = {};
    const MAX_DONE_TIMELINES = 30;
    /** 每通道已处理的最大 agent_stream seq，跨轮次保留，防止重放旧事件导致 Thought 串台 */
    let channelLastSeq = {};
    /** 每通道最近一次 thought_stream 活跃时间，用于暂时屏蔽 chat-store/cdp 混入 */
    let thoughtStreamActiveAt = {};
    /** 每通道 CLI 助手正文实时草稿（镜像 Cursor 窗口中间话，最终由 record_reply 定稿） */
    let draftReplyMap = {}; // channelId -> { node, text, updatedAt }
    let pendingAttachments = []; // [{ name, data(base64), type: 'file'|'image', size, localId }]
    /** 排队待发消息（不进主对话流，暂存在指令台上方）：{ [channelId]: [{ id, text, attachments, createdAt }] } */
    let outboxMap = loadOutbox();
    let outboxEditingId = '';
    let outboxFlushTimer = null;
    let outboxDispatching = false;
    let unreadCount = 0;
    let searchKeyword = '';
    let historyDrawerFilterText = '';
    let historyDrawerFilterType = 'all';
    let recoveryTransferMode = 'current';
    let recoveryTransferTargetId = '';
    let teamSnapshot = null;
    let activeTeamGroupId = '';
    let teamRecoveryMode = 'restore';
    let teamRecoveryModeratorId = '';
    let teamRecoverySourceId = '';
    let teamRecoveryTargetId = '';
    let mentionResults = [];
    let mentionIndex = -1;
    let mentionDebounce = null;

    function mountTeamRecoveryEntry() {
        if (!elTeamToolbar || !elBtnTeamRecovery) return;
        const legacySection = elBtnTeamRecovery.closest('.sidebar-section');
        if (legacySection) legacySection.classList.add('team-recovery-sidebar-hidden');
        elBtnTeamRecovery.classList.add('team-recovery-toolbar-button');
        elBtnTeamRecovery.textContent = '恢复当前群聊';
        if (elBtnTeamRefresh && elBtnTeamRefresh.parentNode === elTeamToolbar) {
            elTeamToolbar.insertBefore(elBtnTeamRecovery, elBtnTeamRefresh.nextSibling);
        } else {
            elTeamToolbar.appendChild(elBtnTeamRecovery);
        }
    }
    let teamLiveStreams = {};

    // ── History helpers ──
    function loadHistory() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return [];
            const arr = JSON.parse(raw);
            if (!Array.isArray(arr)) return [];
            // 确保每条历史都有稳定 id，供完成时间线锚定
            return arr.slice(-MAX_HISTORY).map((item) => {
                if (!item || typeof item !== 'object') return item;
                if (!item.id) item.id = 'h_' + String(item.timestamp || Date.now()) + '_' + Math.random().toString(36).slice(2, 7);
                return item;
            });
        } catch {
            return [];
        }
    }

    function saveHistory() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(-MAX_HISTORY)));
        } catch {}
    }

    function appendHistory(item) {
        if (item && !item.id) item.id = genId();
        history.push(item);
        if (history.length > MAX_HISTORY) {
            history = history.slice(-MAX_HISTORY);
        }
        saveHistory();
        renderMessages();
    }

    function lastHistoryIdForChannel(channelId) {
        const key = String(channelId || '');
        for (let i = history.length - 1; i >= 0; i--) {
            const item = history[i];
            if (!item) continue;
            if (!item.channelId || String(item.channelId) === key) {
                return item.id || null;
            }
        }
        return null;
    }

    /** 当前通道最近一条用户消息 id（完成块必须钉在它后面，才能夹在「用户 → 思维链 → 回复」中间） */
    function latestUserIdForChannel(channelId) {
        const key = String(channelId || '');
        for (let i = history.length - 1; i >= 0; i--) {
            const h = history[i];
            if (!h || !h.id) continue;
            if (h.channelId && String(h.channelId) !== key) continue;
            if (h.type === 'user') return h.id;
        }
        return null;
    }

    /** 当前通道倒数第二条用户消息 id（发新消息时用于钉住上一轮 Thought） */
    function previousUserIdForChannel(channelId) {
        const key = String(channelId || '');
        let seen = 0;
        for (let i = history.length - 1; i >= 0; i--) {
            const h = history[i];
            if (!h || !h.id) continue;
            if (h.channelId && String(h.channelId) !== key) continue;
            if (h.type !== 'user') continue;
            seen += 1;
            if (seen === 2) return h.id;
        }
        return null;
    }

    /** 完成块锚定：优先用时间线创建时钉住的用户消息，避免发新消息后跑到下一轮下面 */
    function nextAnchorIdForChannel(channelId, preferredId) {
        if (preferredId) return preferredId;
        return latestUserIdForChannel(channelId) || lastHistoryIdForChannel(channelId);
    }

    function normalizeDoneEntry(entry) {
        // 兼容旧结构：纯 DOM 节点 → { node, afterId }
        if (entry && entry.node) return entry;
        if (entry && entry.nodeType === 1) return { node: entry, afterId: null };
        return null;
    }

    // ── Rendering ──
    function renderMessages(force) {
        if (currentViewMode === 'team') {
            renderTeamMessages(force);
            return;
        }
        const filtered = history.filter(h => !h.channelId || h.channelId === currentChannel);
        const doneList = (doneTimelines[currentChannel] || [])
            .map(normalizeDoneEntry)
            .filter(Boolean);
        const st = streamMap[currentChannel];
        const fingerprint = filtered.map(h => h.id || `${h.timestamp}_${h.type}`).join('|')
            + '|done=' + doneList.map(d => (d.afterId || '') + ':' + (d.node && d.node.dataset ? (d.node.dataset.timelineId || '') : '')).join(',')
            + '|live=' + (st && st.node ? (st.node.dataset.timelineId || '1') : '0')
            + '|kw=' + (searchKeyword || '');
        if (!force && fingerprint === renderedHistoryFingerprint) return;
        renderedHistoryFingerprint = fingerprint;

        elMessages.innerHTML = '';
        let firstMatchNode = null;
        const placedDone = new Set();
        const appendDoneAfter = (historyId) => {
            if (!historyId) return;
            for (let i = 0; i < doneList.length; i++) {
                if (placedDone.has(i)) continue;
                if (doneList[i].afterId === historyId) {
                    elMessages.appendChild(doneList[i].node);
                    placedDone.add(i);
                }
            }
        };

        let lastUserIdx = -1;
        for (let i = 0; i < filtered.length; i++) {
            if (filtered[i] && filtered[i].type === 'user') lastUserIdx = i;
        }

        if (filtered.length === 0) {
            const sys = document.createElement('div');
            sys.className = 'msg msg-system';
            sys.innerHTML = '<div class="msg-bubble">' +
                '<p>当前通道（<strong>CH-' + escapeHtml(currentChannel) + '</strong>）还没有消息。</p>' +
                '<p>输入内容按 <code>' + (localSettings.sendMode === 'ctrl-enter' ? 'Ctrl/⌘ + Enter' : 'Enter') + '</code> 发送，AI 回复会以聊天气泡形式显示。</p>' +
                '</div>';
            elMessages.appendChild(sys);
        } else {
            let matched = 0;
            let livePlaced = false;
            const searchTerms = searchKeyword ? parseSearchTerms(searchKeyword) : [];
            for (let i = 0; i < filtered.length; i++) {
                const item = filtered[i];
                const node = renderMsg(item);
                if (searchKeyword) {
                    if (!textMatchesSearchTerms(item.text, searchTerms)) {
                        node.classList.add('dim-by-search');
                    } else {
                        matched++;
                        if (!firstMatchNode) firstMatchNode = node;
                        applySearchHighlightInNode(node, searchTerms);
                    }
                }
                elMessages.appendChild(node);
                // 已完成时间线紧跟对应轮次（优先用户消息）
                appendDoneAfter(item.id);

                // 进行中的时间线紧跟当前轮用户消息（在回复气泡之前）
                // 顺序：用户消息 → 思维链 → 回复
                if (!livePlaced && st && st.node && item.type === 'user' && i === lastUserIdx) {
                    elMessages.appendChild(st.node);
                    livePlaced = true;
                }
            }
            if (searchKeyword && elSearchCount) {
                elSearchCount.textContent = String(matched);
            } else if (elSearchCount) {
                elSearchCount.textContent = '0';
            }
            if (!livePlaced && st && st.node) {
                elMessages.appendChild(st.node);
                livePlaced = true;
            }
        }
        // 无锚点或锚点已不在当前历史中的完成块：按顺序插在活跃流之前
        for (let i = 0; i < doneList.length; i++) {
            if (!placedDone.has(i)) {
                if (st && st.node && st.node.parentNode === elMessages) {
                    elMessages.insertBefore(doneList[i].node, st.node);
                } else {
                    elMessages.appendChild(doneList[i].node);
                }
            }
        }
        // 无活跃流时，思考气泡仍贴底
        if (!(st && st.node && st.node.parentNode === elMessages)) {
            const tk = thinkingMap[currentChannel];
            if (tk && tk.node) elMessages.appendChild(tk.node);
        }
        // CLI 中间话草稿：贴在思维链之后、历史回复之前的「正在生成」位
        const draft = draftReplyMap[currentChannel];
        if (draft && draft.node) {
            elMessages.appendChild(draft.node);
        }
        // 搜索时定位到首个命中，其余情况保持贴底
        if (searchKeyword && firstMatchNode) {
            firstMatchNode.scrollIntoView({ block: 'center' });
        } else {
            elMessages.scrollTop = elMessages.scrollHeight;
        }
    }

    function renderTeamMessages(force) {
        const group = getActiveTeamGroup();
        if (teamSnapshot && teamSnapshot.enabled === false) {
            renderedHistoryFingerprint = 'team-disabled';
            elMessages.innerHTML = '<div class="msg msg-system"><div class="msg-bubble"><p><strong>多 Agent 群聊已关闭</strong></p><p>开启后浏览器端可在这里发送群聊消息，并按成员聚合查看回复。</p></div></div>';
            if (elSearchCount) elSearchCount.textContent = '0';
            return;
        }
        const events = group && teamSnapshot && Array.isArray(teamSnapshot.events)
            ? teamSnapshot.events.filter(event => String(event.groupId || '') === String(group.groupId || ''))
            : [];
        cleanupFinalizedTeamStreams(events);
        const liveStreams = Object.values(teamLiveStreams).filter(stream =>
            group &&
            String(stream.groupId || '') === String(group.groupId || '') &&
            (String(stream.text || '').trim() || stream.status !== 'start')
        );
        const fingerprint = 'team|' + (group ? group.groupId : '') + '|' +
            events.map(event => event.eventId || event.createdAt || event.title).join('|') + '|' +
            liveStreams.map(stream => stream.streamId + ':' + stream.status + ':' + String(stream.text || '').length).join('|') +
            '|kw=' + (searchKeyword || '');
        if (!force && fingerprint === renderedHistoryFingerprint) return;
        renderedHistoryFingerprint = fingerprint;

        elMessages.innerHTML = '';
        if (!group) {
            const sys = document.createElement('div');
            sys.className = 'msg msg-system';
            sys.innerHTML = '<div class="msg-bubble"><p><strong>暂无群聊</strong></p><p>请先在 Cursor 群聊工作台创建群组，或刷新浏览器端群聊状态。</p></div>';
            elMessages.appendChild(sys);
            if (elSearchCount) elSearchCount.textContent = '0';
            return;
        }
        const allItems = events.map(event => ({ kind: 'event', event }))
            .concat(liveStreams.map(stream => ({ kind: 'stream', stream })));
        if (allItems.length === 0) {
            const sys = document.createElement('div');
            sys.className = 'msg msg-system';
            sys.innerHTML = '<div class="msg-bubble"><p><strong>' + escapeHtml(group.name || '群聊') + '</strong></p><p>这个群聊还没有浏览器端可见消息。输入内容后会投递给群内成员。</p></div>';
            elMessages.appendChild(sys);
            if (elSearchCount) elSearchCount.textContent = '0';
            return;
        }
        let matched = 0;
        const searchTerms = searchKeyword ? parseSearchTerms(searchKeyword) : [];
        let firstMatchNode = null;
        for (const item of allItems) {
            const node = item.kind === 'stream' ? renderTeamStream(item.stream) : renderTeamEvent(item.event);
            const text = item.kind === 'stream'
                ? String(item.stream.text || '')
                : String((item.event.title || '') + '\n' + (item.event.body || ''));
            if (searchKeyword) {
                if (!textMatchesSearchTerms(text, searchTerms)) {
                    node.classList.add('dim-by-search');
                } else {
                    matched++;
                    if (!firstMatchNode) firstMatchNode = node;
                    applySearchHighlightInNode(node, searchTerms);
                }
            }
            elMessages.appendChild(node);
        }
        if (elSearchCount) elSearchCount.textContent = searchKeyword ? String(matched) : '0';
        if (searchKeyword && firstMatchNode) {
            firstMatchNode.scrollIntoView({ block: 'center' });
        } else {
            elMessages.scrollTop = elMessages.scrollHeight;
        }
    }

    function getTeamRoleName(channelId) {
        const status = getTeamRecoveryStatus();
        const members = status && Array.isArray(status.members) ? status.members : [];
        const member = members.find(item => String(item.channelId || '') === String(channelId || ''));
        if (member && member.roleName) return member.roleName;
        const agents = teamSnapshot && Array.isArray(teamSnapshot.agents) ? teamSnapshot.agents : [];
        const agent = agents.find(item => String(item.channelId || '') === String(channelId || ''));
        const roles = teamSnapshot && Array.isArray(teamSnapshot.roles) ? teamSnapshot.roles : [];
        const role = agent ? roles.find(item => String(item.roleId || '') === String(agent.roleId || '')) : null;
        return role ? role.name : 'Agent';
    }

    function teamEventLabel(event) {
        const ch = String(event.sourceChannelId || '');
        if (ch) return 'CH-' + ch + ' · ' + getTeamRoleName(ch);
        const targets = event.metadata && Array.isArray(event.metadata.targetChannelIds) ? event.metadata.targetChannelIds : [];
        if (targets.length) return '用户 -> ' + targets.map(id => 'CH-' + id).join(', ');
        return event.title || '群聊';
    }

    function renderTeamEvent(event) {
        const type = String(event.type || '');
        const isUser = type === 'user_message' || type === 'team_user_message';
        const wrap = document.createElement('div');
        wrap.className = 'msg ' + (isUser ? 'msg-user' : 'msg-reply') + ' msg-team-event';
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        const body = String(event.body || event.title || '');
        bubble.innerHTML = isUser
            ? escapeHtml(body).replace(/\n/g, '<br>')
            : renderMarkdown(body);
        wrap.appendChild(bubble);
        const meta = document.createElement('div');
        meta.className = 'msg-meta';
        meta.innerHTML = '<span class="msg-channel">' + escapeHtml(teamEventLabel(event)) + '</span><span>' + formatTime(event.createdAt || Date.now()) + '</span>';
        wrap.appendChild(meta);
        return wrap;
    }

    function renderTeamStream(stream) {
        const wrap = document.createElement('div');
        wrap.className = 'msg msg-reply msg-team-stream';
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.innerHTML = renderMarkdown(String(stream.text || ''));
        wrap.appendChild(bubble);
        const meta = document.createElement('div');
        meta.className = 'msg-meta';
        meta.innerHTML = '<span class="msg-channel">CH-' + escapeHtml(stream.channelId || '') + ' · ' + escapeHtml(getTeamRoleName(stream.channelId)) + '</span><span>' + (stream.status === 'done' ? '已完成' : '流式回复中') + '</span>';
        wrap.appendChild(meta);
        return wrap;
    }

    function getTeamEventStreamId(event) {
        const metadata = event && event.metadata && typeof event.metadata === 'object' ? event.metadata : {};
        return metadata.streamId ? String(metadata.streamId) : '';
    }

    function cleanupFinalizedTeamStreams(events) {
        const finalized = new Set();
        for (const event of events || []) {
            const streamId = getTeamEventStreamId(event);
            if (streamId) finalized.add(streamId);
        }
        finalized.forEach(streamId => {
            if (teamLiveStreams[streamId]) delete teamLiveStreams[streamId];
        });
    }

    function findCompatibleTeamLiveStream(streamId, chunk) {
        const direct = streamId && teamLiveStreams[streamId] ? teamLiveStreams[streamId] : null;
        if (direct) return direct;
        const channelId = String(chunk && chunk.channelId || '');
        const groupId = String(chunk && chunk.groupId || '');
        const candidates = Object.values(teamLiveStreams)
            .filter(stream => stream && stream.status !== 'done')
            .filter(stream => !channelId || String(stream.channelId || '') === channelId)
            .filter(stream => !groupId || !stream.groupId || String(stream.groupId || '') === groupId)
            .sort((a, b) => Number(b.updatedAtMs || 0) - Number(a.updatedAtMs || 0));
        return candidates[0] || null;
    }

    function mergeTeamStreamText(stream, incomingText) {
        const incoming = String(incomingText || '');
        if (!incoming) return;
        const previous = String(stream.text || '');
        if (!previous) {
            stream.text = incoming;
            return;
        }
        if (incoming.startsWith(previous)) {
            stream.text = incoming;
            return;
        }
        if (previous.endsWith(incoming) || previous.includes(incoming)) {
            return;
        }
        let overlap = 0;
        const max = Math.min(previous.length, incoming.length);
        for (let size = 1; size <= max; size++) {
            if (previous.endsWith(incoming.slice(0, size))) overlap = size;
        }
        stream.text = previous + incoming.slice(overlap);
    }

    // 把搜索输入拆成词：空格分隔视为 AND，"引号"内视为整体短语
    function parseSearchTerms(kw) {
        const s = String(kw || '').trim().toLowerCase();
        if (!s) return [];
        const terms = [];
        const re = /"([^"]+)"|(\S+)/g;
        let m;
        while ((m = re.exec(s))) {
            const t = (m[1] || m[2] || '').trim();
            if (t) terms.push(t);
        }
        return terms;
    }

    // 文本是否命中全部搜索词（AND 语义）
    function textMatchesSearchTerms(text, terms) {
        if (!terms || !terms.length) return true;
        const hay = String(text || '').toLowerCase();
        return terms.every(t => hay.indexOf(t) >= 0);
    }

    // 高亮气泡内「所有搜索词的所有出现」，重叠区间自动合并
    function applySearchHighlightInNode(node, terms) {
        const list = Array.isArray(terms)
            ? terms.filter(Boolean)
            : parseSearchTerms(terms);
        if (!list.length) return;
        const bubble = node.querySelector('.msg-bubble');
        if (!bubble) return;
        const walker = document.createTreeWalker(bubble, NodeFilter.SHOW_TEXT, null);
        const textNodes = [];
        let n;
        while ((n = walker.nextNode())) textNodes.push(n);
        for (const tn of textNodes) {
            const t = tn.nodeValue;
            if (!t) continue;
            const lo = t.toLowerCase();
            const ranges = [];
            for (const term of list) {
                let from = 0, idx;
                while ((idx = lo.indexOf(term, from)) >= 0) {
                    ranges.push([idx, idx + term.length]);
                    from = idx + term.length;
                }
            }
            if (!ranges.length) continue;
            ranges.sort((a, b) => a[0] - b[0]);
            const merged = [];
            for (const r of ranges) {
                const last = merged[merged.length - 1];
                if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
                else merged.push(r.slice());
            }
            const frag = document.createDocumentFragment();
            let cur = 0;
            for (const [a, b] of merged) {
                if (a > cur) frag.appendChild(document.createTextNode(t.slice(cur, a)));
                const mark = document.createElement('mark');
                mark.className = 'search-highlight';
                mark.textContent = t.slice(a, b);
                frag.appendChild(mark);
                cur = b;
            }
            if (cur < t.length) frag.appendChild(document.createTextNode(t.slice(cur)));
            tn.parentNode.replaceChild(frag, tn);
        }
    }

    function renderMsg(item) {
        const wrap = document.createElement('div');
        wrap.className = 'msg msg-' + (item.type || 'user');

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        if (item.type === 'reply') {
            // AI 回复走 markdown 渲染
            bubble.innerHTML = renderMarkdown(String(item.text || ''));
        } else {
            // 用户消息 / 系统 / 错误：只转义 + 保留换行
            bubble.innerHTML = escapeHtml(String(item.text || '')).replace(/\n/g, '<br>');
        }
        wrap.appendChild(bubble);

        // 非系统消息加 meta（通道 + 时间）
        if (item.type !== 'system') {
            const meta = document.createElement('div');
            meta.className = 'msg-meta';
            meta.innerHTML =
                '<span class="msg-channel">CH-' + escapeHtml(item.channelId || '1') + '</span>' +
                '<span class="msg-time">' + formatTime(item.timestamp) + '</span>';
            wrap.appendChild(meta);
        }
        return wrap;
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }

    function formatTime(ts) {
        if (!ts) return '';
        const t = typeof ts === 'number' ? ts : Date.parse(String(ts));
        if (Number.isNaN(t)) return String(ts);
        const d = new Date(t);
        const pad = (n) => String(n).padStart(2, '0');
        return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    }

    /**
     * 极简 Markdown 渲染器。
     * 支持：代码块 ```lang\n...``` / 行内 `code` / **bold** / *em* / 链接 [text](url)
     *       无序列表 - item / 有序列表 1. item / 换行
     * 足够渲染 AI 的常规回复，不支持嵌套表格、HTML 透传等高级特性。
     */
    function renderMarkdown(src) {
        if (!src) return '';
        // 先抽出代码块（避免内部再被行内规则污染）
        const blocks = [];
        let text = src.replace(/```([\w+-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
            const idx = blocks.length;
            blocks.push(
                '<pre><code class="lang-' + escapeHtml(lang || '') + '">' +
                escapeHtml(code.replace(/\n$/, '')) + '</code></pre>'
            );
            return '\x00BLOCK' + idx + '\x00';
        });

        // 转义其余 HTML
        text = escapeHtml(text);

        // 行内 code
        text = text.replace(/`([^`\n]+)`/g, (_, c) => '<code>' + c + '</code>');
        // bold
        text = text.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
        // italic（单星号，避开 ** 和 */）
        text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
        // 链接
        text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, url) => {
            return '<a href="' + url + '" target="_blank" rel="noopener noreferrer">' + label + '</a>';
        });

        // 列表：逐行处理
        const lines = text.split('\n');
        const out = [];
        let inUl = false, inOl = false;
        const closeLists = () => {
            if (inUl) { out.push('</ul>'); inUl = false; }
            if (inOl) { out.push('</ol>'); inOl = false; }
        };
        for (let raw of lines) {
            const ulMatch = raw.match(/^\s*[-*]\s+(.+)$/);
            const olMatch = raw.match(/^\s*\d+\.\s+(.+)$/);
            if (ulMatch) {
                if (!inUl) { closeLists(); out.push('<ul>'); inUl = true; }
                out.push('<li>' + ulMatch[1] + '</li>');
            } else if (olMatch) {
                if (!inOl) { closeLists(); out.push('<ol>'); inOl = true; }
                out.push('<li>' + olMatch[1] + '</li>');
            } else if (/^\s*$/.test(raw)) {
                closeLists();
                out.push('');
            } else {
                closeLists();
                out.push(raw);
            }
        }
        closeLists();

        // 段落：空行分段，段内换行转 <br>
        text = out.join('\n').replace(/\n{2,}/g, '\n\n');
        const paragraphs = text.split(/\n\n+/).map(p => {
            p = p.trim();
            if (!p) return '';
            if (/^<(ul|ol|pre)/.test(p)) return p;
            return '<p>' + p.replace(/\n/g, '<br>') + '</p>';
        }).join('');

        // 还原代码块
        return paragraphs.replace(/\x00BLOCK(\d+)\x00/g, (_, i) => blocks[Number(i)] || '');
    }

    function replyFingerprint(channelId, reply, timestamp) {
        return String(channelId) + '|' + String(timestamp || '') + '|' + (reply || '').slice(0, 120);
    }

    // ── Theme ──
    function applyTheme(theme) {
        const finalTheme = theme === 'light' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', finalTheme);
        if (elBtnThemeToggle) {
            elBtnThemeToggle.textContent = finalTheme === 'dark' ? '🌙' : '☀';
            elBtnThemeToggle.title = finalTheme === 'dark' ? '切到浅色主题' : '切到深色主题';
        }
        try { localStorage.setItem(STORAGE_THEME, finalTheme); } catch {}
    }
    function toggleTheme() {
        const cur = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = cur === 'dark' ? 'light' : 'dark';
        // 主题切换按钮更新本地偏好，以便持久化 & 与设置抽屉同步
        localSettings.themeMode = next;
        saveLocalSettings();
        if (elSettingThemeMode) elSettingThemeMode.value = next;
        applyTheme(next);
    }

    // ── Sidebar ──
    function applySidebar(collapsed) {
        if (collapsed) elApp.classList.add('sidebar-collapsed');
        else elApp.classList.remove('sidebar-collapsed');
        try { localStorage.setItem(STORAGE_SIDEBAR, collapsed ? '1' : '0'); } catch {}
    }
    function toggleSidebar() {
        applySidebar(!elApp.classList.contains('sidebar-collapsed'));
    }

    // ── Status display ──
    function setWsState(state, label) {
        wsState = state;
        elStatus.setAttribute('data-state', state);
        elStatusLabel.textContent = label;
    }

    // ── WebSocket ──
    function connect() {
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${location.host}/`;
        setWsState('connecting', '连接中…');

        try {
            ws = new WebSocket(url);
        } catch (e) {
            setWsState('error', '连接失败');
            scheduleReconnect();
            return;
        }

        ws.addEventListener('open', () => {
            setWsState('open', '已连接');
            send({ type: 'getStatus' });
            send({ type: 'getTeamSnapshot', groupId: activeTeamGroupId || '' });
            // 重连时补齐当前通道最近一条 AI 回复
            send({ type: 'getLatestReply', channelId: currentChannel });
            // 拉取快捷指令 / 配置 / 许可 / MCP 状态
            send({ type: 'getQuickCommands' });
            send({ type: 'getConfig' });
            send({ type: 'getLicense' });
            send({ type: 'getMCPStatus' });
            // 断线续传 agent_stream 环形缓冲（用通道级 lastSeq，避免新轮次从 0 重放历史 Thought）
            // 刷新后 channelLastSeq 会从 localStorage 恢复，避免把已落盘时间线再重放一遍
            const st = streamMap[currentChannel];
            const sinceSeq = Math.max(
                Number(channelLastSeq[currentChannel] || 0),
                Number(st && st.lastSeq || 0)
            );
            send({
                type: 'agent_stream_resume',
                channelId: currentChannel,
                sinceSeq
            });
        });

        ws.addEventListener('message', (ev) => {
            let data;
            try {
                data = JSON.parse(ev.data);
            } catch {
                return;
            }
            handleServerMessage(data);
        });

        ws.addEventListener('close', () => {
            setWsState('closed', '已断开');
            scheduleReconnect();
        });

        ws.addEventListener('error', () => {
            setWsState('error', '错误');
        });
    }

    function scheduleReconnect() {
        if (reconnectTimer) return;
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            connect();
        }, RECONNECT_DELAY_MS);
    }

    function send(obj) {
        if (!ws || ws.readyState !== WebSocket.OPEN) return false;
        try {
            ws.send(JSON.stringify(obj));
            return true;
        } catch {
            return false;
        }
    }

    // ── Server message handlers ──
    function handleServerMessage(msg) {
        switch (msg.type) {
            case 'hello':
                clientId = msg.clientId;
                syncStreamSeqFromServer(msg.streamSeqByChannel);
                break;
            case 'status':
                updateChannelList(msg.data);
                if (msg.data && msg.data.streamSeqByChannel) {
                    syncStreamSeqFromServer(msg.data.streamSeqByChannel);
                }
                break;
            case 'submitResult':
                if (!msg.ok) {
                    appendHistory({
                        id: genId(),
                        type: 'error',
                        channelId: msg.channelId,
                        text: '发送失败: ' + (msg.message || '未知错误'),
                        timestamp: Date.now()
                    });
                }
                break;
            case 'reply': {
                const fp = replyFingerprint(msg.channelId, msg.reply, msg.timestamp);
                if (seenReplies.has(fp)) break;
                seenReplies.add(fp);
                appendHistory({
                    id: genId(),
                    type: 'reply',
                    channelId: msg.channelId,
                    text: msg.reply || '',
                    timestamp: msg.timestamp ? Date.parse(msg.timestamp) || Date.now() : Date.now(),
                    fingerprint: fp
                });
                hideThinking(msg.channelId);
                collapseAgentTimeline(msg.channelId, { collapseDelayMs: THOUGHT_DONE_AUTO_COLLAPSE_MS });
                clearDraftReply(msg.channelId);
                handleNewReplyNotification(msg.channelId, msg.reply || '');
                // 上一条执行完成：把排队队首自动弹上主对话流
                scheduleOutboxFlush(msg.channelId);
                break;
            }
            case 'latestReply': {
                // 重连后服务端推的当前通道最新回复
                if (!msg.reply) break;
                const fp = replyFingerprint(msg.channelId, msg.reply, msg.timestamp);
                if (seenReplies.has(fp)) break;
                seenReplies.add(fp);
                appendHistory({
                    id: genId(),
                    type: 'reply',
                    channelId: msg.channelId,
                    text: msg.reply || '',
                    timestamp: msg.timestamp ? Date.parse(msg.timestamp) || Date.now() : Date.now(),
                    fingerprint: fp
                });
                hideThinking(msg.channelId);
                collapseAgentTimeline(msg.channelId, { collapseDelayMs: THOUGHT_DONE_AUTO_COLLAPSE_MS });
                clearDraftReply(msg.channelId);
                scheduleOutboxFlush(msg.channelId);
                break;
            }
            case 'agent_stream': {
                handleAgentStreamEvent(msg);
                break;
            }
            case 'agent_stream_resume_result': {
                if (Array.isArray(msg.events)) {
                    const channelId = String(msg.channelId || currentChannel || '');
                    // 刷新后若本地已恢复完成态时间线，只推进 lastSeq，避免把历史工具/思考重放成新的 live Thinking
                    const hasRestored = !!(doneTimelines[channelId] && doneTimelines[channelId].length);
                    if (hasRestored && isChannelTurnComplete(channelId)) {
                        for (const ev of msg.events) {
                            const seq = Number(ev && ev.seq || 0);
                            if (seq) {
                                channelLastSeq[channelId] = Math.max(
                                    Number(channelLastSeq[channelId] || 0),
                                    seq
                                );
                            }
                        }
                    } else {
                        for (const ev of msg.events) handleAgentStreamEvent(ev);
                    }
                }
                break;
            }
            case 'filePreviewResult': {
                const requestId = String(msg.requestId || '');
                const pending = pendingFilePreviews.get(requestId);
                if (pending) {
                    clearTimeout(pending.timer);
                    pendingFilePreviews.delete(requestId);
                    pending.resolve({
                        ok: !!msg.ok,
                        message: msg.message || '',
                        path: msg.path || '',
                        startLine: msg.startLine,
                        totalLines: msg.totalLines,
                        truncated: msg.truncated,
                        lines: msg.lines || []
                    });
                }
                break;
            }
            case 'quickCommands':
                if (Array.isArray(msg.list)) {
                    quickCommands = msg.list.slice();
                    renderQuickCommands();
                }
                break;
            case 'quickCommandsResult':
                if (Array.isArray(msg.list)) {
                    quickCommands = msg.list.slice();
                    renderQuickCommands();
                }
                if (msg.ok && (msg.action === 'add' || msg.action === 'reset')) {
                    toggleQuickCommandForm(false);
                }
                if (!msg.ok && msg.message) {
                    showToast('快捷指令失败：' + msg.message);
                }
                break;
            case 'config':
                if (msg.data) {
                    serverConfig = { ...serverConfig, ...msg.data };
                    renderServerConfigControls();
                }
                break;
            case 'configResult':
                if (msg.ok && msg.data) {
                    serverConfig = { ...serverConfig, ...msg.data };
                    renderServerConfigControls();
                }
                if (!msg.ok && msg.message) {
                    showToast('保存失败：' + msg.message);
                }
                break;
            case 'license':
                if (msg.data) {
                    licenseInfo = { ...licenseInfo, ...msg.data };
                    renderLicenseInfo();
                }
                break;
            case 'mcpStatus':
                if (msg.data) {
                    mcpStatusInfo = { ...mcpStatusInfo, ...msg.data };
                    renderMcpStatusCard();
                }
                break;
            case 'teamSnapshot':
                if (msg.data) {
                    teamSnapshot = msg.data;
                    if (!activeTeamGroupId) activeTeamGroupId = teamSnapshot.activeGroupId || '';
                    const groups = Array.isArray(teamSnapshot.groups) ? teamSnapshot.groups : [];
                    if (groups.length && !groups.some(g => g.groupId === activeTeamGroupId)) {
                        activeTeamGroupId = teamSnapshot.activeGroupId || groups[0].groupId || '';
                    } else if (!groups.length) {
                        activeTeamGroupId = '';
                    }
                    renderTeamRecoveryCard();
                    renderTeamRecoveryModal();
                    renderTeamToolbar();
                    if (currentViewMode === 'team') renderMessages(true);
                }
                break;
            case 'teamCreateResult': {
                const result = msg.data || {};
                if (result.ok && result.group) {
                    activeTeamGroupId = result.group.groupId || activeTeamGroupId;
                    closeTeamGroupModal();
                    showToast('群聊已创建');
                } else {
                    showToast(result.message || '群聊创建失败');
                }
                break;
            }
            case 'teamDeleteResult': {
                const result = msg.data || {};
                if (result.ok) {
                    activeTeamGroupId = result.activeGroupId || '';
                    showToast('群聊已删除');
                } else {
                    showToast(result.message || '群聊删除失败');
                }
                break;
            }
            case 'teamSendResult': {
                const result = msg.data || {};
                if (!result.ok) {
                    showToast(result.message || '群聊消息发送失败');
                }
                break;
            }
            case 'teamReplyStreamChunk':
                handleTeamReplyStreamChunk(msg.data || {});
                break;
            case 'remoteSubmit': {
                const liveBefore = streamMap[String(msg.channelId || '')];
                const prevAnchor = (liveBefore && liveBefore.anchorUserId) || latestUserIdForChannel(msg.channelId) || null;
                if (liveBefore && !liveBefore.done) {
                    collapseAgentTimeline(msg.channelId, { afterId: prevAnchor });
                }
                appendHistory({
                    id: genId(),
                    type: 'user',
                    channelId: msg.channelId,
                    text: msg.text,
                    timestamp: msg.timestamp || Date.now(),
                    remote: true
                });
                showThinking(msg.channelId);
                break;
            }
            case 'startPrompt': {
                if (msg.error) {
                    showToast('复制启动协议失败：' + msg.error);
                    break;
                }
                const text = msg.text || '';
                if (!text) { showToast('启动协议为空'); break; }
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text)
                        .then(() => showToast(`启动协议已复制 · CH-${msg.channelId}`))
                        .catch(() => fallbackCopy(text, msg.channelId));
                } else {
                    fallbackCopy(text, msg.channelId);
                }
                break;
            }
            case 'recoveryPacket': {
                const packet = msg.data || {};
                if (!packet.ok || !packet.prompt) {
                    showToast('恢复包生成失败' + (packet.message ? '：' + packet.message : ''));
                    break;
                }
                const text = packet.prompt || '';
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text)
                        .then(() => showToast(`恢复包已复制 · CH-${msg.channelId || currentChannel}`))
                        .catch(() => fallbackCopy(text, msg.channelId || currentChannel, '恢复包'));
                } else {
                    fallbackCopy(text, msg.channelId || currentChannel, '恢复包');
                }
                break;
            }
            case 'restoreRecoveryResult':
                if (msg.ok) {
                    appendHistory({
                        id: genId(),
                        type: 'system',
                        channelId: msg.targetChannelId || currentChannel,
                        text: String(msg.sourceChannelId || '') && String(msg.targetChannelId || '') && String(msg.sourceChannelId) !== String(msg.targetChannelId)
                            ? ('已将 CH-' + msg.sourceChannelId + ' 的上下文转移到 CH-' + msg.targetChannelId + '。')
                            : '已自动投递恢复上下文，当前或下一个绑定该通道的窗口会自动接手。',
                        timestamp: Date.now()
                    });
                    showToast(
                        String(msg.sourceChannelId || '') && String(msg.targetChannelId || '') && String(msg.sourceChannelId) !== String(msg.targetChannelId)
                            ? (`已转移到 CH-${msg.targetChannelId}`)
                            : (`恢复上下文已投递 · CH-${msg.targetChannelId || currentChannel}`)
                    );
                    setRecoveryTransferOpen(false);
                } else {
                    appendHistory({
                        id: genId(),
                        type: 'error',
                        channelId: msg.targetChannelId || currentChannel,
                        text: '自动恢复上下文失败' + (msg.message ? '：' + msg.message : ''),
                        timestamp: Date.now()
                    });
                    showToast('自动恢复上下文失败' + (msg.message ? '：' + msg.message : ''));
                }
                break;
            case 'teamRecoveryResult': {
                const result = msg.data || {};
                appendHistory({
                    id: genId(),
                    type: result.ok ? 'system' : 'error',
                    channelId: result.targetChannelId || result.moderatorChannelId || currentChannel,
                    text: result.message || (result.ok ? '群聊恢复已投递' : '群聊恢复失败'),
                    timestamp: Date.now()
                });
                showToast(result.message || (result.ok ? '群聊恢复已投递' : '群聊恢复失败'));
                if (result.ok) closeTeamRecoveryModal();
                break;
            }
            case 'startPromptSendResult': {
                const result = msg.data || {};
                if (elBtnSendStartPrompt) elBtnSendStartPrompt.classList.remove('is-busy');
                showToast(result.message || (result.ok ? '开场白已投递' : '一键发送开场白失败'));
                break;
            }
            case 'clearResult':
                if (msg.ok) showToast(`CH-${msg.channelId} 已清空`);
                else showToast('清空失败：' + (msg.message || '未知错误'));
                break;
            case 'channelActionResult':
                if (msg.action === 'add') {
                    showToast(msg.ok ? '已新增通道' : ('新增失败：' + (msg.message || '未知')));
                } else if (msg.action === 'remove') {
                    showToast(msg.ok ? '已删除最后一个通道' : ('删除失败：' + (msg.message || '未知')));
                }
                break;
            case 'error': {
                const errText = String(msg.message || '');
                // 源码预览在旧服务上会触发 Unknown message type: filePreview，
                // 不应写入聊天区（会跳屏），而是回落到预览 Promise。
                if (/filePreview/i.test(errText) || /Unknown message type/i.test(errText)) {
                    for (const [requestId, pending] of pendingFilePreviews.entries()) {
                        clearTimeout(pending.timer);
                        pendingFilePreviews.delete(requestId);
                        pending.resolve({
                            ok: false,
                            message: '预览协议未就绪，请重启 qingtian-standalone 服务'
                        });
                    }
                    break;
                }
                appendHistory({
                    id: genId(),
                    type: 'error',
                    channelId: currentChannel,
                    text: '服务器错误: ' + errText,
                    timestamp: Date.now()
                });
                break;
            }
            case 'pong':
                break;
        }
    }

    function handleTeamReplyStreamChunk(chunk) {
        if (!chunk || !chunk.streamId) return;
        const incomingId = String(chunk.streamId);
        const compatible = findCompatibleTeamLiveStream(incomingId, chunk);
        const id = compatible ? compatible.streamId : incomingId;
        const prev = compatible || teamLiveStreams[id] || {
            streamId: id,
            groupId: String(chunk.groupId || ''),
            channelId: String(chunk.channelId || ''),
            agentId: String(chunk.agentId || ''),
            text: '',
            status: 'start',
            seenChunkIds: [],
            updatedAtMs: 0
        };
        const chunkId = String(chunk.chunkId || '');
        if (chunkId) {
            prev.seenChunkIds = Array.isArray(prev.seenChunkIds) ? prev.seenChunkIds : [];
            if (prev.seenChunkIds.includes(chunkId)) return;
            prev.seenChunkIds.push(chunkId);
            if (prev.seenChunkIds.length > 80) prev.seenChunkIds = prev.seenChunkIds.slice(-80);
        }
        if (incomingId && incomingId !== id && teamLiveStreams[incomingId]) {
            delete teamLiveStreams[incomingId];
        }
        prev.groupId = String(chunk.groupId || prev.groupId || '');
        prev.channelId = String(chunk.channelId || prev.channelId || '');
        prev.agentId = String(chunk.agentId || prev.agentId || '');
        prev.status = String(chunk.status || prev.status || 'delta');
        prev.updatedAt = String(chunk.timestamp || new Date().toISOString());
        prev.updatedAtMs = Date.parse(prev.updatedAt) || Date.now();
        if (chunk.status === 'done') {
            mergeTeamStreamText(prev, chunk.finalText || chunk.delta || '');
            prev.status = 'done';
            teamLiveStreams[id] = prev;
            if (wsState === 'open') send({ type: 'getTeamSnapshot', groupId: activeTeamGroupId || prev.groupId || '' });
            setTimeout(() => {
                if (teamLiveStreams[id] && teamLiveStreams[id].status === 'done' && wsState === 'open') {
                    send({ type: 'getTeamSnapshot', groupId: activeTeamGroupId || prev.groupId || '' });
                }
            }, 900);
        } else if (chunk.status === 'error') {
            prev.text = String(chunk.finalText || chunk.delta || prev.text || '群聊回复同步失败');
            teamLiveStreams[id] = prev;
        } else {
            mergeTeamStreamText(prev, chunk.delta || '');
            teamLiveStreams[id] = prev;
        }
        if (currentViewMode === 'team') renderMessages(true);
    }

    function updateChannelList(data) {
        if (!data || !Array.isArray(data.channels)) return;
        channelCount = Number(data.channelCount || data.channels.length || 1);
        const prev = currentChannel;
        const available = data.channels.map(c => c.channelId);
        channelInfoMap = {};
        for (const ch of data.channels) {
            channelInfoMap[ch.channelId] = ch;
        }
        // 恢复选中
        if (available.indexOf(prev) >= 0) {
            currentChannel = prev;
        } else if (available.length > 0) {
            currentChannel = available[0];
            localStorage.setItem(STORAGE_LAST_CHANNEL, currentChannel);
        }
        renderChannelList();
        renderTopbar();
        renderMessages();
        renderAiPresence();
        refreshChannelButtons();
        renderRecoveryTransferModal();
        renderTeamRecoveryCard();
        renderTeamRecoveryModal();
    }

    function renderChannelList() {
        if (!elChannelList) return;
        const ids = Object.keys(channelInfoMap).sort((a, b) => Number(a) - Number(b));
        elChannelList.innerHTML = '';
        for (const id of ids) {
            const info = channelInfoMap[id];
            const item = document.createElement('div');
            item.className = 'channel-item' + (id === currentChannel ? ' active' : '');
            item.dataset.channelId = id;
            const onlineDot = info.online ? '<span class="channel-online-dot" title="AI 在线"></span>' : '';
            const queueBadge = info.queueLength > 0 ? `<span class="channel-queue-badge">${info.queueLength}</span>` : '';
            const staged = outboxLength(id);
            const stagedBadge = staged > 0
                ? `<span class="channel-outbox-badge" title="${staged} 条排队待发（本地暂存）">${staged}</span>`
                : '';
            item.innerHTML =
                '<span class="channel-item-name">CH-' + escapeHtml(id) + '</span>' +
                '<span class="channel-item-meta">' + stagedBadge + queueBadge + onlineDot + '</span>';
            item.addEventListener('click', () => selectChannel(id));
            elChannelList.appendChild(item);
        }
    }

    function renderTopbar() {
        const group = getActiveTeamGroup();
        if (elTopbarChannelName) {
            elTopbarChannelName.textContent = currentViewMode === 'team'
                ? (group ? (group.name || 'Group Chat') : 'Group Chat')
                : 'CH-' + currentChannel;
        }
        if (elTopbarChannelSub) {
            if (currentViewMode === 'team') {
                const count = group && Array.isArray(group.channelIds) ? group.channelIds.length : 0;
                elTopbarChannelSub.textContent = count ? ('Team workspace · ' + count + ' members') : 'Team workspace';
            } else {
                const info = channelInfoMap[currentChannel];
                if (info) {
                    const stateText = info.online ? 'AI online' : (info.queueLength > 0 ? (info.queueLength + ' queued') : 'idle');
                    elTopbarChannelSub.textContent = 'Current channel · ' + stateText;
                } else {
                    elTopbarChannelSub.textContent = 'Current channel';
                }
            }
        }
        if (elComposerHint) {
            const modeText = localSettings.sendMode === 'ctrl-enter' ? 'Ctrl/Cmd + Enter to send' : 'Enter to send';
            elComposerHint.textContent = currentViewMode === 'team'
                ? ((group ? (group.name || 'Group Chat') : 'Group Chat') + ' · ' + modeText)
                : 'CH-' + currentChannel + ' · ' + modeText;
        }
        if (elInput) {
            elInput.placeholder = currentViewMode === 'team'
                ? 'Type a group message and send it to all selected members'
                : 'Type a message. Enter sends; drag or paste images here';
        }
        if (elTopbarChannelName && currentViewMode === 'team') {
            elTopbarChannelName.textContent = group ? (group.name || '群聊') : '群聊';
        }
        if (elTopbarChannelSub) {
            if (currentViewMode === 'team') {
                const count = group && Array.isArray(group.channelIds) ? group.channelIds.length : 0;
                elTopbarChannelSub.textContent = count ? ('群聊工作台 · ' + count + ' 个成员') : '群聊工作台';
            } else {
                const info = channelInfoMap[currentChannel];
                const stateText = info
                    ? (info.online ? 'AI 在线' : (info.queueLength > 0 ? (info.queueLength + ' 条待取') : '空闲'))
                    : '';
                elTopbarChannelSub.textContent = stateText ? ('当前通道 · ' + stateText) : '当前通道';
            }
        }
        if (elComposerHint) {
            const modeTextCn = localSettings.sendMode === 'ctrl-enter' ? 'Ctrl/Command + Enter 发送' : 'Enter 发送';
            elComposerHint.textContent = currentViewMode === 'team'
                ? ((group ? (group.name || '群聊') : '群聊') + ' · ' + modeTextCn)
                : 'CH-' + currentChannel + ' · ' + modeTextCn;
        }
        if (elInput) {
            elInput.placeholder = currentViewMode === 'team'
                ? '输入群聊消息，@全体成员 / @CH-1 / @文件 引用，Enter 发送'
                : '输入消息，@引用文件，拖拽/粘贴图片，Enter 发送';
        }
        renderViewModeControls();
    }

    function renderViewModeControls() {
        if (elBtnViewChannel) elBtnViewChannel.classList.toggle('active', currentViewMode === 'channel');
        if (elBtnViewTeam) elBtnViewTeam.classList.toggle('active', currentViewMode === 'team');
        if (elTeamToolbar) elTeamToolbar.classList.toggle('hidden', currentViewMode !== 'team');
        if (elAiPresence) elAiPresence.classList.toggle('hidden', currentViewMode === 'team');
        renderTeamToolbar();
    }

    function setViewMode(mode) {
        currentViewMode = mode === 'team' ? 'team' : 'channel';
        localStorage.setItem(STORAGE_VIEW_MODE, currentViewMode);
        renderedHistoryFingerprint = '';
        renderTopbar();
        renderMessages(true);
        renderOutbox();
    }

    function refreshChannelButtons() {
        if (elBtnRemove) elBtnRemove.disabled = channelCount <= 1;
        // MAX_CHANNELS 以服务端推送为准，这里不做上限检查，由服务端回信时给错误
    }

    function selectChannel(id) {
        if (!id || id === currentChannel) return;
        currentChannel = String(id);
        localStorage.setItem(STORAGE_LAST_CHANNEL, currentChannel);
        renderedHistoryFingerprint = '';
        renderChannelList();
        renderTopbar();
        renderMessages();
        renderAiPresence();
        renderRecoveryTransferModal();
        outboxEditingId = '';
        renderOutbox();
        // 切换通道时，如果本地没这个通道的任何 reply，主动补一次
        const hasAnyReply = history.some(h => h.type === 'reply' && h.channelId === currentChannel);
        if (!hasAnyReply && wsState === 'open') {
            send({ type: 'getLatestReply', channelId: currentChannel });
        }
        // 切通道补拉思维链时间线：该通道正在思考/执行工具时，切过来必须立即看到
        // 已发生的步骤，而不是等下一个新事件才出现。
        requestStreamResume(currentChannel);
    }

    // 按通道请求环形缓冲补拉；resume 有序幂等（服务端按 seq 过滤），重复调用无副作用
    function requestStreamResume(channelId) {
        const ch = String(channelId || '');
        if (!ch || wsState !== 'open') return;
        const st = streamMap[ch];
        const sinceSeq = Math.max(
            Number(channelLastSeq[ch] || 0),
            Number(st && st.lastSeq || 0)
        );
        send({ type: 'agent_stream_resume', channelId: ch, sinceSeq });
    }

    // ── Toast ──
    let toastTimer = null;
    function showToast(text) {
        let el = document.getElementById('toast');
        if (!el) {
            el = document.createElement('div');
            el.id = 'toast';
            el.style.cssText = 'position:fixed;left:50%;bottom:32px;transform:translateX(-50%);' +
                'background:var(--bg-elev-2);color:var(--text);padding:10px 18px;border-radius:10px;' +
                'border:1px solid var(--border);box-shadow:var(--shadow-soft);font-size:13px;' +
                'z-index:100;opacity:0;transition:opacity 0.2s;pointer-events:none;max-width:80vw;';
            document.body.appendChild(el);
        }
        el.textContent = text;
        el.style.opacity = '1';
        if (toastTimer) clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { el.style.opacity = '0'; }, 2200);
    }

    function fallbackCopy(text, channelId, label) {
        const copyLabel = label || '启动协议';
        try {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
            showToast(`${copyLabel}已复制 · CH-${channelId}`);
        } catch {
            showToast('复制失败，请手动复制');
        }
    }

    function formatPresenceAge(ms) {
        if (!Number.isFinite(ms) || ms < 0) return '';
        const s = Math.floor(ms / 1000);
        if (s < 60) return s + ' 秒前';
        const m = Math.floor(s / 60);
        if (m < 60) return m + ' 分钟前';
        const h = Math.floor(m / 60);
        if (h < 24) return h + ' 小时前';
        const d = Math.floor(h / 24);
        return d + ' 天前';
    }

    function renderAiPresence() {
        if (!elAiPresence || !elAiPresenceText) return;
        const info = channelInfoMap[currentChannel] || null;
        elAiPresence.classList.remove('online', 'queued', 'unknown');
        if (!info) {
            elAiPresence.classList.add('unknown');
            elAiPresenceText.textContent = 'AI 未连接';
            elAiPresence.title = 'CH-' + currentChannel + ' 无状态，等待服务端推送…';
            return;
        }
        const online = info.online === true;
        const queueLen = Number(info.queueLength || 0);
        const lastSeen = Number(info.lastSeen || 0);
        if (online) {
            elAiPresence.classList.add('online');
            const age = lastSeen ? formatPresenceAge(Date.now() - lastSeen) : '';
            elAiPresenceText.textContent = 'AI 在线' + (age ? ' · ' + age : '');
            elAiPresence.title = 'CH-' + currentChannel + ' AI 正在轮询消息队列（保活中）';
        } else if (queueLen > 0) {
            elAiPresence.classList.add('queued');
            elAiPresenceText.textContent = '消息待取 (' + queueLen + ')';
            elAiPresence.title = 'CH-' + currentChannel + ' 有 ' + queueLen + ' 条消息待取，AI 未轮询\n请确认该通道 Cursor Agent 正在运行';
        } else {
            elAiPresence.classList.add('unknown');
            elAiPresenceText.textContent = 'AI 未连接';
            elAiPresence.title = 'CH-' + currentChannel + ' 未检测到 AI 心跳\n请打开该通道的 Cursor Agent 并粘贴启动协议';
        }
    }

    function getMentionKeyword() {
        if (!elInput) return null;
        const value = elInput.value || '';
        const cursor = Number.isFinite(elInput.selectionStart) ? elInput.selectionStart : value.length;
        let start = cursor - 1;
        while (start >= 0 && value[start] !== '@' && value[start] !== '\n' && value[start] !== ' ') start--;
        if (start >= 0 && value[start] === '@') {
            return { start, end: cursor, keyword: value.slice(start + 1, cursor) };
        }
        return null;
    }

    function getMentionCandidates(keyword) {
        const lower = String(keyword || '').trim().toLowerCase();
        const items = [];
        const push = (item) => {
            if (!item || !item.insertText) return;
            const haystack = [item.label, item.detail, item.insertText, item.path, item.channelId]
                .map(value => String(value || '').toLowerCase())
                .join(' ');
            if (lower && haystack.indexOf(lower) < 0) return;
            items.push(item);
        };
        const group = getActiveTeamGroup();
        if (currentViewMode === 'team' && group) {
            push({ kind: 'all', label: '全体成员', detail: '投递给当前群聊全部在线成员', insertText: '@全体成员 ' });
            const groupIds = new Set(Array.isArray(group.channelIds) ? group.channelIds.map(String) : []);
            const agents = teamSnapshot && Array.isArray(teamSnapshot.agents) ? teamSnapshot.agents : [];
            const roles = teamSnapshot && Array.isArray(teamSnapshot.roles) ? teamSnapshot.roles : [];
            const roleSeen = new Set();
            for (const agent of agents) {
                if (!groupIds.has(String(agent.channelId))) continue;
                const roleName = getTeamRoleName(agent.channelId);
                push({
                    kind: 'agent',
                    label: 'CH-' + agent.channelId,
                    detail: roleName + (agent.status === 'online' ? ' · 已接入' : ' · 未接入'),
                    insertText: '@CH-' + agent.channelId + ' ',
                    channelId: String(agent.channelId || '')
                });
                const role = roles.find(item => String(item.roleId || '') === String(agent.roleId || ''));
                if (role && !roleSeen.has(role.name)) {
                    roleSeen.add(role.name);
                    push({
                        kind: 'agent',
                        label: role.name,
                        detail: '按角色投递',
                        insertText: '@' + role.name + ' '
                    });
                }
            }
        }
        const mentions = teamSnapshot && Array.isArray(teamSnapshot.mentions) ? teamSnapshot.mentions : [];
        for (const item of mentions) {
            if (item && (item.kind === 'file' || item.kind === 'folder')) {
                push({
                    kind: item.kind,
                    label: item.label || item.path || item.insertText,
                    detail: item.detail || (item.kind === 'folder' ? '工作区文件夹' : '工作区文件'),
                    insertText: item.insertText || ('@' + (item.path || item.label) + ' '),
                    path: item.path || item.label || ''
                });
            }
        }
        return items.slice(0, 30);
    }

    function renderMentionDropdown() {
        if (!elMentionDropdown) return;
        if (!mentionResults.length) {
            elMentionDropdown.innerHTML = '<div class="mention-empty">未找到可引用内容</div>';
            elMentionDropdown.classList.remove('hidden');
            return;
        }
        const iconMap = { all: '全', agent: 'AI', folder: '夹', file: '文' };
        elMentionDropdown.innerHTML = mentionResults.map((item, index) =>
            '<button class="mention-item' + (index === mentionIndex ? ' active' : '') + '" type="button" data-index="' + index + '">' +
            '<span class="mention-icon">' + escapeHtml(iconMap[item.kind] || '@') + '</span>' +
            '<span class="mention-main">' +
            '<span class="mention-label">' + escapeHtml(item.label || item.insertText || '') + '</span>' +
            '<span class="mention-detail">' + escapeHtml(item.detail || item.path || '') + '</span>' +
            '</span>' +
            '</button>'
        ).join('');
        elMentionDropdown.classList.remove('hidden');
        elMentionDropdown.querySelectorAll('.mention-item').forEach(item => {
            item.addEventListener('click', () => selectMentionResult(Number(item.dataset.index || 0)));
        });
    }

    function hideMentionDropdown() {
        if (elMentionDropdown) elMentionDropdown.classList.add('hidden');
        mentionResults = [];
        mentionIndex = -1;
    }

    function selectMentionResult(index) {
        const item = mentionResults[index];
        const atInfo = getMentionKeyword();
        if (!item || !atInfo || !elInput) return;
        const value = elInput.value || '';
        const insertText = item.insertText || '';
        elInput.value = value.slice(0, atInfo.start) + insertText + value.slice(atInfo.end);
        const next = atInfo.start + insertText.length;
        elInput.focus();
        elInput.selectionStart = elInput.selectionEnd = next;
        hideMentionDropdown();
    }

    function updateMentionDropdownSoon() {
        if (mentionDebounce) clearTimeout(mentionDebounce);
        const atInfo = getMentionKeyword();
        if (!atInfo) {
            hideMentionDropdown();
            return;
        }
        mentionDebounce = setTimeout(() => {
            mentionResults = getMentionCandidates(atInfo.keyword);
            mentionIndex = mentionResults.length ? 0 : -1;
            renderMentionDropdown();
        }, 80);
    }

    // ── User actions ──
    async function submit() {
        const text = elInput.value.trim();
        if (!text && pendingAttachments.length === 0) return;
        if (wsState !== 'open') {
            appendHistory({
                id: genId(),
                type: 'error',
                channelId: currentChannel,
                text: '未连接服务端，请等待重连后重试',
                timestamp: Date.now()
            });
            return;
        }
        if (currentViewMode === 'team') {
            const group = getActiveTeamGroup();
            if (teamSnapshot && teamSnapshot.enabled === false) {
                showToast('多 Agent 群聊已关闭');
                return;
            }
            if (!group) {
                showToast('暂无可用群聊');
                return;
            }
            if (!text) return;
            if (pendingAttachments.length > 0) {
                showToast('浏览器群聊暂不支持附件，请在通道模式发送附件');
                return;
            }
            const ok = send({ type: 'sendTeamMessage', groupId: group.groupId, text });
            if (ok) {
                elInput.value = '';
                hideMentionDropdown();
                showToast('群聊消息已发送');
                setTimeout(() => send({ type: 'getTeamSnapshot', groupId: group.groupId }), 300);
            }
            return;
        }
        // 上一轮还没跑完：不直接上主对话流，先暂存到指令台上方排队，等这一轮收尾自动送出
        if (isChannelBusy(currentChannel)) {
            const queued = enqueueOutbox(currentChannel, text, pendingAttachments);
            if (!queued) return;
            elInput.value = '';
            pendingAttachments = [];
            renderAttachments();
            hideMentionDropdown();
            showToast('已排队，等上一条执行完成后自动发送');
            return;
        }

        const attachments = pendingAttachments.slice();
        elInput.value = '';
        pendingAttachments = [];
        renderAttachments();
        const sent = await dispatchMessage(currentChannel, text, attachments);
        if (!sent) {
            // 发送失败时把内容退回输入框，避免用户白打一遍
            elInput.value = text;
            pendingAttachments = attachments;
            renderAttachments();
        }
    }

    /**
     * 真正把一条消息发给服务端并上主对话流。
     * 只负责「上传附件 → submit → 落 history → 起思考气泡」，不碰输入框状态。
     */
    async function dispatchMessage(channelId, text, attachments) {
        const ch = String(channelId || currentChannel);
        const list = Array.isArray(attachments) ? attachments : [];
        let filePaths = [];
        let imagePaths = [];
        if (list.length > 0) {
            setSendButtonBusy(true, '上传附件…');
            try {
                const uploaded = await uploadAttachments(list);
                for (let i = 0; i < uploaded.length; i++) {
                    if (list[i].type === 'image') imagePaths.push(uploaded[i]);
                    else filePaths.push(uploaded[i]);
                }
            } catch (e) {
                setSendButtonBusy(false);
                appendHistory({
                    id: genId(),
                    type: 'error',
                    channelId: ch,
                    text: '附件上传失败：' + (e && e.message ? e.message : String(e)),
                    timestamp: Date.now()
                });
                return false;
            }
            setSendButtonBusy(false);
        }

        const payload = { type: 'submit', channelId: ch, text: text };
        if (filePaths.length) payload.file_paths = filePaths;
        if (imagePaths.length) payload.image_paths = imagePaths;
        if (!send(payload)) {
            appendHistory({
                id: genId(),
                type: 'error',
                channelId: ch,
                text: '发送失败（WebSocket 未就绪）',
                timestamp: Date.now()
            });
            return false;
        }

        const displayText = composeUserDisplayText(text, filePaths, imagePaths);
        // 先记住上一轮用户消息，避免 append 后 latest 变成新消息导致旧 Thought 锚错/消失
        const prevUserId = latestUserIdForChannel(ch);
        const liveBeforeSend = streamMap[ch];
        const prevAnchor = (liveBeforeSend && liveBeforeSend.anchorUserId) || prevUserId || null;
        appendHistory({
            id: genId(),
            type: 'user',
            channelId: ch,
            text: displayText,
            timestamp: Date.now(),
            filePaths: filePaths.slice(),
            imagePaths: imagePaths.slice()
        });
        // 新一轮：把上一轮未完成时间线固化到「旧用户消息」后面，保留 Thought for Xs / 折叠链路
        if (liveBeforeSend && !liveBeforeSend.done) {
            collapseAgentTimeline(ch, { afterId: prevAnchor });
        }
        // 显示思考气泡（贴在最新用户消息下方，真实 Thought 到达后会被时间线替换）
        showThinking(ch);
        return true;
    }

    function composeUserDisplayText(text, filePaths, imagePaths) {
        const lines = [];
        if (text) lines.push(text);
        if (Array.isArray(filePaths) && filePaths.length) {
            lines.push('[文件] ' + filePaths.map(p => p.split(/[\\/]/).pop()).join('、'));
        }
        if (Array.isArray(imagePaths) && imagePaths.length) {
            lines.push('[图片] ' + imagePaths.map(p => p.split(/[\\/]/).pop()).join('、'));
        }
        return lines.join('\n');
    }

    function setSendButtonBusy(busy, label) {
        if (!elSend) return;
        if (busy) {
            elSend.dataset.origText = elSend.dataset.origText || elSend.textContent;
            elSend.disabled = true;
            elSend.textContent = label || '发送中…';
        } else {
            elSend.disabled = false;
            if (elSend.dataset.origText) {
                elSend.textContent = elSend.dataset.origText;
                delete elSend.dataset.origText;
            }
        }
    }

    // ── 排队待发消息（指令台上方暂存区）──
    function loadOutbox() {
        try {
            const raw = localStorage.getItem(STORAGE_OUTBOX);
            if (!raw) return {};
            const obj = JSON.parse(raw);
            if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return {};
            const out = {};
            for (const key of Object.keys(obj)) {
                if (!Array.isArray(obj[key])) continue;
                const items = obj[key]
                    .filter(it => it && typeof it === 'object' && typeof it.text === 'string')
                    .slice(0, MAX_OUTBOX_PER_CHANNEL)
                    .map(it => ({
                        id: String(it.id || genId()),
                        text: it.text,
                        attachments: Array.isArray(it.attachments) ? it.attachments : [],
                        createdAt: Number(it.createdAt) || Date.now()
                    }));
                if (items.length) out[String(key)] = items;
            }
            return out;
        } catch {
            return {};
        }
    }

    function saveOutbox() {
        try {
            localStorage.setItem(STORAGE_OUTBOX, JSON.stringify(outboxMap));
        } catch {
            // 附件 base64 可能撑爆配额：退化成只留文本，至少排队内容不丢
            try {
                const lite = {};
                for (const key of Object.keys(outboxMap)) {
                    lite[key] = (outboxMap[key] || []).map(it => ({
                        id: it.id,
                        text: it.text,
                        attachments: [],
                        createdAt: it.createdAt
                    }));
                }
                localStorage.setItem(STORAGE_OUTBOX, JSON.stringify(lite));
            } catch { /* ignore */ }
        }
    }

    function getOutbox(channelId) {
        const key = String(channelId || '');
        if (!Array.isArray(outboxMap[key])) outboxMap[key] = [];
        return outboxMap[key];
    }

    function outboxLength(channelId) {
        const key = String(channelId || '');
        return Array.isArray(outboxMap[key]) ? outboxMap[key].length : 0;
    }

    /**
     * 该通道是否还在跑上一轮：有未收尾的时间线，或最近一条本地消息是 user 且还没等到 reply。
     * 刷新页面后时间线/思考气泡都没了，只剩历史里一条永远等不到回复的 user 消息，
     * 所以超过 OUTBOX_BUSY_STALE_MS 的「悬空轮次」按空闲处理，避免排队卡死。
     */
    function isChannelBusy(channelId) {
        const key = String(channelId || '');
        const live = streamMap[key];
        if (live && !live.done) return true;
        if (thinkingMap[key]) return true;
        for (let i = history.length - 1; i >= 0; i--) {
            const h = history[i];
            if (!h) continue;
            if (h.channelId && String(h.channelId) !== key) continue;
            if (h.type === 'reply') return false;
            if (h.type === 'user') {
                const ts = typeof h.timestamp === 'number' ? h.timestamp : Date.parse(String(h.timestamp || '')) || 0;
                return !ts || (Date.now() - ts) < OUTBOX_BUSY_STALE_MS;
            }
        }
        return false;
    }

    function enqueueOutbox(channelId, text, attachments) {
        const queue = getOutbox(channelId);
        if (queue.length >= MAX_OUTBOX_PER_CHANNEL) {
            showToast('排队已满（' + MAX_OUTBOX_PER_CHANNEL + ' 条），请先发送或删除部分消息');
            return null;
        }
        const item = {
            id: genId(),
            text: String(text || ''),
            attachments: Array.isArray(attachments) ? attachments.slice() : [],
            createdAt: Date.now()
        };
        queue.push(item);
        saveOutbox();
        renderOutbox();
        renderChannelList();
        return item;
    }

    function removeOutboxItem(channelId, itemId) {
        const queue = getOutbox(channelId);
        const idx = queue.findIndex(it => it.id === itemId);
        if (idx < 0) return null;
        const [item] = queue.splice(idx, 1);
        if (!queue.length) delete outboxMap[String(channelId)];
        if (outboxEditingId === itemId) outboxEditingId = '';
        saveOutbox();
        renderOutbox();
        renderChannelList();
        return item;
    }

    function moveOutboxItem(channelId, itemId, delta) {
        const queue = getOutbox(channelId);
        const idx = queue.findIndex(it => it.id === itemId);
        const next = idx + delta;
        if (idx < 0 || next < 0 || next >= queue.length) return;
        const [item] = queue.splice(idx, 1);
        queue.splice(next, 0, item);
        saveOutbox();
        renderOutbox();
    }

    function updateOutboxItemText(channelId, itemId, text) {
        const queue = getOutbox(channelId);
        const item = queue.find(it => it.id === itemId);
        if (!item) return;
        const trimmed = String(text || '').trim();
        if (!trimmed && !(item.attachments && item.attachments.length)) {
            removeOutboxItem(channelId, itemId);
            showToast('内容为空，该排队消息已删除');
            return;
        }
        item.text = trimmed;
        saveOutbox();
    }

    function renderOutbox() {
        if (!elOutboxRow || !elOutboxList) return;
        const queue = currentViewMode === 'team' ? [] : getOutbox(currentChannel);
        if (!queue.length) {
            elOutboxRow.classList.add('hidden');
            elOutboxList.innerHTML = '';
            outboxEditingId = '';
            return;
        }
        elOutboxRow.classList.remove('hidden');
        if (elOutboxCount) elOutboxCount.textContent = String(queue.length);
        updateOutboxHint();
        elOutboxList.innerHTML = queue.map((item, i) => {
            const editing = item.id === outboxEditingId;
            const attach = (item.attachments || []).length
                ? '<div class="outbox-item-attach">' +
                    escapeHtml((item.attachments || []).map(a => (a.type === 'image' ? '🖼 ' : '📎 ') + a.name).join('、')) +
                  '</div>'
                : '';
            const body = editing
                ? '<textarea class="outbox-item-edit" rows="3">' + escapeHtml(item.text) + '</textarea>' +
                  '<div class="outbox-edit-actions">' +
                      '<button class="outbox-mini-btn primary" type="button" data-act="save">保存</button>' +
                      '<button class="outbox-mini-btn" type="button" data-act="cancel">取消</button>' +
                  '</div>'
                : '<div class="outbox-item-text" title="双击编辑">' + escapeHtml(item.text) + '</div>' + attach;
            const actions = editing
                ? ''
                : '<div class="outbox-item-actions">' +
                      '<button class="outbox-icon-btn" type="button" data-act="up" title="上移">↑</button>' +
                      '<button class="outbox-icon-btn" type="button" data-act="down" title="下移">↓</button>' +
                      '<button class="outbox-icon-btn" type="button" data-act="edit" title="编辑">✎</button>' +
                      '<button class="outbox-icon-btn" type="button" data-act="send" title="立即发送这一条">↥</button>' +
                      '<button class="outbox-icon-btn danger" type="button" data-act="del" title="删除">×</button>' +
                  '</div>';
            return '<div class="outbox-item' + (editing ? ' editing' : '') + '" data-id="' + escapeHtml(item.id) + '">' +
                '<span class="outbox-item-index">' + (i + 1) + '</span>' +
                '<div class="outbox-item-body">' + body + '</div>' +
                actions +
            '</div>';
        }).join('');
        if (outboxEditingId) {
            const ta = elOutboxList.querySelector('.outbox-item.editing .outbox-item-edit');
            if (ta) {
                ta.focus();
                ta.selectionStart = ta.selectionEnd = ta.value.length;
            }
        }
    }

    /** 只刷提示文案，避免整块重渲染打断正在编辑的输入框 */
    function updateOutboxHint() {
        if (!elOutboxHint || !elOutboxRow || elOutboxRow.classList.contains('hidden')) return;
        elOutboxHint.textContent = isChannelBusy(currentChannel)
            ? 'CH-' + currentChannel + ' 执行中 · 上一条完成后自动发送'
            : 'CH-' + currentChannel + ' 空闲 · 即将自动发送';
    }

    function handleOutboxListClick(ev) {
        const btn = ev.target.closest('button[data-act]');
        const row = ev.target.closest('.outbox-item');
        if (!row) return;
        const itemId = row.dataset.id || '';
        if (!btn) {
            if (ev.detail === 2 && ev.target.closest('.outbox-item-text')) {
                outboxEditingId = itemId;
                renderOutbox();
            }
            return;
        }
        const act = btn.dataset.act;
        if (act === 'up') moveOutboxItem(currentChannel, itemId, -1);
        else if (act === 'down') moveOutboxItem(currentChannel, itemId, 1);
        else if (act === 'edit') { outboxEditingId = itemId; renderOutbox(); }
        else if (act === 'del') removeOutboxItem(currentChannel, itemId);
        else if (act === 'send') sendOutboxItemNow(currentChannel, itemId);
        else if (act === 'cancel') { outboxEditingId = ''; renderOutbox(); }
        else if (act === 'save') {
            const ta = row.querySelector('.outbox-item-edit');
            updateOutboxItemText(currentChannel, itemId, ta ? ta.value : '');
            outboxEditingId = '';
            renderOutbox();
        }
    }

    function handleOutboxListKeydown(ev) {
        const ta = ev.target.closest('.outbox-item-edit');
        if (!ta) return;
        const row = ev.target.closest('.outbox-item');
        if (!row) return;
        if (ev.key === 'Escape') {
            ev.preventDefault();
            outboxEditingId = '';
            renderOutbox();
            return;
        }
        // 编辑框里 Enter 保存，Shift+Enter 换行
        if (ev.key === 'Enter' && !ev.shiftKey) {
            ev.preventDefault();
            updateOutboxItemText(currentChannel, row.dataset.id || '', ta.value);
            outboxEditingId = '';
            renderOutbox();
        }
    }

    async function sendOutboxItemNow(channelId, itemId) {
        if (outboxDispatching) return;
        if (wsState !== 'open') { showToast('未连接服务端，稍后重试'); return; }
        const queue = getOutbox(channelId);
        const item = queue.find(it => it.id === itemId);
        if (!item) return;
        outboxDispatching = true;
        try {
            const ok = await dispatchMessage(channelId, item.text, item.attachments);
            if (ok) removeOutboxItem(channelId, itemId);
        } finally {
            outboxDispatching = false;
        }
        renderOutbox();
    }

    /** 上一条收尾后把队首消息「弹」上主对话流 */
    function scheduleOutboxFlush(channelId, delayMs) {
        if (channelId && !outboxLength(channelId)) {
            renderOutbox();
            return;
        }
        if (outboxFlushTimer) clearTimeout(outboxFlushTimer);
        outboxFlushTimer = setTimeout(() => {
            outboxFlushTimer = null;
            flushOutbox();
        }, typeof delayMs === 'number' ? delayMs : OUTBOX_FLUSH_DELAY_MS);
        renderOutbox();
    }

    async function flushOutbox() {
        if (outboxDispatching || wsState !== 'open') return;
        const channels = Object.keys(outboxMap).filter(ch => outboxLength(ch) > 0);
        if (!channels.length) return;
        outboxDispatching = true;
        try {
            for (const ch of channels) {
                if (isChannelBusy(ch)) continue;
                const queue = getOutbox(ch);
                const item = queue[0];
                if (!item) continue;
                if (item.id === outboxEditingId) continue; // 正在编辑的不抢跑
                const ok = await dispatchMessage(ch, item.text, item.attachments);
                if (ok) removeOutboxItem(ch, item.id);
            }
        } finally {
            outboxDispatching = false;
        }
        renderOutbox();
        renderChannelList();
    }

    function clearLocalHistory() {
        if (!confirm('清空当前通道（CH-' + currentChannel + '）的浏览器本地历史？\n\n注意：这只清浏览器的显示记录，不影响 Cursor 插件或 MCP 队列。')) return;
        history = history.filter(h => h.channelId !== currentChannel);
        for (const fp of Array.from(seenReplies)) {
            if (fp.startsWith(currentChannel + '|')) seenReplies.delete(fp);
        }
        saveHistory();
        renderedHistoryFingerprint = '';
        if (doneTimelines[currentChannel]) delete doneTimelines[currentChannel];
        if (streamMap[currentChannel]) delete streamMap[currentChannel];
        clearDraftReply(currentChannel);
        saveDoneTimelines();
        renderMessages();
        showToast('本地历史已清空');
    }

    function clearRemoteChannel() {
        if (!confirm('清空当前通道（CH-' + currentChannel + '）在 MCP 服务端的消息队列和最新 AI 回复？\n\n这会同步影响 Cursor 插件端。')) return;
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        send({ type: 'clearChannel', channelId: currentChannel });
    }

    function defaultRecoveryTransferTarget() {
        for (let i = 1; i <= channelCount; i++) {
            const id = String(i);
            const info = channelInfoMap[id] || {};
            if (id !== String(currentChannel) && info.waitingActive) return id;
        }
        return '';
    }

    function setRecoveryTransferOpen(open) {
        if (!elRecoveryTransferModal) return;
        elRecoveryTransferModal.classList.toggle('hidden', !open);
        if (open) {
            recoveryTransferMode = 'current';
            recoveryTransferTargetId = defaultRecoveryTransferTarget();
            renderRecoveryTransferModal();
        }
    }

    function renderRecoveryTransferModal() {
        if (!elRecoveryTransferModal) return;
        if (elRecoveryTransferSource) {
            elRecoveryTransferSource.textContent = 'CH-' + currentChannel;
        }
        elRecoveryTransferModal.querySelectorAll('input[name="recovery-transfer-mode"]').forEach((el) => {
            el.checked = el.value === recoveryTransferMode;
        });
        const isTransfer = recoveryTransferMode === 'transfer';
        if (elRecoveryTransferTargetSection) {
            elRecoveryTransferTargetSection.classList.toggle('hidden', !isTransfer);
        }
        if (elRecoveryTransferTargetList) {
            const items = [];
            for (let i = 1; i <= channelCount; i++) {
                const id = String(i);
                if (id === String(currentChannel)) continue;
                const info = channelInfoMap[id] || {};
                if (!info.waitingActive) continue;
                const meta = ['待命中'];
                if ((info.queueLength || 0) > 0) meta.push('队列 ' + info.queueLength);
                items.push(
                    '<button type="button" class="recovery-target-item' + (id === recoveryTransferTargetId ? ' active' : '') + '" data-target-channel="' + id + '">' +
                        '<span class="recovery-target-main">' +
                            '<span class="recovery-target-title">CH-' + id + '</span>' +
                            '<span class="recovery-target-meta">' + escapeHtml(meta.join(' · ')) + '</span>' +
                        '</span>' +
                    '</button>'
                );
            }
            elRecoveryTransferTargetList.innerHTML = items.length > 0
                ? items.join('')
                : '<div class="history-item"><div class="history-item-text" style="opacity:0.6">暂无已接入 Cursor 待命的其他通道</div></div>';
            elRecoveryTransferTargetList.querySelectorAll('[data-target-channel]').forEach((el) => {
                el.addEventListener('click', () => {
                    recoveryTransferTargetId = String(el.getAttribute('data-target-channel') || '');
                    renderRecoveryTransferModal();
                });
            });
        }
        if (elBtnRecoveryTransferConfirm) {
            elBtnRecoveryTransferConfirm.textContent = isTransfer ? '开始接管' : '开始恢复';
            elBtnRecoveryTransferConfirm.disabled = isTransfer && !recoveryTransferTargetId;
        }
    }

    function copyStartPromptForCurrent() {
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        send({ type: 'getStartPrompt', channelId: currentChannel });
    }

    function sendStartPromptForCurrent() {
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        if (elBtnSendStartPrompt) elBtnSendStartPrompt.classList.add('is-busy');
        send({ type: 'sendStartPrompt', channelId: currentChannel });
        showToast('正在投递开场白');
    }

    function restoreRecoveryContextForCurrent() {
        setRecoveryTransferOpen(true);
    }

    function getTeamGroups() {
        return teamSnapshot && Array.isArray(teamSnapshot.groups) ? teamSnapshot.groups : [];
    }

    function getActiveTeamGroup() {
        const groups = getTeamGroups();
        return groups.find(g => g.groupId === activeTeamGroupId) || groups[0] || null;
    }

    function getTeamRecoveryStatus() {
        return teamSnapshot && teamSnapshot.recoveryStatus && teamSnapshot.recoveryStatus.groupId === activeTeamGroupId
            ? teamSnapshot.recoveryStatus
            : null;
    }

    function getTeamRecoveryMembers() {
        const status = getTeamRecoveryStatus();
        return status && Array.isArray(status.members) ? status.members : [];
    }

    function getTeamTakeoverTargets() {
        const group = getActiveTeamGroup();
        const groupIds = new Set(group && Array.isArray(group.channelIds) ? group.channelIds.map(String) : []);
        const agents = teamSnapshot && Array.isArray(teamSnapshot.agents) ? teamSnapshot.agents : [];
        return agents.filter(agent => agent && agent.status === 'online' && !groupIds.has(String(agent.channelId)));
    }

    function getTeamRoles() {
        return teamSnapshot && Array.isArray(teamSnapshot.roles) ? teamSnapshot.roles : [];
    }

    function getOnlineTeamAgents() {
        const agents = teamSnapshot && Array.isArray(teamSnapshot.agents) ? teamSnapshot.agents : [];
        return agents.filter(agent => agent && agent.status === 'online');
    }

    function getRoleById(roleId) {
        return getTeamRoles().find(role => String(role.roleId || '') === String(roleId || '')) || null;
    }

    function defaultRoleIdForAgent(agent) {
        const roles = getTeamRoles();
        if (agent && getRoleById(agent.roleId)) return String(agent.roleId);
        return roles[0] ? String(roles[0].roleId || '') : '';
    }

    function openTeamGroupModal() {
        if (!elTeamGroupModal) return;
        if (teamSnapshot && teamSnapshot.enabled === false) {
            showToast('多 Agent 群聊已关闭，开启后才能创建群聊');
            return;
        }
        if (wsState !== 'open') {
            showToast('未连接服务端');
            return;
        }
        if (elTeamGroupName) elTeamGroupName.value = '';
        if (elTeamGroupGoal) elTeamGroupGoal.value = '';
        renderTeamGroupModal();
        elTeamGroupModal.classList.remove('hidden');
        setTimeout(() => { if (elTeamGroupName) elTeamGroupName.focus(); }, 0);
    }

    function closeTeamGroupModal() {
        if (elTeamGroupModal) elTeamGroupModal.classList.add('hidden');
    }

    function renderTeamGroupModal() {
        if (!elTeamGroupMemberList) return;
        const agents = getOnlineTeamAgents();
        const roles = getTeamRoles();
        if (!agents.length) {
            elTeamGroupMemberList.innerHTML = '<div class="history-empty">暂无已接入 Cursor 的通道，请先在 Cursor 对话窗口激活 MCP。</div>';
            if (elBtnTeamGroupConfirm) elBtnTeamGroupConfirm.disabled = true;
            return;
        }
        const roleOptions = (selectedRoleId) => roles.map(role => {
            const roleId = String(role.roleId || '');
            return '<option value="' + escapeHtml(roleId) + '"' + (roleId === selectedRoleId ? ' selected' : '') + '>' + escapeHtml(role.name || roleId) + '</option>';
        }).join('');
        elTeamGroupMemberList.innerHTML = agents.map((agent, index) => {
            const channelId = String(agent.channelId || '');
            const selectedRoleId = defaultRoleIdForAgent(agent);
            const role = getRoleById(selectedRoleId);
            return '<label class="team-group-member-row">' +
                '<input class="team-group-member-check" type="checkbox" value="' + escapeHtml(channelId) + '"' + (index < Math.min(3, agents.length) ? ' checked' : '') + ' />' +
                '<span class="team-group-member-main">' +
                    '<span class="team-group-member-head">CH-' + escapeHtml(channelId) + '<small>已接入 Cursor</small></span>' +
                    '<select class="team-group-member-role" data-channel-id="' + escapeHtml(channelId) + '">' + roleOptions(selectedRoleId) + '</select>' +
                    '<textarea class="team-group-member-rule" data-channel-id="' + escapeHtml(channelId) + '" rows="3" placeholder="可编辑该成员的群内身份规则">' + escapeHtml(role ? (role.description || '') : '') + '</textarea>' +
                '</span>' +
            '</label>';
        }).join('');
        elTeamGroupMemberList.querySelectorAll('.team-group-member-role').forEach(select => {
            select.addEventListener('change', () => {
                const role = getRoleById(select.value);
                const row = select.closest('.team-group-member-row');
                const textarea = row ? row.querySelector('.team-group-member-rule') : null;
                if (textarea) textarea.value = role ? (role.description || '') : '';
            });
        });
        if (elBtnTeamGroupConfirm) elBtnTeamGroupConfirm.disabled = false;
    }

    function collectTeamGroupMembers() {
        if (!elTeamGroupMemberList) return [];
        return Array.from(elTeamGroupMemberList.querySelectorAll('.team-group-member-row')).map(row => {
            const checkbox = row.querySelector('.team-group-member-check');
            if (!checkbox || !checkbox.checked) return null;
            const select = row.querySelector('.team-group-member-role');
            const textarea = row.querySelector('.team-group-member-rule');
            return {
                channelId: String(checkbox.value || ''),
                roleId: select ? String(select.value || '') : '',
                ruleOverride: textarea ? String(textarea.value || '') : ''
            };
        }).filter(Boolean);
    }

    function confirmCreateTeamGroup() {
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        const name = (elTeamGroupName && elTeamGroupName.value || '').trim();
        const goal = (elTeamGroupGoal && elTeamGroupGoal.value || '').trim();
        const members = collectTeamGroupMembers();
        if (!name) {
            showToast('请填写群名称');
            if (elTeamGroupName) elTeamGroupName.focus();
            return;
        }
        if (!members.length) {
            showToast('请至少选择一个已接入 Cursor 的通道');
            return;
        }
        send({ type: 'createTeamGroup', name, goal, members });
        showToast('正在创建群聊');
    }

    function deleteActiveTeamGroup() {
        const group = getActiveTeamGroup();
        if (!group) { showToast('暂无可删除的群聊'); return; }
        if (group.groupId === 'default' || (group.metadata && group.metadata.system === true)) {
            showToast('默认协作群不能删除');
            return;
        }
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        const name = group.name || group.groupId;
        if (!confirm('确认删除群聊「' + name + '」吗？该群聊的群内记录也会被移除。')) return;
        send({ type: 'deleteTeamGroup', groupId: group.groupId });
        showToast('正在删除群聊');
    }

    function renderTeamRecoveryCard() {
        const groups = getTeamGroups();
        if (elTeamRecoveryGroup) {
            elTeamRecoveryGroup.innerHTML = groups.length
                ? groups.map(group => '<option value="' + escapeHtml(group.groupId) + '"' + (group.groupId === activeTeamGroupId ? ' selected' : '') + '>' + escapeHtml(group.name || group.groupId) + '</option>').join('')
                : '<option value="">暂无群组</option>';
            elTeamRecoveryGroup.disabled = groups.length === 0;
        }
        const status = getTeamRecoveryStatus();
        if (elTeamRecoveryStatus) {
            elTeamRecoveryStatus.textContent = status && status.message
                ? status.message
                : (groups.length ? '选择群组后查看恢复状态' : '暂无群组，请先在群聊工作台创建群聊');
        }
        if (elBtnTeamRecovery) {
            elBtnTeamRecovery.disabled = groups.length === 0;
        }
    }

    function renderTeamToolbar() {
        const groups = getTeamGroups();
        const group = getActiveTeamGroup();
        if (elTeamGroupSelect) {
            elTeamGroupSelect.innerHTML = groups.length
                ? groups.map(group => '<option value="' + escapeHtml(group.groupId) + '"' + (group.groupId === activeTeamGroupId ? ' selected' : '') + '>' + escapeHtml(group.name || group.groupId) + '</option>').join('')
                : '<option value="">暂无群组</option>';
            elTeamGroupSelect.disabled = groups.length === 0;
        }
        if (elTeamToolbarStatus) {
            const count = group && Array.isArray(group.channelIds) ? group.channelIds.length : 0;
            const recoveryStatus = getTeamRecoveryStatus();
            elTeamToolbarStatus.textContent = teamSnapshot && teamSnapshot.enabled === false
                ? '多 Agent 群聊已关闭'
                : group
                ? ('当前群聊 · ' + count + ' 个成员')
                : '暂无群组，请先在 Cursor 群聊工作台创建';
        }
        if (elBtnTeamCreate) {
            elBtnTeamCreate.disabled = wsState !== 'open' || (teamSnapshot && teamSnapshot.enabled === false);
        }
        if (elBtnTeamDelete) {
            const canDelete = !!group && group.groupId !== 'default' && !(group.metadata && group.metadata.system === true);
            elBtnTeamDelete.disabled = wsState !== 'open' || !canDelete || (teamSnapshot && teamSnapshot.enabled === false);
        }
        syncTeamRecoveryToolbarState();
    }

    function syncTeamRecoveryToolbarState() {
        const group = getActiveTeamGroup();
        const recoveryStatus = getTeamRecoveryStatus();
        if (elTeamToolbarStatus && group && recoveryStatus && recoveryStatus.message) {
            elTeamToolbarStatus.textContent = recoveryStatus.message;
        }
        if (elBtnTeamRecovery) {
            elBtnTeamRecovery.disabled = !group || (teamSnapshot && teamSnapshot.enabled === false);
            elBtnTeamRecovery.title = group ? '恢复当前选中的群聊上下文' : '请先选择群聊';
        }
    }

    function openTeamRecoveryModal() {
        if (!elTeamRecoveryModal) return;
        const group = getActiveTeamGroup();
        if (!group) { showToast('暂无群组'); return; }
        teamRecoveryMode = 'restore';
        const readyMembers = getTeamRecoveryMembers().filter(member => member.ready);
        teamRecoveryModeratorId = readyMembers[0] ? String(readyMembers[0].channelId) : '';
        const members = getTeamRecoveryMembers();
        const preferredSource = members.find(member => !member.ready) || members[0];
        teamRecoverySourceId = preferredSource ? String(preferredSource.channelId) : '';
        const targets = getTeamTakeoverTargets();
        teamRecoveryTargetId = targets[0] ? String(targets[0].channelId) : '';
        renderTeamRecoveryModal();
        elTeamRecoveryModal.classList.remove('hidden');
    }

    function closeTeamRecoveryModal() {
        if (elTeamRecoveryModal) elTeamRecoveryModal.classList.add('hidden');
    }

    function renderTeamRecoveryModal() {
        if (!elTeamRecoveryModal) return;
        const group = getActiveTeamGroup();
        const status = getTeamRecoveryStatus();
        const members = getTeamRecoveryMembers();
        const readyMembers = members.filter(member => member.ready);
        const targets = getTeamTakeoverTargets();
        if (elTeamRecoveryModalSubtitle) {
            elTeamRecoveryModalSubtitle.textContent = group
                ? ((group.name || group.groupId) + ' · ' + ((status && status.message) || '正在检查群聊成员结构'))
                : '请先选择群组';
        }
        if (elTeamRecoveryRestoreDesc) {
            elTeamRecoveryRestoreDesc.textContent = status && status.message
                ? status.message
                : '当前群聊不能直接恢复。';
        }
        elTeamRecoveryModal.querySelectorAll('input[name="team-recovery-mode"]').forEach(input => {
            input.checked = input.value === teamRecoveryMode;
        });
        if (elTeamRecoveryModeratorPanel) {
            elTeamRecoveryModeratorPanel.classList.toggle('hidden', teamRecoveryMode !== 'restore');
        }
        if (elTeamRecoveryModeratorList) {
            if (!readyMembers.length) {
                elTeamRecoveryModeratorList.textContent = '暂无已接入的群成员，当前不能恢复。';
            } else {
                const selectedMember = readyMembers.find(member => String(member.channelId) === String(teamRecoveryModeratorId)) || readyMembers[0];
                teamRecoveryModeratorId = String(selectedMember.channelId || '');
                elTeamRecoveryModeratorList.textContent = '将自动由 CH-' + selectedMember.channelId + ' · ' + (selectedMember.roleName || 'Agent') + ' 发出可见确认，其它成员静默吸收。';
            }
        }
        if (elTeamRecoveryTakeoverPanel) {
            elTeamRecoveryTakeoverPanel.classList.toggle('hidden', teamRecoveryMode !== 'takeover');
        }
        if (elTeamRecoveryTakeoverGrid) {
            const hasTakeoverChoices = members.length > 0 && targets.length > 0;
            elTeamRecoveryTakeoverGrid.classList.toggle('hidden', !hasTakeoverChoices);
        }
        if (elTeamRecoverySource) {
            elTeamRecoverySource.innerHTML = members.length
                ? members.map(member => '<option value="' + escapeHtml(member.channelId) + '"' + (String(member.channelId) === teamRecoverySourceId ? ' selected' : '') + '>CH-' + escapeHtml(member.channelId) + ' · ' + escapeHtml(member.roleName || 'Agent') + (member.ready ? ' · 已接入' : ' · 未接入') + '</option>').join('')
                : '<option value="">暂无成员</option>';
            elTeamRecoverySource.disabled = members.length === 0;
        }
        if (elTeamRecoveryTarget) {
            elTeamRecoveryTarget.innerHTML = targets.length
                ? targets.map(agent => '<option value="' + escapeHtml(agent.channelId) + '"' + (String(agent.channelId) === teamRecoveryTargetId ? ' selected' : '') + '>CH-' + escapeHtml(agent.channelId) + ' · 已接入</option>').join('')
                : '<option value="">暂无可接管通道</option>';
            elTeamRecoveryTarget.disabled = targets.length === 0;
        }
        if (elBtnTeamRecoveryConfirm) {
            const canRestore = !!(status && status.canRestoreCurrent && teamRecoveryModeratorId);
            const canTakeover = !!(teamRecoverySourceId && teamRecoveryTargetId);
            elBtnTeamRecoveryConfirm.textContent = teamRecoveryMode === 'takeover' ? '开始接管' : '开始恢复';
            elBtnTeamRecoveryConfirm.disabled = teamRecoveryMode === 'takeover' ? !canTakeover : !canRestore;
        }
    }

    function confirmTeamRecovery() {
        const group = getActiveTeamGroup();
        if (!group) { showToast('暂无群组'); return; }
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        if (teamRecoveryMode === 'takeover') {
            if (!teamRecoverySourceId || !teamRecoveryTargetId) {
                showToast('请选择来源成员和接管目标');
                return;
            }
            send({
                type: 'takeoverTeamGroupMember',
                groupId: group.groupId,
                sourceChannelId: teamRecoverySourceId,
                targetChannelId: teamRecoveryTargetId,
                maxChars: 18000
            });
            showToast('正在投递成员接管上下文');
            return;
        }
        const status = getTeamRecoveryStatus();
        if (!status || !status.canRestoreCurrent) {
            showToast((status && status.message) || '当前群聊不能直接恢复');
            return;
        }
        send({
            type: 'restoreTeamGroupContext',
            groupId: group.groupId,
            moderatorChannelId: teamRecoveryModeratorId,
            maxChars: 16000
        });
        showToast('正在投递群聊恢复上下文');
    }

    function doAddChannel() {
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        send({ type: 'addChannel' });
    }

    function doRemoveChannel() {
        if (channelCount <= 1) { showToast('至少保留 1 个通道'); return; }
        if (!confirm('删除最后一个通道？该通道的队列和回复会被清空。')) return;
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        send({ type: 'removeChannel' });
    }

    function genId() {
        return Date.now() + '-' + Math.random().toString(36).slice(2, 8);
    }

    // ── Thinking Bubble（无 agent_stream 时的兜底）──
    function showThinking(channelId) {
        if (!channelId) return;
        // 已有流式时间线时不再挂旧气泡
        if (streamMap[channelId] && !streamMap[channelId].collapsed) return;
        if (!elThinkingTemplate) return;
        hideThinking(channelId);
        const frag = elThinkingTemplate.content.cloneNode(true);
        const node = frag.querySelector('.msg-thinking');
        if (!node) return;
        node.dataset.channelId = String(channelId);
        const closeBtn = node.querySelector('.thinking-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => hideThinking(channelId));
        }
        const entry = {
            node,
            startedAt: Date.now(),
            degradeTimer: setTimeout(() => degradeThinking(channelId), THINKING_DEGRADE_MS)
        };
        thinkingMap[channelId] = entry;
        if (channelId === currentChannel) {
            elMessages.appendChild(node);
            maybeScrollMessages();
        }
    }
    function hideThinking(channelId) {
        const entry = thinkingMap[channelId];
        if (!entry) return;
        if (entry.degradeTimer) clearTimeout(entry.degradeTimer);
        if (entry.node && entry.node.parentNode) {
            entry.node.parentNode.removeChild(entry.node);
        }
        delete thinkingMap[channelId];
    }
    function degradeThinking(channelId) {
        const entry = thinkingMap[channelId];
        if (!entry || !entry.node) return;
        entry.node.classList.add('degraded');
        const textEl = entry.node.querySelector('.thinking-text');
        if (textEl) textEl.textContent = 'AI 响应延迟，可能需要手动唤醒';
    }

    // ── Agent Stream Timeline ──
    const FILE_PREVIEW_LINES = 4;
    const PREVIEW_AUTO_COLLAPSE_MS = 5000;

    function isNearBottom() {
        if (!elMessages) return true;
        const gap = elMessages.scrollHeight - elMessages.scrollTop - elMessages.clientHeight;
        return gap < 80;
    }
    function forceScrollMessages() {
        if (!elMessages) return;
        elMessages.scrollTop = elMessages.scrollHeight;
        const st = streamMap[currentChannel];
        if (st) {
            const btn = st.node.querySelector('.agent-scroll-bottom');
            if (btn) btn.classList.add('hidden');
        }
    }
    function maybeScrollMessages() {
        if (!elMessages) return;
        const st = streamMap[currentChannel];
        // 生成中强制贴底；空闲时仅在接近底部时跟随
        if ((st && !st.done) || isNearBottom()) {
            forceScrollMessages();
        } else if (st) {
            const btn = st.node.querySelector('.agent-scroll-bottom');
            if (btn) btn.classList.remove('hidden');
        }
    }
    function scheduleAutoCollapse(target, collapseFn, ms) {
        if (!target || typeof collapseFn !== 'function') return;
        if (target._collapseTimer) clearTimeout(target._collapseTimer);
        target._collapseTimer = setTimeout(() => {
            try {
                collapseFn();
            } catch {
                /* ignore */
            }
            forceScrollMessages();
        }, ms == null ? PREVIEW_AUTO_COLLAPSE_MS : ms);
    }

    function bindTimelineFoldToggle(node, entryState) {
        if (!node) return;
        const summaryBtn = node.querySelector('.agent-timeline-toggle');
        const root = node.querySelector('.agent-timeline');
        const summaryText = node.querySelector('.summary-text');
        if (!summaryBtn || !root || summaryBtn.dataset.foldBound === '1') return;
        summaryBtn.dataset.foldBound = '1';
        summaryBtn.addEventListener('click', () => {
            const next = !root.classList.contains('collapsed');
            root.classList.toggle('collapsed', next);
            if (entryState) entryState.collapsed = next;
            if (summaryText) {
                const base = String(summaryText.textContent || '')
                    .replace(/\s*（点击展开）\s*$/, '')
                    .replace(/\s*（点击折叠）\s*$/, '')
                    .trim();
                summaryText.textContent = base + (next ? ' （点击展开）' : ' （点击折叠）');
            }
        });
    }

    function syncStreamSeqFromServer(streamSeqByChannel) {
        if (!streamSeqByChannel || typeof streamSeqByChannel !== 'object') return;
        let changed = false;
        Object.keys(streamSeqByChannel).forEach((ch) => {
            const serverSeq = Number(streamSeqByChannel[ch] || 0);
            const localSeq = Number(channelLastSeq[ch] || 0);
            // 服务端重启后 seq 从 0/小值重计；若本地仍握着旧高位，后续事件会被全部丢掉
            if (serverSeq < localSeq) {
                channelLastSeq[ch] = serverSeq;
                changed = true;
            } else if (serverSeq > localSeq) {
                // 服务端领先：断线期间（或后台通道）漏了事件，从环形缓冲补拉。
                // resume 后 localSeq 会追平，不会每次 status 都重发。
                requestStreamResume(ch);
            }
        });
        // 本地有、服务端没有的通道，也重置（服务端全新启动）
        Object.keys(channelLastSeq).forEach((ch) => {
            if (!Object.prototype.hasOwnProperty.call(streamSeqByChannel, ch)) {
                if (Number(channelLastSeq[ch] || 0) > 0 && Object.keys(streamSeqByChannel).length === 0) {
                    channelLastSeq[ch] = 0;
                    changed = true;
                }
            }
        });
        if (changed) {
            try {
                localStorage.setItem(STORAGE_STREAM_SEQ, JSON.stringify(channelLastSeq || {}));
            } catch { /* ignore */ }
        }
    }

    function ensureAgentTimeline(channelId) {
        const id = String(channelId);
        const currentUserId = latestUserIdForChannel(id);
        // 复用已有时间线：节点可能被 renderMessages 的 innerHTML='' 卸出 DOM（后台通道尤其如此），
        // 此时绝不能新建空时间线冲掉已累积的思考/工具步骤，只需在当前通道下把节点重新挂回即可。
        // 但若已换到新的用户消息轮次，必须先固化上一轮，避免旧 Thought 跟着跑到新消息下面甚至被合并掉。
        const existing = streamMap[id];
        if (existing && existing.node && existing.itemsEl) {
            const existingAnchor = existing.anchorUserId || null;
            const newTurn = !!(currentUserId && existingAnchor && existingAnchor !== currentUserId);
            // 已收尾却仍留在 streamMap，或已进入新用户轮次：固化旧块并新建
            if (existing.done || (newTurn && !existing.done)) {
                if (!existing.done) {
                    collapseAgentTimeline(id, { afterId: existingAnchor });
                } else {
                    delete streamMap[id];
                }
                // 继续往下创建新时间线
            } else {
                if (!existing.anchorUserId && currentUserId) existing.anchorUserId = currentUserId;
                if (id === currentChannel && !existing.node.isConnected) {
                    renderMessages(true);
                    maybeScrollMessages();
                }
                return existing;
            }
        }
        if (!elTimelineTemplate) return null;
        hideThinking(id);
        const frag = elTimelineTemplate.content.cloneNode(true);
        const node = frag.querySelector('.msg-agent-timeline');
        if (!node) return null;
        node.dataset.channelId = id;
        const itemsEl = node.querySelector('.agent-timeline-items');
        const summaryBtn = node.querySelector('.agent-timeline-toggle');
        const scrollBtn = node.querySelector('.agent-scroll-bottom');
        const entry = {
            node,
            itemsEl,
            startedAt: Date.now(),
            stepCount: 0,
            lastSeq: Number(channelLastSeq[id] || 0),
            collapsed: false,
            typing: null,
            omitCount: 0,
            anchorUserId: currentUserId || null
        };
        if (summaryBtn) {
            bindTimelineFoldToggle(node, entry);
        }
        if (scrollBtn) {
            scrollBtn.addEventListener('click', () => {
                elMessages.scrollTop = elMessages.scrollHeight;
                scrollBtn.classList.add('hidden');
            });
        }
        streamMap[id] = entry;
        if (id === currentChannel) {
            // 走统一布局：活跃时间线跟在当前轮用户消息后
            renderMessages(true);
            maybeScrollMessages();
        }
        return entry;
    }

    function isMetaStatusLine(line) {
        const t = String(line || '').trim();
        if (!t) return true;
        if (/keepalive/i.test(t)) return true;
        if (/保活|静默(重试|循环|调用|轮询)|不向用户展示/.test(t)) return true;
        if (/Request timed out|MCP error\s*-?32001|__TIMEOUT_RENEW__/i.test(t)) return true;
        if (/check_messages|check\s*接口|q[tw]wx-mcp-\d+/i.test(t)) return true;
        if (/持续(对话|重试|等待|轮询|接收)|在岗待命|超时后/.test(t)) return true;
        if (/等待(插件|消息|下一条)|消息到达|连接正常建立/.test(t)) return true;
        if (/我会继续循环|用户规则要求|尝试再次调用\s*check/i.test(t)) return true;
        if (/已进入持续对话|持续对话已接通|通道 MCP 未就绪/.test(t)) return true;
        if (/这是一条来自插件端的用户消息|持续对话协议|任务完成后必须立即调用/.test(t)) return true;
        if (/Generating…|Generating\.\.\./i.test(t)) return true;
        if (/^(你好|hi|hello|测试|ping|test|收到)[。.!！]?$/i.test(t)) return true;
        return false;
    }

    function stripMetaStatusLines(text) {
        return String(text || '')
            .split(/\r?\n/)
            .filter((line) => !isMetaStatusLine(line))
            .join('\n')
            .replace(/\n{3,}/g, '\n\n')
            .trim();
    }

    function isMetaStatusThinkingText(text) {
        const raw = String(text || '').trim();
        if (!raw) return true;
        const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
        if (!lines.length) return true;
        let metaCount = 0;
        for (let i = 0; i < lines.length; i++) {
            if (isMetaStatusLine(lines[i])) metaCount++;
        }
        if (metaCount === lines.length) return true;
        // 短文本里只要谈保活/check_messages 协议自述，整段丢弃
        if (
            raw.length < 800 &&
            metaCount > 0 &&
            /keepalive|保活|静默|check_messages|q[tw]wx-mcp|在岗待命|Request timed out|__TIMEOUT_RENEW__/i.test(raw)
        ) {
            return true;
        }
        if (metaCount / lines.length >= 0.6) return true;
        return false;
    }

    function normalizeThinkCmp(text) {
        return String(text || '')
            .replace(/\s+/g, '')
            .replace(/[`*_>#\-]/g, '')
            .slice(0, 240);
    }

    function isDuplicateOfLatestReply(channelId, text) {
        const key = String(channelId || currentChannel || '');
        const needle = normalizeThinkCmp(text);
        if (!needle || needle.length < 8) return false;
        for (let i = history.length - 1; i >= 0; i--) {
            const h = history[i];
            if (!h) continue;
            if (h.channelId && String(h.channelId) !== key) continue;
            if (h.type === 'reply') {
                const body = normalizeThinkCmp(h.text || '');
                if (!body) return false;
                if (body === needle || body.includes(needle) || needle.includes(body.slice(0, 80))) return true;
                return false;
            }
            if (h.type === 'user') return false;
        }
        return false;
    }

    function isGarbageThinkingText(text) {
        const t = String(text || '').trim();
        if (!t) return true;
        if (isMetaStatusThinkingText(t)) return true;
        if (/\bW?call-[a-f0-9-]{8,}/i.test(t)) return true;
        if (/\bfc[_-]?[a-f0-9-]{6,}/i.test(t)) return true;
        if (/Generating…|Generating\.\.\./i.test(t)) return true;
        if (/已完成\s*·/.test(t)) return true;
        // 整段几乎没有汉字且像 id
        const cjk = (t.match(/[\u4e00-\u9fff]/g) || []).length;
        if (cjk < 2 && (/^[\w\-]+$/.test(t) || (t.match(/-/g) || []).length >= 2)) return true;
        return false;
    }

    function sanitizeThinkingText(text) {
        const lines = String(text || '').split(/\r?\n/);
        const kept = lines.filter((line) => !isGarbageThinkingText(line) && !isMetaStatusLine(line));
        // 若过滤后太空，再尝试去掉行内 call-id
        let out = kept.join('\n').replace(/\bW?call-[a-f0-9-]{8,}/gi, '').replace(/\bfc[_-]?[a-f0-9-]{6,}/gi, '');
        out = out.replace(/[ \t]{2,}/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
        out = stripMetaStatusLines(out);
        if (isMetaStatusThinkingText(out)) return '';
        return out;
    }

    function finishOpenThinkingBlocks(entry, opts) {
        if (!entry || !entry.itemsEl) return;
        if (entry.typing && entry.typing.flush) entry.typing.flush();
        const autoCollapse = !opts || opts.autoCollapse !== false;
        const collapseDelay = opts && opts.collapseDelayMs != null ? opts.collapseDelayMs : 450;
        const onlyThoughtId = opts && opts.thoughtId ? String(opts.thoughtId) : '';
        const onlySource = opts && opts.source ? String(opts.source) : '';
        entry.itemsEl.querySelectorAll('.atl-thinking.live').forEach((el) => {
            if (onlyThoughtId && String(el.dataset.thoughtId || '') !== onlyThoughtId) return;
            if (onlySource && String(el.dataset.source || '') !== onlySource) return;
            el.classList.remove('live');
            const started = Number(el.dataset.startedAt || 0) || Date.now();
            const secs = Math.max(1, Math.round((Date.now() - started) / 1000));
            const headLabel = el.querySelector('.atl-thinking-label');
            if (headLabel) {
                headLabel.classList.remove('atl-shimmer');
                headLabel.textContent = 'Thought for ' + secs + 's';
            }
            const textEl = el.querySelector('.atl-thinking-text');
            let text = String(el._thinkingBuffer || (textEl && textEl.textContent) || '').trim();
            text = sanitizeThinkingText(text);
            el._thinkingBuffer = text;
            if (textEl) textEl.textContent = text;
            if (!text) {
                el.remove();
                return;
            }
            // 完成后：标题改为 Thought for Xs，正文默认折叠（点击可再展开）
            el.classList.add('atl-narrative');
            el.classList.add('collapsed-done');
            const cursor = el.querySelector('.atl-cursor');
            if (cursor) cursor.remove();
            const pulse = el.querySelector('.atl-thinking-pulse');
            if (pulse) pulse.remove();
            if (autoCollapse) {
                // 稍留展开态让用户瞥见末段，再自动折叠只留 Thought 描述
                el.classList.add('open');
                scheduleAutoCollapse(el, () => {
                    el.classList.remove('open');
                }, collapseDelay);
            } else {
                el.classList.remove('open');
            }
        });
        entry.itemsEl.querySelectorAll('.atl-term:not(.done)').forEach((el) => {
            el.classList.add('done');
            const st = el.querySelector('.atl-term-status');
            if (st) {
                st.textContent = '✓';
                st.classList.remove('atl-spin');
            }
        });
    }

    function findThinkingLive(entry, thoughtId, source) {
        if (!entry || !entry.itemsEl) return null;
        const tid = String(thoughtId || '');
        const src = String(source || '');
        const nodes = entry.itemsEl.querySelectorAll('.atl-thinking.live');
        if (tid) {
            for (let i = nodes.length - 1; i >= 0; i--) {
                if (String(nodes[i].dataset.thoughtId || '') === tid) return nodes[i];
            }
        }
        if (src) {
            for (let i = nodes.length - 1; i >= 0; i--) {
                if (String(nodes[i].dataset.source || '') === src) return nodes[i];
            }
        }
        return null;
    }

    function hasActiveThoughtStream(entry, channelId) {
        if (entry && entry.itemsEl && entry.itemsEl.querySelector('.atl-thinking.live[data-source="thought_stream"]')) {
            return true;
        }
        const ts = Number(thoughtStreamActiveAt[String(channelId)] || 0);
        return ts > 0 && (Date.now() - ts) < 120000;
    }

    /** 本轮用户消息之后是否已有最终回复（用于丢弃迟到的 Thinking，避免再次卡在 Thinking） */
    function isChannelTurnComplete(channelId) {
        const key = String(channelId || '');
        for (let i = history.length - 1; i >= 0; i--) {
            const h = history[i];
            if (!h) continue;
            if (h.channelId && String(h.channelId) !== key) continue;
            if (h.type === 'reply') return true;
            if (h.type === 'user') return false;
        }
        return false;
    }

    function latestReplyTimestamp(channelId) {
        const key = String(channelId || '');
        for (let i = history.length - 1; i >= 0; i--) {
            const h = history[i];
            if (!h) continue;
            if (h.channelId && String(h.channelId) !== key) continue;
            if (h.type === 'reply') {
                const t = typeof h.timestamp === 'number' ? h.timestamp : Date.parse(String(h.timestamp || '')) || 0;
                return t;
            }
            if (h.type === 'user') return 0;
        }
        return 0;
    }

    function handleAgentStreamEvent(msg) {
        if (!msg || msg.type !== 'agent_stream') return;
        const channelId = String(msg.channelId || '');
        if (!channelId) return;

        const seq = Number(msg.seq || 0);
        const seen = Number(channelLastSeq[channelId] || 0);
        if (seq && seq <= seen) {
            // 服务端 seq 重置（重启）后，首批新事件可能小于本地 seen；用时间戳兜底接纳
            const ts = Number(msg.ts || 0);
            const fresh = ts > 0 && (Date.now() - ts) < 120000;
            if (fresh && seq + 50 < seen) {
                channelLastSeq[channelId] = Math.max(0, seq - 1);
            } else {
                return;
            }
        }
        if (seq) {
            channelLastSeq[channelId] = seq;
            try {
                localStorage.setItem(STORAGE_STREAM_SEQ, JSON.stringify(channelLastSeq || {}));
            } catch { /* ignore */ }
        }

        // 最终回复已上屏且活跃时间线已收尾时，忽略「早于最后一条回复」的迟到事件。
        // 注意：只要最新一条是 user（新一轮已开始），绝不能早退，否则第二轮起只剩「AI 正在思考…」。
        const kind = String((msg.event && msg.event.kind) || '');
        if (!streamMap[channelId] && isChannelTurnComplete(channelId)) {
            const lastReplyTs = latestReplyTimestamp(channelId);
            const evTs = Number(msg.ts || 0);
            const isLate = !evTs || !lastReplyTs || evTs <= lastReplyTs;
            if (
                isLate &&
                (kind === 'thinking' ||
                    kind === 'tool_call' ||
                    kind === 'reply_done' ||
                    kind === 'session_bound' ||
                    kind === 'assistant_narration')
            ) {
                return;
            }
        }

        // CLI 中间话：直接镜像到回复草稿，不依赖思维链时间线
        if (kind === 'assistant_narration') {
            applyAssistantNarration(channelId, (msg && msg.event) || {});
            return;
        }

        const entry = ensureAgentTimeline(channelId);
        if (!entry) return;
        if (seq) entry.lastSeq = Math.max(Number(entry.lastSeq || 0), seq);

        applyAgentStreamEvent(channelId, entry, msg);
    }

    function clearDraftReply(channelId) {
        const key = String(channelId || '');
        const draft = draftReplyMap[key];
        if (!draft) return;
        if (draft.node && draft.node.parentNode) draft.node.parentNode.removeChild(draft.node);
        delete draftReplyMap[key];
    }

    function applyAssistantNarration(channelId, event) {
        const key = String(channelId || '');
        if (!key) return;
        let text = String((event && event.text) || '').trim();
        if (!text || isMetaStatusThinkingText(text)) return;
        // 与已定稿回复完全重复时不再刷草稿
        if (isDuplicateOfLatestReply(key, text)) return;

        let draft = draftReplyMap[key];
        if (!draft || !draft.node) {
            const wrap = document.createElement('div');
            wrap.className = 'msg msg-reply msg-reply-draft';
            wrap.dataset.channelId = key;
            wrap.innerHTML =
                '<div class="msg-bubble"></div>' +
                '<div class="msg-meta">' +
                '<span class="msg-channel">CH-' + escapeHtml(key) + ' · 生成中</span>' +
                '<span class="msg-time">实时同步 CLI</span>' +
                '</div>';
            draft = { node: wrap, text: '', updatedAt: Date.now() };
            draftReplyMap[key] = draft;
        }

        const streaming = !!(event && event.streaming);
        if (streaming) {
            // 增量：若已是前缀则续写，否则追加段落
            if (text.startsWith(draft.text)) {
                draft.text = text;
            } else if (draft.text && text.startsWith(draft.text.slice(-20))) {
                draft.text += text;
            } else if (!draft.text) {
                draft.text = text;
            } else if (!draft.text.endsWith(text)) {
                draft.text = draft.text + (draft.text.endsWith('\n') ? '' : '\n') + text;
            }
        } else {
            if (!draft.text) {
                draft.text = text;
            } else if (text.startsWith(draft.text) || text.length >= draft.text.length + 10) {
                // 完整行到达：用更长文本覆盖，或追加新段
                if (text.startsWith(draft.text)) draft.text = text;
                else if (!draft.text.includes(text)) {
                    draft.text = draft.text + (draft.text.endsWith('\n') ? '' : '\n\n') + text;
                }
            } else if (!draft.text.includes(text)) {
                draft.text = draft.text + (draft.text.endsWith('\n') ? '' : '\n\n') + text;
            }
        }
        draft.updatedAt = Date.now();
        const bubble = draft.node.querySelector('.msg-bubble');
        if (bubble) bubble.innerHTML = renderMarkdown(draft.text);
        if (key === String(currentChannel) && currentViewMode !== 'team') {
            if (!draft.node.isConnected) renderMessages(true);
            else maybeScrollMessages();
        }
    }

    function applyAgentStreamEvent(channelId, entry, msg) {
        const event = (msg && msg.event) || {};

        if (event.kind === 'session_bound') {
            const bound = entry.node.querySelector('.agent-timeline-bound');
            if (bound) {
                bound.textContent = '已连接 Agent 会话';
                bound.classList.remove('hidden');
            }
            return;
        }

        if (event.kind === 'reply_done') {
            // 若仍有假动画，先冲掉再收尾；真流式 delta 已即时上屏
            if (entry.typing && entry.typing.flush) entry.typing.flush();
            collapseAgentTimeline(channelId, { collapseDelayMs: THOUGHT_DONE_AUTO_COLLAPSE_MS });
            return;
        }

        if (event.kind === 'thinking') {
            const status = String(event.status || event.phase || '').toLowerCase();
            const source = String(event.source || event.summary || '');
            const thoughtId = String(event.thoughtId || event.chunkId || '');
            let text = String(event.text || '');

            // thought_stream 活跃时，屏蔽 chat-store/cdp 混入同一框，避免多轮计时/正文串台
            if (source && source !== 'thought_stream' && hasActiveThoughtStream(entry, channelId)) {
                return;
            }
            if (source === 'thought_stream') {
                thoughtStreamActiveAt[String(channelId)] = Date.now();
            }

            // 保活文案、协议提示、与最终回复重复的正文，一律不进思维链
            if (text && (isMetaStatusThinkingText(text) || isDuplicateOfLatestReply(channelId, text))) {
                if (status !== 'start' && status !== 'done' && status !== 'error') return;
                text = '';
            }

            // chat-store / 脏增量：丢掉 call-id 等乱码碎片
            if (text && (source === 'chat-store' || event.summary === 'chat-store')) {
                text = sanitizeThinkingText(text);
                if (!text || isGarbageThinkingText(text)) {
                    if (status !== 'start' && status !== 'done' && status !== 'error') return;
                    text = '';
                }
            } else if (text && isGarbageThinkingText(text)) {
                if (status !== 'start' && status !== 'done' && status !== 'error') return;
                text = '';
            }

            // thought_stream：按 thoughtId 独立开框/收尾，互不污染
            if (status === 'start') {
                // 新开一段前，只收尾同 source 的旧 live 框（保留其他源 / 已折叠历史）
                finishOpenThinkingBlocks(entry, {
                    autoCollapse: true,
                    collapseDelayMs: 0,
                    source: source || 'thought_stream'
                });
                appendThinkingItem(entry, text, {
                    instant: true,
                    thoughtId: thoughtId || undefined,
                    source: source || 'thought_stream'
                });
                entry.stepCount++;
            } else if (status === 'done' || status === 'error') {
                let live = findThinkingLive(entry, thoughtId, source || 'thought_stream');
                if (live && text) {
                    const sanitized = sanitizeThinkingText(text);
                    const prev = String(live._thinkingBuffer || '').trim();
                    if (sanitized && (!prev || sanitized.length >= prev.length)) {
                        live._thinkingBuffer = sanitized;
                        const textEl = live.querySelector('.atl-thinking-text');
                        if (textEl) textEl.textContent = sanitized;
                    }
                } else if (!live && text) {
                    appendThinkingItem(entry, text, {
                        instant: true,
                        thoughtId: thoughtId || undefined,
                        source: source || 'thought_stream'
                    });
                    entry.stepCount++;
                    live = findThinkingLive(entry, thoughtId, source || 'thought_stream');
                }
                finishOpenThinkingBlocks(entry, {
                    autoCollapse: true,
                    collapseDelayMs: 450,
                    thoughtId: thoughtId || (live && live.dataset.thoughtId) || '',
                    source: source || 'thought_stream'
                });
            } else {
                if (!text) return;
                let live = findThinkingLive(entry, thoughtId, source);
                // 无 thoughtId 时，绝不复用其他 source 的 live 框
                if (!live && !thoughtId && source) {
                    live = findThinkingLive(entry, '', source);
                }
                if (event.streaming) {
                    if (live) appendThinkingDelta(entry, live, text);
                    else {
                        appendThinkingItem(entry, text, {
                            instant: true,
                            thoughtId: thoughtId || undefined,
                            source: source || undefined
                        });
                        entry.stepCount++;
                    }
                } else if (live) {
                    extendThinkingBurst(entry, live, text);
                } else {
                    appendThinkingItem(entry, text, {
                        instant: false,
                        thoughtId: thoughtId || undefined,
                        source: source || undefined
                    });
                    entry.stepCount++;
                }
            }
        } else if (event.kind === 'tool_call') {
            // 工具打断：只收尾同时间线里仍 live 的思考，不影响已折叠历史 Thought
            finishOpenThinkingBlocks(entry, { autoCollapse: true, collapseDelayMs: 0 });
            appendToolItem(entry, event.tool || 'tool', event.summary || '', event.detail || null);
            entry.stepCount++;
        }

        while (entry.itemsEl.children.length > MAX_TIMELINE_ITEMS) {
            entry.itemsEl.removeChild(entry.itemsEl.firstChild);
            entry.omitCount++;
        }
        if (entry.omitCount > 0) {
            let omit = entry.itemsEl.querySelector('.agent-timeline-omit');
            if (!omit) {
                omit = document.createElement('div');
                omit.className = 'agent-timeline-omit';
                entry.itemsEl.insertBefore(omit, entry.itemsEl.firstChild);
            }
            omit.textContent = '已省略 ' + entry.omitCount + ' 条';
        }

        const gen = entry.node.querySelector('.agent-timeline-generating');
        if (gen) gen.classList.remove('hidden');
        entry.collapsed = false;
        const root = entry.node.querySelector('.agent-timeline');
        if (root) root.classList.remove('collapsed');
        const summary = entry.node.querySelector('.agent-timeline-summary');
        if (summary) summary.classList.add('hidden');

        if (String(channelId) === String(currentChannel)) forceScrollMessages();
    }

    function prefersReducedMotion() {
        return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    }

    function appendThinkingDelta(entry, wrap, delta) {
        const textEl = wrap.querySelector('.atl-thinking-text');
        if (!textEl) return;
        // 真流式增量：取消假动画，直接追加上屏
        if (entry.typing && entry.typing.cancel) entry.typing.cancel();
        entry.typing = null;
        entry.thinkingAnimating = false;
        wrap._thinkingBuffer = String(wrap._thinkingBuffer || textEl.textContent || '') + String(delta || '');
        textEl.textContent = wrap._thinkingBuffer;
        forceScrollMessages();
    }

    function startThinkingBurst(entry, wrap, textEl, getFullText, setFullText) {
        const reduced = prefersReducedMotion();
        const isBackground = !!(entry.node && entry.node.dataset && String(entry.node.dataset.channelId) !== String(currentChannel));
        if (reduced || isBackground) {
            textEl.textContent = getFullText();
            entry.thinkingAnimating = false;
            entry.typing = null;
            return;
        }

        if (entry.typing && entry.typing.cancel) entry.typing.cancel();

        let i = Math.min(textEl.textContent.length, getFullText().length);
        let timer = 0;
        entry.thinkingAnimating = true;

        const finish = () => {
            entry.typing = null;
            entry.thinkingAnimating = false;
        };

        const tick = () => {
            const fullText = getFullText();
            if (i >= fullText.length) {
                finish();
                return;
            }
            let burst = 2 + Math.floor(Math.random() * 8);
            const next = fullText.slice(i);
            const space = next.search(/[\s\n，。；！？、,.!?;:]/);
            if (space >= 0 && space < burst + 6) burst = Math.max(1, space + 1);
            i = Math.min(fullText.length, i + burst);
            textEl.textContent = fullText.slice(0, i);
            forceScrollMessages();
            const delay = 18 + Math.random() * 36;
            timer = setTimeout(tick, delay);
        };

        entry.typing = {
            cancel() {
                if (timer) clearTimeout(timer);
                timer = 0;
            },
            flush() {
                if (timer) clearTimeout(timer);
                timer = 0;
                textEl.textContent = getFullText();
                finish();
            },
            extend(more) {
                setFullText(getFullText() + (getFullText() && more ? '\n' : '') + more);
                entry.thinkingAnimating = true;
                if (!timer) timer = setTimeout(tick, 16);
            }
        };
        timer = setTimeout(tick, 16);
    }

    function appendThinkingItem(entry, fullText, opts) {
        const wrap = document.createElement('div');
        wrap.className = 'atl-item atl-thinking live open atl-enter';
        wrap.dataset.startedAt = String(Date.now());
        const thoughtId = opts && opts.thoughtId ? String(opts.thoughtId) : ('th_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 7));
        const source = opts && opts.source ? String(opts.source) : '';
        wrap.dataset.thoughtId = thoughtId;
        if (source) wrap.dataset.source = source;
        wrap.innerHTML =
            '<button type="button" class="atl-thinking-head">' +
            '<span class="chevron">›</span>' +
            '<span class="atl-thinking-label atl-shimmer">Thinking</span>' +
            '<span class="atl-thinking-pulse"></span>' +
            '</button>' +
            '<div class="atl-thinking-body"><span class="atl-thinking-text"></span><span class="atl-cursor">▍</span></div>';
        const head = wrap.querySelector('.atl-thinking-head');
        head.addEventListener('click', () => wrap.classList.toggle('open'));
        entry.itemsEl.appendChild(wrap);

        const textEl = wrap.querySelector('.atl-thinking-text');
        wrap._thinkingBuffer = String(fullText || '');
        if (opts && opts.instant) {
            textEl.textContent = wrap._thinkingBuffer;
            entry.thinkingAnimating = false;
            entry.typing = null;
            return;
        }
        startThinkingBurst(
            entry,
            wrap,
            textEl,
            () => wrap._thinkingBuffer || '',
            (v) => { wrap._thinkingBuffer = v; }
        );
    }

    function extendThinkingBurst(entry, wrap, moreText) {
        const more = String(moreText || '');
        if (!more) return;
        const textEl = wrap.querySelector('.atl-thinking-text');
        if (!textEl) return;
        if (entry.typing && typeof entry.typing.extend === 'function') {
            entry.typing.extend(more);
            return;
        }
        const prev = wrap._thinkingBuffer || textEl.textContent || '';
        wrap._thinkingBuffer = prev + (prev ? '\n' : '') + more;
        startThinkingBurst(
            entry,
            wrap,
            textEl,
            () => wrap._thinkingBuffer || '',
            (v) => { wrap._thinkingBuffer = v; }
        );
    }

    function toolVerb(tool, summary) {
        if (READ_TOOLS.has(tool)) {
            if (tool === 'Grep' || tool === 'fs_grep') return 'Searched ' + (summary ? '"' + summary + '"' : '');
            if (tool === 'Glob' || tool === 'fs_glob') return 'Globbed ' + (summary || '');
            return 'Read ' + (summary || tool);
        }
        if (tool.indexOf('mcp') >= 0 || summary.indexOf('qtwx-mcp') >= 0) {
            return 'Called ' + (summary || tool);
        }
        return tool + (summary ? ' · ' + summary : '');
    }

    function countLines(s) {
        if (!s) return 0;
        return String(s).split(/\r?\n/).length;
    }

    function shortPath(p) {
        const s = String(p || '');
        if (!s) return '';
        const parts = s.split(/[/\\]/);
        return parts.slice(-2).join('/') || s;
    }

    function buildDiffRows(detail) {
        const oldS = detail && detail.old_string != null ? String(detail.old_string) : '';
        const newS = detail && detail.new_string != null ? String(detail.new_string) : '';
        const contents = detail && detail.contents != null ? String(detail.contents) : '';
        const patch = detail && detail.patch != null ? String(detail.patch) : '';
        const rows = [];
        if (oldS || newS) {
            oldS.split(/\r?\n/).forEach((line) => rows.push({ cls: 'del', sign: '-', text: line }));
            newS.split(/\r?\n/).forEach((line) => rows.push({ cls: 'add', sign: '+', text: line }));
        } else if (contents) {
            contents.split(/\r?\n/).forEach((line) => rows.push({ cls: 'add', sign: '+', text: line }));
        } else if (patch) {
            patch.split(/\r?\n/).forEach((line) => {
                let cls = 'ctx';
                let sign = ' ';
                if (line.startsWith('+') && !line.startsWith('+++')) { cls = 'add'; sign = '+'; }
                else if (line.startsWith('-') && !line.startsWith('---')) { cls = 'del'; sign = '-'; }
                else if (line.startsWith('@@')) { cls = 'hunk'; sign = '@'; }
                rows.push({ cls, sign, text: line.replace(/^[-+ ]/, '') });
            });
        }
        return rows;
    }

    function buildDiffStats(detail) {
        const rows = buildDiffRows(detail);
        let del = 0;
        let add = 0;
        rows.forEach((r) => {
            if (r.cls === 'del') del++;
            if (r.cls === 'add') add++;
        });
        return { del, add, mode: rows.length ? 'diff' : 'none', rows };
    }

    function renderDiffRowsHtml(rows) {
        if (!rows || !rows.length) return '<div class="atl-diff-empty">暂无变更预览</div>';
        return '<div class="atl-diff-body">' + rows.map((r) =>
            '<div class="atl-diff-line ' + r.cls + '"><span class="atl-diff-sign">' + r.sign + '</span><span class="atl-diff-code">' + escapeHtml(r.text) + '</span></div>'
        ).join('') + '</div>';
    }

    function guessEditLineRange(detail, stats) {
        const start = Number(detail && detail.offset) || 0;
        if (start > 0) {
            const span = Math.max(stats.del, stats.add, 1);
            return start + '-' + (start + span - 1);
        }
        // 无 offset 时用变更行数作为展示范围提示
        if (stats.add || stats.del) {
            return '1-' + Math.max(stats.add, stats.del, 1);
        }
        return '';
    }

    function renderCodeLines(lines, startHint) {
        if (!lines || !lines.length) {
            return '<div class="atl-code-empty">无内容</div>';
        }
        return '<div class="atl-code-body">' + lines.map((row, idx) => {
            const n = row.n != null ? row.n : ((startHint || 1) + idx);
            return '<div class="atl-code-line"><span class="atl-code-ln">' + n + '</span><span class="atl-code-tx">' + escapeHtml(row.text == null ? row : row.text) + '</span></div>';
        }).join('') + '</div>';
    }

    async function fetchFilePreview(filePath, offset, limit) {
        const pathVal = String(filePath || '').trim();
        if (!pathVal) return { ok: false, message: '无文件路径' };

        // 先走 HTTP：旧服务未知 WS 类型时不会污染聊天区
        try {
            const q = new URLSearchParams();
            q.set('path', pathVal);
            if (offset) q.set('offset', String(offset));
            if (limit) q.set('limit', String(limit));
            const res = await fetch('/api/file-preview?' + q.toString());
            const ct = String(res.headers.get('content-type') || '');
            if (ct.includes('application/json')) {
                const data = await res.json();
                if (data && data.ok) return data;
                // JSON 但失败时，再尝试 WS（新服务）
                if (data && data.message && !/不在工作区|不存在/.test(String(data.message))) {
                    // continue to WS
                } else if (data) {
                    return data;
                }
            }
        } catch {
            /* fall through */
        }

        if (ws && ws.readyState === WebSocket.OPEN) {
            const requestId = 'fp_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
            const resultPromise = new Promise((resolve) => {
                const timer = setTimeout(() => {
                    pendingFilePreviews.delete(requestId);
                    resolve({ ok: false, message: '预览超时，请重启 qingtian-standalone 服务' });
                }, 4000);
                pendingFilePreviews.set(requestId, { resolve, reject: resolve, timer });
            });
            try {
                ws.send(JSON.stringify({
                    type: 'filePreview',
                    requestId,
                    path: pathVal,
                    offset: offset || 1,
                    limit: limit || FILE_PREVIEW_LINES
                }));
                return await resultPromise;
            } catch (e) {
                return { ok: false, message: (e && e.message) ? e.message : '加载失败' };
            }
        }

        return { ok: false, message: '预览接口不可用，请重启 qingtian-standalone 服务后再试' };
    }

    function isSearchTool(tool) {
        return tool === 'Grep' || tool === 'fs_grep' || tool === 'Glob' || tool === 'fs_glob' ||
            tool === 'SemanticSearch' || tool === 'rg';
    }

    function isReadExploreTool(tool) {
        return tool === 'Read' || tool === 'fs_read' || tool === 'fs_list' || tool === 'search_replace_read';
    }

    function isExploreTool(tool) {
        return isSearchTool(tool) || isReadExploreTool(tool) || READ_TOOLS.has(tool);
    }

    function exploredLabel(state) {
        const uniqFiles = (state.paths && state.paths.length)
            ? state.paths.length
            : (state.reads || 0);
        const parts = [];
        if (uniqFiles > 0) parts.push(uniqFiles + (uniqFiles === 1 ? ' file' : ' files'));
        if (state.searches > 0) parts.push(state.searches + (state.searches === 1 ? ' search' : ' searches'));
        if (!parts.length) parts.push((state.total || 0) + ' items');
        return 'Explored ' + parts.join(', ');
    }

    function looksLikeFilePath(text) {
        const t = String(text || '').trim();
        if (!t || t.length < 3 || t.length > 400) return false;
        if (/\s/.test(t)) return false;
        if (/[\\/]/.test(t)) return true;
        if (/\.(js|ts|tsx|jsx|py|go|rs|java|css|html|json|md|mjs|cjs)$/i.test(t)) return true;
        return false;
    }

    /** 把散落的路径小条收进 Explored，避免历史上出现一长串裸路径 */
    function absorbPathChipsIntoExplored(entry) {
        if (!entry || !entry.itemsEl) return;
        const children = Array.from(entry.itemsEl.children);
        let exploreWrap = null;
        children.forEach((el) => {
            if (el.classList.contains('atl-explored')) {
                exploreWrap = el;
                return;
            }
            if (el.classList.contains('atl-thinking') || el.classList.contains('atl-term') || el.querySelector('.atl-edit')) {
                exploreWrap = null; // 被思考/终端/编辑打断后开新组
                return;
            }
            const light = el.querySelector('.atl-light');
            if (!light) return;
            const raw = String(light.getAttribute('title') || light.textContent || '').trim();
            if (!looksLikeFilePath(raw)) return;
            if (!exploreWrap) {
                // 借用 appendExploredItem 的结构
                appendExploredItem(entry, 'Read', raw, { path: raw });
                exploreWrap = entry.itemsEl.querySelector('.atl-explored:last-of-type');
                // append 会加到末尾，挪到当前 chip 位置
                if (exploreWrap && el.parentNode) {
                    el.parentNode.insertBefore(exploreWrap, el);
                }
            } else {
                const list = exploreWrap.querySelector('.atl-explored-list');
                const paths = exploreWrap._explored.paths || (exploreWrap._explored.paths = []);
                const norm = raw.replace(/\\/g, '/');
                if (!paths.includes(norm)) {
                    paths.push(norm);
                    exploreWrap._explored.reads = (exploreWrap._explored.reads || 0) + 1;
                    exploreWrap._explored.total = (exploreWrap._explored.total || 0) + 1;
                    if (list) {
                        const row = document.createElement('div');
                        row.className = 'atl-explored-row';
                        row.innerHTML =
                            '<span class="atl-explored-kind">Read</span>' +
                            '<span class="atl-explored-path" title="' + escapeAttr(raw) + '">' +
                            escapeHtml(shortPath(raw) || raw) + '</span>';
                        list.appendChild(row);
                        list.classList.add('hidden');
                    }
                    updateExploredSummary(exploreWrap);
                }
            }
            el.remove();
        });
        mergeAdjacentExploredBlocks(entry);
    }

    function updateExploredSummary(wrap) {
        const state = wrap._explored || { reads: 0, searches: 0, total: 0 };
        const label = wrap.querySelector('.atl-explored-label');
        if (label) label.textContent = exploredLabel(state);
    }

    function mergeAdjacentExploredBlocks(entry) {
        if (!entry || !entry.itemsEl) return;
        const items = Array.from(entry.itemsEl.children);
        for (let i = 0; i < items.length - 1; i++) {
            const cur = items[i];
            const next = items[i + 1];
            if (!cur || !next) continue;
            if (!cur.classList.contains('atl-explored') || !next.classList.contains('atl-explored')) continue;
            const curList = cur.querySelector('.atl-explored-list');
            const nextList = next.querySelector('.atl-explored-list');
            if (!curList || !nextList) continue;
            const seen = new Set();
            curList.querySelectorAll('.atl-explored-path').forEach((p) => {
                seen.add(String(p.getAttribute('title') || p.textContent || '').trim());
            });
            Array.from(nextList.children).forEach((row) => {
                const pathEl = row.querySelector('.atl-explored-path');
                const key = String((pathEl && (pathEl.getAttribute('title') || pathEl.textContent)) || '').trim();
                if (key && seen.has(key)) return;
                if (key) seen.add(key);
                curList.appendChild(row);
            });
            const a = cur._explored || { reads: 0, searches: 0, total: 0, paths: [] };
            const b = next._explored || { reads: 0, searches: 0, total: 0, paths: [] };
            cur._explored = {
                reads: (a.reads || 0) + (b.reads || 0),
                searches: (a.searches || 0) + (b.searches || 0),
                total: (a.total || 0) + (b.total || 0),
                paths: Array.from(seen)
            };
            updateExploredSummary(cur);
            next.remove();
            items[i + 1] = null;
        }
    }

    function appendExploredItem(entry, tool, summary, detail) {
        const last = entry.itemsEl.lastElementChild;
        let wrap = last && last.classList.contains('atl-explored') ? last : null;
        if (!wrap) {
            wrap = document.createElement('div');
            wrap.className = 'atl-item atl-explored atl-enter';
            wrap._explored = { reads: 0, searches: 0, total: 0, paths: [] };
            wrap.innerHTML =
                '<div class="atl-explored-card atl-glass-chip">' +
                '<button type="button" class="atl-explored-head">' +
                '<span class="chevron">›</span>' +
                '<span class="atl-explored-label">Explored</span>' +
                '</button>' +
                '<div class="atl-explored-list hidden"></div>' +
                '</div>';
            const head = wrap.querySelector('.atl-explored-head');
            const list = wrap.querySelector('.atl-explored-list');
            head.addEventListener('click', () => {
                const open = list.classList.toggle('hidden') === false;
                wrap.querySelector('.atl-explored-card').classList.toggle('open', open);
            });
            entry.itemsEl.appendChild(wrap);
        }
        const d = detail && typeof detail === 'object' ? detail : {};
        const filePath = String(d.path || summary || '').trim();
        const normPath = filePath.replace(/\\/g, '/');
        // 连续重复读同一文件：只更新计数，不堆一排相同路径
        const paths = wrap._explored.paths || (wrap._explored.paths = []);
        if (normPath && paths.length && paths[paths.length - 1] === normPath) {
            wrap._explored.total++;
            if (isSearchTool(tool)) wrap._explored.searches++;
            else wrap._explored.reads++;
            updateExploredSummary(wrap);
            scheduleAutoCollapse(wrap, () => {
                const list = wrap.querySelector('.atl-explored-list');
                const card = wrap.querySelector('.atl-explored-card');
                if (list) list.classList.add('hidden');
                if (card) card.classList.remove('open');
            }, 1200);
            forceScrollMessages();
            return;
        }
        if (normPath) paths.push(normPath);

        const state = wrap._explored;
        state.total++;
        if (isSearchTool(tool)) state.searches++;
        else state.reads++;
        updateExploredSummary(wrap);

        const row = document.createElement('div');
        row.className = 'atl-explored-row';
        const start = Number(d.offset || d.previewStartLine || 0) || 0;
        const shown = Math.min(FILE_PREVIEW_LINES, Number(d.previewLimit) || FILE_PREVIEW_LINES);
        const range = start ? (String(start) + '-' + (start + shown - 1)) : '';
        row.innerHTML =
            '<span class="atl-explored-kind">' + escapeHtml(isSearchTool(tool) ? 'Search' : 'Read') + '</span>' +
            '<span class="atl-explored-path" title="' + escapeAttr(filePath) + '">' + escapeHtml(shortPath(filePath) || summary || tool) + '</span>' +
            (range ? '<span class="atl-explored-range">' + escapeHtml(range) + '</span>' : '') +
            '<button type="button" class="atl-explored-open">源码</button>';
        const list = wrap.querySelector('.atl-explored-list');
        list.appendChild(row);
        // 默认折叠：只显示「Explored N files」，点击再展开路径列表
        list.classList.add('hidden');
        const card = wrap.querySelector('.atl-explored-card');
        if (card) card.classList.remove('open');

        const panel = document.createElement('div');
        panel.className = 'atl-explored-preview';
        const previewLines = Array.isArray(d.previewLines)
            ? d.previewLines.slice(0, FILE_PREVIEW_LINES)
            : [];
        if (previewLines.length) {
            panel.innerHTML = renderCodeLines(previewLines, d.previewStartLine || start || 1);
            row.appendChild(panel);
            scheduleAutoCollapse(panel, () => {
                panel.classList.add('hidden');
            }, PREVIEW_AUTO_COLLAPSE_MS);
        } else {
            panel.classList.add('hidden');
            panel.innerHTML = '<div class="atl-code-loading">加载源码…</div>';
            row.appendChild(panel);
        }

        // 整块 Explored 稍晚折叠，只留标题
        scheduleAutoCollapse(wrap, () => {
            if (list) list.classList.add('hidden');
            if (card) card.classList.remove('open');
        }, PREVIEW_AUTO_COLLAPSE_MS + 1500);

        const btn = row.querySelector('.atl-explored-open');
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            if (previewLines.length && panel.dataset.loaded !== 'fetch') {
                panel.classList.toggle('hidden');
                if (!panel.classList.contains('hidden')) {
                    scheduleAutoCollapse(panel, () => panel.classList.add('hidden'), PREVIEW_AUTO_COLLAPSE_MS);
                }
                return;
            }
            panel.classList.remove('hidden');
            if (panel.dataset.loaded === '1' || panel.dataset.loaded === 'fetch') {
                if (panel.dataset.loaded === 'fetch') {
                    panel.classList.toggle('hidden');
                    if (!panel.classList.contains('hidden')) {
                        scheduleAutoCollapse(panel, () => panel.classList.add('hidden'), PREVIEW_AUTO_COLLAPSE_MS);
                    }
                }
                return;
            }
            if (!filePath) {
                panel.innerHTML = '<div class="atl-code-empty">无文件路径</div>';
                panel.dataset.loaded = '1';
                return;
            }
            try {
                const data = await fetchFilePreview(
                    filePath,
                    d.offset || d.previewStartLine || 1,
                    FILE_PREVIEW_LINES
                );
                if (!data || !data.ok) {
                    panel.innerHTML = '<div class="atl-code-empty">' + escapeHtml((data && data.message) || '读取失败') + '</div>';
                } else {
                    const lines = Array.isArray(data.lines) ? data.lines.slice(0, FILE_PREVIEW_LINES) : [];
                    panel.innerHTML = renderCodeLines(lines, data.startLine);
                }
                panel.dataset.loaded = 'fetch';
                scheduleAutoCollapse(panel, () => panel.classList.add('hidden'), PREVIEW_AUTO_COLLAPSE_MS);
            } catch (err) {
                panel.innerHTML = '<div class="atl-code-empty">' + escapeHtml((err && err.message) || '加载失败') + '</div>';
                panel.dataset.loaded = '1';
            }
        });
        forceScrollMessages();
    }

    function appendToolItem(entry, tool, summary, detail) {
        const d = detail && typeof detail === 'object' ? detail : {};
        const filePath = d.path || summary || '';
        const toolName = String(tool || '');
        if (toolName === 'record_reply' || toolName === 'team_reply_stream' || toolName === 'thought_stream') return;

        if (isExploreTool(tool) && !EDIT_TOOLS.has(tool) && !SHELL_TOOLS.has(tool)) {
            appendExploredItem(entry, tool, summary, d);
            return;
        }

        const wrap = document.createElement('div');
        wrap.className = 'atl-item atl-enter';

        if (SHELL_TOOLS.has(tool)) {
            wrap.innerHTML =
                '<div class="atl-term atl-glass-chip">' +
                '<div class="atl-term-row">' +
                '<span class="atl-term-status atl-spin">⟳</span>' +
                '<span class="atl-term-cmd"></span>' +
                '</div>' +
                '<button type="button" class="atl-term-expand hidden">展开</button>' +
                '</div>';
            const cmd = wrap.querySelector('.atl-term-cmd');
            cmd.textContent = '$ ' + (d.command || summary);
            const expand = wrap.querySelector('.atl-term-expand');
            requestAnimationFrame(() => {
                if (cmd.scrollHeight > cmd.clientHeight + 4) {
                    expand.classList.remove('hidden');
                    expand.addEventListener('click', () => {
                        cmd.classList.toggle('clamped');
                        expand.textContent = cmd.classList.contains('clamped') ? '展开' : '收起';
                    });
                }
            });
        } else if (EDIT_TOOLS.has(tool)) {
            const stats = buildDiffStats(d);
            const base = shortPath(filePath) || tool;
            const range = guessEditLineRange(d, stats);
            const previewRows = (stats.rows || []).slice(0, FILE_PREVIEW_LINES);
            const rest = Math.max(0, (stats.rows || []).length - FILE_PREVIEW_LINES);
            wrap.innerHTML =
                '<div class="atl-edit atl-card atl-glass-chip open">' +
                '<div class="atl-edit-title">' +
                '<span class="atl-edit-path" title="' + escapeAttr(filePath) + '">' + escapeHtml(base) + '</span>' +
                (range ? '<span class="atl-edit-range">:' + escapeHtml(range) + '</span>' : '') +
                '<span class="atl-edit-stats">' +
                (stats.del ? '<span class="atl-stat-del">-' + stats.del + '</span>' : '') +
                (stats.add ? '<span class="atl-stat-add">+' + stats.add + '</span>' : '') +
                '</span>' +
                '</div>' +
                '<div class="atl-edit-preview">' + renderDiffRowsHtml(previewRows) + '</div>' +
                (rest > 0
                    ? ('<button type="button" class="atl-edit-more">Expand ' + rest + ' lines</button>' +
                        '<div class="atl-edit-panel hidden">' + renderDiffRowsHtml(stats.rows) + '</div>')
                    : '') +
                '</div>';
            const more = wrap.querySelector('.atl-edit-more');
            const panel = wrap.querySelector('.atl-edit-panel');
            const preview = wrap.querySelector('.atl-edit-preview');
            const card = wrap.querySelector('.atl-edit');
            if (more && panel && preview) {
                more.addEventListener('click', () => {
                    const expanding = panel.classList.contains('hidden');
                    panel.classList.toggle('hidden', !expanding);
                    preview.classList.toggle('hidden', expanding);
                    more.textContent = expanding ? 'Collapse' : ('Expand ' + rest + ' lines');
                    if (expanding) {
                        scheduleAutoCollapse(wrap, () => {
                            panel.classList.add('hidden');
                            preview.classList.remove('hidden');
                            more.textContent = 'Expand ' + rest + ' lines';
                            if (card) card.classList.remove('open');
                        }, PREVIEW_AUTO_COLLAPSE_MS);
                    }
                });
            }
            scheduleAutoCollapse(wrap, () => {
                if (preview) preview.classList.add('hidden');
                if (panel) panel.classList.add('hidden');
                if (more) more.classList.add('hidden');
                if (card) card.classList.remove('open');
            }, PREVIEW_AUTO_COLLAPSE_MS);
        } else {
            wrap.innerHTML =
                '<div class="atl-light atl-card atl-glass-chip" title="' + escapeAttr(summary) + '">' +
                '<span class="atl-light-dot"></span>' +
                escapeHtml(toolVerb(tool, summary)) +
                '</div>';
        }
        entry.itemsEl.appendChild(wrap);
        forceScrollMessages();
    }

    function collapseAgentTimeline(channelId, opts) {
        const key = String(channelId);
        const entry = streamMap[key];
        if (!entry) return;
        if (entry.typing && entry.typing.flush) entry.typing.flush();

        const collapseDelayMs = opts && opts.collapseDelayMs != null
            ? opts.collapseDelayMs
            : THOUGHT_DONE_AUTO_COLLAPSE_MS;
        finishOpenThinkingBlocks(entry, {
            autoCollapse: true,
            collapseDelayMs: collapseDelayMs
        });

        // 收尾时再清一遍已结束的 Thought，去掉 call-id / 保活文案 / 与最终回复重复的正文
        entry.itemsEl.querySelectorAll('.atl-thinking').forEach((el) => {
            const textEl = el.querySelector('.atl-thinking-text');
            let cleaned = sanitizeThinkingText(
                el._thinkingBuffer || (textEl && textEl.textContent) || ''
            );
            if (cleaned && (isMetaStatusThinkingText(cleaned) || isDuplicateOfLatestReply(key, cleaned))) {
                cleaned = '';
            }
            el._thinkingBuffer = cleaned;
            if (textEl) textEl.textContent = cleaned;
            if (!cleaned) el.remove();
        });

        // 去掉 record_reply 等不应出现在思维链中的工具条目（绝不动 .atl-thinking）
        entry.itemsEl.querySelectorAll('.atl-item').forEach((item) => {
            if (item.classList.contains('atl-thinking')) return;
            const label = String(item.textContent || '').toLowerCase();
            if (label.indexOf('record_reply') >= 0 || label.indexOf('team_reply_stream') >= 0 || label.indexOf('thought_stream') >= 0) {
                item.remove();
            }
        });

        // 裸路径小条收进 Explored；相邻 Explored 合并
        absorbPathChipsIntoExplored(entry);
        mergeAdjacentExploredBlocks(entry);

        // 历史会话更简洁：Explored 默认折叠成一行标题
        entry.itemsEl.querySelectorAll('.atl-explored').forEach((wrap) => {
            const list = wrap.querySelector('.atl-explored-list');
            const card = wrap.querySelector('.atl-explored-card');
            if (list) list.classList.add('hidden');
            if (card) card.classList.remove('open');
            // 收起各行源码预览
            wrap.querySelectorAll('.atl-explored-preview').forEach((p) => p.classList.add('hidden'));
        });

        // 仅当完全没有可见步骤时才丢弃；有思维链内容则必须保留在「用户消息」和「回复」之间
        if (!entry.itemsEl.children.length) {
            if (entry.node && entry.node.parentNode) entry.node.parentNode.removeChild(entry.node);
            delete streamMap[key];
            return;
        }

        const gen = entry.node.querySelector('.agent-timeline-generating');
        if (gen) gen.classList.add('hidden');
        const elapsed = Math.max(1, Math.round((Date.now() - entry.startedAt) / 1000));
        const summary = entry.node.querySelector('.agent-timeline-summary');
        const summaryText = entry.node.querySelector('.summary-text');
        const steps = entry.itemsEl.querySelectorAll('.atl-item').length || entry.stepCount || 0;
        if (summary) summary.classList.remove('hidden');

        // 历史会话默认总折叠：只留「已完成 · N 步」大按钮，点击再展开思考+工具
        entry.collapsed = true;
        entry.done = true;
        const root = entry.node.querySelector('.agent-timeline');
        if (root) {
            root.classList.add('atl-done-open');
            root.classList.add('collapsed');
        }
        if (summaryText) {
            summaryText.textContent = '已完成 · ' + steps + ' 步 · 用时 ' + elapsed + 's （点击展开）';
        }
        // 确保摘要按钮可切换总折叠（即使 entry 已移出 streamMap）
        bindTimelineFoldToggle(entry.node, entry);

        if (!entry.node.dataset.timelineId) {
            entry.node.dataset.timelineId = 'tl_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 7);
        }

        // 锚定到本轮用户消息（创建时的 anchorUserId），绝不能用「当前最新用户」以免发新消息后旧 Thought 消失/错位
        const afterId = nextAnchorIdForChannel(
            key,
            (opts && opts.afterId) || entry.anchorUserId || null
        );
        if (afterId && !entry.anchorUserId) entry.anchorUserId = afterId;
        if (!doneTimelines[key]) doneTimelines[key] = [];
        const existingIdx = doneTimelines[key].findIndex(
            (e) => e && e.afterId && afterId && e.afterId === afterId && e.node && e.node !== entry.node
        );
        if (existingIdx >= 0) {
            // 同一轮用户消息下的重复收尾：合并到已有完成块，保持历史可见
            const prev = doneTimelines[key][existingIdx];
            const prevItems = prev.node.querySelector('.agent-timeline-items');
            if (prevItems && entry.itemsEl) {
                while (entry.itemsEl.firstChild) {
                    prevItems.appendChild(entry.itemsEl.firstChild);
                }
            }
            if (entry.node.parentNode) entry.node.parentNode.removeChild(entry.node);
            const steps2 = prevItems ? prevItems.querySelectorAll('.atl-item').length : steps;
            const st2 = prev.node.querySelector('.summary-text');
            if (st2) st2.textContent = '已完成 · ' + steps2 + ' 步 · 用时 ' + elapsed + 's';
            const root2 = prev.node.querySelector('.agent-timeline');
            if (root2) {
                root2.classList.add('atl-done-open');
                root2.classList.add('collapsed');
            }
            if (st2) st2.textContent = '已完成 · ' + steps2 + ' 步 · 用时 ' + elapsed + 's （点击展开）';
            bindTimelineFoldToggle(prev.node, { collapsed: true, done: true });
        } else {
            doneTimelines[key].push({ node: entry.node, afterId: afterId || null });
        }
        while (doneTimelines[key].length > MAX_DONE_TIMELINES) {
            const oldEntry = normalizeDoneEntry(doneTimelines[key].shift());
            if (oldEntry && oldEntry.node && oldEntry.node.parentNode) {
                oldEntry.node.parentNode.removeChild(oldEntry.node);
            }
        }
        delete streamMap[key];
        saveDoneTimelines();

        if (key === String(currentChannel) && currentViewMode !== 'team') {
            renderMessages(true);
        }
    }

    function serializeDoneTimelineEntry(entry) {
        const norm = normalizeDoneEntry(entry);
        if (!norm || !norm.node) return null;
        const itemsEl = norm.node.querySelector('.agent-timeline-items');
        if (!itemsEl) return null;
        const steps = [];
        itemsEl.querySelectorAll('.atl-item').forEach((el) => {
            if (el.classList.contains('atl-thinking')) {
                const label = (el.querySelector('.atl-thinking-label') || {}).textContent || 'Thought';
                const text = String(el._thinkingBuffer || (el.querySelector('.atl-thinking-text') || {}).textContent || '').trim();
                if (!text || isMetaStatusThinkingText(text)) return;
                const secsMatch = String(label).match(/(\d+)\s*s/i);
                steps.push({
                    kind: 'thinking',
                    label: String(label).trim() || 'Thought',
                    text,
                    secs: secsMatch ? Number(secsMatch[1]) : 1
                });
                return;
            }
            if (el.classList.contains('atl-explored')) {
                const label = String((el.querySelector('.atl-explored-label') || {}).textContent || 'Explored').trim();
                const paths = [];
                el.querySelectorAll('.atl-explored-path').forEach((p) => {
                    const t = String(p.getAttribute('title') || p.textContent || '').trim();
                    if (t && paths[paths.length - 1] !== t) paths.push(t);
                });
                if (!paths.length && el._explored && Array.isArray(el._explored.paths)) {
                    el._explored.paths.forEach((p) => {
                        if (p && paths[paths.length - 1] !== p) paths.push(p);
                    });
                }
                if (paths.length) {
                    steps.push({
                        kind: 'explored_group',
                        label: label || ('Explored ' + paths.length + ' files'),
                        paths
                    });
                }
                return;
            }
            const termCmd = el.querySelector('.atl-term-cmd');
            if (termCmd) {
                steps.push({ kind: 'shell', summary: String(termCmd.textContent || '').replace(/^\$\s*/, '').trim() });
                return;
            }
            const editPath = el.querySelector('.atl-edit-path');
            if (editPath) {
                steps.push({
                    kind: 'edit',
                    summary: String(editPath.getAttribute('title') || editPath.textContent || '').trim()
                });
                return;
            }
            const light = el.querySelector('.atl-light');
            const summary = String((light && light.textContent) || el.textContent || '').trim();
            if (summary && !isMetaStatusThinkingText(summary)) steps.push({ kind: 'tool', summary });
        });
        if (!steps.length) return null;
        const summaryText = (norm.node.querySelector('.summary-text') || {}).textContent || '';
        return {
            afterId: norm.afterId || null,
            timelineId: norm.node.dataset.timelineId || '',
            summary: String(summaryText || ''),
            steps
        };
    }

    function rebuildTimelineFromSnapshot(snap) {
        if (!snap || !Array.isArray(snap.steps) || !snap.steps.length) return null;
        if (!elTimelineTemplate) return null;
        const frag = elTimelineTemplate.content.cloneNode(true);
        const node = frag.querySelector('.msg-agent-timeline');
        if (!node) return null;
        const itemsEl = node.querySelector('.agent-timeline-items');
        const gen = node.querySelector('.agent-timeline-generating');
        if (gen) gen.classList.add('hidden');
        const summary = node.querySelector('.agent-timeline-summary');
        const summaryText = node.querySelector('.summary-text');
        const baseSummary = String(snap.summary || ('已完成 · ' + snap.steps.length + ' 步'))
            .replace(/\s*（点击展开）\s*$/, '')
            .replace(/\s*（点击折叠）\s*$/, '')
            .trim();
        if (summaryText) summaryText.textContent = baseSummary + ' （点击展开）';
        if (summary) summary.classList.remove('hidden');
        const root = node.querySelector('.agent-timeline');
        if (root) {
            root.classList.add('atl-done-open');
            root.classList.add('collapsed');
        }
        node.dataset.timelineId = snap.timelineId || ('tl_restore_' + Date.now().toString(36));
        bindTimelineFoldToggle(node, { collapsed: true, done: true });
        snap.steps.forEach((step) => {
            if (!step) return;
            if (step.kind === 'thinking') {
                if (isMetaStatusThinkingText(step.text)) return;
                const wrap = document.createElement('div');
                wrap.className = 'atl-item atl-thinking atl-narrative collapsed-done';
                wrap._thinkingBuffer = String(step.text || '');
                wrap.innerHTML =
                    '<button type="button" class="atl-thinking-head">' +
                    '<span class="chevron">›</span>' +
                    '<span class="atl-thinking-label">' + escapeHtml(step.label || ('Thought for ' + (step.secs || 1) + 's')) + '</span>' +
                    '</button>' +
                    '<div class="atl-thinking-body"><span class="atl-thinking-text"></span></div>';
                const textEl = wrap.querySelector('.atl-thinking-text');
                if (textEl) textEl.textContent = wrap._thinkingBuffer;
                const head = wrap.querySelector('.atl-thinking-head');
                if (head) head.addEventListener('click', () => wrap.classList.toggle('open'));
                itemsEl.appendChild(wrap);
                return;
            }
            if (step.kind === 'explored_group' || step.kind === 'explore') {
                const paths = Array.isArray(step.paths)
                    ? step.paths
                    : (step.summary ? [step.summary] : []);
                const uniq = [];
                paths.forEach((p) => {
                    const s = String(p || '').trim();
                    if (s && uniq[uniq.length - 1] !== s) uniq.push(s);
                });
                if (!uniq.length) return;
                const wrap = document.createElement('div');
                wrap.className = 'atl-item atl-explored';
                wrap._explored = { reads: uniq.length, searches: 0, total: uniq.length, paths: uniq.slice() };
                wrap.innerHTML =
                    '<div class="atl-explored-card atl-glass-chip">' +
                    '<button type="button" class="atl-explored-head">' +
                    '<span class="chevron">›</span>' +
                    '<span class="atl-explored-label">' + escapeHtml(step.label || ('Explored ' + uniq.length + ' files')) + '</span>' +
                    '</button>' +
                    '<div class="atl-explored-list hidden"></div>' +
                    '</div>';
                const list = wrap.querySelector('.atl-explored-list');
                const head = wrap.querySelector('.atl-explored-head');
                uniq.forEach((p) => {
                    const row = document.createElement('div');
                    row.className = 'atl-explored-row';
                    row.innerHTML =
                        '<span class="atl-explored-kind">Read</span>' +
                        '<span class="atl-explored-path" title="' + escapeAttr(p) + '">' + escapeHtml(shortPath(p) || p) + '</span>';
                    list.appendChild(row);
                });
                if (head) {
                    head.addEventListener('click', () => {
                        const open = list.classList.toggle('hidden') === false;
                        wrap.querySelector('.atl-explored-card').classList.toggle('open', open);
                    });
                }
                itemsEl.appendChild(wrap);
                return;
            }
            const wrap = document.createElement('div');
            wrap.className = 'atl-item';
            const summary = String(step.summary || '').trim();
            if (!summary || isMetaStatusThinkingText(summary)) return;
            if (step.kind === 'shell') {
                wrap.innerHTML =
                    '<div class="atl-term atl-glass-chip done">' +
                    '<div class="atl-term-row"><span class="atl-term-status">✓</span><span class="atl-term-cmd"></span></div>' +
                    '</div>';
                const cmd = wrap.querySelector('.atl-term-cmd');
                if (cmd) cmd.textContent = '$ ' + summary;
            } else {
                wrap.innerHTML =
                    '<div class="atl-light atl-card atl-glass-chip" title="' + escapeAttr(summary) + '">' +
                    '<span class="atl-light-dot"></span>' +
                    escapeHtml(summary) +
                    '</div>';
            }
            itemsEl.appendChild(wrap);
        });
        if (!itemsEl.children.length) return null;
        return { node, afterId: snap.afterId || null };
    }

    function saveDoneTimelines() {
        try {
            const out = {};
            Object.keys(doneTimelines).forEach((ch) => {
                const list = (doneTimelines[ch] || [])
                    .map(serializeDoneTimelineEntry)
                    .filter(Boolean)
                    .slice(-MAX_DONE_TIMELINES);
                if (list.length) out[ch] = list;
            });
            localStorage.setItem(STORAGE_TIMELINES, JSON.stringify(out));
            localStorage.setItem(STORAGE_STREAM_SEQ, JSON.stringify(channelLastSeq || {}));
        } catch {
            /* ignore quota */
        }
    }

    function loadDoneTimelines() {
        try {
            const raw = localStorage.getItem(STORAGE_TIMELINES);
            if (raw) {
                const obj = JSON.parse(raw);
                if (obj && typeof obj === 'object') {
                    Object.keys(obj).forEach((ch) => {
                        const list = Array.isArray(obj[ch]) ? obj[ch] : [];
                        doneTimelines[ch] = list.map(rebuildTimelineFromSnapshot).filter(Boolean);
                        // 旧快照可能把同一文件拆成多条 explore → 合并折叠
                        doneTimelines[ch].forEach((entry) => {
                            if (entry && entry.node) {
                                mergeAdjacentExploredBlocks({
                                    itemsEl: entry.node.querySelector('.agent-timeline-items')
                                });
                            }
                        });
                    });
                }
            }
        } catch {
            /* ignore */
        }
        try {
            const seqRaw = localStorage.getItem(STORAGE_STREAM_SEQ);
            if (seqRaw) {
                const seqObj = JSON.parse(seqRaw);
                if (seqObj && typeof seqObj === 'object') {
                    Object.keys(seqObj).forEach((ch) => {
                        const n = Number(seqObj[ch] || 0);
                        if (n > 0) channelLastSeq[ch] = Math.max(Number(channelLastSeq[ch] || 0), n);
                    });
                }
            }
        } catch {
            /* ignore */
        }
    }

    function escapeAttr(s) {
        return String(s || '')
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/</g, '&lt;');
    }

    // ── Quick Commands ──
    function renderQuickCommands() {
        if (!elQuickCommands) return;
        if (!Array.isArray(quickCommands) || quickCommands.length === 0) {
            elQuickCommands.innerHTML = '<div class="quick-cmd-empty">暂无快捷指令，点击 ＋ 新增</div>';
            return;
        }
        elQuickCommands.innerHTML = quickCommands.map((q) => {
            const label = escapeHtml(q.label || '');
            const id = escapeHtml(q.id || '');
            return '<div class="quick-cmd-item" data-id="' + id + '">' +
                '<button class="quick-cmd" type="button">' + label + '</button>' +
                '<button class="quick-cmd-del" type="button" title="删除">×</button>' +
                '</div>';
        }).join('');
    }
    function toggleQuickCommandForm(show) {
        if (!elQuickCmdForm) return;
        if (show) {
            elQuickCmdForm.classList.remove('hidden');
            if (elQuickCmdLabelInput) { elQuickCmdLabelInput.value = ''; elQuickCmdLabelInput.focus(); }
            if (elQuickCmdTextInput) elQuickCmdTextInput.value = '';
        } else {
            elQuickCmdForm.classList.add('hidden');
        }
    }

    // ── Attachments ──
    async function attachFile(file, kind) {
        if (!file) return;
        if (pendingAttachments.length >= 10) {
            showToast('最多 10 个附件');
            return;
        }
        const reader = new FileReader();
        const dataUrl = await new Promise((resolve, reject) => {
            reader.onload = () => resolve(String(reader.result || ''));
            reader.onerror = () => reject(reader.error || new Error('读取失败'));
            reader.readAsDataURL(file);
        });
        const att = {
            localId: genId(),
            name: file.name || (kind === 'image' ? ('paste_' + Date.now() + '.png') : 'file.bin'),
            data: dataUrl,
            size: file.size || 0,
            type: kind === 'image' ? 'image' : 'file'
        };
        pendingAttachments.push(att);
        renderAttachments();
    }
    function removeAttachment(localId) {
        pendingAttachments = pendingAttachments.filter(a => a.localId !== localId);
        renderAttachments();
    }
    function renderAttachments() {
        if (!elAttachmentsList || !elAttachmentsRow) return;
        if (pendingAttachments.length === 0) {
            elAttachmentsList.innerHTML = '';
            elAttachmentsRow.classList.add('hidden');
            return;
        }
        elAttachmentsRow.classList.remove('hidden');
        elAttachmentsList.innerHTML = pendingAttachments.map((a) => {
            const icon = a.type === 'image' ? '🖼' : '📎';
            const sizeKb = a.size ? ' · ' + (a.size / 1024).toFixed(1) + ' KB' : '';
            return '<span class="attachment-chip" data-id="' + escapeHtml(a.localId) + '">' +
                '<span>' + icon + '</span>' +
                '<span class="attachment-chip-name" title="' + escapeHtml(a.name) + '">' +
                escapeHtml(a.name) + sizeKb + '</span>' +
                '<button class="attachment-chip-del" type="button">×</button>' +
                '</span>';
        }).join('');
    }
    async function uploadAttachments(list) {
        const body = {
            files: list.map(a => ({ name: a.name, data: a.data }))
        };
        const resp = await fetch('/api/upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        let json;
        try { json = await resp.json(); } catch { json = null; }
        if (!resp.ok || !json || !json.ok) {
            const err = (json && json.message) ? json.message : ('HTTP ' + resp.status);
            throw new Error(err);
        }
        return Array.isArray(json.paths) ? json.paths : [];
    }

    // ── Settings (local) ──
    function loadLocalSettings() {
        try {
            const raw = localStorage.getItem(STORAGE_SETTINGS);
            if (!raw) return;
            const data = JSON.parse(raw);
            if (data && typeof data === 'object') {
                localSettings = { ...localSettings, ...data };
            }
        } catch {}
    }
    function saveLocalSettings() {
        try { localStorage.setItem(STORAGE_SETTINGS, JSON.stringify(localSettings)); } catch {}
    }
    function applySendMode() {
        if (elSettingSendMode) elSettingSendMode.value = localSettings.sendMode;
        if (elComposerHint) {
            const modeText = localSettings.sendMode === 'ctrl-enter' ? 'Ctrl/⌘ + Enter 发送' : 'Enter 发送';
            elComposerHint.textContent = 'CH-' + currentChannel + ' · ' + modeText;
        }
    }
    function applyFontSize() {
        document.body.classList.remove('fontsize-small', 'fontsize-medium', 'fontsize-large');
        document.body.classList.add('fontsize-' + (localSettings.fontSize || 'medium'));
        if (elSettingFontSize) elSettingFontSize.value = localSettings.fontSize || 'medium';
    }
    function applyThemeMode() {
        const mode = localSettings.themeMode || 'system';
        if (elSettingThemeMode) elSettingThemeMode.value = mode;
        if (mode === 'system') {
            const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            applyTheme(prefersDark ? 'dark' : 'light');
            return;
        }
        applyTheme(mode);
    }
    function applyNotifyDesktop() {
        if (elSettingNotifyDesktop) elSettingNotifyDesktop.checked = Boolean(localSettings.notifyDesktop);
        if (localSettings.notifyDesktop) {
            requestDesktopNotifyPermission();
        }
    }
    function requestDesktopNotifyPermission() {
        if (!('Notification' in window)) return;
        if (Notification.permission === 'default') {
            try { Notification.requestPermission(); } catch {}
        }
    }

    // ── Notifications / Unread ──
    function handleNewReplyNotification(channelId, text) {
        if (document.hidden) {
            unreadCount++;
            updateTabTitle();
            document.body.classList.add('has-unread');
            if (localSettings.notifyDesktop && 'Notification' in window && Notification.permission === 'granted') {
                const snippet = String(text || '').replace(/\s+/g, ' ').slice(0, 80);
                try {
                    const n = new Notification('CH-' + channelId + ' · AI 回复', {
                        body: snippet,
                        tag: 'qt-mcp-' + channelId,
                        silent: false
                    });
                    n.onclick = () => { window.focus(); n.close(); };
                } catch {}
            }
        }
    }
    function updateTabTitle() {
        if (unreadCount > 0) {
            document.title = '(' + unreadCount + ') ' + DOC_TITLE_ORIGINAL;
        } else {
            document.title = DOC_TITLE_ORIGINAL;
        }
    }
    function clearUnread() {
        unreadCount = 0;
        updateTabTitle();
        document.body.classList.remove('has-unread');
    }

    // ── Drawers ──
    function openDrawer(which) {
        closeAllDrawers();
        if (which === 'settings' && elSettingsDrawer) elSettingsDrawer.classList.remove('hidden');
        if (which === 'history' && elHistoryDrawer) {
            elHistoryDrawer.classList.remove('hidden');
            renderHistoryDrawer();
        }
        if (elDrawerBackdrop) elDrawerBackdrop.classList.remove('hidden');
    }
    function closeAllDrawers() {
        if (elSettingsDrawer) elSettingsDrawer.classList.add('hidden');
        if (elHistoryDrawer) elHistoryDrawer.classList.add('hidden');
        if (elDrawerBackdrop) elDrawerBackdrop.classList.add('hidden');
    }

    // ── Server config (keepalive / notify on reply) ──
    function renderServerConfigControls() {
        if (elSettingKeepaliveEnabled) elSettingKeepaliveEnabled.checked = Boolean(serverConfig.keepaliveEnabled);
        if (elSettingKeepaliveMinutes) {
            if (document.activeElement !== elSettingKeepaliveMinutes) {
                elSettingKeepaliveMinutes.value = String(serverConfig.keepaliveMinutes || 45);
            }
            elSettingKeepaliveMinutes.disabled = !serverConfig.keepaliveEnabled;
        }
        if (elSettingKeepaliveMinutesRow) {
            elSettingKeepaliveMinutesRow.style.opacity = serverConfig.keepaliveEnabled ? '1' : '0.5';
        }
    }

    // ── License / MCP Status ──
    function renderLicenseInfo() {
        if (!elLicenseDesc) return;
        if (!licenseInfo || !licenseInfo.activated) {
            elLicenseDesc.textContent = '未激活';
        } else if (licenseInfo.permanent) {
            elLicenseDesc.textContent = '已激活 · 永久授权';
        } else {
            const txt = formatRemainingCompact(licenseInfo.remainingMs);
            elLicenseDesc.textContent = '已激活 · 剩余 ' + txt;
        }
        if (elLicenseRemaining) {
            if (!licenseInfo || !licenseInfo.activated) {
                elLicenseRemaining.textContent = '未激活';
            } else if (licenseInfo.permanent) {
                elLicenseRemaining.textContent = '永久';
            } else {
                elLicenseRemaining.textContent = formatRemainingCompact(licenseInfo.remainingMs);
            }
        }
    }
    function formatRemainingCompact(ms) {
        if (ms === null || ms === undefined || !isFinite(ms) || ms <= 0) return '已过期';
        const total = Math.floor(ms / 1000);
        const days = Math.floor(total / 86400);
        const h = Math.floor((total % 86400) / 3600);
        const m = Math.floor((total % 3600) / 60);
        const s = total % 60;
        const pad2 = (n) => String(n).padStart(2, '0');
        if (days > 0) return days + 'd ' + pad2(h) + ':' + pad2(m);
        return pad2(h) + ':' + pad2(m) + ':' + pad2(s);
    }
    function renderMcpStatusCard() {
        if (elMcpMode) {
            elMcpMode.textContent = mcpStatusInfo.mcpConnected ? 'Stdio · 在线' : 'Stdio · 离线';
        }
        if (elMcpClientCount) {
            const chCount = Number(mcpStatusInfo.channelCount || 0);
            elMcpClientCount.textContent = chCount + ' 通道';
        }
    }

    // ── History Drawer ──
    // 历史抽屉过滤：类型过滤 + 多词 AND 文本匹配（与聊天区搜索同语义）
    function filterHistoryForDrawer() {
        const terms = historyDrawerFilterText ? parseSearchTerms(historyDrawerFilterText) : [];
        return history.filter(h => {
            if (historyDrawerFilterType !== 'all' && h.type !== historyDrawerFilterType) return false;
            return textMatchesSearchTerms(h.text, terms);
        });
    }

    function renderHistoryDrawer() {
        if (!elHistoryListDrawer) return;
        const filtered = filterHistoryForDrawer();
        if (elHistorySummary) {
            elHistorySummary.textContent = '共 ' + filtered.length + ' 条（本地保存最多 ' + MAX_HISTORY + '）';
        }
        if (filtered.length === 0) {
            elHistoryListDrawer.innerHTML = '<div class="history-item"><div class="history-item-text" style="opacity:0.6">暂无记录</div></div>';
            return;
        }
        elHistoryListDrawer.innerHTML = filtered.slice().reverse().map((h) => {
            const text = String(h.text || '').slice(0, 600);
            const time = formatTime(h.timestamp);
            const typeLabel = h.type === 'user' ? '用户' : (h.type === 'reply' ? 'AI' : (h.type === 'error' ? '错误' : '系统'));
            return '<div class="history-item" data-type="' + escapeHtml(h.type || 'user') + '" data-id="' + escapeHtml(h.id || '') + '">' +
                '<div class="history-item-meta">' +
                '<span>CH-' + escapeHtml(h.channelId || '?') + ' · ' + typeLabel + '</span>' +
                '<span>' + time + '</span>' +
                '</div>' +
                '<div class="history-item-text">' + escapeHtml(text) + '</div>' +
                '</div>';
        }).join('');
    }
    function exportHistoryAsMarkdown() {
        const filtered = filterHistoryForDrawer();
        if (filtered.length === 0) { showToast('没有可导出的记录'); return; }
        const now = new Date();
        const pad2 = (n) => String(n).padStart(2, '0');
        const stamp = now.getFullYear() + pad2(now.getMonth() + 1) + pad2(now.getDate()) + '-' +
            pad2(now.getHours()) + pad2(now.getMinutes());
        const lines = ['# 晴天无限 MCP · 浏览器端历史', '', '导出时间：' + now.toLocaleString(), ''];
        for (const h of filtered) {
            const typeLabel = h.type === 'user' ? '👤 用户' : (h.type === 'reply' ? '🤖 AI' : (h.type === 'error' ? '⚠️ 错误' : '📎 系统'));
            lines.push('## ' + typeLabel + ' · CH-' + (h.channelId || '?') + ' · ' + formatTime(h.timestamp));
            lines.push('');
            lines.push(String(h.text || ''));
            lines.push('');
            lines.push('---');
            lines.push('');
        }
        downloadText('qingtian-mcp-browser-' + stamp + '.md', lines.join('\n'));
    }
    function exportHistoryAsJson() {
        const filtered = filterHistoryForDrawer();
        if (filtered.length === 0) { showToast('没有可导出的记录'); return; }
        const now = new Date();
        const pad2 = (n) => String(n).padStart(2, '0');
        const stamp = now.getFullYear() + pad2(now.getMonth() + 1) + pad2(now.getDate()) + '-' +
            pad2(now.getHours()) + pad2(now.getMinutes());
        downloadText('qingtian-mcp-browser-' + stamp + '.json', JSON.stringify(filtered, null, 2));
    }
    function downloadText(filename, content) {
        try {
            const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(url), 500);
        } catch (e) {
            showToast('导出失败：' + (e && e.message ? e.message : e));
        }
    }

    // ── Search bar ──
    function applySearchKeyword(kw) {
        searchKeyword = String(kw || '').trim().toLowerCase();
        renderMessages(true);
    }

    // ── Event bindings ──
    elSend.addEventListener('click', submit);

    if (elOutboxList) {
        elOutboxList.addEventListener('click', handleOutboxListClick);
        elOutboxList.addEventListener('keydown', handleOutboxListKeydown);
    }
    if (elBtnOutboxSendNow) {
        elBtnOutboxSendNow.addEventListener('click', () => {
            const queue = getOutbox(currentChannel);
            if (!queue.length) return;
            sendOutboxItemNow(currentChannel, queue[0].id);
        });
    }
    if (elBtnOutboxClear) {
        elBtnOutboxClear.addEventListener('click', () => {
            if (!outboxLength(currentChannel)) return;
            if (!confirm('清空 CH-' + currentChannel + ' 的排队待发消息？')) return;
            delete outboxMap[String(currentChannel)];
            outboxEditingId = '';
            saveOutbox();
            renderOutbox();
            renderChannelList();
        });
    }

    elInput.addEventListener('keydown', (e) => {
        if (elMentionDropdown && !elMentionDropdown.classList.contains('hidden')) {
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (mentionResults.length) {
                    mentionIndex = (mentionIndex + 1) % mentionResults.length;
                    renderMentionDropdown();
                }
                return;
            }
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (mentionResults.length) {
                    mentionIndex = mentionIndex <= 0 ? mentionResults.length - 1 : mentionIndex - 1;
                    renderMentionDropdown();
                }
                return;
            }
            if ((e.key === 'Enter' || e.key === 'Tab') && mentionResults.length) {
                e.preventDefault();
                selectMentionResult(mentionIndex >= 0 ? mentionIndex : 0);
                return;
            }
            if (e.key === 'Escape') {
                hideMentionDropdown();
                return;
            }
        }
        if (e.key !== 'Enter') return;
        if (localSettings.sendMode === 'ctrl-enter') {
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                submit();
            }
            return;
        }
        // sendMode === 'enter'：Enter 发送，Shift+Enter 换行
        if (!e.shiftKey && !e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            submit();
        }
    });

    elInput.addEventListener('input', updateMentionDropdownSoon);

    document.addEventListener('click', (e) => {
        if (!elMentionDropdown || elMentionDropdown.classList.contains('hidden')) return;
        if (elInput && elInput.contains(e.target)) return;
        if (elMentionDropdown.contains(e.target)) return;
        hideMentionDropdown();
    });

    // 粘贴图片
    elInput.addEventListener('paste', (e) => {
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (const it of items) {
            if (it.kind === 'file' && it.type && it.type.startsWith('image/')) {
                const file = it.getAsFile();
                if (file) {
                    e.preventDefault();
                    attachFile(file, 'image').catch(() => {});
                }
            }
        }
    });

    mountTeamRecoveryEntry();

    if (elBtnClearHistory) elBtnClearHistory.addEventListener('click', clearLocalHistory);
    if (elBtnClearChannel) elBtnClearChannel.addEventListener('click', clearRemoteChannel);
    if (elBtnCopyPrompt) elBtnCopyPrompt.addEventListener('click', copyStartPromptForCurrent);
    if (elBtnSendStartPrompt) elBtnSendStartPrompt.addEventListener('click', sendStartPromptForCurrent);
    if (elBtnCopyRecovery) elBtnCopyRecovery.addEventListener('click', restoreRecoveryContextForCurrent);
    if (elBtnViewChannel) elBtnViewChannel.addEventListener('click', () => setViewMode('channel'));
    if (elBtnViewTeam) elBtnViewTeam.addEventListener('click', () => {
        setViewMode('team');
        if (wsState === 'open') send({ type: 'getTeamSnapshot', groupId: activeTeamGroupId || '' });
    });
    if (elTeamGroupSelect) elTeamGroupSelect.addEventListener('change', () => {
        activeTeamGroupId = elTeamGroupSelect.value || '';
        if (wsState === 'open') send({ type: 'getTeamSnapshot', groupId: activeTeamGroupId });
        renderTeamToolbar();
        renderMessages(true);
    });
    if (elBtnTeamRefresh) elBtnTeamRefresh.addEventListener('click', () => {
        if (wsState === 'open') send({ type: 'getTeamSnapshot', groupId: activeTeamGroupId || '' });
    });
    if (elBtnTeamCreate) elBtnTeamCreate.addEventListener('click', openTeamGroupModal);
    if (elBtnTeamDelete) elBtnTeamDelete.addEventListener('click', deleteActiveTeamGroup);
    if (elTeamGroupOverlay) elTeamGroupOverlay.addEventListener('click', closeTeamGroupModal);
    if (elBtnTeamGroupClose) elBtnTeamGroupClose.addEventListener('click', closeTeamGroupModal);
    if (elBtnTeamGroupCancel) elBtnTeamGroupCancel.addEventListener('click', closeTeamGroupModal);
    if (elBtnTeamGroupConfirm) elBtnTeamGroupConfirm.addEventListener('click', confirmCreateTeamGroup);
    if (elTeamRecoveryGroup) elTeamRecoveryGroup.addEventListener('change', () => {
        activeTeamGroupId = elTeamRecoveryGroup.value || '';
        if (wsState === 'open') send({ type: 'getTeamSnapshot', groupId: activeTeamGroupId });
        renderTeamRecoveryCard();
        renderTeamRecoveryModal();
        renderTeamToolbar();
        if (currentViewMode === 'team') renderMessages(true);
    });
    if (elBtnTeamRecovery) elBtnTeamRecovery.addEventListener('click', openTeamRecoveryModal);
    if (elTeamRecoveryOverlay) elTeamRecoveryOverlay.addEventListener('click', closeTeamRecoveryModal);
    if (elBtnTeamRecoveryClose) elBtnTeamRecoveryClose.addEventListener('click', closeTeamRecoveryModal);
    if (elBtnTeamRecoveryCancel) elBtnTeamRecoveryCancel.addEventListener('click', closeTeamRecoveryModal);
    if (elTeamRecoveryModal) {
        elTeamRecoveryModal.querySelectorAll('input[name="team-recovery-mode"]').forEach((el) => {
            el.addEventListener('change', () => {
                teamRecoveryMode = el.value === 'takeover' ? 'takeover' : 'restore';
                renderTeamRecoveryModal();
            });
        });
    }
    if (elTeamRecoverySource) elTeamRecoverySource.addEventListener('change', () => {
        teamRecoverySourceId = elTeamRecoverySource.value || '';
        renderTeamRecoveryModal();
    });
    if (elTeamRecoveryTarget) elTeamRecoveryTarget.addEventListener('change', () => {
        teamRecoveryTargetId = elTeamRecoveryTarget.value || '';
        renderTeamRecoveryModal();
    });
    if (elBtnTeamRecoveryConfirm) elBtnTeamRecoveryConfirm.addEventListener('click', confirmTeamRecovery);
    if (elRecoveryTransferOverlay) elRecoveryTransferOverlay.addEventListener('click', () => setRecoveryTransferOpen(false));
    if (elBtnRecoveryTransferClose) elBtnRecoveryTransferClose.addEventListener('click', () => setRecoveryTransferOpen(false));
    if (elBtnRecoveryTransferCancel) elBtnRecoveryTransferCancel.addEventListener('click', () => setRecoveryTransferOpen(false));
    if (elRecoveryTransferModal) {
        elRecoveryTransferModal.querySelectorAll('input[name="recovery-transfer-mode"]').forEach((el) => {
            el.addEventListener('change', () => {
                recoveryTransferMode = el.value === 'transfer' ? 'transfer' : 'current';
                if (recoveryTransferMode === 'transfer' && !recoveryTransferTargetId) {
                    recoveryTransferTargetId = defaultRecoveryTransferTarget();
                }
                renderRecoveryTransferModal();
            });
        });
    }
    if (elBtnRecoveryTransferConfirm) elBtnRecoveryTransferConfirm.addEventListener('click', () => {
        const targetChannelId = recoveryTransferMode === 'transfer' ? recoveryTransferTargetId : currentChannel;
        if (!targetChannelId || wsState !== 'open') {
            if (wsState !== 'open') showToast('未连接服务端');
            return;
        }
        send({
            type: 'restoreRecoveryContext',
            sourceChannelId: currentChannel,
            targetChannelId: targetChannelId,
            scope: 'channel',
            maxChars: 12000
        });
    });
    if (elBtnAdd) elBtnAdd.addEventListener('click', doAddChannel);
    if (elBtnRemove) elBtnRemove.addEventListener('click', doRemoveChannel);
    if (elBtnThemeToggle) elBtnThemeToggle.addEventListener('click', toggleTheme);
    if (elBtnToggleSidebar) elBtnToggleSidebar.addEventListener('click', toggleSidebar);

    // 快捷指令事件委托
    if (elQuickCommands) {
        elQuickCommands.addEventListener('click', (e) => {
            const target = e.target;
            if (!target || !target.closest) return;
            const item = target.closest('.quick-cmd-item');
            if (!item) return;
            const id = item.getAttribute('data-id');
            if (target.closest('.quick-cmd-del')) {
                if (!id || wsState !== 'open') { showToast(wsState !== 'open' ? '未连接服务端' : ''); return; }
                send({ type: 'removeQuickCommand', id });
                return;
            }
            if (target.closest('.quick-cmd')) {
                const q = quickCommands.find(x => x.id === id);
                if (q && q.text && elInput) {
                    elInput.value = q.text;
                    submit();
                }
            }
        });
    }
    if (elBtnQuickAdd) elBtnQuickAdd.addEventListener('click', () => {
        if (elQuickCmdForm && elQuickCmdForm.classList.contains('hidden')) {
            toggleQuickCommandForm(true);
        } else {
            toggleQuickCommandForm(false);
        }
    });
    if (elBtnQuickCancel) elBtnQuickCancel.addEventListener('click', () => toggleQuickCommandForm(false));
    if (elBtnQuickSave) elBtnQuickSave.addEventListener('click', () => {
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        const label = (elQuickCmdLabelInput && elQuickCmdLabelInput.value || '').trim();
        const text = (elQuickCmdTextInput && elQuickCmdTextInput.value || '');
        if (!label || !text) { showToast('请填写标签和内容'); return; }
        send({ type: 'addQuickCommand', label, text });
    });
    if (elBtnQuickReset) elBtnQuickReset.addEventListener('click', () => {
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        if (!confirm('重置快捷指令为默认值？这会同步到插件端。')) return;
        send({ type: 'resetQuickCommands' });
    });

    // 附件按钮
    if (elBtnAttachFile) elBtnAttachFile.addEventListener('click', () => {
        if (elHiddenFilePicker) elHiddenFilePicker.click();
    });
    if (elBtnAttachImage) elBtnAttachImage.addEventListener('click', () => {
        if (elHiddenImagePicker) elHiddenImagePicker.click();
    });
    if (elHiddenFilePicker) elHiddenFilePicker.addEventListener('change', async (e) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;
        for (const f of files) await attachFile(f, 'file');
        e.target.value = '';
    });
    if (elHiddenImagePicker) elHiddenImagePicker.addEventListener('change', async (e) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;
        for (const f of files) await attachFile(f, 'image');
        e.target.value = '';
    });
    if (elAttachmentsList) {
        elAttachmentsList.addEventListener('click', (e) => {
            const del = e.target && e.target.closest && e.target.closest('.attachment-chip-del');
            if (!del) return;
            const chip = del.closest('.attachment-chip');
            if (!chip) return;
            removeAttachment(chip.getAttribute('data-id'));
        });
    }
    // Drag & drop 到 composer
    if (elComposerDrop) {
        elComposerDrop.addEventListener('dragover', (e) => {
            e.preventDefault();
            elComposerDrop.classList.add('drag-over');
        });
        elComposerDrop.addEventListener('dragleave', () => elComposerDrop.classList.remove('drag-over'));
        elComposerDrop.addEventListener('drop', async (e) => {
            e.preventDefault();
            elComposerDrop.classList.remove('drag-over');
            const files = e.dataTransfer && e.dataTransfer.files;
            if (!files || files.length === 0) return;
            for (const f of files) {
                const kind = f.type && f.type.startsWith('image/') ? 'image' : 'file';
                await attachFile(f, kind);
            }
        });
    }

    // 设置抽屉
    if (elBtnOpenSettings) elBtnOpenSettings.addEventListener('click', () => openDrawer('settings'));
    if (elBtnCloseSettings) elBtnCloseSettings.addEventListener('click', closeAllDrawers);
    if (elBtnOpenHistory) elBtnOpenHistory.addEventListener('click', () => openDrawer('history'));
    if (elBtnCloseHistory) elBtnCloseHistory.addEventListener('click', closeAllDrawers);
    if (elDrawerBackdrop) elDrawerBackdrop.addEventListener('click', closeAllDrawers);

    if (elSettingSendMode) elSettingSendMode.addEventListener('change', () => {
        localSettings.sendMode = elSettingSendMode.value === 'ctrl-enter' ? 'ctrl-enter' : 'enter';
        saveLocalSettings();
        applySendMode();
        renderTopbar();
    });
    if (elSettingFontSize) elSettingFontSize.addEventListener('change', () => {
        localSettings.fontSize = elSettingFontSize.value;
        saveLocalSettings();
        applyFontSize();
    });
    if (elSettingThemeMode) elSettingThemeMode.addEventListener('change', () => {
        localSettings.themeMode = elSettingThemeMode.value;
        saveLocalSettings();
        applyThemeMode();
    });
    if (elSettingNotifyDesktop) elSettingNotifyDesktop.addEventListener('change', () => {
        localSettings.notifyDesktop = elSettingNotifyDesktop.checked;
        saveLocalSettings();
        if (localSettings.notifyDesktop) requestDesktopNotifyPermission();
    });
    if (elSettingKeepaliveEnabled) elSettingKeepaliveEnabled.addEventListener('change', () => {
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        send({ type: 'setConfig', data: { keepaliveEnabled: elSettingKeepaliveEnabled.checked } });
    });
    if (elSettingKeepaliveMinutes) elSettingKeepaliveMinutes.addEventListener('change', () => {
        if (wsState !== 'open') { showToast('未连接服务端'); return; }
        const val = Math.max(1, Math.min(120, Number(elSettingKeepaliveMinutes.value) || 45));
        elSettingKeepaliveMinutes.value = String(val);
        send({ type: 'setConfig', data: { keepaliveMinutes: val } });
    });

    // 搜索条
    if (elBtnToggleSearch) elBtnToggleSearch.addEventListener('click', () => {
        if (!elSearchBar) return;
        if (elSearchBar.classList.contains('hidden')) {
            elSearchBar.classList.remove('hidden');
            if (elSearchInput) elSearchInput.focus();
        } else {
            elSearchBar.classList.add('hidden');
            applySearchKeyword('');
        }
    });
    if (elBtnCloseSearch) elBtnCloseSearch.addEventListener('click', () => {
        if (elSearchBar) elSearchBar.classList.add('hidden');
        if (elSearchInput) elSearchInput.value = '';
        applySearchKeyword('');
    });
    if (elSearchInput) {
        let searchDebounce = null;
        elSearchInput.addEventListener('input', () => {
            if (searchDebounce) clearTimeout(searchDebounce);
            searchDebounce = setTimeout(() => {
                applySearchKeyword(elSearchInput.value);
            }, 150);
        });
    }

    // 历史抽屉
    if (elHistorySearch) {
        let histDebounce = null;
        elHistorySearch.addEventListener('input', () => {
            if (histDebounce) clearTimeout(histDebounce);
            histDebounce = setTimeout(() => {
                historyDrawerFilterText = String(elHistorySearch.value || '').trim().toLowerCase();
                renderHistoryDrawer();
            }, 150);
        });
    }
    if (elHistoryTypeFilter) elHistoryTypeFilter.addEventListener('change', () => {
        historyDrawerFilterType = elHistoryTypeFilter.value || 'all';
        renderHistoryDrawer();
    });
    if (elBtnExportMd) elBtnExportMd.addEventListener('click', exportHistoryAsMarkdown);
    if (elBtnExportJson) elBtnExportJson.addEventListener('click', exportHistoryAsJson);

    // 焦点切换清零未读
    window.addEventListener('focus', clearUnread);
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) clearUnread();
    });
    // 系统主题变化（仅 themeMode=system 时跟随）
    if (window.matchMedia) {
        const mq = window.matchMedia('(prefers-color-scheme: dark)');
        const onPrefChange = () => {
            if (localSettings.themeMode === 'system') applyThemeMode();
        };
        try { mq.addEventListener('change', onPrefChange); } catch { mq.addListener && mq.addListener(onPrefChange); }
    }

    // ── Initial ──
    loadLocalSettings();

    // 主题：若用户在设置里选择了 themeMode 则按它来，否则兼容老版 STORAGE_THEME
    if (localSettings.themeMode && localSettings.themeMode !== 'system') {
        applyThemeMode();
    } else {
        const savedTheme = localStorage.getItem(STORAGE_THEME);
        if (savedTheme && !localSettings.themeMode) {
            localSettings.themeMode = savedTheme;
            saveLocalSettings();
            applyThemeMode();
        } else {
            // themeMode=system 或未设置
            if (!localSettings.themeMode) localSettings.themeMode = 'system';
            applyThemeMode();
        }
    }

    // 侧栏折叠状态
    if (localStorage.getItem(STORAGE_SIDEBAR) === '1') {
        applySidebar(true);
    }
    if (window.innerWidth <= 820 && !localStorage.getItem(STORAGE_SIDEBAR)) {
        applySidebar(true);
    }

    applySendMode();
    applyFontSize();
    applyNotifyDesktop();
    renderQuickCommands();
    renderServerConfigControls();
    renderLicenseInfo();
    renderMcpStatusCard();
    renderAttachments();
    renderOutbox();
    renderTopbar();
    loadDoneTimelines();
    renderMessages();
    renderAiPresence();
    connect();

    // 心跳（每 20 秒 ping，保持连接稳定；也兜底检测连接状态）
    setInterval(() => {
        if (wsState === 'open') send({ type: 'ping' });
    }, 20000);

    // 周期性拉取许可/MCP 状态，以便实时刷新剩余时间
    setInterval(() => {
        if (wsState === 'open') {
            send({ type: 'getLicense' });
            send({ type: 'getMCPStatus' });
            send({ type: 'getTeamSnapshot', groupId: activeTeamGroupId || '' });
        }
    }, 30000);

    // 排队兜底：reply 之外的收尾路径（超时降级、手动关闭思考气泡等）也要能自动弹下一条
    setInterval(() => {
        if (wsState !== 'open' || outboxDispatching) return;
        const ready = Object.keys(outboxMap).some(ch => outboxLength(ch) > 0 && !isChannelBusy(ch));
        if (ready) flushOutbox();
    }, 1500);

    // 每秒刷新：AI 心跳、topbar、许可剩余时间
    setInterval(() => {
        renderAiPresence();
        renderTopbar();
        updateOutboxHint();
        if (licenseInfo && licenseInfo.activated && !licenseInfo.permanent && licenseInfo.expiresAt) {
            licenseInfo.remainingMs = Math.max(0, licenseInfo.expiresAt - Date.now());
            renderLicenseInfo();
        }
    }, 1000);
})();
