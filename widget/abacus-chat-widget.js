/**
 * Abacus Digital Chat Widget
 * Lightweight embeddable chat widget for Framer sites.
 * Single <script> tag embedding, no framework dependencies.
 *
 * Usage:
 *   <script src="https://your-backend.com/widget/abacus-chat-widget.js"
 *           data-api-url="https://your-backend.com"
 *           data-calendly-url="https://calendly.com/abacusdigital/discovery"
 *           data-mode="public">
 *   </script>
 *
 * data-mode="client" renders the authenticated client-support surface instead,
 * which talks to /api/client/* and never shares a session with the public bot.
 *
 * Public mode treats every conversation as its own chat (own AI-generated title,
 * own transcript), listed in a "Chats" panel behind a back button — like Intercom/Fin.
 * A browser-scoped visitor_id (not a login) ties those chats together so the bot can
 * carry context across them without asking the same questions twice.
 */

(function () {
    'use strict';

    // Bump on every meaningful widget change. Appended as a cache-busting query param
    // to the CSS link so a browser that already cached an older stylesheet actually
    // fetches the new one instead of silently rendering stale styles against the
    // current JS. (The script tag itself is controlled by the embedding page — see
    // widget/index.html and widget/client.html, which carry the same param.)
    const WIDGET_VERSION = '2026-08-13.2';

    // ---- Configuration ----
    const SCRIPT = document.currentScript;
    const attr = (name, fallback) => (SCRIPT && SCRIPT.getAttribute(name)) || fallback;

    // Default to the origin the script itself was served from — on the Framer site the
    // script comes from the backend, and locally it keeps the test pages on whatever
    // port uvicorn happens to be running.
    function defaultApiUrl() {
        try {
            if (SCRIPT && SCRIPT.src) {
                const origin = new URL(SCRIPT.src, window.location.href).origin;
                if (origin && origin !== 'null') return origin;
            }
        } catch (e) { /* fall through */ }
        return window.location.origin && window.location.origin !== 'null'
            ? window.location.origin
            : 'http://localhost:8000';
    }

    const API_URL = attr('data-api-url', defaultApiUrl()).replace(/\/$/, '');
    const CONTACT_URL = attr('data-contact-url', 'https://www.abacusdigital.net/contact');
    const MODE = attr('data-mode', 'public') === 'client' ? 'client' : 'public';

    const VISITOR_KEY = 'abacus_visitor_id';
    const CURRENT_SESSION_KEY = 'abacus_current_session_' + MODE;
    const TOKEN_KEY = 'abacus_client_token';
    const TRANSCRIPT_PREFIX = 'abacus_transcript_' + MODE + '_';

    // ---- State ----
    let sessionId = null;
    let visitorId = null;
    let isOpen = false;
    let isWaiting = false;
    let messageCount = 0;
    let consentGiven = false;
    let backendUp = true;
    let clientToken = null;
    let clientInfo = null;
    let view = 'chat'; // 'chat' | 'list' — public mode only
    let bootedOnce = false;

    // ---- Storage helpers (private browsing can throw on access) ----
    function store(key, value) {
        try { value === null ? localStorage.removeItem(key) : localStorage.setItem(key, value); }
        catch (e) { /* storage unavailable; state simply won't persist */ }
    }
    function read(key) {
        try { return localStorage.getItem(key); } catch (e) { return null; }
    }
    function storeJSON(key, value) { store(key, JSON.stringify(value)); }
    function readJSON(key, fallback) {
        try { const raw = read(key); return raw ? JSON.parse(raw) : fallback; }
        catch (e) { return fallback; }
    }

    // ---- Load CSS ----
    function loadCSS() {
        if (document.getElementById('abacus-chat-css')) return;
        const link = document.createElement('link');
        link.id = 'abacus-chat-css';
        link.rel = 'stylesheet';
        const scriptSrc = SCRIPT ? SCRIPT.src : '';
        const baseUrl = scriptSrc.substring(0, scriptSrc.lastIndexOf('/'));
        link.href = (baseUrl || '.') + '/abacus-chat-widget.css?v=' + WIDGET_VERSION;
        document.head.appendChild(link);
    }

    // ---- Load GSAP (progressive enhancement) ----
    // The widget is fully functional and smooth on CSS transitions alone — GSAP is
    // layered on top for nicer easing and staggered reveals. If the CDN is blocked or
    // slow, hasGSAP() simply stays false and every animated call site below falls
    // through to its CSS-driven default, so nothing breaks or looks unfinished.
    function hasGSAP() { return typeof window.gsap !== 'undefined'; }

    function loadGSAP() {
        if (hasGSAP() || document.getElementById('abacus-gsap-lib')) return;
        const script = document.createElement('script');
        script.id = 'abacus-gsap-lib';
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js';
        script.async = true;
        script.onload = () => {
            // Flips a CSS switch (see .abacus-gsap rules) that hands entrance/exit
            // animation over to the tweens below, so CSS and GSAP never fight.
            const root = document.getElementById('abacus-chat-widget');
            if (root) root.classList.add('abacus-gsap');
        };
        document.head.appendChild(script);
    }

    function generateId() {
        if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
        return 'abacus_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 11);
    }

    function getVisitorId() {
        let id = read(VISITOR_KEY);
        if (!id) { id = generateId(); store(VISITOR_KEY, id); }
        return id;
    }

    // ---- Build Widget DOM ----
    function createWidget() {
        const container = document.createElement('div');
        container.id = 'abacus-chat-widget';

        const isClient = MODE === 'client';
        const title = isClient ? 'Abacus Digital Support' : 'Abacus Digital';
        const quickActions = isClient
            ? [
                ['What\'s the status of my project?', 'Project Status'],
                ['What deliverables are outstanding?', 'Deliverables'],
                ['When is my next milestone due?', 'Next Milestone'],
                ['I\'d like to speak to my account manager', 'Talk to a Human'],
            ]
            : [
                ['What services do you offer?', 'Our Services'],
                ['I need a website for my business', 'I Need a Website'],
                ['I\'m interested in AI & automation solutions', 'AI & Automation'],
                ['I\'d like to book a discovery call', 'Book a Call'],
            ];

        container.innerHTML = `
            <button class="abacus-trigger" id="abacus-trigger" aria-label="Open chat">
                <svg class="chat-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
                    <circle cx="8" cy="10" r="1.2"/>
                    <circle cx="12" cy="10" r="1.2"/>
                    <circle cx="16" cy="10" r="1.2"/>
                </svg>
                <svg class="close-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                </svg>
            </button>

            <div class="abacus-chat-window" id="abacus-chat-window" role="dialog" aria-label="${title} chat">
                <div class="abacus-header">
                    ${isClient ? '' : `
                    <button class="abacus-back-btn" id="abacus-back" aria-label="Back to all chats">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M15 18l-6-6 6-6"/>
                        </svg>
                    </button>`}
                    <div class="abacus-header-info">
                        <div class="abacus-header-title">${title}</div>
                        <div class="abacus-header-status" id="abacus-status"></div>
                    </div>
                    ${isClient ? '' : `
                    <button class="abacus-newchat-btn" id="abacus-newchat" aria-label="Start a new chat" title="New chat">
                        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 5v14M5 12h14"/>
                        </svg>
                    </button>`}
                    <button class="abacus-header-close" id="abacus-close" aria-label="Close chat">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
                            <path d="M18 6L6 18M6 6l12 12"/>
                        </svg>
                    </button>
                </div>

                ${isClient ? '' : `
                <div class="abacus-chatlist" id="abacus-chatlist">
                    <button class="abacus-chatlist-new" id="abacus-chatlist-new">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 5v14M5 12h14"/>
                        </svg>
                        Start a new chat
                    </button>
                    <div class="abacus-chatlist-rows" id="abacus-chatlist-rows"></div>
                </div>`}

                <div class="abacus-consent ${isClient ? '' : 'visible'}" id="abacus-consent">
                    <span class="abacus-consent-icon">🔒</span>
                    <span>This chat may be recorded to follow up with you. By continuing, you agree.</span>
                </div>

                ${isClient ? `
                <div class="abacus-login" id="abacus-login">
                    <div class="abacus-login-title">Client sign-in</div>
                    <p class="abacus-login-copy">Enter the email address on your account and we'll send you a secure sign-in link.</p>
                    <input class="abacus-input abacus-login-input" id="abacus-login-email" type="email" placeholder="you@company.com" autocomplete="email">
                    <button class="abacus-login-btn" id="abacus-login-send">Send sign-in link</button>
                    <div class="abacus-login-msg" id="abacus-login-msg"></div>
                </div>` : ''}

                <div class="abacus-messages" id="abacus-messages"></div>

                <div class="abacus-typing" id="abacus-typing">
                    <div class="abacus-typing-dot"></div>
                    <div class="abacus-typing-dot"></div>
                    <div class="abacus-typing-dot"></div>
                </div>

                <div class="abacus-quick-actions" id="abacus-quick-actions">
                    ${quickActions.map(([msg, label]) =>
                        `<button class="abacus-quick-btn" data-msg="${escapeAttr(msg)}">${escapeHtml(label)}</button>`
                    ).join('')}
                </div>

                <div class="abacus-input-area" id="abacus-input-area">
                    <div class="abacus-input-row">
                        <textarea class="abacus-input" id="abacus-input" placeholder="Type your message..." rows="1" aria-label="Chat message"></textarea>
                        <button class="abacus-send-btn" id="abacus-send" aria-label="Send message">
                            <svg viewBox="0 0 24 24" fill="none" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 19V5M5 12l7-7 7 7"/>
                            </svg>
                        </button>
                    </div>
                    <div class="abacus-powered">
                        Powered by <a href="https://www.abacusdigital.net" target="_blank" rel="noopener">Abacus Digital</a>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(container);
    }

    // ---- Escaping / rendering ----
    function escapeHtml(text) {
        return String(text == null ? '' : text)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function escapeAttr(text) { return escapeHtml(text); }

    // Only http(s) and mailto links are rendered; anything else (javascript:, data:)
    // is dropped so model output can never inject an executable URL.
    function safeUrl(url) {
        const trimmed = String(url || '').trim();
        return /^(https?:\/\/|mailto:)/i.test(trimmed) ? trimmed : '';
    }

    function renderMarkdown(text) {
        if (!text) return '';

        // Inline formatting first (bold, links) — safe to run before splitting into
        // blocks since neither pattern spans a line break.
        const inline = escapeHtml(text)
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (match, label, url) => {
                const href = safeUrl(url);
                return href
                    ? `<a href="${escapeAttr(href)}" target="_blank" rel="noopener noreferrer">${label}</a>`
                    : label;
            });

        const BULLET_RE = /^[•●\-*]\s+/;
        const NUMBERED_RE = /^\d+\.\s+/;

        // Split on blank lines into blocks, then walk each block's lines grouping
        // consecutive bullets into one clean <ul> (no stray <br> between items — that
        // was inserting a spurious extra line break after every bullet) and consecutive
        // prose lines into one <p> (single line breaks preserved as <br>). A block often
        // mixes an intro line with a bullet run right underneath it, with no blank line
        // in between, so list detection has to work within a block, not just across them.
        // Block-to-block and run-to-run spacing comes from CSS margins, never literal
        // <br><br>.
        return inline.split(/\n{2,}/).map(block => {
            const lines = block.split('\n').filter(l => l.trim().length);
            if (!lines.length) return '';

            const parts = [];
            let run = [];
            let runIsList = false;

            function flush() {
                if (!run.length) return;
                parts.push(runIsList
                    ? '<ul>' + run.map(item => `<li>${item}</li>`).join('') + '</ul>'
                    : `<p>${run.join('<br>')}</p>`);
                run = [];
            }

            lines.forEach(line => {
                const isBullet = BULLET_RE.test(line) || NUMBERED_RE.test(line);
                if (run.length && isBullet !== runIsList) flush();
                runIsList = isBullet;
                run.push(isBullet ? line.replace(BULLET_RE, '').replace(NUMBERED_RE, '') : line);
            });
            flush();

            return parts.join('');
        }).join('');
    }

    function relativeTime(isoString) {
        if (!isoString) return '';
        const then = new Date(isoString.endsWith('Z') ? isoString : isoString + 'Z').getTime();
        if (Number.isNaN(then)) return '';
        const diffSec = Math.max(0, Math.round((Date.now() - then) / 1000));
        if (diffSec < 60) return 'now';
        const diffMin = Math.round(diffSec / 60);
        if (diffMin < 60) return diffMin + 'm';
        const diffHr = Math.round(diffMin / 60);
        if (diffHr < 24) return diffHr + 'h';
        const diffDay = Math.round(diffHr / 24);
        if (diffDay < 7) return diffDay + 'd';
        return Math.round(diffDay / 7) + 'w';
    }

    // ---- Message rendering ----
    function addMessage(role, content, options) {
        options = options || {};
        const messagesEl = document.getElementById('abacus-messages');
        if (!messagesEl) return;

        // Only the most recently attached suggestion chips should stay actionable —
        // once the conversation moves on, older chip rows are stripped.
        messagesEl.querySelectorAll('.abacus-suggestions').forEach(el => el.remove());

        const isUser = role === 'user';
        const msgDiv = document.createElement('div');
        msgDiv.className = `abacus-message ${isUser ? 'user' : 'bot'}`;

        let html = `<div class="abacus-msg-content">`;
        if (!isUser) html += `<div class="abacus-msg-label">Abacus</div>`;
        html += renderMarkdown(content);

        const sourceUrl = options.sourceLink && safeUrl(options.sourceLink.url);
        if (sourceUrl) {
            html += `<div class="abacus-source-link">🔗 <a href="${escapeAttr(sourceUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(options.sourceLink.label || 'Learn more')}</a></div>`;
        }

        const bookingHref = safeUrl(options.bookingUrl);
        if (options.showBooking && bookingHref) {
            html += `<a class="abacus-booking-btn" href="${escapeAttr(bookingHref)}" target="_blank" rel="noopener noreferrer">📅 Book Discovery Call</a>`;
        }

        msgDiv.innerHTML = html + '</div>';
        messagesEl.appendChild(msgDiv);

        // Restored/cached transcripts render instantly; only genuinely new turns
        // (a live send or reply) get the entrance animation.
        if (hasGSAP() && !options.skipAnimation) {
            gsap.fromTo(msgDiv,
                { opacity: 0, y: 12, scale: 0.98 },
                { opacity: 1, y: 0, scale: 1, duration: 0.4, ease: 'power3.out' }
            );
        }

        if (!isUser && options.suggestions && options.suggestions.length) {
            const chipRow = document.createElement('div');
            chipRow.className = 'abacus-suggestions';
            chipRow.innerHTML = options.suggestions.map(s =>
                `<button class="abacus-suggestion-btn" data-msg="${escapeAttr(s)}">${escapeHtml(s)}</button>`
            ).join('');
            messagesEl.appendChild(chipRow);
            const chips = chipRow.querySelectorAll('.abacus-suggestion-btn');
            chips.forEach(btn => {
                btn.addEventListener('click', () => sendMessage(btn.getAttribute('data-msg')));
            });
            if (hasGSAP() && !options.skipAnimation) {
                gsap.fromTo(chips,
                    { opacity: 0, y: 8 },
                    { opacity: 1, y: 0, duration: 0.32, stagger: 0.05, ease: 'power2.out', delay: 0.08 }
                );
            }
        }

        messagesEl.scrollTop = messagesEl.scrollHeight;

        if (!options.skipPersist) persistTranscript(role, content, options);
    }

    // ---- Per-chat transcript cache (instant reopen; server stays the source of truth) ----
    function transcriptKey(id) { return TRANSCRIPT_PREFIX + id; }

    function persistTranscript(role, content, options) {
        if (!sessionId) return;
        const key = transcriptKey(sessionId);
        const history = readJSON(key, []);
        history.push({
            role, content,
            sourceLink: options.sourceLink || null,
            showBooking: !!options.showBooking,
            bookingUrl: options.bookingUrl || null,
        });
        storeJSON(key, history.slice(-60));
    }

    function loadCachedTranscript(id) {
        return readJSON(transcriptKey(id), null);
    }

    function renderTranscript(messages) {
        const messagesEl = document.getElementById('abacus-messages');
        if (messagesEl) messagesEl.innerHTML = '';
        messages.forEach(m => addMessage(m.role, m.content, {
            sourceLink: m.sourceLink, showBooking: m.showBooking, bookingUrl: m.bookingUrl,
            skipPersist: true, skipAnimation: true,
        }));
        messageCount = messages.length;
        if (messageCount > 0) hideQuickActions();
    }

    function hideConsentBanner() {
        const consent = document.getElementById('abacus-consent');
        if (consent) consent.classList.remove('visible');
    }

    function hideQuickActions() {
        const el = document.getElementById('abacus-quick-actions');
        if (el) el.style.display = 'none';
    }

    function showQuickActionsIfEmpty() {
        const el = document.getElementById('abacus-quick-actions');
        if (el) el.style.display = messageCount ? 'none' : '';
    }

    function setTyping(show) {
        const el = document.getElementById('abacus-typing');
        if (el) el.classList.toggle('active', show);
    }

    function setStatus(text) {
        const el = document.getElementById('abacus-status');
        if (el) el.textContent = text;
    }

    // ---- Backend availability: degrade to the contact form (PRD 8) ----
    async function checkBackend() {
        try {
            const res = await fetch(`${API_URL}/health`, { method: 'GET' });
            backendUp = res.ok;
        } catch (e) {
            backendUp = false;
        }
        if (!backendUp) showOfflineFallback();
        return backendUp;
    }

    function showOfflineFallback() {
        setStatus('Offline');
        hideQuickActions();
        setView('chat');
        const inputArea = document.getElementById('abacus-input-area');
        if (inputArea) inputArea.style.display = 'none';
        addMessage('assistant',
            'Our assistant is offline right now, but the team still wants to hear from you. ' +
            `Reach us through the [contact form](${CONTACT_URL}) and we'll get straight back to you.`,
            { skipPersist: true }
        );
    }

    // ---- View switching (public mode: chat list <-> chat) ----
    function setView(next) {
        view = next;
        const listEl = document.getElementById('abacus-chatlist');
        const messagesEl = document.getElementById('abacus-messages');
        const inputArea = document.getElementById('abacus-input-area');
        const quick = document.getElementById('abacus-quick-actions');
        const backBtn = document.getElementById('abacus-back');
        const newBtn = document.getElementById('abacus-newchat');
        const typing = document.getElementById('abacus-typing');
        const consent = document.getElementById('abacus-consent');

        const showList = next === 'list';
        if (listEl) listEl.style.display = showList ? '' : 'none';
        if (messagesEl) messagesEl.style.display = showList ? 'none' : '';
        if (inputArea) inputArea.style.display = showList ? 'none' : '';
        if (typing) typing.style.display = showList ? 'none' : '';
        if (consent) consent.style.display = showList ? 'none' : '';
        if (quick) quick.style.display = showList ? 'none' : (messageCount ? 'none' : '');
        if (backBtn) backBtn.style.display = showList ? 'none' : '';
        if (newBtn) newBtn.style.display = showList ? 'none' : '';

        setStatus(showList ? 'Your conversations' : '');
    }

    async function fetchChatList() {
        try {
            const res = await fetch(
                `${API_URL}/api/chats?visitor_id=${encodeURIComponent(visitorId)}` +
                (sessionId ? `&exclude=${encodeURIComponent(sessionId)}` : '')
            );
            if (!res.ok) return [];
            const data = await res.json();
            return data.chats || [];
        } catch (e) {
            return [];
        }
    }

    function renderChatList(chats) {
        const rowsEl = document.getElementById('abacus-chatlist-rows');
        if (!rowsEl) return;

        if (!chats.length) {
            rowsEl.innerHTML = '<div class="abacus-chatlist-empty">No previous conversations yet.</div>';
            return;
        }

        rowsEl.innerHTML = chats.map(c => `
            <button class="abacus-chatlist-row" data-id="${escapeAttr(c.session_id)}">
                <div class="abacus-chatlist-row-top">
                    <span class="abacus-chatlist-title">${escapeHtml(c.title || 'New conversation')}</span>
                    <span class="abacus-chatlist-time">${escapeHtml(relativeTime(c.updated_at))}</span>
                </div>
                <div class="abacus-chatlist-snippet">${escapeHtml(c.snippet || '')}</div>
            </button>
        `).join('');

        const rows = rowsEl.querySelectorAll('.abacus-chatlist-row');
        rows.forEach(row => {
            row.addEventListener('click', () => openChat(row.getAttribute('data-id')));
        });

        if (hasGSAP()) {
            gsap.fromTo(rows,
                { opacity: 0, y: 10 },
                { opacity: 1, y: 0, duration: 0.32, stagger: 0.04, ease: 'power2.out' }
            );
        }
    }

    async function showChatListView() {
        setView('list');
        const chats = await fetchChatList();
        renderChatList(chats);
    }

    async function openChat(id) {
        sessionId = id;
        store(CURRENT_SESSION_KEY, id);
        messageCount = 0;
        consentGiven = true;
        setView('chat');
        hideConsentBanner();

        const cached = loadCachedTranscript(id);
        if (cached && cached.length) {
            renderTranscript(cached);
            return;
        }

        setTyping(true);
        try {
            const res = await fetch(
                `${API_URL}/api/chats/${encodeURIComponent(id)}?visitor_id=${encodeURIComponent(visitorId)}`
            );
            setTyping(false);
            if (!res.ok) {
                addMessage('assistant', 'I couldn\'t load that conversation. Starting a new one instead.', { skipPersist: true });
                startNewChat();
                return;
            }
            const data = await res.json();
            const messages = (data.transcript || []).map(m => ({ role: m.role, content: m.content }));
            renderTranscript(messages);
            messages.forEach(m => persistTranscript(m.role, m.content, {}));
        } catch (e) {
            setTyping(false);
            startNewChat();
        }
    }

    function startNewChat() {
        sessionId = generateId();
        store(CURRENT_SESSION_KEY, sessionId);
        messageCount = 0;
        consentGiven = false;
        setView('chat');

        const messagesEl = document.getElementById('abacus-messages');
        if (messagesEl) messagesEl.innerHTML = '';
        const consent = document.getElementById('abacus-consent');
        if (consent) consent.classList.add('visible');
        showQuickActionsIfEmpty();

        addMessage('assistant',
            'Hi! 👋 I\'m the Abacus Digital assistant. I can answer questions about our services, ' +
            'help scope a project, or get you booked in with the team.\n\n' +
            'A quick note: this chat may be recorded so we can follow up with you.',
            { skipPersist: true }
        );

        const input = document.getElementById('abacus-input');
        if (input) input.focus();
    }

    // ---- Send Message ----
    async function sendMessage(text) {
        if (!text || !text.trim() || isWaiting) return;
        if (!backendUp) return;
        if (MODE === 'client' && !clientToken) return;
        if (MODE === 'public' && view === 'list') return;

        text = text.trim();
        isWaiting = true;

        addMessage('user', text);

        const input = document.getElementById('abacus-input');
        if (input) { input.value = ''; input.style.height = 'auto'; }

        hideQuickActions();

        if (!consentGiven) {
            consentGiven = true;
            const consent = document.getElementById('abacus-consent');
            if (consent) consent.classList.remove('visible');
        }

        setTyping(true);
        const sendBtn = document.getElementById('abacus-send');
        if (sendBtn) sendBtn.disabled = true;

        try {
            const endpoint = MODE === 'client' ? '/api/client/chat' : '/api/chat';
            const headers = { 'Content-Type': 'application/json' };
            if (MODE === 'client') headers['Authorization'] = 'Bearer ' + clientToken;

            const body = MODE === 'client'
                ? { session_id: sessionId, message: text, source_page: window.location.href }
                : {
                    session_id: sessionId, visitor_id: visitorId, message: text,
                    source_page: window.location.href, consent_given: consentGiven,
                };

            const response = await fetch(API_URL + endpoint, {
                method: 'POST',
                headers,
                body: JSON.stringify(body),
            });

            if (response.status === 401 && MODE === 'client') {
                signOut('Your session expired. Please sign in again.');
                return;
            }

            const data = await response.json();

            if (data.session_id) {
                sessionId = data.session_id;
                if (MODE === 'public') store(CURRENT_SESSION_KEY, sessionId);
            }

            setTyping(false);
            addMessage('assistant', data.message, {
                sourceLink: data.source_link,
                showBooking: data.show_booking,
                bookingUrl: data.booking_url,
                suggestions: data.suggestions,
            });
            messageCount++;

        } catch (error) {
            console.error('Abacus Chat Error:', error);
            setTyping(false);
            addMessage('assistant',
                'I\'m having trouble connecting right now. Please try again in a moment, or reach us ' +
                `through the [contact form](${CONTACT_URL}).`,
                { skipPersist: true }
            );
        } finally {
            isWaiting = false;
            if (sendBtn) sendBtn.disabled = false;
            const inputEl = document.getElementById('abacus-input');
            if (inputEl) inputEl.focus();
        }
    }

    // ---- Client auth (Phase 3) ----
    async function requestMagicLink() {
        const emailEl = document.getElementById('abacus-login-email');
        const msgEl = document.getElementById('abacus-login-msg');
        const email = (emailEl && emailEl.value || '').trim();

        if (!/^[^@\s]+@[^@\s]+\.[a-z]{2,}$/i.test(email)) {
            if (msgEl) msgEl.textContent = 'Please enter a valid email address.';
            return;
        }

        if (msgEl) msgEl.textContent = 'Sending…';
        try {
            const res = await fetch(`${API_URL}/api/client/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email }),
            });
            const data = await res.json();
            if (msgEl) msgEl.textContent = data.message;

            // Local development without an email provider: the API hands back the token
            if (data.debug_token) verifyToken(data.debug_token);
        } catch (e) {
            if (msgEl) msgEl.textContent = 'Could not reach the server. Please try again shortly.';
        }
    }

    async function verifyToken(token) {
        try {
            const res = await fetch(`${API_URL}/api/client/verify`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token }),
            });
            const data = await res.json();
            if (!data.success) {
                const msgEl = document.getElementById('abacus-login-msg');
                if (msgEl) msgEl.textContent = data.message;
                return false;
            }

            clientToken = data.session_token;
            clientInfo = { name: data.client_name, company: data.client_company };
            store(TOKEN_KEY, clientToken);
            showClientChat();
            return true;
        } catch (e) {
            return false;
        }
    }

    function showClientChat() {
        const login = document.getElementById('abacus-login');
        if (login) login.style.display = 'none';
        const inputArea = document.getElementById('abacus-input-area');
        if (inputArea) inputArea.style.display = '';
        const quick = document.getElementById('abacus-quick-actions');
        if (quick && !messageCount) quick.style.display = '';

        setStatus(clientInfo && clientInfo.company ? `Signed in — ${clientInfo.company}` : 'Signed in');

        const cached = sessionId ? loadCachedTranscript(sessionId) : null;
        if (cached && cached.length) {
            renderTranscript(cached);
        } else {
            const name = (clientInfo && clientInfo.name) || 'there';
            addMessage('assistant',
                `Hi ${name} 👋 You're signed in to Abacus Digital support. I can check your project ` +
                'status, deliverables, and support docs. If you\'d rather speak to your account manager, just say so.',
                { skipPersist: true }
            );
        }
    }

    function showLogin() {
        const login = document.getElementById('abacus-login');
        if (login) login.style.display = '';
        const inputArea = document.getElementById('abacus-input-area');
        if (inputArea) inputArea.style.display = 'none';
        hideQuickActions();
        setStatus('Sign in to continue');
    }

    function signOut(message) {
        clientToken = null;
        sessionId = null;
        store(TOKEN_KEY, null);
        store(CURRENT_SESSION_KEY, null);
        const messages = document.getElementById('abacus-messages');
        if (messages) messages.innerHTML = '';
        showLogin();
        const msgEl = document.getElementById('abacus-login-msg');
        if (msgEl && message) msgEl.textContent = message;
    }

    async function restoreClientSession() {
        const token = read(TOKEN_KEY);
        if (!token) { showLogin(); return; }

        try {
            const res = await fetch(`${API_URL}/api/client/me`, {
                headers: { 'Authorization': 'Bearer ' + token },
            });
            if (!res.ok) { signOut(); return; }
            const data = await res.json();
            clientToken = token;
            clientInfo = { name: data.client.name, company: data.client.company };
            sessionId = read(CURRENT_SESSION_KEY) || generateId();
            showClientChat();
        } catch (e) {
            showLogin();
        }
    }

    // ---- Toggle Chat Window ----
    async function toggleChat() {
        isOpen = !isOpen;
        const chatWindow = document.getElementById('abacus-chat-window');
        const trigger = document.getElementById('abacus-trigger');

        if (trigger) trigger.classList.toggle('open', isOpen);

        if (chatWindow) {
            if (hasGSAP()) {
                gsap.killTweensOf(chatWindow);
                if (isOpen) {
                    // Adding .open first flips visibility on immediately (CSS transition
                    // is disabled while .abacus-gsap is active) so the tween is visible.
                    chatWindow.classList.add('open');
                    gsap.fromTo(chatWindow,
                        { opacity: 0, y: 16, scale: 0.96 },
                        { opacity: 1, y: 0, scale: 1, duration: 0.4, ease: 'power3.out' }
                    );
                } else {
                    gsap.to(chatWindow, {
                        opacity: 0, y: 12, scale: 0.97, duration: 0.22, ease: 'power2.in',
                        onComplete: () => {
                            chatWindow.classList.remove('open');
                            gsap.set(chatWindow, { clearProps: 'opacity,transform' });
                        },
                    });
                }
            } else {
                chatWindow.classList.toggle('open', isOpen);
            }
        }

        if (isOpen && !bootedOnce) {
            bootedOnce = true;
            await initConversation();
        }

        if (isOpen) {
            setTimeout(() => {
                const input = document.getElementById('abacus-input');
                if (input && input.offsetParent !== null) input.focus();
            }, 400);
        }
    }

    async function initConversation() {
        const healthy = await checkBackend();
        if (!healthy) return;

        if (MODE === 'client') {
            await restoreClientSession();
            return;
        }

        // Resume whatever chat was open last time this page loaded; otherwise land on
        // the chat list if there's history, or start fresh if this is a first visit.
        const current = read(CURRENT_SESSION_KEY);
        if (current) {
            const cached = loadCachedTranscript(current);
            if (cached && cached.length) {
                sessionId = current;
                consentGiven = true;
                setView('chat');
                hideConsentBanner();
                renderTranscript(cached);
                return;
            }
            // A pointer with nothing cached locally (e.g. storage was cleared) is still
            // worth resuming from the server rather than dropping the visitor into a
            // brand-new chat.
            await openChat(current);
            return;
        }

        const chats = await fetchChatList();
        if (chats.length > 0) {
            await showChatListView();
        } else {
            startNewChat();
        }
    }

    // ---- Auto-resize Textarea ----
    function autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 100) + 'px';
    }

    // ---- Event wiring ----
    function bindEvents() {
        const trigger = document.getElementById('abacus-trigger');
        if (trigger) trigger.addEventListener('click', toggleChat);

        const close = document.getElementById('abacus-close');
        if (close) close.addEventListener('click', toggleChat);

        const sendBtn = document.getElementById('abacus-send');
        if (sendBtn) sendBtn.addEventListener('click', () => {
            const input = document.getElementById('abacus-input');
            if (input) sendMessage(input.value);
        });

        const input = document.getElementById('abacus-input');
        if (input) {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage(input.value);
                }
            });
            input.addEventListener('input', () => autoResize(input));
        }

        document.querySelectorAll('.abacus-quick-btn').forEach(btn => {
            btn.addEventListener('click', () => sendMessage(btn.getAttribute('data-msg')));
        });

        const loginBtn = document.getElementById('abacus-login-send');
        if (loginBtn) loginBtn.addEventListener('click', requestMagicLink);

        const loginInput = document.getElementById('abacus-login-email');
        if (loginInput) loginInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') requestMagicLink();
        });

        // Public-mode chat list navigation
        const backBtn = document.getElementById('abacus-back');
        if (backBtn) backBtn.addEventListener('click', showChatListView);

        const newBtn = document.getElementById('abacus-newchat');
        if (newBtn) newBtn.addEventListener('click', startNewChat);

        const listNewBtn = document.getElementById('abacus-chatlist-new');
        if (listNewBtn) listNewBtn.addEventListener('click', startNewChat);

        document.addEventListener('keydown', e => {
            if (e.key === 'Escape' && isOpen) toggleChat();
        });

        // Close out the session so the backend can summarise it for the sales team
        window.addEventListener('pagehide', () => {
            if (MODE === 'public' && sessionId && messageCount > 1 && navigator.sendBeacon) {
                navigator.sendBeacon(`${API_URL}/api/sessions/${sessionId}/end`);
            }
        });
    }

    // ---- Boot ----
    function init() {
        loadCSS();
        createWidget();
        loadGSAP();
        bindEvents();

        visitorId = getVisitorId();

        if (MODE === 'client') {
            sessionId = read(CURRENT_SESSION_KEY);
            // A magic link lands on the page as ?token=...; consume it and clean the URL
            const params = new URLSearchParams(window.location.search);
            const token = params.get('token');
            if (token) {
                verifyToken(token).then(() => {
                    params.delete('token');
                    const query = params.toString();
                    history.replaceState({}, '', window.location.pathname + (query ? '?' + query : ''));
                    if (!isOpen) toggleChat();
                });
            }
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
