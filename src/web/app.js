/**
 * Nexus AI — Frontend Application Logic
 * Uses Server-Sent Events (SSE) for real-time agent pipeline updates.
 */

const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const typingIndicator = document.getElementById('typing-indicator');
const statusBadge = document.getElementById('status-badge');
const newChatBtn = document.getElementById('new-chat-btn');

const pipelineSteps = {
    clarity: document.getElementById('step-clarity'),
    research: document.getElementById('step-research'),
    validator: document.getElementById('step-validator'),
    synthesis: document.getElementById('step-synthesis'),
};
const connectors = document.querySelectorAll('.pipeline-connector');

let threadId = null;
const BASE_URL = window.location.origin;

// ─── Pipeline Visualization ──────────────────────────────

function resetPipeline() {
    Object.values(pipelineSteps).forEach(step => {
        step.classList.remove('active', 'done');
    });
    connectors.forEach(c => c.classList.remove('active'));
}

/** Marks an agent step as actively processing (pulsing dot). */
function setAgentActive(agentName) {
    const step = pipelineSteps[agentName];
    if (step) {
        step.classList.remove('done');
        step.classList.add('active');
    }
    // Handle clarification as a clarity step
    if (agentName === 'clarification') {
        pipelineSteps.clarity?.classList.add('active');
    }
}

/** Marks an agent step as completed (green dot). */
function setAgentDone(agentName) {
    const step = pipelineSteps[agentName];
    if (step) {
        step.classList.remove('active');
        step.classList.add('done');
    }
    if (agentName === 'clarification') {
        pipelineSteps.clarity?.classList.remove('active');
        pipelineSteps.clarity?.classList.add('done');
    }
}

function showPipelineResult(agentsUsed) {
    const order = ['clarity', 'research', 'validator', 'synthesis'];
    let lastIdx = -1;
    order.forEach((name, idx) => {
        if (agentsUsed.includes(name) || (name === 'clarity' && agentsUsed.includes('clarification'))) {
            pipelineSteps[name]?.classList.add('done');
            lastIdx = idx;
        }
    });
    if (lastIdx >= 1) connectors[0]?.classList.add('active');
    if (lastIdx >= 2) connectors[1]?.classList.add('active');
    if (lastIdx >= 3) connectors[2]?.classList.add('active');
}

function updateStatus(text, processing = true) {
    statusBadge.textContent = text;
    statusBadge.classList.toggle('processing', processing);
}

// ─── Agent Labels ────────────────────────────────────────
const AGENT_LABELS = {
    clarity: '🔍 Clarity',
    research: '📡 Research',
    validator: '✅ Validator',
    synthesis: '📝 Synthesis',
    clarification: '❓ Clarification',
};

// ─── Message Rendering ───────────────────────────────────

function appendMessage(role, content, meta = null) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', role === 'user' ? 'user-message' : 'ai-message');

    if (role === 'ai') {
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.innerHTML = marked.parse(content);
        msgDiv.appendChild(contentDiv);

        if (meta && meta.agents_used && meta.agents_used.length > 0) {
            const metaDiv = document.createElement('div');
            metaDiv.classList.add('agent-meta');

            meta.agents_used.forEach(agent => {
                const tag = document.createElement('span');
                tag.classList.add('agent-tag');
                tag.textContent = AGENT_LABELS[agent] || agent;
                metaDiv.appendChild(tag);
            });

            if (meta.confidence !== null && meta.confidence !== undefined) {
                const confTag = document.createElement('span');
                confTag.classList.add('confidence-tag');
                const score = meta.confidence;
                if (score >= 7) confTag.classList.add('confidence-high');
                else if (score >= 4) confTag.classList.add('confidence-mid');
                else confTag.classList.add('confidence-low');
                confTag.textContent = `Confidence: ${score}/10`;
                metaDiv.appendChild(confTag);
            }

            msgDiv.appendChild(metaDiv);
        }
    } else {
        msgDiv.textContent = content;
    }

    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ─── SSE Streaming Chat ──────────────────────────────────

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    userInput.value = '';
    userInput.disabled = true;
    sendBtn.disabled = true;
    typingIndicator.classList.remove('hidden');
    resetPipeline();
    updateStatus('Agents Processing...', true);

    let currentAgent = null;
    const agentsUsed = [];

    try {
        const payload = { message: text };
        if (threadId) payload.thread_id = threadId;

        const response = await fetch(`${BASE_URL}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Server error');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            let eventType = null;
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    eventType = line.slice(7).trim();
                } else if (line.startsWith('data: ') && eventType) {
                    const data = JSON.parse(line.slice(6));

                    if (eventType === 'agent') {
                        // Mark previous agent as done, new one as active
                        if (currentAgent) setAgentDone(currentAgent);
                        currentAgent = data.agent;
                        agentsUsed.push(data.agent);
                        setAgentActive(currentAgent);
                        updateStatus(`${AGENT_LABELS[data.agent] || data.agent} processing...`, true);
                    }

                    if (eventType === 'result') {
                        if (currentAgent) setAgentDone(currentAgent);
                        threadId = data.thread_id;
                        showPipelineResult(data.agents_used || agentsUsed);
                        appendMessage('ai', data.response, {
                            agents_used: data.agents_used || agentsUsed,
                            confidence: data.confidence,
                        });
                    }

                    if (eventType === 'error') {
                        throw new Error(data.detail);
                    }

                    eventType = null;
                }
            }
        }
    } catch (error) {
        console.error(error);
        resetPipeline();
        appendMessage('ai', `⚠️ Error: ${error.message}. Make sure the FastAPI server is running.`);
    } finally {
        typingIndicator.classList.add('hidden');
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
        updateStatus('Agents Standby', false);
    }
}

// ─── New Conversation ────────────────────────────────────

async function startNewChat() {
    try {
        const res = await fetch(`${BASE_URL}/new`, { method: 'POST' });
        const data = await res.json();
        threadId = data.thread_id;
    } catch {
        threadId = null;
    }
    chatWindow.innerHTML = '';
    resetPipeline();
    appendMessage('ai', 'New conversation started! What company would you like me to research?');
}

// ─── Event Listeners ─────────────────────────────────────

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
newChatBtn.addEventListener('click', startNewChat);
