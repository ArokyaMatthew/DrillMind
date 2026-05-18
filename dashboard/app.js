/**
 * DrillMind RTOC Dashboard — Application Logic
 * ==============================================
 * Fetches data from the FastAPI backend, renders real-time charts
 * using Chart.js, populates KPIs and anomaly events.
 */

const API_BASE = 'http://localhost:8000';
const MAX_CHART_POINTS = 500;
const SPARKLINE_SIZE = 100;

// ---- State ----
let charts = {};
let chartData = {
    timestamps: [],
    spp: [],
    hookload: [],
    torque: [],
    pitVolume: [],
    anomalyScore: [],
    isAnomaly: [],
};
let allEvents = [];
let currentFilter = 'all';

// ---- Sparkline Ring Buffers (fixed-size, zero-alloc after init) ----
const SPARK_CHANNELS = ['Depth','WOB','SPP','Hookload','Torque','RPM','MSE','Anomaly'];
const sparkBuffers = {};
const sparkMinMax = {};
SPARK_CHANNELS.forEach(ch => {
    sparkBuffers[ch] = new Float32Array(SPARKLINE_SIZE);
    sparkMinMax[ch] = { min: Infinity, max: -Infinity, idx: 0 };
});

function pushSparkValue(channel, val) {
    const buf = sparkBuffers[channel];
    const mm = sparkMinMax[channel];
    const idx = mm.idx % SPARKLINE_SIZE;
    buf[idx] = val;
    mm.idx++;
    // Recompute min/max every SPARKLINE_SIZE pushes (amortized)
    if (mm.idx % SPARKLINE_SIZE === 0 || val < mm.min || val > mm.max) {
        let lo = Infinity, hi = -Infinity;
        const len = Math.min(mm.idx, SPARKLINE_SIZE);
        for (let i = 0; i < len; i++) {
            if (buf[i] < lo) lo = buf[i];
            if (buf[i] > hi) hi = buf[i];
        }
        mm.min = lo;
        mm.max = hi;
    }
}

// ---- Sparkline Canvas Renderer ----
const sparkCanvasCache = {};
function renderSparkline(canvasId, channel, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    if (!sparkCanvasCache[canvasId]) {
        sparkCanvasCache[canvasId] = canvas.getContext('2d', { alpha: true });
        canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
        canvas.height = canvas.offsetHeight * (window.devicePixelRatio || 1);
    }
    const ctx = sparkCanvasCache[canvasId];
    const w = canvas.width;
    const h = canvas.height;
    const buf = sparkBuffers[channel];
    const mm = sparkMinMax[channel];
    const len = Math.min(mm.idx, SPARKLINE_SIZE);
    if (len < 2) return;

    ctx.clearRect(0, 0, w, h);
    const range = mm.max - mm.min || 1;
    const startIdx = mm.idx >= SPARKLINE_SIZE ? mm.idx % SPARKLINE_SIZE : 0;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2 * (window.devicePixelRatio || 1);
    for (let i = 0; i < len; i++) {
        const dataIdx = (startIdx + i) % SPARKLINE_SIZE;
        const x = (i / (len - 1)) * w;
        const y = h - ((buf[dataIdx] - mm.min) / range) * (h * 0.85);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

function renderAllSparklines() {
    renderSparkline('sparkDepth', 'Depth', '#60a5fa');
    renderSparkline('sparkWOB', 'WOB', '#a78bfa');
    renderSparkline('sparkSPP', 'SPP', '#818cf8');
    renderSparkline('sparkHookload', 'Hookload', '#67e8f9');
    renderSparkline('sparkTorque', 'Torque', '#fcd34d');
    renderSparkline('sparkRPM', 'RPM', '#6ee7b7');
    renderSparkline('sparkMSE', 'MSE', '#c084fc');
    renderSparkline('sparkAnomaly', 'Anomaly', '#f87171');
}

// ---- MSE Client-Side Computation (Teale, 1965) ----
const BIT_DIAMETER_INCHES = 12.25;
const BIT_DIAMETER_M = BIT_DIAMETER_INCHES * 0.0254;
const BIT_AREA_M2 = Math.PI / 4 * BIT_DIAMETER_M * BIT_DIAMETER_M;

function computeMSE(wob, torque_kNm, rpm, rop_mh) {
    if (!rop_mh || rop_mh <= 0 || !rpm || rpm <= 0) return null;
    const torque_Nm = torque_kNm * 1000;
    const rop_ms = rop_mh / 3600;
    const rpm_rps = rpm / 60;
    const rotary = (2 * Math.PI * torque_Nm * rpm_rps) / (BIT_AREA_M2 * rop_ms);
    const thrust = (wob || 0) / BIT_AREA_M2;
    const mse_mpa = (rotary + thrust) / 1e6;
    return mse_mpa > 0 && mse_mpa < 1e5 ? mse_mpa : null;
}

// ---- Chart Configuration ----
const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
        legend: { display: false },
        tooltip: {
            backgroundColor: 'rgba(26, 31, 46, 0.95)',
            borderColor: 'rgba(99, 102, 241, 0.3)',
            borderWidth: 1,
            titleFont: { family: "'Inter', sans-serif", size: 11 },
            bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
            padding: 10,
            cornerRadius: 8,
        },
    },
    scales: {
        x: {
            display: true,
            grid: { color: 'rgba(42, 48, 66, 0.5)', lineWidth: 0.5 },
            ticks: {
                color: '#64748b',
                font: { family: "'JetBrains Mono', monospace", size: 9 },
                maxTicksLimit: 8,
                maxRotation: 0,
            },
        },
        y: {
            display: true,
            grid: { color: 'rgba(42, 48, 66, 0.5)', lineWidth: 0.5 },
            ticks: {
                color: '#64748b',
                font: { family: "'JetBrains Mono', monospace", size: 10 },
                maxTicksLimit: 5,
            },
        },
    },
};

function createChart(canvasId, color, label) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 180);
    gradient.addColorStop(0, color.replace(')', ', 0.25)').replace('rgb', 'rgba'));
    gradient.addColorStop(1, color.replace(')', ', 0.01)').replace('rgb', 'rgba'));

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: label,
                data: [],
                borderColor: color,
                backgroundColor: gradient,
                fill: true,
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.3,
            }],
        },
        options: { ...CHART_DEFAULTS },
    });
}

function createAnomalyChart() {
    const ctx = document.getElementById('chartAnomaly').getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Anomaly Score',
                    data: [],
                    borderColor: '#f87171',
                    backgroundColor: 'rgba(248, 113, 113, 0.06)',
                    fill: true,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.2,
                },
                {
                    label: 'Threshold',
                    data: [],
                    borderColor: 'rgba(251, 191, 36, 0.4)',
                    borderWidth: 1,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false,
                },
            ],
        },
        options: {
            ...CHART_DEFAULTS,
            plugins: {
                ...CHART_DEFAULTS.plugins,
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        color: '#94a3b8',
                        font: { family: "'Inter', sans-serif", size: 10 },
                        boxWidth: 12,
                        padding: 12,
                    },
                },
            },
        },
    });
}

// ---- Initialize Charts ----
function initCharts() {
    charts.spp = createChart('chartSPP', 'rgb(129, 140, 248)', 'SPP');
    charts.hookload = createChart('chartHookload', 'rgb(103, 232, 249)', 'Hookload');
    charts.torque = createChart('chartTorque', 'rgb(252, 211, 77)', 'Torque');
    charts.pit = createChart('chartPit', 'rgb(110, 231, 183)', 'Pit Volume');
    charts.anomaly = createAnomalyChart();
}

// ---- Update Charts with Data ----
function updateCharts(data) {
    const labels = data.map(d => {
        const ts = d.timestamp;
        return ts.substring(11, 19); // HH:MM:SS
    });

    function updateSingle(chart, values) {
        chart.data.labels = labels;
        chart.data.datasets[0].data = values;
        chart.update('none');
    }

    updateSingle(charts.spp, data.map(d => d.spp));
    updateSingle(charts.hookload, data.map(d => d.weight_on_hook));
    updateSingle(charts.torque, data.map(d => d.torque_averaged));
    updateSingle(charts.pit, data.map(d => d.pit_volume_active));
}

function updateAnomalyChart(scores) {
    const labels = scores.map(s => s.timestamp.substring(11, 19));
    const values = scores.map(s => s.combined);

    charts.anomaly.data.labels = labels;
    charts.anomaly.data.datasets[0].data = values;

    // Threshold line
    const thresholdVal = scores.length > 0 ? 0.3 : 0; // Will be updated from API
    charts.anomaly.data.datasets[1].data = labels.map(() => thresholdVal);
    charts.anomaly.update('none');
}

// ---- KPI Updates ----

let _hookloadBaseline = null;
const ON_BOTTOM_THRESHOLD = 5.0;
let _sparkRafPending = false;

function updateKPIs(record) {
    const setValue = (id, val, decimals = 1) => {
        const el = document.getElementById(id);
        if (el && val != null && !isNaN(val)) {
            el.textContent = Number(val).toFixed(decimals);
        }
    };
    const setMinMax = (minId, maxId, channel) => {
        const mm = sparkMinMax[channel];
        if (mm && mm.idx > 0) {
            const minEl = document.getElementById(minId);
            const maxEl = document.getElementById(maxId);
            if (minEl) minEl.textContent = `▼ ${mm.min.toFixed(1)}`;
            if (maxEl) maxEl.textContent = `▲ ${mm.max.toFixed(1)}`;
        }
    };

    // Push values to sparkline buffers
    if (record.bit_depth != null) pushSparkValue('Depth', record.bit_depth);
    const wobVal = (record.wob_avg != null && record.wob_avg < 0) ? 0 : (record.wob_avg || 0);
    pushSparkValue('WOB', wobVal);
    if (record.spp != null) pushSparkValue('SPP', record.spp);
    if (record.weight_on_hook != null) pushSparkValue('Hookload', record.weight_on_hook);
    if (record.torque_averaged != null) pushSparkValue('Torque', record.torque_averaged);
    pushSparkValue('RPM', record.rpm_avg || 0);

    // MSE computation (Teale)
    const rop = record.rop || record.rop_5ft_avg || 0;
    const mse = computeMSE(wobVal, record.torque_averaged || 0, record.rpm_avg || 0, rop);
    if (mse != null) pushSparkValue('MSE', mse);

    // Set display values
    setValue('kpiDepthValue', record.bit_depth, 1);
    setValue('kpiROPValue', wobVal, 2);
    setValue('kpiSPPValue', record.spp, 0);
    setValue('kpiHookloadValue', record.weight_on_hook, 1);
    setValue('kpiTorqueValue', record.torque_averaged, 2);
    setValue('kpiRPMValue', record.rpm_avg, 0);

    // MSE display
    const mseEl = document.getElementById('kpiMSEValue');
    if (mseEl) mseEl.textContent = mse != null ? mse.toFixed(1) : '—';

    // Min/max
    setMinMax('kpiDepthMin', 'kpiDepthMax', 'Depth');
    setMinMax('kpiWOBMin', 'kpiWOBMax', 'WOB');
    setMinMax('kpiSPPMin', 'kpiSPPMax', 'SPP');
    setMinMax('kpiHookMin', 'kpiHookMax', 'Hookload');
    setMinMax('kpiTorqueMin', 'kpiTorqueMax', 'Torque');
    setMinMax('kpiRPMMin', 'kpiRPMMax', 'RPM');
    setMinMax('kpiMSEMin', 'kpiMSEMax', 'MSE');

    // Timestamp
    const tsEl = document.getElementById('currentTimestamp');
    if (tsEl && record.timestamp) {
        tsEl.textContent = record.timestamp.substring(0, 19).replace('T', ' ');
    }

    // Well schematic bit depth
    const bitLabel = document.getElementById('bitDepthLabel');
    if (bitLabel && record.bit_depth != null) {
        bitLabel.textContent = `Bit: ${record.bit_depth.toFixed(0)}m`;
    }

    // On-Bottom detection
    updateOnBottomStatus(record);

    // Schedule sparkline render (throttled to ~4 FPS)
    if (!_sparkRafPending) {
        _sparkRafPending = true;
        requestAnimationFrame(() => {
            renderAllSparklines();
            _sparkRafPending = false;
        });
    }
}

function updateOnBottomStatus(record) {
    const hookload = record.weight_on_hook;
    const rpm = record.rpm_avg || 0;

    if (hookload != null && !isNaN(hookload)) {
        if (_hookloadBaseline === null) {
            _hookloadBaseline = hookload;
        } else {
            _hookloadBaseline = Math.max(hookload, _hookloadBaseline * 0.999);
        }
    }

    const isOnBottom = (
        _hookloadBaseline !== null &&
        hookload != null &&
        hookload < (_hookloadBaseline - ON_BOTTOM_THRESHOLD) &&
        rpm > 0
    );

    const card = document.getElementById('kpiOnBottom');
    const valueEl = document.getElementById('kpiOnBottomValue');
    if (valueEl) valueEl.textContent = isOnBottom ? 'ON BTM' : 'OFF BTM';
    if (card) card.classList.toggle('is-on-bottom', isOnBottom);
}

function updateAnomalyKPI(score) {
    const el = document.getElementById('kpiAnomalyScore');
    const card = document.getElementById('kpiAnomaly');
    if (el) el.textContent = score != null ? score.toFixed(3) : '—';
    if (card) card.classList.toggle('alert', score > 0.3);
    if (score != null) pushSparkValue('Anomaly', score);
}

// ---- Events ----
function renderEvents(events, filter) {
    const container = document.getElementById('eventsList');
    if (!container) return;

    let filtered = events;
    if (filter && filter !== 'all') {
        filtered = events.filter(e => e.severity === filter);
    }

    if (filtered.length === 0) {
        container.innerHTML = '<div class="loading-events">No events found</div>';
        return;
    }

    container.innerHTML = filtered.slice(0, 50).map(event => `
        <div class="event-item" data-severity="${event.severity}">
            <div class="event-severity ${event.severity}"></div>
            <div class="event-content">
                <div class="event-header">
                    <span class="event-type ${event.event_type}">${event.event_type.replace('_', ' ')}</span>
                    <span class="event-time">${event.timestamp.substring(0, 19).replace('T', ' ')}</span>
                    <span class="event-score">${event.score.toFixed(3)}</span>
                </div>
                <div class="event-description">${event.description}</div>
                <div class="event-action">→ ${event.recommended_action}</div>
            </div>
        </div>
    `).join('');
}

function renderSummary(summary) {
    const container = document.getElementById('summaryContent');
    if (!container || !summary) return;

    const anomalyPct = (summary.anomaly_rate * 100).toFixed(1);
    const normalPct = (100 - summary.anomaly_rate * 100).toFixed(1);

    let typeBreakdown = '';
    if (summary.by_type) {
        const typeColors = {
            kick: '#ef4444', lost_circulation: '#f97316', stuck_pipe: '#f97316',
            bit_dysfunction: '#f59e0b', washout: '#f59e0b', connection_gas: '#22c55e',
            unknown: '#64748b',
        };
        typeBreakdown = Object.entries(summary.by_type).map(([type, count]) => `
            <div class="type-item">
                <div class="type-dot" style="background: ${typeColors[type] || '#64748b'}"></div>
                <span class="type-name">${type.replace('_', ' ')}</span>
                <span class="type-count">${count}</span>
            </div>
        `).join('');
    }

    container.innerHTML = `
        <div class="summary-stat">
            <span class="summary-stat-label">Total Samples</span>
            <span class="summary-stat-value">${summary.total_samples?.toLocaleString() || '—'}</span>
        </div>
        <div class="summary-stat">
            <span class="summary-stat-label">Anomalous Samples</span>
            <span class="summary-stat-value" style="color: #ef4444">${summary.total_anomalous_samples?.toLocaleString() || '—'}</span>
        </div>
        <div class="summary-stat">
            <span class="summary-stat-label">Anomaly Rate</span>
            <span class="summary-stat-value">${anomalyPct}%</span>
        </div>
        <div class="summary-stat">
            <span class="summary-stat-label">Total Events</span>
            <span class="summary-stat-value">${summary.total_events || '—'}</span>
        </div>
        <div class="summary-bar">
            <div class="summary-bar-header">
                <span>Normal: ${normalPct}%</span>
                <span>Anomalous: ${anomalyPct}%</span>
            </div>
            <div class="bar-container">
                <div class="bar-segment normal" style="width: ${normalPct}%"></div>
                <div class="bar-segment anomalous" style="width: ${anomalyPct}%"></div>
            </div>
        </div>
        <div class="type-breakdown">
            ${typeBreakdown}
        </div>
    `;
}

// ---- API Calls ----
async function fetchJSON(endpoint) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`Fetch error for ${endpoint}:`, err);
        return null;
    }
}

async function loadWellInfo() {
    const info = await fetchJSON('/api/well/info');
    if (info) {
        document.getElementById('wellName').textContent = info.well || '—';
        document.getElementById('fieldName').textContent = info.field || '—';
        document.getElementById('operatorName').textContent = info.operator || '—';
        setConnected(true);
    }
}

async function loadTimeseries(start = 0, limit = 500) {
    const data = await fetchJSON(`/api/data/timeseries?start=${start}&limit=${limit}`);
    if (data && data.data) {
        updateCharts(data.data);
        if (data.data.length > 0) {
            updateKPIs(data.data[data.data.length - 1]);
        }
    }
}

async function loadAnomalyScores(start = 0, limit = 500) {
    const data = await fetchJSON(`/api/anomalies/scores?start=${start}&limit=${limit}`);
    if (data && data.scores) {
        updateAnomalyChart(data.scores);
        if (data.scores.length > 0) {
            const last = data.scores[data.scores.length - 1];
            updateAnomalyKPI(last.combined);
        }
    }
}

async function loadEvents() {
    const data = await fetchJSON('/api/anomalies/events?limit=100');
    if (data && data.events) {
        allEvents = data.events;
        renderEvents(allEvents, currentFilter);
        updateAlertTicker(allEvents);
    }
}

function updateAlertTicker(events) {
    const ticker = document.getElementById('tickerContent');
    if (!ticker || !events || events.length === 0) return;

    const severityClass = (s) => {
        if (s === 'critical') return 'ticker-critical';
        if (s === 'high') return 'ticker-high';
        if (s === 'medium') return 'ticker-medium';
        return 'ticker-info';
    };

    // Take top 10 events and duplicate for seamless scroll
    const top = events.slice(0, 10);
    const items = top.map(e => {
        const desc = e.description || e.event_type || 'Unknown event';
        const action = e.recommended_action ? ` → ${e.recommended_action}` : '';
        return `<span class="ticker-item ${severityClass(e.severity)}">[${(e.severity||'low').toUpperCase()}] ${desc}${action}</span>`;
    });

    // Duplicate for infinite scroll illusion
    ticker.innerHTML = items.join('') + items.join('');
}

async function loadSummary() {
    const data = await fetchJSON('/api/anomalies/summary');
    if (data) {
        renderSummary(data);
    }
}

// ---- Connection Status ----
function setConnected(connected) {
    const el = document.getElementById('connectionStatus');
    if (!el) return;
    el.className = 'status-indicator ' + (connected ? 'connected' : 'error');
    el.querySelector('span').textContent = connected ? 'Connected' : 'Disconnected';
}

// ---- Filter Buttons ----
function setupFilters() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.dataset.filter;
            renderEvents(allEvents, currentFilter);
        });
    });
}

// ---- Streaming offset for browsing ----
let browseOffset = 0;
const BROWSE_STEP = 500;

function setupNavigation() {
    // Add keyboard navigation for browsing through the data
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight') {
            browseOffset = Math.min(browseOffset + BROWSE_STEP, 419000);
            loadTimeseries(browseOffset, BROWSE_STEP);
            loadAnomalyScores(browseOffset, BROWSE_STEP);
        } else if (e.key === 'ArrowLeft') {
            browseOffset = Math.max(browseOffset - BROWSE_STEP, 0);
            loadTimeseries(browseOffset, BROWSE_STEP);
            loadAnomalyScores(browseOffset, BROWSE_STEP);
        }
    });
}

// ---- Rig State ----
async function loadRigState() {
    const data = await fetchJSON('/api/rig/summary');
    if (data && data.states) {
        // Find the state with highest count
        const topState = Object.entries(data.states)
            .sort((a, b) => b[1].count - a[1].count)[0];
        if (topState) {
            const stateEl = document.getElementById('kpiRigStateValue');
            const cardEl = document.getElementById('kpiRigState');
            if (stateEl) stateEl.textContent = topState[0].replace('_', ' ');
            if (cardEl) cardEl.setAttribute('data-state', topState[0]);
        }
    }
}

// ---- Copilot Chat ----
function simpleMarkdown(text) {
    // Convert basic markdown to HTML
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n## (.*)/g, '<h2>$1</h2>')
        .replace(/\n### (.*)/g, '<h3>$1</h3>')
        .replace(/^## (.*)/gm, '<h2>$1</h2>')
        .replace(/^### (.*)/gm, '<h3>$1</h3>')
        .replace(/^- (.*)/gm, '<li>$1</li>')
        .replace(/((<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>')
        .replace(/✅/g, '<span style="color:#22c55e">✅</span>')
        .replace(/⚠️/g, '<span style="color:#f59e0b">⚠️</span>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
}

function addMessage(role, content, meta = null) {
    const container = document.getElementById('copilotMessages');
    // Remove welcome screen if present
    const welcome = container.querySelector('.copilot-welcome');
    if (welcome) welcome.remove();

    const msgDiv = document.createElement('div');
    msgDiv.className = `copilot-msg ${role}`;

    let html = `<div class="msg-bubble">${role === 'assistant' ? simpleMarkdown(content) : content}</div>`;
    if (meta) {
        html += `<div class="msg-meta">${meta}</div>`;
    }
    msgDiv.innerHTML = html;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function showTyping() {
    const container = document.getElementById('copilotMessages');
    const typing = document.createElement('div');
    typing.className = 'copilot-msg assistant';
    typing.id = 'typingIndicator';
    typing.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    container.appendChild(typing);
    container.scrollTop = container.scrollHeight;
}

function removeTyping() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

async function sendCopilotQuery(question) {
    if (!question.trim()) return;

    const input = document.getElementById('copilotInput');
    const sendBtn = document.getElementById('copilotSend');
    const status = document.getElementById('copilotStatus');

    // Add user message
    addMessage('user', question);
    input.value = '';
    sendBtn.disabled = true;
    status.textContent = 'Thinking...';
    status.className = 'copilot-status thinking';
    showTyping();

    try {
        const res = await fetch(`${API_BASE}/api/copilot/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        removeTyping();

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        const ctx = data.context_summary || {};
        const toolsList = (ctx.tools_called || []).join(', ');
        const meta = `Agent (${ctx.intent || 'general'}) • tools: ${toolsList} • evidence: ${ctx.evidence_count || 0} • ${ctx.total_time_ms || 0}ms`;
        addMessage('assistant', data.answer, meta);

    } catch (err) {
        removeTyping();
        addMessage('assistant', `Error: ${err.message}. Check that the API server is running.`);
    }

    sendBtn.disabled = false;
    status.textContent = 'Ready';
    status.className = 'copilot-status';
}

function setupCopilot() {
    const input = document.getElementById('copilotInput');
    const sendBtn = document.getElementById('copilotSend');

    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendCopilotQuery(input.value);
            }
        });
    }

    if (sendBtn) {
        sendBtn.addEventListener('click', () => sendCopilotQuery(input.value));
    }

    // Suggestion buttons
    document.querySelectorAll('.suggestion-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.dataset.query;
            if (query) sendCopilotQuery(query);
        });
    });
}

// ---- Offset Well Comparison ----
let offsetChart = null;

const WELL_COLORS = [
    '#818cf8', '#67e8f9', '#6ee7b7', '#fcd34d',
    '#f87171', '#c084fc', '#f472b6', '#2dd4bf',
];

async function loadOffsetWells() {
    const data = await fetchJSON('/api/data/production?limit=5000');
    if (!data || !data.data || data.data.length === 0) return;

    const byWell = {};
    data.data.forEach(row => {
        const rawName = row.wellbore_code || row.wellbore || 'Unknown';
        const well = rawName.replace(/^NO 15\/9-/, '');
        if (!byWell[well]) byWell[well] = [];
        byWell[well].push(row);
    });

    const wellNames = Object.keys(byWell);
    const datasets = wellNames.map((well, i) => {
        const rows = byWell[well].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        let cumOil = 0;
        const points = rows.map((r, day) => {
            cumOil += (r.oil_vol || 0);
            return { x: day, y: cumOil };
        });
        return {
            label: well,
            data: points,
            borderColor: WELL_COLORS[i % WELL_COLORS.length],
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
        };
    });

    const ctx = document.getElementById('chartOffset');
    if (!ctx) return;

    if (offsetChart) offsetChart.destroy();
    offsetChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            ...CHART_DEFAULTS,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { family: "'Inter', sans-serif", size: 10 },
                        boxWidth: 12,
                        padding: 10,
                    },
                },
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Days', color: '#64748b' },
                    grid: { color: 'rgba(42, 48, 66, 0.3)' },
                    ticks: { color: '#64748b' },
                },
                y: {
                    title: { display: true, text: 'Cumulative Oil (Sm³)', color: '#64748b' },
                    grid: { color: 'rgba(42, 48, 66, 0.3)' },
                    ticks: { color: '#64748b' },
                },
            },
        },
    });
}

// ---- Theme Toggle ----
function setupThemeToggle() {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    // Load saved preference
    const saved = localStorage.getItem('drillmind-theme');
    if (saved) {
        document.documentElement.setAttribute('data-theme', saved);
        btn.textContent = saved === 'light' ? '☀' : '☾';
    }
    btn.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('drillmind-theme', next);
        btn.textContent = next === 'light' ? '☀' : '☾';
    });
}

// ---- Copilot Panel Toggle ----
function setupCopilotToggle() {
    const toggleBtn = document.getElementById('copilotToggle');
    const panel = document.getElementById('copilotSection');
    if (!toggleBtn || !panel) return;

    toggleBtn.addEventListener('click', () => {
        const isOpen = !panel.classList.contains('collapsed');
        panel.classList.toggle('collapsed', isOpen);
        document.body.classList.toggle('copilot-open', !isOpen);
        toggleBtn.textContent = isOpen ? 'AI' : '✕';
    });
}

// ---- Pre-fetch Init Sequence ----
async function initializeSparklineBuffers() {
    // Load last 100 points into sparkline buffers before WebSocket connects
    const data = await fetchJSON('/api/data/timeseries?start=0&limit=100');
    if (data && data.data) {
        data.data.forEach(record => {
            if (record.bit_depth != null) pushSparkValue('Depth', record.bit_depth);
            const wob = (record.wob_avg != null && record.wob_avg < 0) ? 0 : (record.wob_avg || 0);
            pushSparkValue('WOB', wob);
            if (record.spp != null) pushSparkValue('SPP', record.spp);
            if (record.weight_on_hook != null) pushSparkValue('Hookload', record.weight_on_hook);
            if (record.torque_averaged != null) pushSparkValue('Torque', record.torque_averaged);
            pushSparkValue('RPM', record.rpm_avg || 0);
            const rop = record.rop || record.rop_5ft_avg || 0;
            const mse = computeMSE(wob, record.torque_averaged || 0, record.rpm_avg || 0, rop);
            if (mse != null) pushSparkValue('MSE', mse);
        });
        // Render sparklines immediately
        renderAllSparklines();
        // Set initial KPI values from last record
        if (data.data.length > 0) {
            updateKPIs(data.data[data.data.length - 1]);
        }
    }
}

// ---- Replay Engine ----
const TOTAL_ROWS = 419745;
let replaySpeed = 1;       // multiplier: 1, 10, 100
let replayInterval = null;
const REPLAY_TICK_MS = 2000;  // advance every 2 seconds
const ROWS_PER_TICK = { 1: 50, 10: 500, 100: 5000 };

function setupReplayControls() {
    document.querySelectorAll('.replay-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.replay-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            replaySpeed = parseInt(btn.dataset.speed) || 1;
        });
    });
}

function replayTick() {
    const step = ROWS_PER_TICK[replaySpeed] || 50;
    browseOffset += step;

    if (browseOffset >= TOTAL_ROWS - BROWSE_STEP) {
        browseOffset = 0; // loop back to start
    }

    loadTimeseries(browseOffset, BROWSE_STEP);
    loadAnomalyScores(browseOffset, BROWSE_STEP);
}

function startReplay() {
    if (replayInterval) clearInterval(replayInterval);
    replayInterval = setInterval(replayTick, REPLAY_TICK_MS);
}

// ---- Initialize ----
async function init() {
    initCharts();
    setupFilters();
    setupNavigation();
    setupCopilot();
    setupThemeToggle();
    setupCopilotToggle();
    setupReplayControls();

    // 1. Pre-fill sparklines from historical data (eliminates empty state)
    await initializeSparklineBuffers();

    // 2. Load all remaining data in parallel
    await Promise.all([
        loadWellInfo(),
        loadTimeseries(0, 500),
        loadAnomalyScores(0, 500),
        loadEvents(),
        loadSummary(),
        loadRigState(),
        loadOffsetWells(),
    ]);

    // 3. Start auto-replay (advances through the dataset)
    startReplay();
}

// Start when DOM ready
document.addEventListener('DOMContentLoaded', init);
