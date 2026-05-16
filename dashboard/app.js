/**
 * DrillMind RTOC Dashboard — Application Logic
 * ==============================================
 * Fetches data from the FastAPI backend, renders real-time charts
 * using Chart.js, populates KPIs and anomaly events.
 */

const API_BASE = 'http://localhost:8000';
const MAX_CHART_POINTS = 500;

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
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.08)',
                    fill: true,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.2,
                },
                {
                    label: 'Threshold',
                    data: [],
                    borderColor: 'rgba(245, 158, 11, 0.5)',
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
    charts.spp = createChart('chartSPP', 'rgb(99, 102, 241)', 'SPP');
    charts.hookload = createChart('chartHookload', 'rgb(6, 182, 212)', 'Hookload');
    charts.torque = createChart('chartTorque', 'rgb(245, 158, 11)', 'Torque');
    charts.pitVolume = createChart('chartPitVolume', 'rgb(34, 197, 94)', 'Pit Volume');
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
    updateSingle(charts.pitVolume, data.map(d => d.pit_volume_active));
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
function updateKPIs(record) {
    const setValue = (id, val, decimals = 1) => {
        const el = document.getElementById(id);
        if (el && val != null && !isNaN(val)) {
            el.textContent = Number(val).toFixed(decimals);
        }
    };

    setValue('kpiDepthValue', record.bit_depth, 1);
    setValue('kpiROPValue', record.wob_avg, 2);
    setValue('kpiSPPValue', record.spp, 0);
    setValue('kpiHookloadValue', record.weight_on_hook, 1);
    setValue('kpiTorqueValue', record.torque_averaged, 2);
    setValue('kpiRPMValue', record.rpm_avg, 0);

    // Update timestamp
    const tsEl = document.getElementById('currentTimestamp');
    if (tsEl && record.timestamp) {
        tsEl.textContent = record.timestamp.substring(0, 19).replace('T', ' ');
    }
}

function updateAnomalyKPI(score) {
    const el = document.getElementById('kpiAnomalyScore');
    const card = document.getElementById('kpiAnomaly');
    if (el) {
        el.textContent = score != null ? score.toFixed(3) : '—';
    }
    if (card) {
        card.classList.toggle('alert', score > 0.3);
    }
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
    }
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

// ---- Initialize ----
async function init() {
    initCharts();
    setupFilters();
    setupNavigation();

    // Load all data
    await Promise.all([
        loadWellInfo(),
        loadTimeseries(0, 500),
        loadAnomalyScores(0, 500),
        loadEvents(),
        loadSummary(),
    ]);

    // Auto-refresh every 30 seconds
    setInterval(() => {
        loadTimeseries(browseOffset, BROWSE_STEP);
        loadAnomalyScores(browseOffset, BROWSE_STEP);
    }, 30000);
}

// Start when DOM ready
document.addEventListener('DOMContentLoaded', init);
