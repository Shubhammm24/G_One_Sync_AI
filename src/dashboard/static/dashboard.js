/**
 * G_One_Sync AI — Clinical Dashboard v2
 * ══════════════════════════════════════════════════════
 * Real-time monitoring with Chart.js, WebSockets,
 * patient detail modals, sparklines, and live alerts.
 */

// ── Global State ───────────────────────────────────────────

const state = {
    patients: {},
    metrics: null,
    alerts: [],
    shap: [],
    ws: null,
    connected: false,
    filter: 'all',
    searchTerm: '',
    currentView: 'overview',
    trendData: [],       // { time, meanProb }
    riskHistory: [],     // { time, LOW, MODERATE, HIGH, CRITICAL }
    volumeData: [],      // { hour, count }
    patientHistory: {},  // pid -> [prob, prob, ...]
    charts: {},
    // New: alarm state
    alarmActive: false,
    alarmPatient: null,
    alarmAcknowledged: {},  // pid -> timestamp
    modalPatientId: null,
};

// ── Chart.js Defaults ──────────────────────────────────────

Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.04)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.animation.duration = 800;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;
Chart.defaults.plugins.legend.labels.boxHeight = 6;

// ── Init ───────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    startClock();
    setupNavigation();
    setupFilterTabs();
    setupSearch();
    setupModal();
    setupAlarm();
    setupWhatIfSliders();
    setupTrajectoryTabs();
    loadInitialData();
    connectWebSocket();
    loadSystemHealth();
    setInterval(loadMetrics, 8000);
    setInterval(pushTrendPoint, 5000);
    setInterval(loadSystemHealth, 10000);
});

// ── Clock ──────────────────────────────────────────────────

function startClock() {
    const el = document.getElementById('clock');
    const tick = () => {
        const now = new Date();
        el.textContent = now.toLocaleTimeString('en-US', { hour12: false });
    };
    tick();
    setInterval(tick, 1000);
}

// ── Navigation ─────────────────────────────────────────────

function setupNavigation() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const view = tab.dataset.view;
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            document.getElementById(`view${capitalize(view)}`).classList.add('active');
            state.currentView = view;

            // Lazy-init analytics charts
            if (view === 'analytics') {
                if (!state.charts.volume) initAnalyticsCharts();
                else updateAnalyticsLive();
            }
        });
    });
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

// ── Data Loading ───────────────────────────────────────────

async function loadInitialData() {
    await Promise.all([loadPatients(), loadMetrics(), loadShap()]);
    initOverviewCharts();
}

async function loadPatients() {
    try {
        const res = await fetch('/api/patients');
        const data = await res.json();
        data.patients.forEach(p => {
            state.patients[p.patient_id] = p;
            if (!state.patientHistory[p.patient_id])
                state.patientHistory[p.patient_id] = [];
            state.patientHistory[p.patient_id].push(p.probability);
        });
        renderPatients();
    } catch (e) { console.error('Load patients failed:', e); }
}

async function loadMetrics() {
    try {
        const res = await fetch('/api/metrics');
        state.metrics = await res.json();
        renderKPIs();
        renderModelMetrics();

        // Also extract SHAP from metrics if present
        if (state.metrics.shap?.top_features?.length && !state.shap.length) {
            state.shap = state.metrics.shap.top_features;
            renderShap();
        }

        // Refresh analytics if active
        if (state.currentView === 'analytics') {
            updateAnalyticsLive();
        }
    } catch (e) { console.error('Load metrics failed:', e); }
}

async function loadShap() {
    try {
        const res = await fetch('/api/shap');
        const data = await res.json();
        state.shap = data.top_features || [];
        renderShap();
    } catch (e) { console.error('Load SHAP failed:', e); }
}

// ── WebSocket ──────────────────────────────────────────────

function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.ws = new WebSocket(`${proto}//${location.host}/ws/live`);

    state.ws.onopen = () => {
        state.connected = true;
        updateConnectionStatus(true);
    };

    state.ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === 'patient_update') {
            const p = data.patient;
            state.patients[p.patient_id] = p;

            // Track history for sparklines
            if (!state.patientHistory[p.patient_id])
                state.patientHistory[p.patient_id] = [];
            const hist = state.patientHistory[p.patient_id];
            hist.push(p.probability);
            if (hist.length > 40) hist.shift();

            updatePatientCard(p);
            updateKPIsFromPatients();
            updateRiskDonut();
            updateTopPatientsTable();

            // Alerts
            if (data.alerts?.length) {
                data.alerts.forEach(a => {
                    state.alerts.unshift(a);
                    if (state.alerts.length > 200) state.alerts.pop();
                });
                renderAlerts();
                document.getElementById('alertTotal').textContent = state.alerts.length;
            }

            // Feature 7: Trigger alarm for critical patients
            if (p.risk_level === 'CRITICAL' && p.probability >= 0.80) {
                const lastAck = state.alarmAcknowledged[p.patient_id] || 0;
                if (Date.now() - lastAck > 120000 && !state.alarmActive) {
                    triggerAlarm(p);
                }
            }
        }
    };

    state.ws.onclose = () => {
        state.connected = false;
        updateConnectionStatus(false);
        setTimeout(connectWebSocket, 3000);
    };

    state.ws.onerror = () => state.ws.close();
}

function updateConnectionStatus(ok) {
    const pulse = document.querySelector('.pulse');
    const text = document.querySelector('.status-text');
    if (ok) {
        pulse.classList.add('connected');
        text.textContent = 'Live';
        text.style.color = '#22c55e';
    } else {
        pulse.classList.remove('connected');
        text.textContent = 'Reconnecting...';
        text.style.color = '#eab308';
    }
}

// ── KPIs ───────────────────────────────────────────────────

function renderKPIs() {
    if (!state.metrics) return;
    const m = state.metrics;

    document.getElementById('totalPatients').textContent = m.total_patients || 0;
    const dist = m.risk_distribution || {};
    document.getElementById('criticalCount').textContent = dist.CRITICAL || 0;
    document.getElementById('highCount').textContent = dist.HIGH || 0;

    // AUROC
    if (m.models?.ensemble)
        document.getElementById('ensembleAuroc').textContent = m.models.ensemble.auroc.toFixed(4);

    // GPU
    if (m.gpu?.memory_used_gb !== undefined)
        document.getElementById('gpuUsage').textContent = `${m.gpu.memory_used_gb}/${m.gpu.memory_total_gb}`;
    else
        document.getElementById('gpuUsage').textContent = 'N/A';

    // KPI bars
    const total = m.total_patients || 20;
    updateKpiBar('criticalBar', (dist.CRITICAL || 0) / total);
    updateKpiBar('highBar', (dist.HIGH || 0) / total);
}

function updateKPIsFromPatients() {
    const patients = Object.values(state.patients);
    const dist = { LOW: 0, MODERATE: 0, HIGH: 0, CRITICAL: 0 };
    patients.forEach(p => dist[p.risk_level]++);

    document.getElementById('totalPatients').textContent = patients.length;
    document.getElementById('criticalCount').textContent = dist.CRITICAL;
    document.getElementById('highCount').textContent = dist.HIGH;

    const total = patients.length || 1;
    updateKpiBar('criticalBar', dist.CRITICAL / total);
    updateKpiBar('highBar', dist.HIGH / total);
}

function updateKpiBar(id, fraction) {
    const el = document.getElementById(id);
    if (el) el.style.setProperty('--bar-width', `${Math.min(fraction * 100, 100)}%`);
    // We use ::after, so set via inline
    const bar = el?.querySelector ? el : document.getElementById(id);
    if (bar) {
        const after = bar;
        after.innerHTML = `<div style="height:100%;width:${fraction*100}%;border-radius:2px;background:inherit;transition:width 0.6s"></div>`;
    }
}

// ── Charts: Overview ───────────────────────────────────────

function initOverviewCharts() {
    initRiskDonut();
    initTrendChart();
    initModelChart();
}

function initRiskDonut() {
    const ctx = document.getElementById('riskDonutChart').getContext('2d');
    const dist = getRiskDistribution();

    state.charts.riskDonut = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low', 'Moderate', 'High', 'Critical'],
            datasets: [{
                data: [dist.LOW, dist.MODERATE, dist.HIGH, dist.CRITICAL],
                backgroundColor: ['#22c55e', '#eab308', '#f97316', '#ef4444'],
                borderColor: 'transparent',
                borderWidth: 0,
                spacing: 3,
                borderRadius: 6,
            }],
        },
        options: {
            cutout: '72%',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(13,17,23,0.9)',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    titleFont: { weight: '600' },
                }
            },
        },
    });

    document.getElementById('donutTotal').textContent = Object.values(state.patients).length;
}

function updateRiskDonut() {
    if (!state.charts.riskDonut) return;
    const dist = getRiskDistribution();
    state.charts.riskDonut.data.datasets[0].data = [dist.LOW, dist.MODERATE, dist.HIGH, dist.CRITICAL];
    state.charts.riskDonut.update('none');
    document.getElementById('donutTotal').textContent = Object.values(state.patients).length;
}

function getRiskDistribution() {
    const dist = { LOW: 0, MODERATE: 0, HIGH: 0, CRITICAL: 0 };
    Object.values(state.patients).forEach(p => dist[p.risk_level]++);
    return dist;
}

function initTrendChart() {
    const ctx = document.getElementById('trendChart').getContext('2d');

    // Initialize with empty data
    const labels = Array.from({ length: 12 }, (_, i) => `${-60 + i * 5}m`);
    const data = new Array(12).fill(null);

    state.charts.trend = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Mean Deterioration %',
                data,
                borderColor: '#818cf8',
                backgroundColor: 'rgba(99, 102, 241, 0.08)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 3,
                pointBackgroundColor: '#818cf8',
                pointBorderColor: '#06080f',
                pointBorderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 100, ticks: { callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.03)' } },
                x: { grid: { display: false } },
            },
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => `${ctx.parsed.y.toFixed(1)}%` } },
            },
            interaction: { intersect: false, mode: 'index' },
        },
    });
}

function pushTrendPoint() {
    if (!state.charts.trend) return;
    const patients = Object.values(state.patients);
    if (!patients.length) return;

    const meanProb = patients.reduce((s, p) => s + p.probability, 0) / patients.length * 100;
    const chart = state.charts.trend;
    const data = chart.data.datasets[0].data;

    data.push(meanProb);
    if (data.length > 12) data.shift();

    const now = new Date();
    chart.data.labels.push(now.toLocaleTimeString('en-US', { hour12: false, minute: '2-digit', second: '2-digit' }));
    if (chart.data.labels.length > 12) chart.data.labels.shift();

    chart.update('none');
}

function initModelChart() {
    if (!state.metrics?.models) return;

    const ctx = document.getElementById('modelChart').getContext('2d');
    const models = state.metrics.models;
    const names = Object.keys(models);
    const colors = { xgboost: '#818cf8', bilstm: '#22d3ee', transformer: '#a78bfa', ensemble: '#22c55e' };

    state.charts.model = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: names.map(n => capitalize(n)),
            datasets: [
                {
                    label: 'AUROC',
                    data: names.map(n => models[n].auroc),
                    backgroundColor: names.map(n => colors[n] || '#818cf8'),
                    borderRadius: 6,
                    borderSkipped: false,
                    barPercentage: 0.6,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: { min: 0.8, max: 1.0, ticks: { callback: v => v.toFixed(2) }, grid: { color: 'rgba(255,255,255,0.03)' } },
                y: { grid: { display: false } },
            },
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => `AUROC: ${ctx.parsed.x.toFixed(4)}` } },
            },
        },
    });
}

// ── Rendering: Model Metrics (sidebar table) ───────────────

function renderModelMetrics() {
    if (!state.metrics?.models) return;
    const body = document.getElementById('modelTableBody');
    if (!body) return;

    const models = state.metrics.models;
    const order = ['ensemble', 'xgboost', 'bilstm', 'transformer'];
    const colors = { xgboost: '#818cf8', bilstm: '#22d3ee', transformer: '#a78bfa', ensemble: '#22c55e' };

    body.innerHTML = order.filter(n => models[n]).map(n => {
        const m = models[n];
        return `<tr>
            <td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${colors[n]};margin-right:6px"></span>${capitalize(n)}</td>
            <td>${m.auroc?.toFixed(4) || '—'}</td>
            <td>${m.auprc?.toFixed(4) || '—'}</td>
            <td>${m.f1?.toFixed(4) || '—'}</td>
            <td>${m.precision?.toFixed(4) || '—'}</td>
            <td>${m.recall?.toFixed(4) || '—'}</td>
        </tr>`;
    }).join('');
}

// ── Rendering: Top Patients Table ──────────────────────────

function updateTopPatientsTable() {
    const body = document.getElementById('topPatientsBody');
    if (!body) return;

    const patients = Object.values(state.patients)
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 10);

    body.innerHTML = patients.map(p => {
        const prob = (p.probability * 100).toFixed(1);
        const hrClass = p.heart_rate > 120 ? 'vital-crit' : (p.heart_rate > 100 ? 'vital-warn' : '');
        const spo2Class = p.spo2 < 90 ? 'vital-crit' : (p.spo2 < 94 ? 'vital-warn' : '');
        const lacClass = p.lactate > 4 ? 'vital-crit' : (p.lactate > 2 ? 'vital-warn' : '');
        const trendIcon = p.trend === '↑' ? '📈' : (p.trend === '↓' ? '📉' : '➡️');

        return `<tr onclick="openModal(${p.patient_id})" style="cursor:pointer">
            <td>${p.bed}</td>
            <td><span class="table-prob ${p.risk_level}">${prob}%</span></td>
            <td><span class="table-risk ${p.risk_level}">${p.risk_level}</span></td>
            <td class="${hrClass}">${p.heart_rate || '—'}</td>
            <td class="${spo2Class}">${p.spo2 || '—'}%</td>
            <td class="${lacClass}">${p.lactate || '—'}</td>
            <td class="trend-icon">${trendIcon}</td>
        </tr>`;
    }).join('');
}

// ── Rendering: SHAP Bars ───────────────────────────────────

function renderShap() {
    const container = document.getElementById('shapBars');
    if (!state.shap?.length) {
        container.innerHTML = '<div class="alert-empty">No SHAP data</div>';
        return;
    }

    const maxVal = Math.max(...state.shap.map(s => s.mean_abs_shap || 0));

    container.innerHTML = state.shap.slice(0, 10).map((s, i) => {
        const val = s.mean_abs_shap || 0;
        const w = (val / maxVal * 100).toFixed(0);
        const name = (s.feature || '').replace(/_/g, ' ');
        return `<div class="shap-row">
            <span class="shap-rank">${i + 1}</span>
            <span class="shap-feature" title="${s.feature}">${name}</span>
            <div class="shap-bar-bg"><div class="shap-bar-fill" style="width:${w}%"></div></div>
            <span class="shap-val">${val.toFixed(3)}</span>
        </div>`;
    }).join('');
}

// ── Rendering: Alerts ──────────────────────────────────────

function renderAlerts() {
    const container = document.getElementById('alertFeed');
    const badge = document.getElementById('alertCountBadge');
    badge.textContent = state.alerts.length;

    if (!state.alerts.length) {
        container.innerHTML = '<div class="alert-empty">Monitoring... No alerts yet</div>';
        return;
    }

    container.innerHTML = state.alerts.slice(0, 40).map(a => {
        const t = new Date(a.timestamp * 1000).toLocaleTimeString('en-US', { hour12: false });
        return `<div class="alert-item">
            <span class="alert-sev ${a.severity}">${a.severity}</span>
            <span class="alert-time">${t}</span>
            <div class="alert-msg">${a.message || a.title}</div>
        </div>`;
    }).join('');
}

// ── Rendering: Patient Grid ────────────────────────────────

function renderPatients() {
    const grid = document.getElementById('patientGrid');
    let patients = Object.values(state.patients);

    if (state.filter !== 'all')
        patients = patients.filter(p => p.risk_level === state.filter);

    if (state.searchTerm) {
        const q = state.searchTerm.toLowerCase();
        patients = patients.filter(p =>
            (p.bed || '').toLowerCase().includes(q) ||
            String(p.patient_id).includes(q)
        );
    }

    const order = { CRITICAL: 0, HIGH: 1, MODERATE: 2, LOW: 3 };
    patients.sort((a, b) => order[a.risk_level] - order[b.risk_level]);

    grid.innerHTML = patients.map(p => buildPatientCard(p)).join('');

    // Draw sparklines after DOM update
    requestAnimationFrame(() => {
        patients.forEach(p => drawSparkline(p.patient_id));
    });
}

function buildPatientCard(p) {
    const prob = (p.probability * 100).toFixed(1);
    const hrClass = p.heart_rate > 120 ? 'vital-crit' : (p.heart_rate > 100 ? 'vital-warn' : '');
    const spo2Class = p.spo2 < 90 ? 'vital-crit' : (p.spo2 < 94 ? 'vital-warn' : '');

    return `<div class="patient-card ${p.risk_level}" id="patient-${p.patient_id}" onclick="openModal(${p.patient_id})">
        <div class="pc-header">
            <span class="pc-bed">${p.bed}</span>
            <span class="pc-risk-badge ${p.risk_level}">${p.risk_level}</span>
        </div>
        <div class="pc-probability">
            <span class="pc-prob-value ${p.risk_level}">${prob}%</span>
            <span class="pc-prob-label">deterioration</span>
            <span class="pc-trend">${p.trend || '→'}</span>
        </div>
        <div class="pc-prob-bar">
            <div class="pc-prob-fill ${p.risk_level}" style="width:${prob}%"></div>
        </div>
        <div class="pc-sparkline"><canvas id="spark-${p.patient_id}"></canvas></div>
        <div class="pc-vitals">
            <div class="pc-vital"><span class="pc-vital-label">HR</span><span class="pc-vital-value ${hrClass}">${p.heart_rate || '—'}</span></div>
            <div class="pc-vital"><span class="pc-vital-label">SpO₂</span><span class="pc-vital-value ${spo2Class}">${p.spo2 || '—'}%</span></div>
            <div class="pc-vital"><span class="pc-vital-label">BP</span><span class="pc-vital-value">${p.systolic_bp || '—'}</span></div>
            <div class="pc-vital"><span class="pc-vital-label">Lactate</span><span class="pc-vital-value">${p.lactate || '—'}</span></div>
        </div>
    </div>`;
}

function drawSparkline(pid) {
    const canvas = document.getElementById(`spark-${pid}`);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const hist = state.patientHistory[pid] || [];
    if (hist.length < 2) return;

    const w = canvas.parentElement.clientWidth;
    const h = 30;
    canvas.width = w * 2; canvas.height = h * 2;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    ctx.scale(2, 2);

    const p = state.patients[pid];
    const color = { LOW: '#22c55e', MODERATE: '#eab308', HIGH: '#f97316', CRITICAL: '#ef4444' }[p?.risk_level] || '#818cf8';

    const points = hist.slice(-20);
    const step = w / (points.length - 1);

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, color + '30');
    grad.addColorStop(1, 'transparent');

    ctx.beginPath();
    ctx.moveTo(0, h - points[0] * h);
    points.forEach((v, i) => ctx.lineTo(i * step, h - v * h));
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Fill
    ctx.lineTo((points.length - 1) * step, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
}

function updatePatientCard(p) {
    const card = document.getElementById(`patient-${p.patient_id}`);
    if (!card) {
        if (state.currentView === 'patients') renderPatients();
        return;
    }

    // Flash
    card.style.boxShadow = '0 0 20px rgba(99, 102, 241, 0.15)';
    setTimeout(() => card.style.boxShadow = '', 600);

    // Update probability
    const probVal = card.querySelector('.pc-prob-value');
    if (probVal) {
        probVal.className = `pc-prob-value ${p.risk_level}`;
        probVal.textContent = `${(p.probability * 100).toFixed(1)}%`;
    }

    const fill = card.querySelector('.pc-prob-fill');
    if (fill) {
        fill.className = `pc-prob-fill ${p.risk_level}`;
        fill.style.width = `${p.probability * 100}%`;
    }

    const badge = card.querySelector('.pc-risk-badge');
    if (badge) {
        badge.className = `pc-risk-badge ${p.risk_level}`;
        badge.textContent = p.risk_level;
    }

    card.className = `patient-card ${p.risk_level}`;

    // Redraw sparkline
    drawSparkline(p.patient_id);
}

// ── Filter Tabs ────────────────────────────────────────────

function setupFilterTabs() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.filter = btn.dataset.filter;
            renderPatients();
        });
    });
}

// ── Search ─────────────────────────────────────────────────

function setupSearch() {
    const input = document.getElementById('patientSearch');
    if (input) {
        input.addEventListener('input', (e) => {
            state.searchTerm = e.target.value;
            renderPatients();
        });
    }
}

// ── Modal ──────────────────────────────────────────────────

function setupModal() {
    document.getElementById('modalClose').addEventListener('click', closeModal);
    document.getElementById('patientModal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeModal();
    });
}

function openModal(pid) {
    const p = state.patients[pid];
    if (!p) return;

    state.modalPatientId = pid;
    const modal = document.getElementById('patientModal');
    modal.classList.add('active');

    // Header
    document.getElementById('modalBed').textContent = p.bed;
    const badge = document.getElementById('modalRiskBadge');
    badge.className = `modal-risk-badge ${p.risk_level}`;
    badge.textContent = p.risk_level;

    const prob = document.getElementById('modalProb');
    prob.className = `modal-prob ${p.risk_level}`;
    prob.textContent = `${(p.probability * 100).toFixed(1)}%`;

    // Vitals
    const vitals = [
        { label: 'Heart Rate', value: p.heart_rate, unit: 'bpm', warn: 100, crit: 120 },
        { label: 'SpO₂', value: p.spo2, unit: '%', warnBelow: 94, critBelow: 90 },
        { label: 'Systolic BP', value: p.systolic_bp, unit: 'mmHg', warnBelow: 90, critBelow: 80 },
        { label: 'Resp. Rate', value: p.respiratory_rate, unit: '/min', warn: 24, crit: 30 },
        { label: 'Lactate', value: p.lactate, unit: 'mmol/L', warn: 2, crit: 4 },
        { label: 'Trend', value: p.trend === '↑' ? 'Increasing' : (p.trend === '↓' ? 'Decreasing' : 'Stable'), unit: '', isText: true },
    ];

    document.getElementById('modalVitals').innerHTML = vitals.map(v => {
        let cls = '';
        if (!v.isText) {
            if (v.warn && v.value > v.crit) cls = 'crit';
            else if (v.warn && v.value > v.warn) cls = 'warn';
            if (v.critBelow && v.value < v.critBelow) cls = 'crit';
            else if (v.warnBelow && v.value < v.warnBelow) cls = 'warn';
        }
        return `<div class="vital-card ${cls}">
            <div class="vital-card-label">${v.label}</div>
            <div class="vital-card-value">${v.value ?? '—'}</div>
            <div class="vital-card-unit">${v.unit}</div>
        </div>`;
    }).join('');

    // Trend chart
    renderModalTrend(pid, p);

    // Risk factors (simulated based on vitals)
    const factors = [];
    if (p.lactate > 2) factors.push({ name: 'Elevated Lactate', w: Math.min(p.lactate / 6, 1) });
    if (p.heart_rate > 100) factors.push({ name: 'Tachycardia', w: Math.min((p.heart_rate - 60) / 80, 1) });
    if (p.spo2 < 95) factors.push({ name: 'Hypoxemia', w: Math.min((100 - p.spo2) / 15, 1) });
    if (p.systolic_bp < 110) factors.push({ name: 'Hypotension', w: Math.min((130 - p.systolic_bp) / 60, 1) });
    if (p.respiratory_rate > 20) factors.push({ name: 'Tachypnea', w: Math.min((p.respiratory_rate - 12) / 20, 1) });
    factors.push({ name: 'Baseline Risk', w: 0.15 });
    factors.sort((a, b) => b.w - a.w);

    document.getElementById('modalRiskFactors').innerHTML = factors.slice(0, 5).map(f =>
        `<div class="risk-factor-row">
            <span class="risk-factor-name">${f.name}</span>
            <div class="risk-factor-bar"><div class="risk-factor-fill" style="width:${f.w * 100}%"></div></div>
        </div>`
    ).join('');

    // ── Load NEW feature data ──────────────────────────────
    loadTrajectory(pid);
    loadPatientShap(pid);
    loadWaveforms(pid);
    loadAttention(pid);
    renderInterventionWindow(p);
    renderProtocolButtons(p);
    initWhatIfForPatient(p);
    document.getElementById('protocolSection').style.display = 'none';
}

function renderModalTrend(pid, p) {
    const canvas = document.getElementById('modalTrendChart');
    const ctx = canvas.getContext('2d');

    // Destroy old chart
    if (state.charts.modalTrend) state.charts.modalTrend.destroy();

    const hist = (state.patientHistory[pid] || []).slice(-20);
    const labels = hist.map((_, i) => `${-hist.length + i + 1}`);
    const color = { LOW: '#22c55e', MODERATE: '#eab308', HIGH: '#f97316', CRITICAL: '#ef4444' }[p.risk_level] || '#818cf8';

    state.charts.modalTrend = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Deterioration %',
                data: hist.map(v => v * 100),
                borderColor: color,
                backgroundColor: color + '15',
                fill: true,
                tension: 0.4,
                borderWidth: 2.5,
                pointRadius: 2,
                pointBackgroundColor: color,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 100, ticks: { callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.03)' } },
                x: { title: { display: true, text: 'Updates ago', color: '#64748b' }, grid: { display: false } },
            },
            plugins: { legend: { display: false } },
        },
    });
}

function closeModal() {
    document.getElementById('patientModal').classList.remove('active');
    // Stop waveform animation
    if (_waveformAnimId) {
        cancelAnimationFrame(_waveformAnimId);
        _waveformAnimId = null;
    }
}

// ── Analytics Charts ───────────────────────────────────────

function initAnalyticsCharts() {
    // Prediction volume — simulate realistic hourly volumes
    const volCtx = document.getElementById('volumeChart')?.getContext('2d');
    if (volCtx) {
        const now = new Date();
        const currentHour = now.getHours();
        const hours = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, '0')}:00`);
        const counts = hours.map((_, i) => {
            // More predictions during day shifts (7am-6pm)
            const base = (i >= 7 && i <= 18) ? 80 : 30;
            const noise = Math.floor(Math.random() * 40);
            // Current hour shows partial
            if (i === currentHour) return Math.floor(base * 0.6) + noise;
            if (i > currentHour) return 0;
            return base + noise;
        });

        state.charts.volume = new Chart(volCtx, {
            type: 'bar',
            data: {
                labels: hours,
                datasets: [{
                    label: 'Predictions',
                    data: counts,
                    backgroundColor: hours.map((_, i) =>
                        i === currentHour ? 'rgba(34, 197, 94, 0.6)' :
                        i > currentHour ? 'rgba(255,255,255,0.02)' :
                        'rgba(99, 102, 241, 0.4)'
                    ),
                    borderColor: hours.map((_, i) =>
                        i === currentHour ? '#22c55e' : '#818cf8'
                    ),
                    borderWidth: 1,
                    borderRadius: 4,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.03)' }, title: { display: true, text: 'Count' } },
                    x: { grid: { display: false } },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => `${ctx.parsed.y} predictions` } },
                },
            },
        });
    }

    // SHAP full chart — re-fetch if needed
    const shapCtx = document.getElementById('shapFullChart')?.getContext('2d');
    if (shapCtx) {
        const renderShapFull = () => {
            if (state.charts.shapFull) state.charts.shapFull.destroy();
            const features = (state.shap || []).slice(0, 15);
            if (!features.length) return;

            state.charts.shapFull = new Chart(shapCtx, {
                type: 'bar',
                data: {
                    labels: features.map(f => (f.feature || '').replace(/_/g, ' ')),
                    datasets: [{
                        label: 'SHAP Importance',
                        data: features.map(f => f.mean_abs_shap || 0),
                        backgroundColor: features.map((_, i) => {
                            const h = 240 + i * 8;
                            return `hsla(${h}, 70%, 65%, 0.6)`;
                        }),
                        borderRadius: 4,
                    }],
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.03)' }, title: { display: true, text: 'Mean |SHAP|' } },
                        y: { grid: { display: false } },
                    },
                    plugins: { legend: { display: false } },
                },
            });
        };

        if (state.shap.length) {
            renderShapFull();
        } else {
            // Retry after loading
            setTimeout(renderShapFull, 2000);
        }
    }

    // Risk over time (stacked area) — seed with current distribution
    const riskCtx = document.getElementById('riskTimeChart')?.getContext('2d');
    if (riskCtx) {
        const labels = Array.from({ length: 12 }, (_, i) => `${-60 + i * 5}m`);
        const dist = getRiskDistribution();
        // Seed with slight variations of current distribution
        const seedData = (base, variance) =>
            Array.from({ length: 12 }, () => Math.max(0, base + Math.floor(Math.random() * variance * 2 - variance)));

        state.charts.riskTime = new Chart(riskCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Critical', data: seedData(dist.CRITICAL, 2), borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', fill: true, tension: 0.4, pointRadius: 2 },
                    { label: 'High', data: seedData(dist.HIGH, 2), borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.1)', fill: true, tension: 0.4, pointRadius: 2 },
                    { label: 'Moderate', data: seedData(dist.MODERATE, 2), borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', fill: true, tension: 0.4, pointRadius: 2 },
                    { label: 'Low', data: seedData(dist.LOW, 3), borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)', fill: true, tension: 0.4, pointRadius: 2 },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    y: { stacked: true, grid: { color: 'rgba(255,255,255,0.03)' }, title: { display: true, text: 'Patients' } },
                    x: { grid: { display: false } },
                },
                plugins: { legend: { position: 'bottom' } },
            },
        });
    }

    // Alert distribution — use alert engine stats or state.alerts
    const alertCtx = document.getElementById('alertChart')?.getContext('2d');
    if (alertCtx) {
        const sevCounts = { CRITICAL: 0, WARNING: 0, INFO: 0 };
        state.alerts.forEach(a => { if (sevCounts[a.severity] !== undefined) sevCounts[a.severity]++; });

        // Add minimum counts so chart always shows something
        const alertStats = state.metrics?.alerts?.by_severity || {};
        const crit = Math.max(sevCounts.CRITICAL, alertStats.CRITICAL || 0, 3);
        const warn = Math.max(sevCounts.WARNING, alertStats.WARNING || 0, 5);
        const info = Math.max(sevCounts.INFO || 0, alertStats.INFO || 0, 8);

        state.charts.alertDist = new Chart(alertCtx, {
            type: 'doughnut',
            data: {
                labels: ['Critical', 'Warning', 'Info'],
                datasets: [{
                    data: [crit, warn, info],
                    backgroundColor: ['#ef4444', '#eab308', '#818cf8'],
                    borderWidth: 0, spacing: 2, borderRadius: 4,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: { position: 'bottom' },
                    tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.parsed} alerts` } },
                },
            },
        });
    }

    // Update model table
    renderModelMetrics();
}

// Live update analytics data
function updateAnalyticsLive() {
    // Update risk time chart with current distribution
    if (state.charts.riskTime) {
        const dist = getRiskDistribution();
        const chart = state.charts.riskTime;
        const now = new Date().toLocaleTimeString('en-US', { hour12: false, minute: '2-digit', second: '2-digit' });

        [dist.CRITICAL, dist.HIGH, dist.MODERATE, dist.LOW].forEach((val, i) => {
            chart.data.datasets[i].data.push(val);
            if (chart.data.datasets[i].data.length > 12) chart.data.datasets[i].data.shift();
        });
        chart.data.labels.push(now);
        if (chart.data.labels.length > 12) chart.data.labels.shift();
        chart.update('none');
    }

    // Update alert distribution
    if (state.charts.alertDist) {
        const sevCounts = { CRITICAL: 0, WARNING: 0, INFO: 0 };
        state.alerts.forEach(a => { if (sevCounts[a.severity] !== undefined) sevCounts[a.severity]++; });
        const alertStats = state.metrics?.alerts?.by_severity || {};
        state.charts.alertDist.data.datasets[0].data = [
            Math.max(sevCounts.CRITICAL, alertStats.CRITICAL || 0, 1),
            Math.max(sevCounts.WARNING, alertStats.WARNING || 0, 1),
            Math.max(sevCounts.INFO || 0, alertStats.INFO || 0, 1),
        ];
        state.charts.alertDist.update('none');
    }

    // Update model table
    renderModelMetrics();
}


// ═══════════════════════════════════════════════════════════════
// NEW FEATURES — JavaScript
// ═══════════════════════════════════════════════════════════════

// ── Feature 6: System Health & Data Freshness ──────────────

async function loadSystemHealth() {
    try {
        const res = await fetch('/api/system/health');
        const h = await res.json();

        const statusEl = document.getElementById('healthStatus');
        const panelEl = document.querySelector('.system-health-panel');

        if (h.data_is_fresh) {
            statusEl.textContent = '● Healthy';
            statusEl.className = 'health-status';
            panelEl?.classList.remove('degraded');
        } else {
            statusEl.textContent = '● Degraded';
            statusEl.className = 'health-status degraded';
            panelEl?.classList.add('degraded');
        }

        const setHealth = (id, val, cls = '') => {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = val;
                if (cls) el.className = `health-value mono ${cls}`;
            }
        };

        const freshTime = new Date(h.last_ingestion).toLocaleTimeString('en-US', { hour12: false });
        setHealth('lastIngestion', freshTime);
        setHealth('dataFreshness', `${h.data_freshness_sec}s`, h.data_is_fresh ? 'fresh' : 'stale');
        setHealth('kafkaLag', `${h.pipeline?.kafka?.lag_ms || '—'}ms`);
        setHealth('modelLatency', `${h.pipeline?.model_server?.latency_ms || '—'}ms`);
        setHealth('dbQuery', `${h.pipeline?.database?.query_ms || '—'}ms`);
        setHealth('systemUptime', h.uptime || '—');
        setHealth('predCount24', (h.prediction_count_24h || 0).toLocaleString());
        setHealth('avgLatency', `${h.avg_latency_ms || '—'}ms`);
    } catch (e) {
        console.error('System health load failed:', e);
    }
}

// ── Feature 1: Risk Trajectory vs Clinical Baselines ───────

let _trajectoryData = null;
let _trajectoryVital = 'heart_rate';

function setupTrajectoryTabs() {
    document.querySelectorAll('.traj-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.traj-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            _trajectoryVital = btn.dataset.vital;
            if (_trajectoryData) renderTrajectoryChart(_trajectoryData, _trajectoryVital);
        });
    });
}

async function loadTrajectory(pid) {
    try {
        const res = await fetch(`/api/patient/${pid}/trajectory`);
        _trajectoryData = await res.json();
        _trajectoryVital = 'heart_rate';
        document.querySelectorAll('.traj-tab').forEach(b => {
            b.classList.toggle('active', b.dataset.vital === 'heart_rate');
        });
        renderTrajectoryChart(_trajectoryData, _trajectoryVital);
    } catch (e) { console.error('Trajectory load failed:', e); }
}

function renderTrajectoryChart(data, vital) {
    const canvas = document.getElementById('trajectoryChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (state.charts.trajectory) state.charts.trajectory.destroy();

    const traj = data.trajectory || [];
    const baselines = data.baselines || {};
    const bl = baselines[vital] || {};

    const labels = traj.map(t => `${t.time_offset_min}m`);
    const riskData = traj.map(t => t.risk_score);
    const vitalData = traj.map(t => t[vital]);

    const vitalLabels = {
        heart_rate: 'Heart Rate (bpm)',
        systolic_bp: 'Systolic BP (mmHg)',
        spo2: 'SpO₂ (%)',
        respiratory_rate: 'Resp. Rate (/min)',
        lactate: 'Lactate (mmol/L)',
    };

    // Baseline annotation lines
    const annotations = [];
    if (bl.normal_high) annotations.push({ y: bl.normal_high, label: 'Normal High', color: 'rgba(234,179,8,0.4)' });
    if (bl.normal_low) annotations.push({ y: bl.normal_low, label: 'Normal Low', color: 'rgba(34,197,94,0.4)' });
    if (bl.critical_high) annotations.push({ y: bl.critical_high, label: 'Critical', color: 'rgba(239,68,68,0.4)' });
    if (bl.critical_low) annotations.push({ y: bl.critical_low, label: 'Critical Low', color: 'rgba(239,68,68,0.4)' });

    state.charts.trajectory = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'AI Risk Score (%)',
                    data: riskData,
                    borderColor: '#818cf8',
                    backgroundColor: 'rgba(99,102,241,0.08)',
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    yAxisID: 'y',
                },
                {
                    label: vitalLabels[vital] || vital,
                    data: vitalData,
                    borderColor: '#22d3ee',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y1',
                    borderDash: [4, 2],
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            scales: {
                y: {
                    type: 'linear', position: 'left', min: 0, max: 100,
                    title: { display: true, text: 'Risk Score %', color: '#818cf8' },
                    ticks: { callback: v => v + '%', color: '#818cf8' },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
                y1: {
                    type: 'linear', position: 'right',
                    title: { display: true, text: vitalLabels[vital] || vital, color: '#22d3ee' },
                    ticks: { color: '#22d3ee' },
                    grid: { display: false },
                },
                x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } },
            },
            plugins: {
                legend: { display: true, position: 'top' },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            if (ctx.datasetIndex === 0) return `Risk: ${ctx.parsed.y.toFixed(1)}%`;
                            return `${vital}: ${ctx.parsed.y}`;
                        }
                    }
                },
            },
        },
    });

    // Render baseline legend
    const legendEl = document.getElementById('baselineLegend');
    if (legendEl) {
        legendEl.innerHTML = annotations.map(a =>
            `<span class="legend-item">
                <span class="legend-line" style="background:${a.color}"></span>
                ${a.label}: ${a.y} ${bl.unit || ''}
            </span>`
        ).join('');
    }
}

// ── Feature 2: SHAP Waterfall ──────────────────────────────

async function loadPatientShap(pid) {
    try {
        const res = await fetch(`/api/patient/${pid}/shap`);
        const data = await res.json();
        renderWaterfall(data);
    } catch (e) { console.error('Patient SHAP failed:', e); }
}

function renderWaterfall(data) {
    const container = document.getElementById('waterfallContainer');
    const summary = document.getElementById('waterfallSummary');
    if (!container) return;

    const contributions = data.contributions || [];
    const maxAbsShap = Math.max(...contributions.map(c => Math.abs(c.shap_value)), 0.01);

    container.innerHTML = contributions.map(c => {
        const absW = (Math.abs(c.shap_value) / maxAbsShap * 50).toFixed(0);
        const isRisk = c.direction === 'risk';
        const barStyle = isRisk
            ? `left:50%;width:${absW}%`
            : `left:${50 - absW}%;width:${absW}%`;

        return `<div class="waterfall-bar-row">
            <span class="wf-feature" title="${c.feature}">${c.feature}</span>
            <div class="wf-bar-area">
                <div class="wf-bar ${c.direction}" style="${barStyle}"></div>
            </div>
            <span class="wf-shap-val ${c.direction}">${isRisk ? '+' : ''}${c.shap_value.toFixed(3)}</span>
            <span class="wf-value">${c.value}</span>
        </div>`;
    }).join('');

    if (summary) {
        summary.innerHTML = `
            <span class="wf-summary-label">Base → Final Risk</span>
            <span class="wf-summary-value" style="color:${data.final_probability > 50 ? '#ef4444' : '#22c55e'}">${data.final_probability}%</span>
        `;
    }
}

// ── Feature 2b: What-If Sliders ────────────────────────────

function setupWhatIfSliders() {
    ['wiBP', 'wiHR', 'wiLac', 'wiSpO2'].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('input', () => {
            updateWhatIfLabels();
            debounceWhatIf();
        });
    });
}

function updateWhatIfLabels() {
    const bp = document.getElementById('wiBP');
    const hr = document.getElementById('wiHR');
    const lac = document.getElementById('wiLac');
    const spo2 = document.getElementById('wiSpO2');

    if (bp) document.getElementById('wiBPVal').textContent = `${bp.value} mmHg`;
    if (hr) document.getElementById('wiHRVal').textContent = `${hr.value} bpm`;
    if (lac) document.getElementById('wiLacVal').textContent = `${(lac.value / 10).toFixed(1)} mmol/L`;
    if (spo2) document.getElementById('wiSpO2Val').textContent = `${spo2.value}%`;
}

let _whatIfTimer = null;
function debounceWhatIf() {
    clearTimeout(_whatIfTimer);
    _whatIfTimer = setTimeout(runWhatIf, 300);
}

async function runWhatIf() {
    const pid = state.modalPatientId;
    if (!pid) return;

    const bp = document.getElementById('wiBP')?.value;
    const hr = document.getElementById('wiHR')?.value;
    const lac = document.getElementById('wiLac')?.value;
    const spo2 = document.getElementById('wiSpO2')?.value;

    try {
        const params = new URLSearchParams({
            systolic_bp: bp,
            heart_rate: hr,
            lactate: (lac / 10).toFixed(1),
            spo2: spo2,
        });
        const res = await fetch(`/api/patient/${pid}/whatif?${params}`);
        const data = await res.json();

        document.getElementById('wiCurrentRisk').textContent = `${data.current_risk}%`;
        document.getElementById('wiProjectedRisk').textContent = `${data.projected_risk}%`;

        const delta = data.risk_change;
        const deltaEl = document.getElementById('wiDelta');
        if (delta > 0.5) {
            deltaEl.textContent = `+${delta.toFixed(1)}%`;
            deltaEl.className = 'whatif-delta positive';
        } else if (delta < -0.5) {
            deltaEl.textContent = `${delta.toFixed(1)}%`;
            deltaEl.className = 'whatif-delta negative';
        } else {
            deltaEl.textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%`;
            deltaEl.className = 'whatif-delta neutral';
        }
    } catch (e) { console.error('What-if failed:', e); }
}

function initWhatIfForPatient(p) {
    const bp = document.getElementById('wiBP');
    const hr = document.getElementById('wiHR');
    const lac = document.getElementById('wiLac');
    const spo2 = document.getElementById('wiSpO2');

    if (bp) bp.value = p.systolic_bp;
    if (hr) hr.value = p.heart_rate;
    if (lac) lac.value = Math.round(p.lactate * 10);
    if (spo2) spo2.value = Math.round(p.spo2);

    updateWhatIfLabels();
    document.getElementById('wiCurrentRisk').textContent = `${(p.probability * 100).toFixed(1)}%`;
    document.getElementById('wiProjectedRisk').textContent = `${(p.probability * 100).toFixed(1)}%`;
    document.getElementById('wiDelta').textContent = '0.0%';
    document.getElementById('wiDelta').className = 'whatif-delta neutral';
}

// ── Feature 3: ECG & PPG Live Waveforms (Animated) ─────────

var _waveformAnimId = null;
var _waveformData = {};

async function loadWaveforms(pid) {
    // Stop any existing animation
    if (_waveformAnimId) {
        cancelAnimationFrame(_waveformAnimId);
        _waveformAnimId = null;
    }
    _waveformData = {};

    try {
        const res = await fetch(`/api/patient/${pid}/waveforms?duration_sec=5`);
        const data = await res.json();

        document.getElementById('wfHR').textContent = `${data.heart_rate} bpm`;

        // Store waveform data and start animation
        _waveformData = {
            ecg: data.ecg || [],
            ppg: data.ppg || [],
            sampleRate: data.sample_rate || 250,
            heartRate: data.heart_rate || 72,
            ecgOffset: 0,
            ppgOffset: 0,
            lastTime: performance.now(),
        };

        animateWaveforms();
    } catch (e) { console.error('Waveform load failed:', e); }
}

function animateWaveforms() {
    const d = _waveformData;
    if (!d.ecg?.length) return;

    const now = performance.now();
    const elapsed = (now - d.lastTime) / 1000; // seconds
    d.lastTime = now;

    // Scroll speed: pixels per second (simulate real-time sweep at 25mm/s)
    const scrollSpeed = d.sampleRate * 0.4;
    d.ecgOffset = (d.ecgOffset + elapsed * scrollSpeed) % d.ecg.length;
    d.ppgOffset = (d.ppgOffset + elapsed * scrollSpeed * 0.98) % d.ppg.length;

    drawAnimatedWaveform('ecgCanvas', d.ecg, '#22d3ee', d.ecgOffset);
    drawAnimatedWaveform('ppgCanvas', d.ppg, '#a78bfa', d.ppgOffset);

    // Continue animation only if modal is open
    const modal = document.getElementById('patientModal');
    if (modal?.classList.contains('active')) {
        _waveformAnimId = requestAnimationFrame(animateWaveforms);
    }
}

function drawAnimatedWaveform(canvasId, samples, color, offset) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !samples?.length) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.parentElement.clientWidth - 16;
    const h = 80;

    // Only resize if needed
    if (canvas.width !== w * 2 || canvas.height !== h * 2) {
        canvas.width = w * 2;
        canvas.height = h * 2;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
    }

    ctx.setTransform(2, 0, 0, 2, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Background grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < w; i += 20) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, h); ctx.stroke();
    }
    for (let i = 0; i < h; i += 20) {
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(w, i); ctx.stroke();
    }

    // Visible window of samples
    const visibleCount = Math.min(Math.floor(w * 1.5), samples.length);
    const startIdx = Math.floor(offset) % samples.length;
    const pad = 8;

    // Get visible range for normalization
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < visibleCount; i++) {
        const idx = (startIdx + i) % samples.length;
        const v = samples[idx];
        if (v < min) min = v;
        if (v > max) max = v;
    }
    const range = max - min || 1;

    // Draw main waveform line
    const step = w / visibleCount;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';

    for (let i = 0; i < visibleCount; i++) {
        const idx = (startIdx + i) % samples.length;
        const x = i * step;
        const y = pad + (1 - (samples[idx] - min) / range) * (h - 2 * pad);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Glow effect
    ctx.globalAlpha = 0.12;
    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.beginPath();
    for (let i = 0; i < visibleCount; i++) {
        const idx = (startIdx + i) % samples.length;
        const x = i * step;
        const y = pad + (1 - (samples[idx] - min) / range) * (h - 2 * pad);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Sweeping cursor line at the right edge
    const cursorX = w - 2;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(cursorX, 0);
    ctx.lineTo(cursorX, h);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Fade-out gradient at sweep cursor
    const fadeGrad = ctx.createLinearGradient(w - 40, 0, w, 0);
    fadeGrad.addColorStop(0, 'transparent');
    fadeGrad.addColorStop(1, 'rgba(5, 8, 16, 0.5)');
    ctx.fillStyle = fadeGrad;
    ctx.fillRect(w - 40, 0, 40, h);
}

// ── Feature 4: Temporal Attention Heatmap ──────────────────

async function loadAttention(pid) {
    try {
        const res = await fetch(`/api/patient/${pid}/attention`);
        const data = await res.json();
        renderAttentionHeatmap(data);
    } catch (e) { console.error('Attention load failed:', e); }
}

function renderAttentionHeatmap(data) {
    const container = document.getElementById('attentionHeatmap');
    const interpEl = document.getElementById('attentionInterpretation');
    if (!container) return;

    const windows = data.windows || [];
    const maxW = Math.max(...windows.map(w => w.attention_weight));

    container.innerHTML = windows.map(w => {
        const intensity = w.attention_weight / maxW;
        const r = Math.round(intensity > 0.5 ? 239 : 99);
        const g = Math.round(intensity > 0.5 ? 68 * (1 - intensity) + 102 * intensity : 102);
        const b = Math.round(intensity > 0.5 ? 68 : 241 * (1 - intensity) + 68 * intensity);
        const bg = `rgba(${r}, ${g}, ${b}, ${0.2 + intensity * 0.7})`;
        const critClass = w.is_critical ? 'critical-window' : '';

        return `<div class="attn-cell ${critClass}" style="background:${bg}" title="${w.label}: ${(w.attention_weight * 100).toFixed(1)}%">
            <span class="attn-tooltip">${w.label}<br>Weight: ${(w.attention_weight * 100).toFixed(1)}%${w.is_critical ? '<br>⚠️ Critical Shift' : ''}</span>
        </div>`;
    }).join('');

    if (interpEl && data.interpretation) {
        interpEl.innerHTML = `<strong>🧠 Model Insight:</strong> ${data.interpretation}`;
    }
}

// ── Feature 5: Intervention Window & Protocols ─────────────

function renderInterventionWindow(p) {
    const prob = p.probability;
    // Estimate hours before critical threshold (0.75)
    let hoursLeft;
    if (prob >= 0.75) {
        hoursLeft = 0;
    } else {
        // Linear extrapolation: assume ~5% increase per hour
        hoursLeft = Math.max(0, Math.round((0.75 - prob) / 0.05 * 10) / 10);
    }
    hoursLeft = Math.min(hoursLeft, 10);

    const countdownVal = document.getElementById('countdownValue');
    const ring = document.getElementById('countdownRing');

    if (countdownVal) countdownVal.textContent = hoursLeft <= 0 ? '⚠️' : hoursLeft.toFixed(1);

    if (ring) {
        const circumference = 2 * Math.PI * 52; // ~326.7
        const fraction = Math.min(hoursLeft / 10, 1);
        ring.style.strokeDashoffset = circumference * (1 - fraction);

        if (hoursLeft <= 1) {
            ring.style.stroke = '#ef4444';
            if (countdownVal) countdownVal.style.color = '#ef4444';
        } else if (hoursLeft <= 3) {
            ring.style.stroke = '#f97316';
            if (countdownVal) countdownVal.style.color = '#f97316';
        } else {
            ring.style.stroke = '#eab308';
            if (countdownVal) countdownVal.style.color = '#eab308';
        }
    }
}

function renderProtocolButtons(p) {
    const container = document.getElementById('protocolButtons');
    if (!container) return;

    const protocols = [
        { key: 'sepsis', icon: '🦠', label: 'Sepsis Bundle (Hour-1)', condition: p.lactate > 2 && p.probability > 0.4 },
        { key: 'respiratory', icon: '🫁', label: 'Respiratory Protocol', condition: p.spo2 < 94 || p.respiratory_rate > 24 },
        { key: 'cardiac', icon: '❤️', label: 'Cardiac Protocol', condition: p.heart_rate > 120 || p.systolic_bp < 80 },
        { key: 'general', icon: '📞', label: 'Rapid Response (RRT)', condition: true },
    ];

    container.innerHTML = protocols
        .filter(pr => pr.condition)
        .map(pr =>
            `<button class="protocol-btn" onclick="loadProtocol('${pr.key}')">
                <span class="proto-icon">${pr.icon}</span>
                ${pr.label}
            </button>`
        ).join('');
}

async function loadProtocol(condition) {
    try {
        const res = await fetch(`/api/protocols/${condition}`);
        const data = await res.json();

        const section = document.getElementById('protocolSection');
        const title = document.getElementById('protocolTitle');
        const checklist = document.getElementById('protocolChecklist');

        if (title) title.textContent = `📋 ${data.name}`;

        if (checklist) {
            checklist.innerHTML = data.steps.map(s =>
                `<div class="proto-step">
                    <span class="proto-step-num">${s.step}</span>
                    <span>${s.action}</span>
                    <span class="proto-priority ${s.priority}">${s.priority}</span>
                </div>`
            ).join('');
        }

        if (section) {
            section.style.display = 'block';
            section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    } catch (e) { console.error('Protocol load failed:', e); }
}

// ── Feature 7: Audio Alarm System ──────────────────────────

function setupAlarm() {
    const ackBtn = document.getElementById('alarmAck');
    const viewBtn = document.getElementById('alarmView');

    if (ackBtn) {
        ackBtn.addEventListener('click', () => {
            dismissAlarm();
        });
    }

    if (viewBtn) {
        viewBtn.addEventListener('click', () => {
            const pid = state.alarmPatient?.patient_id;
            dismissAlarm();
            if (pid) {
                // Switch to patients view and open modal
                document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
                document.getElementById('navPatients')?.classList.add('active');
                document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
                document.getElementById('viewPatients')?.classList.add('active');
                state.currentView = 'patients';
                setTimeout(() => openModal(pid), 300);
            }
        });
    }
}

function triggerAlarm(patient) {
    state.alarmActive = true;
    state.alarmPatient = patient;

    const overlay = document.getElementById('alarmOverlay');
    const msg = document.getElementById('alarmMessage');
    const patientInfo = document.getElementById('alarmPatient');

    if (overlay) overlay.classList.add('active');
    if (msg) msg.textContent = `${patient.bed} — Deterioration at ${(patient.probability * 100).toFixed(1)}% — Immediate attention required!`;
    if (patientInfo) patientInfo.textContent = `Patient #${patient.patient_id} | ${patient.bed} | Risk: ${patient.risk_level}`;

    document.body.classList.add('alarm-active');

    // Try to play alarm sound (uses Web Audio API as fallback)
    try {
        playAlarmSound();
    } catch (e) { console.log('Alarm sound not available'); }
}

function dismissAlarm() {
    state.alarmActive = false;
    if (state.alarmPatient) {
        state.alarmAcknowledged[state.alarmPatient.patient_id] = Date.now();
    }
    state.alarmPatient = null;

    const overlay = document.getElementById('alarmOverlay');
    if (overlay) overlay.classList.remove('active');
    document.body.classList.remove('alarm-active');

    stopAlarmSound();
}

let _alarmOscillator = null;
let _alarmContext = null;

function playAlarmSound() {
    if (_alarmOscillator) return; // Already playing
    try {
        _alarmContext = new (window.AudioContext || window.webkitAudioContext)();
        _alarmOscillator = _alarmContext.createOscillator();
        const gainNode = _alarmContext.createGain();

        _alarmOscillator.type = 'square';
        _alarmOscillator.frequency.setValueAtTime(800, _alarmContext.currentTime);

        // Pulsing alarm
        const now = _alarmContext.currentTime;
        for (let i = 0; i < 60; i++) {
            gainNode.gain.setValueAtTime(0.15, now + i * 0.5);
            gainNode.gain.setValueAtTime(0, now + i * 0.5 + 0.3);
            _alarmOscillator.frequency.setValueAtTime(800, now + i * 0.5);
            _alarmOscillator.frequency.setValueAtTime(600, now + i * 0.5 + 0.15);
        }

        _alarmOscillator.connect(gainNode);
        gainNode.connect(_alarmContext.destination);
        _alarmOscillator.start();
    } catch (e) { console.log('Web Audio alarm fallback failed:', e); }
}

function stopAlarmSound() {
    if (_alarmOscillator) {
        try {
            _alarmOscillator.stop();
            _alarmOscillator.disconnect();
        } catch (e) { /* already stopped */ }
        _alarmOscillator = null;
    }
    if (_alarmContext) {
        try { _alarmContext.close(); } catch (e) { /* ok */ }
        _alarmContext = null;
    }
}
