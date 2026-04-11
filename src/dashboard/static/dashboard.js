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
    loadInitialData();
    connectWebSocket();
    setInterval(loadMetrics, 8000);
    setInterval(pushTrendPoint, 5000);
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
            if (view === 'analytics' && !state.charts.volume) {
                initAnalyticsCharts();
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
}

// ── Analytics Charts ───────────────────────────────────────

function initAnalyticsCharts() {
    // Prediction volume
    const volCtx = document.getElementById('volumeChart')?.getContext('2d');
    if (volCtx) {
        const hours = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, '0')}:00`);
        const counts = hours.map(() => Math.floor(Math.random() * 80 + 20));

        state.charts.volume = new Chart(volCtx, {
            type: 'bar',
            data: {
                labels: hours,
                datasets: [{
                    label: 'Predictions',
                    data: counts,
                    backgroundColor: 'rgba(99, 102, 241, 0.4)',
                    borderColor: '#818cf8',
                    borderWidth: 1,
                    borderRadius: 4,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.03)' } },
                    x: { grid: { display: false } },
                },
                plugins: { legend: { display: false } },
            },
        });
    }

    // SHAP full chart
    const shapCtx = document.getElementById('shapFullChart')?.getContext('2d');
    if (shapCtx && state.shap.length) {
        const features = state.shap.slice(0, 15);
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
                    x: { grid: { color: 'rgba(255,255,255,0.03)' } },
                    y: { grid: { display: false } },
                },
                plugins: { legend: { display: false } },
            },
        });
    }

    // Risk over time (stacked area)
    const riskCtx = document.getElementById('riskTimeChart')?.getContext('2d');
    if (riskCtx) {
        const labels = Array.from({ length: 12 }, (_, i) => `${-60 + i * 5}m`);
        state.charts.riskTime = new Chart(riskCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Critical', data: Array(12).fill(0), borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', fill: true, tension: 0.4 },
                    { label: 'High', data: Array(12).fill(0), borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.1)', fill: true, tension: 0.4 },
                    { label: 'Moderate', data: Array(12).fill(0), borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', fill: true, tension: 0.4 },
                    { label: 'Low', data: Array(12).fill(0), borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)', fill: true, tension: 0.4 },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    y: { stacked: true, grid: { color: 'rgba(255,255,255,0.03)' } },
                    x: { grid: { display: false } },
                },
                plugins: { legend: { position: 'bottom' } },
            },
        });
    }

    // Alert distribution pie
    const alertCtx = document.getElementById('alertChart')?.getContext('2d');
    if (alertCtx) {
        const sevCounts = { CRITICAL: 0, WARNING: 0, INFO: 0 };
        state.alerts.forEach(a => { if (sevCounts[a.severity] !== undefined) sevCounts[a.severity]++; });

        state.charts.alertDist = new Chart(alertCtx, {
            type: 'doughnut',
            data: {
                labels: ['Critical', 'Warning', 'Info'],
                datasets: [{
                    data: [sevCounts.CRITICAL || 1, sevCounts.WARNING || 1, sevCounts.INFO || 1],
                    backgroundColor: ['#ef4444', '#eab308', '#818cf8'],
                    borderWidth: 0, spacing: 2, borderRadius: 4,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                cutout: '65%',
                plugins: { legend: { position: 'bottom' } },
            },
        });
    }

    // Update model table
    renderModelMetrics();
}
