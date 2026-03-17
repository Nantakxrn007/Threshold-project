// static/dashboard/js/charts.js

// ── Global helpers ──────────────────────────────────────────────────
function getAgg()        { return document.getElementById('agg-method')?.value || 'mean'; }
function getShowLabels() { return document.getElementById('show-labels')?.checked ? '1' : '0'; }

// ── UMAP Tab ───────────────────────────────────────────────────────
function toggleUmapParams() {
  const isPCA = document.getElementById('umap-method').value === 'pca';
  document.getElementById('umap-nn').disabled = isPCA;
  document.getElementById('umap-md').disabled = isPCA;
  const op = isPCA ? '0.4' : '1';
  ['label-nn','row-nn','label-md','row-md'].forEach(id =>
    document.getElementById(id).style.opacity = op);
}

async function loadUmap() {
  if (!JOB_ID) {
    document.getElementById('umap-chart').innerHTML =
      '<div style="padding:40px;text-align:center;color:var(--text-3);font-family:\'JetBrains Mono\',monospace;font-size:0.85rem;">Train a model first.</div>';
    return;
  }
  const btn = document.getElementById('btn-recompute');
  btn.disabled = true; btn.innerHTML = '⏳ Computing...';
  const method = document.getElementById('umap-method').value;
  const nn = document.getElementById('umap-nn').value;
  const md = document.getElementById('umap-md').value;
  const cb = document.getElementById('umap-color').value;
  const chartDiv = document.getElementById('umap-chart');

  let spinner = document.getElementById('umap-spinner');
  if (!spinner) {
    spinner = document.createElement('div');
    spinner.id = 'umap-spinner';
    spinner.style.cssText = 'position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;align-items:center;background:rgba(248,250,252,0.85);z-index:10;border-radius:8px;';
    chartDiv.style.position = 'relative';
    chartDiv.appendChild(spinner);
  }
  spinner.style.display = 'flex';
  spinner.innerHTML = `<div style="color:var(--text-2);font-family:'JetBrains Mono',monospace;font-size:0.9rem;font-weight:600;">⏳ Computing ${method.toUpperCase()}…</div>`;
  try {
    const res = await fetch(`/api/charts/umap/?job_id=${JOB_ID}&method=${method}&nn=${nn}&md=${md}&color_by=${cb}`);
    if (!res.ok) throw new Error('Computation failed.');
    const fig = await res.json();
    Plotly.newPlot('umap-chart', fig.data, fig.layout, {displayModeBar:false, responsive:true});
  } catch (err) {
    spinner.innerHTML = `<div style="color:var(--red);">❌ ${err.message}</div>`;
  } finally {
    spinner.style.display = 'none';
    btn.disabled = false; btn.innerHTML = '↻ Recompute';
  }
}

// ── Inspector Tab ──────────────────────────────────────────────────
async function loadInspector() {
  if (!JOB_ID) return;
  const run = document.getElementById('inspect-run').value;
  const agg = getAgg(), sl = getShowLabels();
  const [rawRes, errRes, umapRes] = await Promise.all([
    fetch(`/api/charts/raw/?run=${run}&show_labels=${sl}`),
    fetch(`/api/charts/error/?job_id=${JOB_ID}&run=${run}&agg=${agg}&show_labels=${sl}`),
    fetch(`/api/charts/inspector-umap/?job_id=${JOB_ID}&run=${run}`),
  ]);
  const [rawFig, errFig, umapFig] = await Promise.all([rawRes.json(), errRes.json(), umapRes.json()]);
  Plotly.react('inspect-raw-chart',  rawFig.data,  rawFig.layout,  {displayModeBar:false, responsive:true});
  Plotly.react('inspect-err-chart',  errFig.data,  errFig.layout,  {displayModeBar:false, responsive:true});
  Plotly.react('inspect-umap-chart', umapFig.data, umapFig.layout, {displayModeBar:false, responsive:true});

  const rawEl = document.getElementById('inspect-raw-chart');
  rawEl.removeAllListeners && rawEl.removeAllListeners('plotly_hover');
  rawEl.on('plotly_hover', function(ev) {
    const x = ev.points[0].x;
    Plotly.relayout('inspect-err-chart', {
      shapes: [{type:'line', x0:x, x1:x, y0:0, y1:1, yref:'paper',
                line:{color:'#f59e0b', width:1.5, dash:'dot'}}],
    });
    const umapEl = document.getElementById('inspect-umap-chart');
    const selIdx = umapEl.data?.findIndex(t => t.name === 'selected');
    if (selIdx >= 0) {
      const nPts = umapEl.data[selIdx].x.length;
      const ptIdx = Math.min(Math.round(x), nPts - 1);
      const sizes  = Array(nPts).fill(6); sizes[ptIdx]  = 14;
      const colors = Array(nPts).fill('#2563eb'); colors[ptIdx] = '#f59e0b';
      Plotly.restyle('inspect-umap-chart', {'marker.size':[sizes],'marker.color':[colors]}, [selIdx]);
    }
  });
  rawEl.on('plotly_unhover', function() {
    Plotly.relayout('inspect-err-chart', {shapes: []});
    const umapEl = document.getElementById('inspect-umap-chart');
    const selIdx = umapEl.data?.findIndex(t => t.name === 'selected');
    if (selIdx >= 0) {
      const n = umapEl.data[selIdx].x.length;
      Plotly.restyle('inspect-umap-chart',
        {'marker.size':[Array(n).fill(6)],'marker.color':[Array(n).fill('#2563eb')]}, [selIdx]);
    }
  });
}

// ── Anomaly Tab ────────────────────────────────────────────────────
async function loadAnomalyCharts() {
  if (!JOB_ID) return;
  const container = document.getElementById('anomaly-charts');
  container.innerHTML = '';
  const agg = getAgg(), sl = getShowLabels();
  for (const run of ANOMALY_RUNS) {
    // Raw sensor card
    const rawCard = document.createElement('div');
    rawCard.className = 'card';
    const rawId = `anomaly-raw-${run.replace(/[^a-z0-9]/gi,'_')}`;
    rawCard.innerHTML = `<div style="font-size:0.78rem;font-weight:600;color:var(--text-2);margin-bottom:4px;">📡 ${run} — Raw Sensor</div><div id="${rawId}"></div>`;
    container.appendChild(rawCard);
    const rawRes = await fetch(`/api/charts/raw/?run=${run}&show_labels=${sl}`);
    if (rawRes.ok) {
      const rf = await rawRes.json();
      Plotly.newPlot(rawId, rf.data, rf.layout, {displayModeBar:false, responsive:true});
    }
    // Threshold card
    const card = document.createElement('div');
    card.className = 'card';
    const chartId = `anomaly-chart-${run.replace(/[^a-z0-9]/gi,'_')}`;
    card.innerHTML = `<div id="${chartId}"></div>`;
    container.appendChild(card);
    const res = await fetch(`/api/charts/threshold/?job_id=${JOB_ID}&run=${run}&height=340&agg=${agg}&show_labels=${sl}`);
    if (!res.ok) continue;
    const fig = await res.json();
    Plotly.newPlot(chartId, fig.data, fig.layout, {displayModeBar:false, responsive:true});
  }
}

// ── Legend state helpers ───────────────────────────────────────────
// Save { traceName → visible } from a Plotly div (keyed by trace.name)
function _saveLegend(divId) {
  const el = document.getElementById(divId);
  if (!el?.data) return {};
  const state = {};
  el.data.forEach(t => {
    if (t.name) state[t.name] = t.visible ?? true;
  });
  return state;
}

// Restore visibility after Plotly.react() — match by trace name
function _restoreLegend(divId, state) {
  const el = document.getElementById(divId);
  if (!el?.data || !Object.keys(state).length) return;
  const updates = el.data.map(t => state[t.name] ?? true);  // unknown traces default visible
  Plotly.restyle(divId, { visible: updates });
}

// ── Threshold Tuner ────────────────────────────────────────────────
function debounceTuner() {
  clearTimeout(TUNER_TIMER);
  TUNER_TIMER = setTimeout(loadTuner, 250);
}

async function loadTuner() {
  if (!JOB_ID) return;
  const run = document.getElementById('tuner-run').value;
  const agg = getAgg(), sl = getShowLabels();

  // ── Save legend state BEFORE fetch (user may have hidden some traces) ──
  const rawLegend = _saveLegend('tuner-raw-chart');
  const errLegend = _saveLegend('tuner-chart');

  const params = new URLSearchParams({
    job_id: JOB_ID, run, height: 380, agg, show_labels: sl,
    th1_pct:    document.getElementById('th1-pct').value,
    th1_mode:   document.getElementById('th1-mode').value,
    th1_win:    document.getElementById('th1-win').value,
    th1_recalc: document.getElementById('th1-recalc').value,
    th2_alpha:  document.getElementById('th2-alpha').value,
    th2_win:    document.getElementById('th2-win').value,
    th2_recalc: document.getElementById('th2-recalc').value,
    th3_zmin:   document.getElementById('th3-zmin').value,
    th3_zmax:   document.getElementById('th3-zmax').value,
    th3_win:    document.getElementById('th3-win').value,
    th3_recalc: document.getElementById('th3-recalc').value,
    th4_alpha:  document.getElementById('th4-alpha').value,
    th4_win:    document.getElementById('th4-win').value,
    th4_cons:   document.getElementById('th4-cons').value,
    th4_eth:    document.getElementById('th4-eth').value,
    th4_recalc: document.getElementById('th4-recalc').value,
  });
  const [rawRes, errRes] = await Promise.all([
    fetch(`/api/charts/raw/?run=${run}&show_labels=${sl}`),
    fetch(`/api/charts/threshold/?${params}`),
  ]);
  if (!errRes.ok) return;
  const [rawFig, fig] = await Promise.all([rawRes.json(), errRes.json()]);

  Plotly.react('tuner-raw-chart', rawFig.data, rawFig.layout, {displayModeBar:false, responsive:true});
  Plotly.react('tuner-chart',     fig.data,    fig.layout,    {displayModeBar:false, responsive:true});

  // ── Restore legend state AFTER react ──────────────────────────────
  _restoreLegend('tuner-raw-chart', rawLegend);
  _restoreLegend('tuner-chart',     errLegend);

  if (fig.stats) {
    const TC = {'P99 Static':'#ef4444','Sliding Mu+αStd':'#3b82f6','Adaptive-z':'#10b981','Entropy-lock':'#7c3aed'};
    document.getElementById('tuner-stats').innerHTML =
      Object.entries(fig.stats).map(([name, s]) =>
        `<div class="stat-card" style="border-left-color:${TC[name]}">
           <div class="stat-name">${name}</div>
           <div class="stat-value">${s.flagged} pts</div>
           <div class="stat-sub">${s.pct}% flagged</div>
         </div>`).join('');
  }

  // ── Confusion matrix per threshold ────────────────────────────────
  if (fig.metrics) _renderTunerMetrics(fig.metrics);
}

// ── Tuner Confusion Matrix ─────────────────────────────────────────
function _renderTunerMetrics(metrics) {
  const el = document.getElementById('tuner-cm');
  if (!el) return;

  const TC = {
    'P99 Static':      { color: '#ef4444', bg: '#fef2f2', border: '#fecaca' },
    'Sliding Mu+αStd': { color: '#3b82f6', bg: '#eff6ff', border: '#bfdbfe' },
    'Adaptive-z':      { color: '#10b981', bg: '#f0fdf4', border: '#bbf7d0' },
    'Entropy-lock':    { color: '#7c3aed', bg: '#faf5ff', border: '#e9d5ff' },
  };

  // Check if any threshold has labels
  const hasLabels = Object.values(metrics).some(m => m.has_labels);

  if (!hasLabels) {
    el.innerHTML = `<div style="color:var(--text-3);font-size:0.75rem;font-family:'JetBrains Mono',monospace;padding:12px 0;">
      ไม่มี Anomaly Labels สำหรับ run นี้ (Normal run)
    </div>`;
    return;
  }

  // Bar helper: colored progress bar
  const bar = (val, max, color) => {
    const pct = max > 0 ? Math.round(100 * val / max) : 0;
    return `<div style="height:4px;background:#e2e8f0;border-radius:2px;margin-top:3px;">
      <div style="width:${pct}%;height:100%;background:${color};border-radius:2px;transition:width .3s;"></div>
    </div>`;
  };

  // F1 color: red→amber→green
  const f1Clr = f1 => {
    if (f1 >= 0.7) return '#16a34a';
    if (f1 >= 0.4) return '#d97706';
    return '#dc2626';
  };

  let html = `<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:4px;">`;

  for (const [name, m] of Object.entries(metrics)) {
    if (!m.has_labels) continue;
    const tc   = TC[name] || { color:'#64748b', bg:'#f8fafc', border:'#e2e8f0' };
    const total = m.tp + m.fp + m.tn + m.fn;
    const f1pct = Math.round((m.f1 || 0) * 100);
    const ppct  = Math.round((m.precision || 0) * 100);
    const rpct  = Math.round((m.recall || 0) * 100);
    const apct  = Math.round((m.accuracy || 0) * 100);
    const segTxt = m.n_segments > 0
      ? `${m.n_detected_segs}/${m.n_segments} segs`
      : '—';

    html += `
    <div style="border:1px solid ${tc.border};border-left:3px solid ${tc.color};background:${tc.bg};border-radius:6px;padding:12px;">
      <div style="font-size:0.72rem;font-weight:700;color:${tc.color};margin-bottom:8px;font-family:'JetBrains Mono',monospace;">${name}</div>

      <!-- Confusion matrix 2×2 -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:10px;font-family:'JetBrains Mono',monospace;font-size:0.7rem;text-align:center;">
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:4px;padding:4px;">
          <div style="color:#64748b;font-size:0.62rem;">TP</div>
          <div style="font-size:1rem;font-weight:700;color:#16a34a;">${m.tp}</div>
        </div>
        <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:4px;padding:4px;">
          <div style="color:#64748b;font-size:0.62rem;">FP</div>
          <div style="font-size:1rem;font-weight:700;color:#dc2626;">${m.fp}</div>
        </div>
        <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:4px;padding:4px;">
          <div style="color:#64748b;font-size:0.62rem;">FN</div>
          <div style="font-size:1rem;font-weight:700;color:#d97706;">${m.fn}</div>
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:4px;padding:4px;">
          <div style="color:#64748b;font-size:0.62rem;">TN</div>
          <div style="font-size:1rem;font-weight:700;color:#475569;">${m.tn}</div>
        </div>
      </div>

      <!-- F1 big score -->
      <div style="text-align:center;margin-bottom:8px;">
        <span style="font-size:1.4rem;font-weight:800;color:${f1Clr(m.f1)};font-family:'JetBrains Mono',monospace;">${f1pct}%</span>
        <span style="font-size:0.65rem;color:var(--text-3);margin-left:4px;">F1</span>
      </div>

      <!-- Precision / Recall / Acc bars -->
      <div style="font-size:0.68rem;color:var(--text-2);">
        <div style="display:flex;justify-content:space-between;"><span>Precision</span><span style="font-weight:600;">${ppct}%</span></div>
        ${bar(m.precision, 1, tc.color)}
        <div style="display:flex;justify-content:space-between;margin-top:5px;"><span>Recall</span><span style="font-weight:600;">${rpct}%</span></div>
        ${bar(m.recall, 1, tc.color)}
        <div style="display:flex;justify-content:space-between;margin-top:5px;"><span>Accuracy</span><span style="font-weight:600;">${apct}%</span></div>
        ${bar(m.accuracy, 1, '#94a3b8')}
      </div>

      <!-- Segment detection -->
      <div style="margin-top:8px;font-size:0.68rem;color:var(--text-3);font-family:'JetBrains Mono',monospace;
                  border-top:1px solid ${tc.border};padding-top:6px;">
        Seg detected: <b style="color:${tc.color};">${segTxt}</b>
        &nbsp;|&nbsp; total: ${total} pts
      </div>
    </div>`;
  }

  html += `</div>`;
  el.innerHTML = html;
}

// ── Reset Threshold Sliders ────────────────────────────────────────
const TH_DEFAULTS = {
  1: {
    'th1-pct':    [99.0, v => v.toFixed(1)],
    'th1-win':    [100,  v => v],
    'th1-recalc': [10,   v => v],
  },
  2: { 'th2-alpha': [3.5, v => v.toFixed(1)], 'th2-win': [80, v => v], 'th2-recalc': [50, v => v] },
  3: { 'th3-zmin': [2.0, v => v.toFixed(1)], 'th3-zmax': [10.0, v => v.toFixed(1)], 'th3-win': [80, v => v], 'th3-recalc': [1, v => v] },
  4: { 'th4-alpha': [3.5, v => v.toFixed(1)], 'th4-win': [150, v => v], 'th4-cons': [5, v => v], 'th4-eth': [0.95, v => v.toFixed(2)], 'th4-recalc': [1, v => v] },
};

function resetTh(n) {
  // Reset selects (mode)
  if (n === 1) {
    const modeEl = document.getElementById('th1-mode');
    if (modeEl) { modeEl.value = 'sliding'; _toggleTh1SlidingParams(); }
  }
  for (const [id, [val, fmt]] of Object.entries(TH_DEFAULTS[n])) {
    const el = document.getElementById(id);
    if (el) { el.value = val; document.getElementById(id+'-val').textContent = fmt(val); }
  }
  debounceTuner();
}

// Show/hide TH1 window params based on mode
function _toggleTh1SlidingParams() {
  const isSliding = document.getElementById('th1-mode')?.value === 'sliding';
  const panel = document.getElementById('th1-sliding-params');
  if (panel) panel.style.display = isSliding ? 'block' : 'none';
}