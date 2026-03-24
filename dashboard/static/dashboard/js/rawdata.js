// static/dashboard/js/rawdata.js
// Raw Data Editor — view/edit anomaly labels per run

let RD_DATA   = null;   // current loaded data from API
let RD_MODE   = 'view'; // 'view' | 'add' | 'delete'
let RD_FILTER = '';

const RD_FEAT_COLORS = {
  conductivity: '#2563eb',
  pH:           '#16a34a',
  temperature:  '#d97706',
  voltage:      '#dc2626',
};

const RD_LABEL_COLS_SHORT = {
  'Anomaly V_filled': 'Voltage',
  'Anomaly C_filled': 'Conductivity',
  'Anomaly P_filled': 'pH',
  'Anomaly T_filled': 'Temperature',
};

// ── Mode ───────────────────────────────────────────────────────────
function rdSetMode(mode) {
  RD_MODE = mode;
  document.querySelectorAll('.rd-mode-btn').forEach(b => b.classList.remove('rd-active'));
  document.getElementById(`rd-mode-${mode}`)?.classList.add('rd-active');

  const hints = {
    view:   '👁 View only — เลือก mode เพื่อแก้ไข label',
    add:    '➕ Add mode — คลิกบนกราฟเพื่อ mark anomaly | คลิก+ลาก = mark ช่วง',
    delete: '🗑 Remove mode — คลิกบนกราฟเพื่อลบ anomaly | คลิก+ลาก = ลบช่วง',
  };
  document.getElementById('rd-mode-hint').textContent = hints[mode] || '';
}

// ── Load run ───────────────────────────────────────────────────────
async function rdLoad() {
  const run = document.getElementById('rd-run')?.value;
  if (!run) return;

  document.getElementById('rd-chart-title').textContent = run;
  document.getElementById('rd-chart').innerHTML =
    '<div style="padding:40px;text-align:center;color:var(--text-3);font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;">⏳ Loading…</div>';

  try {
    const res  = await fetch(`/api/rawdata/run/?run=${run}`);
    if (!res.ok) throw new Error(await res.text());
    RD_DATA = await res.json();
    rdRenderChart();
    rdRenderStats();
    rdRenderTable();
    rdUpdateUndoBtn();
  } catch(e) {
    document.getElementById('rd-chart').innerHTML =
      `<div style="padding:20px;color:var(--red);font-family:'JetBrains Mono',monospace;">❌ ${e.message}</div>`;
  }
}

// ── Chart ──────────────────────────────────────────────────────────
function rdRenderChart() {
  if (!RD_DATA) return;
  const d      = RD_DATA;
  const sensor = document.getElementById('rd-sensor')?.value || FEATS[0];
  const vals   = d.sensor_data[sensor] || [];
  const labels = d.label_data;
  const n      = vals.length;
  const xs     = Array.from({length: n}, (_, i) => i);

  // Find label col for this sensor
  const sensorToCol = {
    voltage:      'Anomaly V_filled',
    conductivity: 'Anomaly C_filled',
    pH:           'Anomaly P_filled',
    temperature:  'Anomaly T_filled',
  };
  const thisCol  = sensorToCol[sensor];
  const thisMask = labels[thisCol] || new Array(n).fill(0);
  const overall  = d.overall || new Array(n).fill(0);

  const traces = [];

  // Anomaly vrect via shapes (add as shapes, not traces)
  const shapes = [];
  let inSeg = false, segStart = 0;
  for (let i = 0; i <= n; i++) {
    const flag = i < n ? (overall[i] === 1) : false;
    if (flag && !inSeg) { segStart = i; inSeg = true; }
    else if (!flag && inSeg) {
      shapes.push({ type:'rect', x0: segStart - 0.5, x1: i - 0.5,
        yref:'paper', y0:0, y1:1,
        fillcolor:'rgba(239,68,68,0.10)', line:{width:0}, layer:'below' });
      inSeg = false;
    }
  }

  // Sensor line
  traces.push({
    x: xs, y: vals,
    mode: 'lines',
    name: sensor,
    line: { color: RD_FEAT_COLORS[sensor] || '#2563eb', width: 2 },
    hovertemplate: 'idx=%{x}<br>val=%{y:.4f}<extra></extra>',
  });

  // Highlight anomaly points on this sensor
  const anomIdx = xs.filter(i => thisMask[i] === 1);
  const anomY   = anomIdx.map(i => vals[i]);
  if (anomIdx.length) {
    traces.push({
      x: anomIdx, y: anomY,
      mode: 'markers',
      name: `${sensor} anomaly`,
      marker: { color: '#dc2626', size: 5, symbol: 'circle' },
      hoverinfo: 'skip',
    });
  }

  const layout = {
    height: 300,
    margin: { l:50, r:20, t:30, b:40 },
    paper_bgcolor: '#ffffff', plot_bgcolor: '#f8fafc',
    font: { family: 'Inter,sans-serif', size: 11, color: '#94a3b8' },
    xaxis: { gridcolor:'#e2e8f0', zeroline:false, title:'Index' },
    yaxis: { gridcolor:'#e2e8f0', zeroline:false, title:sensor },
    shapes,
    legend: { orientation:'h', y:1.08 },
    dragmode: 'select',
    selectdirection: 'h',
  };

  const div = document.getElementById('rd-chart');
  div.innerHTML = '';
  Plotly.newPlot(div, traces, layout, { displayModeBar: true, responsive: true,
    modeBarButtonsToRemove: ['lasso2d','autoScale2d'] });

  // Hook selection event for add/delete
  div.on('plotly_selected', (eventData) => {
    if (RD_MODE === 'view' || !eventData?.range) return;
    const [x0, x1] = eventData.range.x;
    const s = Math.max(0, Math.round(x0));
    const e = Math.min(n - 1, Math.round(x1));
    rdPatch(thisCol, s, e, RD_MODE === 'add' ? 1 : 0);
  });

  // Click for single point toggle
  div.on('plotly_click', (eventData) => {
    if (RD_MODE === 'view') return;
    const idx = Math.round(eventData.points[0].x);
    rdPatch(thisCol, idx, idx, RD_MODE === 'add' ? 1 : 0);
  });
}

// ── Patch ──────────────────────────────────────────────────────────
async function rdPatch(labelCol, startIdx, endIdx, value) {
  const run = RD_DATA?.run_id;
  if (!run) return;

  try {
    const res = await fetch('/api/rawdata/patch/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_id: run, label_col: labelCol,
                             start_idx: startIdx, end_idx: endIdx, value }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);

    // Update local data
    RD_DATA.label_data = data.label_data;
    RD_DATA.overall    = data.overall;
    RD_DATA.stats.n_anomaly_pts = data.n_anomaly_pts;

    rdRenderChart();
    rdRenderStats();
    rdRenderTable();
    rdUpdateUndoBtn();
  } catch(e) {
    alert('Patch error: ' + e.message);
  }
}

// ── Undo / Reset ───────────────────────────────────────────────────
async function rdUndo() {
  const run = RD_DATA?.run_id;
  if (!run) return;
  try {
    const res  = await fetch('/api/rawdata/undo/', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ run_id: run }),
    });
    const data = await res.json();
    if (!data.ok) { alert(data.error); return; }
    RD_DATA.label_data = data.label_data;
    RD_DATA.overall    = data.overall;
    rdRenderChart(); rdRenderStats(); rdRenderTable();
    rdUpdateUndoBtn(data.history_remaining);
  } catch(e) { alert('Undo error: ' + e.message); }
}

async function rdReset() {
  if (!confirm('Reset labels กลับไปค่าเดิม?')) return;
  const run = RD_DATA?.run_id;
  await fetch('/api/rawdata/reset/', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ run_id: run }),
  });
  await rdLoad();
}

function rdUpdateUndoBtn(remaining) {
  const btn = document.getElementById('rd-undo-btn');
  if (!btn) return;
  const hist = remaining ?? (RD_DATA?.stats?.has_history ?? 0);
  btn.disabled = hist <= 0;
}

// ── Stats ──────────────────────────────────────────────────────────
function rdRenderStats() {
  if (!RD_DATA) return;
  const s = RD_DATA.stats;
  document.getElementById('rd-stats').innerHTML = `
    <div>📍 Total pts: <b>${s.n_rows.toLocaleString()}</b></div>
    <div>🔴 Anomaly pts: <b style="color:var(--red);">${s.n_anomaly_pts.toLocaleString()}</b> (${s.pct_anomaly}%)</div>
    <div>📦 Segments: <b>${s.n_segments_overall}</b></div>`;

  let perSen = '';
  for (const [col, info] of Object.entries(s.per_sensor || {})) {
    const short = RD_LABEL_COLS_SHORT[col] || col;
    perSen += `<div>${short}: <b>${info.n_pts}</b> pts, <b>${info.n_segs}</b> seg</div>`;
  }
  document.getElementById('rd-per-sensor-stats').innerHTML = perSen || '—';
}

// ── Table ──────────────────────────────────────────────────────────
function rdRenderTable() {
  if (!RD_DATA) return;
  const d = RD_DATA;
  const feats = d.feats || [];
  const lcols = d.label_cols || [];

  // Header
  document.getElementById('rd-thead').innerHTML = `
    <tr style="border-bottom:2px solid var(--border);">
      <th style="padding:4px 8px;text-align:right;color:var(--text-3);font-weight:400;">#</th>
      ${feats.map(f => `<th style="padding:4px 8px;color:var(--text-3);font-weight:400;">${f}</th>`).join('')}
      ${lcols.map(c => `<th style="padding:4px 8px;color:var(--red);font-weight:400;font-size:0.65rem;">${RD_LABEL_COLS_SHORT[c]||c}</th>`).join('')}
      <th style="padding:4px 8px;color:var(--red);font-weight:600;font-size:0.65rem;">Any</th>
    </tr>`;

  rdFilterTable(document.getElementById('rd-search')?.value || '');
}

function rdFilterTable(query) {
  if (!RD_DATA) return;
  const d        = RD_DATA;
  const feats    = d.feats || [];
  const lcols    = d.label_cols || [];
  const anomOnly = document.getElementById('rd-anomaly-only')?.checked;
  const q        = query.trim().toLowerCase();

  const rows = d.rows || [];
  const visible = rows.filter(r => {
    if (anomOnly && !r.anomaly_any) return false;
    if (!q) return true;
    return String(r.idx).includes(q) ||
      feats.some(f => String(r[f] ?? '').includes(q));
  });

  document.getElementById('rd-table-count').textContent = `${visible.length} / ${rows.length} rows`;

  document.getElementById('rd-tbody').innerHTML = visible.slice(0, 2000).map(r => {
    const isAnom = r.anomaly_any === 1;
    const bg = isAnom ? 'background:rgba(239,68,68,0.05);' : '';
    return `<tr style="${bg}border-bottom:1px solid var(--border);">
      <td style="padding:3px 8px;text-align:right;color:var(--text-3);">${r.idx}</td>
      ${feats.map(f => `<td style="padding:3px 8px;text-align:right;">${r[f] ?? '—'}</td>`).join('')}
      ${lcols.map(c => {
        const v = r[c] ?? 0;
        return `<td style="padding:3px 8px;text-align:center;color:${v?'var(--red)':'var(--text-3)'};">${v ? '●' : '○'}</td>`;
      }).join('')}
      <td style="padding:3px 8px;text-align:center;font-weight:${isAnom?700:400};color:${isAnom?'var(--red)':'var(--text-3)'};">${isAnom?'●':'○'}</td>
    </tr>`;
  }).join('');
}

// ── Export CSV ─────────────────────────────────────────────────────
function rdExportCSV() {
  // Always export ALL runs with current labels
  window.location.href = '/api/rawdata/export/?run=all';
}

function rdExportThisRun() {
  const run = RD_DATA?.run_id;
  if (!run) { alert('Load a run first'); return; }
  window.location.href = `/api/rawdata/export/?run=${run}`;
}