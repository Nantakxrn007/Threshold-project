// static/dashboard/js/results.js

// ── Cache helpers ──────────────────────────────────────────────────
function _getThHash() {
  return ['th1-pct','th1-mode','th1-win','th1-recalc',
          'th2-alpha','th2-win','th2-recalc',
          'th3-zmin','th3-zmax','th3-win','th3-recalc',
          'th4-alpha','th4-win','th4-cons','th4-eth','th4-recalc']
    .map(id => document.getElementById(id)?.value ?? '')
    .join(',');
}

// ── Load (with cache) ──────────────────────────────────────────────
async function loadResultsTable(forceRefresh = false) {
  const container = document.getElementById('results-content');
  if (!container) return;

  if (!BATCH_JOB_ID) {
    container.innerHTML = `
      <div style="text-align:center;padding:56px;color:var(--text-3);font-size:0.82rem;font-family:'JetBrains Mono',monospace;">
        ⚠️  Run <b>🏋️ Train All Models</b> first, then come back here.
      </div>`;
    return;
  }

  const cacheKey = `${BATCH_JOB_ID}|${_getThHash()}`;
  if (!forceRefresh && RESULTS_CACHE && RESULTS_CACHE_KEY === cacheKey) {
    renderResultsTable(RESULTS_CACHE, container);
    return;
  }

  container.innerHTML = `
    <div style="text-align:center;padding:40px;color:var(--text-3);font-family:'JetBrains Mono',monospace;">
      ⏳ Computing metrics…
    </div>`;

  // Show Stop button while computing
  const computeBtn = document.getElementById('results-compute-btn');
  const stopBtn    = document.getElementById('results-stop-btn');
  if (computeBtn) computeBtn.style.display = 'none';
  if (stopBtn)    stopBtn.style.display    = 'inline-flex';

  const g = id => document.getElementById(id)?.value ?? '';
  const params = new URLSearchParams({
    job_id:     BATCH_JOB_ID,
    th1_pct:    g('th1-pct')    || 99.0,
    th1_mode:   g('th1-mode')   || 'sliding',
    th1_win:    g('th1-win')    || 100,
    th1_recalc: g('th1-recalc') || 10,
    th2_alpha:  g('th2-alpha')  || 3.5,
    th2_win:    g('th2-win')    || 80,
    th2_recalc: g('th2-recalc') || 50,
    th3_zmin:   g('th3-zmin')   || 2.0,
    th3_zmax:   g('th3-zmax')   || 10.0,
    th3_win:    g('th3-win')    || 80,
    th3_recalc: g('th3-recalc') || 1,
    th4_alpha:  g('th4-alpha')  || 3.5,
    th4_win:    g('th4-win')    || 150,
    th4_cons:   g('th4-cons')   || 5,
    th4_eth:    g('th4-eth')    || 0.95,
    th4_recalc: g('th4-recalc') || 1,
  });

  try {
    const res = await fetch(`/api/results-table/?${params}`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    RESULTS_CACHE     = data;
    RESULTS_CACHE_KEY = cacheKey;

    // Hide stop, show compute
    if (computeBtn) computeBtn.style.display = 'inline-flex';
    if (stopBtn)    stopBtn.style.display    = 'none';

    renderResultsTable(data, container);
  } catch (err) {
    if (computeBtn) computeBtn.style.display = 'inline-flex';
    if (stopBtn)    stopBtn.style.display    = 'none';
    container.innerHTML = `
      <div style="color:var(--red);padding:32px;text-align:center;font-family:'JetBrains Mono',monospace;">
        ❌ ${err.message}
      </div>`;
  }
}

// ── Scoring ────────────────────────────────────────────────────────
function _scoreModel(rows, modelName, scoringMode) {
  // rows filtered to this model
  const modelRows = rows.filter(r => r.model === modelName);
  const f1s  = modelRows.map(r => r.aggregate.f1   || 0);
  const recs = modelRows.map(r => r.aggregate.recall || 0);
  const mean = arr => arr.reduce((a,b)=>a+b,0) / Math.max(arr.length,1);

  if (scoringMode === 'f1_recall') {
    return 0.6 * mean(f1s) + 0.4 * mean(recs);
  }
  return mean(f1s); // default: mean_f1
}

function _computeRankings(rows, scoringMode) {
  // Get unique models
  const models = [...new Set(rows.map(r => r.model))];
  return models
    .map(m => ({ model: m, score: _scoreModel(rows, m, scoringMode) }))
    .sort((a, b) => b.score - a.score);
}

// ── Save CSV ───────────────────────────────────────────────────────
function saveResultsCSV() {
  if (!RESULTS_CACHE) return;
  const { rows, has_labels, runs, normal_runs } = RESULTS_CACHE;

  const headers = ['Model','Threshold','Precision','Recall','F1','Accuracy',
                   'TP','FP','TN','FN','Seg_Detected','Seg_Total',
                   ...runs.map(r => `F1_${r}`)];

  const csvRows = [headers.join(',')];
  for (const row of rows) {
    const agg = row.aggregate;
    const perRun = runs.map(r => {
      const m = row.per_run?.[r] || {};
      return has_labels ? ((m.f1||0)*100).toFixed(1) : (m.pct||0);
    });
    csvRows.push([
      `"${row.model}"`,
      `"${row.threshold}"`,
      has_labels ? ((agg.precision||0)*100).toFixed(1) : '',
      has_labels ? ((agg.recall||0)*100).toFixed(1)    : '',
      has_labels ? ((agg.f1||0)*100).toFixed(1)        : '',
      has_labels ? ((agg.accuracy||0)*100).toFixed(1)  : '',
      agg.tp||0, agg.fp||0, agg.tn||0, agg.fn||0,
      agg.n_detected_segs||0, agg.n_segments||0,
      ...perRun,
    ].join(','));
  }

  const blob = new Blob([csvRows.join('\n')], {type:'text/csv'});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'results.csv'; a.click();
  URL.revokeObjectURL(url);
}

// ── Render ─────────────────────────────────────────────────────────
function renderResultsTable(data, container) {
  const { rows, has_labels, runs, normal_runs } = data;

  const scoringMode = document.getElementById('result-scoring')?.value || 'mean_f1';
  const rankings    = has_labels ? _computeRankings(rows, scoringMode) : [];
  const bestModel   = rankings[0]?.model || null;

  const TH_COLORS_MAP = {
    'P99 Static':'#ef4444','Sliding Mu+αStd':'#3b82f6',
    'Adaptive-z':'#10b981','Entropy-lock':'#7c3aed',
  };
  const f1Color = v => {
    if (v === undefined) return '';
    const r = Math.round(255*(1-v)), g = Math.round(200*v);
    return `background:rgba(${r},${g},60,0.15)`;
  };
  const medal = ['🥇','🥈','🥉'];

  // ── Best Model Banner ──────────────────────────────────────────
  let html = '';
  if (has_labels && rankings.length) {
    const scoringLabel = scoringMode === 'f1_recall'
      ? '0.6×mean(F1) + 0.4×mean(Recall)'
      : 'mean(F1) across all thresholds';

    html += `<div class="card" style="padding:14px;margin-bottom:12px;background:linear-gradient(135deg,#eff6ff,#f0fdf4);border:1px solid #bfdbfe;">
      <div style="font-size:0.72rem;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">
        🏆 Model Ranking
        <span style="font-weight:400;font-size:0.68rem;margin-left:8px;color:var(--text-3);">scoring: ${scoringLabel}</span>
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:stretch;">`;

    rankings.forEach((r, i) => {
      const isBest   = i === 0;
      const scorePct = (r.score * 100).toFixed(1);
      const barW     = rankings[0].score > 0
        ? Math.round(100 * r.score / rankings[0].score) : 0;
      html += `
        <div style="flex:1;min-width:140px;border:${isBest?'2px solid #2563eb':'1px solid var(--border)'};
                    border-radius:8px;padding:10px 12px;background:${isBest?'#eff6ff':'var(--bg)'};
                    position:relative;">
          ${isBest ? '<div style="position:absolute;top:-8px;right:8px;font-size:0.65rem;background:#2563eb;color:#fff;padding:1px 7px;border-radius:10px;font-weight:700;">BEST</div>' : ''}
          <div style="font-size:1.1rem;margin-bottom:2px;">${medal[i] || `#${i+1}`}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;font-weight:700;color:var(--text);margin-bottom:6px;">${r.model}</div>
          <div style="font-size:1.3rem;font-weight:800;color:${isBest?'#2563eb':'var(--text-2)'};font-family:'JetBrains Mono',monospace;">${scorePct}%</div>
          <div style="height:4px;background:#e2e8f0;border-radius:2px;margin-top:6px;">
            <div style="width:${barW}%;height:100%;background:${isBest?'#2563eb':'#94a3b8'};border-radius:2px;"></div>
          </div>
        </div>`;
    });
    html += `</div></div>`;
  }

  // ── Toolbar ────────────────────────────────────────────────────
  html += `<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px;">`;
  if (has_labels) {
    html += `
      <div style="display:flex;align-items:center;gap:6px;font-size:0.75rem;color:var(--text-2);">
        <label style="font-size:0.68rem;font-weight:600;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;">Scoring</label>
        <select id="result-scoring" onchange="renderResultsTable(RESULTS_CACHE, document.getElementById('results-content'))"
                style="font-size:0.75rem;padding:4px 8px;border:1px solid var(--border);border-radius:5px;background:var(--bg);">
          <option value="mean_f1" ${scoringMode==='mean_f1'?'selected':''}>Mean F1</option>
          <option value="f1_recall" ${scoringMode==='f1_recall'?'selected':''}>0.6×F1 + 0.4×Recall</option>
        </select>
      </div>`;
  }
  html += `
    <button onclick="saveResultsCSV()"
            style="margin-left:auto;padding:6px 14px;border-radius:6px;border:1px solid var(--border);
                   background:var(--bg);font-size:0.75rem;cursor:pointer;display:flex;align-items:center;gap:5px;
                   color:var(--text-2);font-weight:500;"
            title="Download table as CSV">
      ⬇ Save CSV
    </button>
  </div>`;

  // ── Table ──────────────────────────────────────────────────────
  html += `<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:0.75rem;">
    <thead><tr style="background:var(--bg-1);border-bottom:2px solid var(--border);">
      <th style="padding:8px 10px;text-align:left;white-space:nowrap;">Model</th>
      <th style="padding:8px 10px;text-align:left;white-space:nowrap;">Threshold</th>`;

  if (has_labels) {
    html += `
      <th style="padding:8px 10px;text-align:center;background:#eff6ff;border-left:2px solid var(--border);" title="Precision">P</th>
      <th style="padding:8px 10px;text-align:center;background:#eff6ff;" title="Recall">R</th>
      <th style="padding:8px 10px;text-align:center;background:#eff6ff;" title="F1">F1 ★</th>
      <th style="padding:8px 10px;text-align:center;background:#eff6ff;" title="Accuracy">Acc</th>
      <th style="padding:8px 10px;text-align:center;background:#f0fdf4;" title="True Positive">TP</th>
      <th style="padding:8px 10px;text-align:center;background:#fef2f2;" title="False Positive">FP</th>
      <th style="padding:8px 10px;text-align:center;background:#f8fafc;" title="True Negative">TN</th>
      <th style="padding:8px 10px;text-align:center;background:#fff7ed;" title="False Negative">FN</th>
      <th style="padding:8px 10px;text-align:center;background:#faf5ff;border-right:2px solid var(--border);" title="Segment detection">Seg</th>`;
  }
  for (const r of runs) {
    const isNorm = normal_runs.includes(r);
    html += `<th style="padding:6px 8px;text-align:center;white-space:nowrap;font-size:0.68rem;color:${isNorm?'var(--green)':'var(--red)'};">${r}</th>`;
  }
  html += `</tr></thead><tbody>`;

  let lastModel = '';
  for (const row of rows) {
    const isNew  = row.model !== lastModel;
    lastModel    = row.model;
    const isBest = row.model === bestModel;
    const rowBg  = isNew
      ? `background:${isBest?'#eff6ff':'var(--bg-1)'};border-top:2px solid ${isBest?'#bfdbfe':'var(--border)'};`
      : isBest ? 'background:#f8fbff;' : '';
    const thColor = TH_COLORS_MAP[row.threshold] || '#64748b';
    const agg     = row.aggregate;

    html += `<tr style="${rowBg}">`;
    if (isNew) {
      html += `<td rowspan="4" style="padding:8px 10px;font-family:'JetBrains Mono',monospace;font-weight:700;
                    vertical-align:middle;border-right:1px solid var(--border);white-space:nowrap;">
        ${isBest ? '🏆 ' : ''}${row.model}
        ${isBest ? '<div style="font-size:0.6rem;color:#2563eb;font-weight:600;margin-top:2px;">BEST MODEL</div>' : ''}
      </td>`;
    }
    html += `<td style="padding:6px 10px;font-size:0.72rem;white-space:nowrap;border-left:3px solid ${thColor};">${row.threshold}</td>`;

    if (has_labels) {
      const seg = `${agg.n_detected_segs||0}/${agg.n_segments||0}`;
      html += `
        <td style="padding:6px 8px;text-align:center;background:#eff6ff;">${((agg.precision||0)*100).toFixed(1)}%</td>
        <td style="padding:6px 8px;text-align:center;background:#eff6ff;">${((agg.recall||0)*100).toFixed(1)}%</td>
        <td style="padding:6px 8px;text-align:center;background:#eff6ff;font-weight:700;${f1Color(agg.f1)}">${((agg.f1||0)*100).toFixed(1)}%</td>
        <td style="padding:6px 8px;text-align:center;background:#eff6ff;">${((agg.accuracy||0)*100).toFixed(1)}%</td>
        <td style="padding:6px 8px;text-align:center;background:#f0fdf4;color:var(--green);">${agg.tp||0}</td>
        <td style="padding:6px 8px;text-align:center;background:#fef2f2;color:var(--red);">${agg.fp||0}</td>
        <td style="padding:6px 8px;text-align:center;background:#f8fafc;">${agg.tn||0}</td>
        <td style="padding:6px 8px;text-align:center;background:#fff7ed;color:var(--amber);">${agg.fn||0}</td>
        <td style="padding:6px 8px;text-align:center;background:#faf5ff;font-weight:600;">${seg}</td>`;
    }
    for (const r of runs) {
      const m      = row.per_run?.[r] || {};
      const cellBg = has_labels ? (normal_runs.includes(r) ? '' : f1Color(m.f1)) : '';
      const val    = has_labels ? (m.f1 !== undefined ? `${(m.f1*100).toFixed(0)}%` : `${m.pct||0}%`) : `${m.pct||0}%`;
      const tip    = has_labels && m.tp !== undefined
        ? `F1:${(m.f1*100).toFixed(1)}% P:${(m.precision*100).toFixed(0)}% R:${(m.recall*100).toFixed(0)}% | TP:${m.tp} FP:${m.fp} FN:${m.fn} | Seg:${m.n_detected_segs||0}/${m.n_segments||0}`
        : `${m.flagged||0} pts`;
      html += `<td style="padding:6px 8px;text-align:center;${cellBg}" title="${tip}">${val}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table></div>';

  // ── PA Explanation ─────────────────────────────────────────────
  html += `
  <div style="margin-top:24px;">
    <div class="card" style="padding:18px;">
      <div style="font-size:0.85rem;font-weight:700;margin-bottom:2px;">⚡ Point-Adjusted Evaluation</div>
      <div style="font-size:0.76rem;color:var(--text-2);margin-bottom:16px;line-height:1.7;">
        ถ้า model ตรวจเจอ anomaly <b>อย่างน้อย 1 จุด</b> ใน segment →
        นับ <b style="color:var(--green);">TP ทั้ง segment</b><br>
        เพราะ operator เจอ alert แล้วก็ตรวจทั้ง segment อยู่ดี —
        ไม่ใช่ fault ที่ model miss บางจุดใน segment เดียวกัน
      </div>
      <div style="font-size:0.75rem;font-weight:600;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;">ตัวอย่าง: 40 data points, anomaly segment = idx 15–25 (11 pts), model flags idx 18–20 (3 pts)</div>
      <div id="pa-demo" style="font-family:'JetBrains Mono',monospace;"></div>
      <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;align-items:center;">
        <span style="font-size:0.72rem;color:var(--text-3);">ขั้นตอน →</span>
        <button onclick="paStep(0)" id="pa-btn-0" style="padding:5px 14px;border-radius:5px;border:1px solid var(--border);background:var(--bg-1);font-size:0.72rem;cursor:pointer;">① ข้อมูล</button>
        <button onclick="paStep(1)" id="pa-btn-1" style="padding:5px 14px;border-radius:5px;border:1px solid #ef4444;color:#ef4444;background:#fef2f2;font-size:0.72rem;cursor:pointer;">② Point-Wise</button>
        <button onclick="paStep(2)" id="pa-btn-2" style="padding:5px 14px;border-radius:5px;border:1px solid #10b981;color:#10b981;background:#f0fdf4;font-size:0.72rem;cursor:pointer;">③ Point-Adjusted</button>
        <button onclick="paAutoPlay()" id="pa-btn-play" style="margin-left:auto;padding:5px 14px;border-radius:5px;border:1px solid var(--accent);color:var(--accent);background:#eff6ff;font-size:0.72rem;cursor:pointer;">▶ Auto Play</button>
      </div>
      <div style="margin-top:14px;padding:10px 14px;background:var(--bg-1);border-radius:6px;border:1px solid var(--border);font-size:0.75rem;line-height:1.9;">
        <b>สูตรที่ใช้ (3 ขั้นตอน):</b><br>
        <span style="color:var(--text-3);">① Gap Fill:</span> ถ้า 2 segments ห่างกัน ≤ 30 pts → merge<br>
        <span style="color:var(--text-3);">② Point-Adjust:</span> ∀ segment s: ถ้า <code>y_pred[s].any()</code> → <code>y_pred_adj[s] = 1</code><br>
        <span style="color:var(--text-3);">③ Metrics:</span>
        <span style="color:var(--accent);">Precision = TP/(TP+FP) | Recall = TP/(TP+FN) | F1 = 2PR/(P+R)</span>
      </div>
    </div>
  </div>`;

  container.innerHTML = html;
  paStep(0);
}

// ── PA Demo ────────────────────────────────────────────────────────
let PA_AUTO_TIMER = null;
let PA_CURRENT    = 0;

function paStep(mode) {
  PA_CURRENT = mode;
  [0,1,2].forEach(i => {
    const btn = document.getElementById(`pa-btn-${i}`);
    if (!btn) return;
    btn.style.fontWeight = (i === mode) ? '700' : '400';
    btn.style.boxShadow  = (i === mode) ? '0 0 0 2px currentColor' : '';
  });
  const el = document.getElementById('pa-demo');
  if (!el) return;

  const n      = 40;
  const y_true = Array.from({length:n}, (_,i) => (i>=15&&i<=25) ? 1 : 0);
  const y_pred = Array.from({length:n}, (_,i) => (i>=18&&i<=20) ? 1 : 0);
  const y_adj  = y_pred.map((v,i) => (i>=15&&i<=25&&y_pred.slice(15,26).some(Boolean)) ? 1 : v);

  const C = { N:'#cbd5e1',A:'#ef4444',F:'#f59e0b',TP:'#10b981',FP:'#ef4444',FN:'#f97316',TN:'#e2e8f0',ADJ:'#2563eb' };
  const box = (c,lbl) => `<span title="${lbl}" style="display:inline-block;width:16px;height:20px;border-radius:3px;background:${c};margin:1px;vertical-align:middle;cursor:default;"></span>`;
  let out = '';

  if (mode === 0) {
    out += `<div style="margin-bottom:6px;font-size:0.72rem;color:var(--text-3);">${box(C.N,'Normal')} Normal &nbsp;${box(C.A,'Anomaly')} Anomaly (y_true) &nbsp;${box(C.F,'Flagged')} Flagged (y_pred)</div>`;
    out += `<div style="margin-bottom:3px;"><span style="display:inline-block;width:52px;font-size:0.71rem;color:var(--text-3);">y_true:</span>`;
    y_true.forEach(v => out += box(v?C.A:C.N, v?'Anomaly':'Normal'));
    out += `</div><div><span style="display:inline-block;width:52px;font-size:0.71rem;color:var(--text-3);">y_pred:</span>`;
    y_pred.forEach(v => out += box(v?C.F:C.N, v?'Flagged':'Normal'));
    out += `</div><div style="margin-top:8px;font-size:0.72rem;color:var(--text-3);">Segment: idx 15–25 (11 pts) | Flagged: idx 18–20 (3 pts)</div>`;

  } else if (mode === 1) {
    const tp=y_true.filter((_,i)=>y_pred[i]&&y_true[i]).length;
    const fp=y_true.filter((_,i)=>y_pred[i]&&!y_true[i]).length;
    const fn=y_true.filter((_,i)=>!y_pred[i]&&y_true[i]).length;
    const tn=y_true.filter((_,i)=>!y_pred[i]&&!y_true[i]).length;
    const p=tp/(tp+fp)||0, r=tp/(tp+fn)||0, f1=2*p*r/(p+r)||0;
    out += `<div style="margin-bottom:6px;font-size:0.72rem;color:var(--text-3);">${box(C.TP,'TP')} TP &nbsp;${box(C.FP,'FP')} FP &nbsp;${box(C.FN,'FN')} FN &nbsp;${box(C.TN,'TN')} TN</div>`;
    out += `<div><span style="display:inline-block;width:52px;font-size:0.71rem;color:var(--text-3);">Point-Wise:</span>`;
    y_true.forEach((vt,i)=>{ const vp=y_pred[i]; out+=box(vp&&vt?C.TP:vp?C.FP:vt?C.FN:C.TN, vp&&vt?'TP':vp?'FP':vt?'FN':'TN'); });
    out += `</div><div style="margin-top:10px;padding:10px 14px;background:#fef2f2;border-radius:6px;border-left:3px solid #ef4444;font-size:0.78rem;line-height:1.8;">
      TP=${tp} FP=${fp} FN=${fn} TN=${tn}<br>P=${(p*100).toFixed(0)}% R=${(r*100).toFixed(0)}% <b>F1=${(f1*100).toFixed(1)}%</b><br>
      <span style="color:#dc2626;font-size:0.72rem;">⚠️ FN=8 ทำให้ Recall ต่ำมาก</span></div>`;

  } else if (mode === 2) {
    const tp=y_true.filter((_,i)=>y_adj[i]&&y_true[i]).length;
    const fp=y_true.filter((_,i)=>y_adj[i]&&!y_true[i]).length;
    const fn=y_true.filter((_,i)=>!y_adj[i]&&y_true[i]).length;
    const tn=y_true.filter((_,i)=>!y_adj[i]&&!y_true[i]).length;
    const p=tp/(tp+fp)||0, r=tp/(tp+fn)||0, f1=2*p*r/(p+r)||0;
    out += `<div style="margin-bottom:6px;font-size:0.72rem;color:var(--text-3);">${box(C.TP,'TP orig')} TP &nbsp;${box(C.ADJ,'TP adj')} TP-adjusted &nbsp;${box(C.TN,'TN')} TN</div>`;
    out += `<div><span style="display:inline-block;width:52px;font-size:0.71rem;color:var(--text-3);">PA-Adjust:</span>`;
    y_true.forEach((vt,i)=>{ const vo=y_pred[i],va=y_adj[i]; out+=box(va&&vt?(vo?C.TP:C.ADJ):va?C.FP:vt?C.FN:C.TN, va&&vt?(vo?'TP':'TP-adj'):va?'FP':vt?'FN':'TN'); });
    out += `</div><div style="margin-top:10px;padding:10px 14px;background:#f0fdf4;border-radius:6px;border-left:3px solid #10b981;font-size:0.78rem;line-height:1.8;">
      TP=${tp} FP=${fp} FN=${fn} TN=${tn}<br>P=${(p*100).toFixed(0)}% R=${(r*100).toFixed(0)}% <b style="color:#16a34a;">F1=${(f1*100).toFixed(1)}%</b><br>
      <span style="color:#16a34a;font-size:0.72rem;">✅ เจอ segment → TP ทั้ง 11 pts, Recall 100%</span></div>`;
  }
  el.innerHTML = out;
}

let PA_PLAY_STEP = 0;
function paAutoPlay() {
  if (PA_AUTO_TIMER) {
    clearInterval(PA_AUTO_TIMER);
    PA_AUTO_TIMER = null;
    document.getElementById('pa-btn-play').textContent = '▶ Auto Play';
    return;
  }
  document.getElementById('pa-btn-play').textContent = '⏹ Stop';
  PA_PLAY_STEP = 0; paStep(0);
  PA_AUTO_TIMER = setInterval(() => {
    PA_PLAY_STEP = (PA_PLAY_STEP + 1) % 3;
    paStep(PA_PLAY_STEP);
    if (PA_PLAY_STEP === 2) {
      clearInterval(PA_AUTO_TIMER); PA_AUTO_TIMER = null;
      document.getElementById('pa-btn-play').textContent = '▶ Auto Play';
    }
  }, 2000);
}