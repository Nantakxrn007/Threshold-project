// static/dashboard/js/grid_search.js

// ── Defaults ───────────────────────────────────────────────────────
const GS_CFG_DEFAULTS = {
  th1_pct:99.0, th1_mode:'sliding', th1_win:100, th1_recalc:10,
  th2_alpha:3.5, th2_win:80,  th2_recalc:50,
  th3_zmin:2.0,  th3_zmax:10.0, th3_win:80, th3_recalc:1,
  th4_alpha:3.5, th4_win:150, th4_cons:5,  th4_eth:0.95, th4_recalc:1,
};

// ── Chip helpers ───────────────────────────────────────────────────
function gsTogChip(el) {
  el.classList.toggle('gs-on');
  gsUpdateCombo();
}

function gsAddChip(groupId, inputId, step) {
  const input = document.getElementById(inputId);
  const raw   = input.value.trim();
  if (!raw) return;
  const val = step && step < 1 ? parseFloat(raw).toFixed(step === 0.01 ? 2 : 1) : raw;
  const group = document.getElementById(groupId);
  // If already exists, just toggle on
  const existing = [...group.querySelectorAll('.gs-chip')].find(c => c.dataset.val === val);
  if (existing) { existing.classList.add('gs-on'); input.value = ''; gsUpdateCombo(); return; }
  // Create new chip
  const chip = document.createElement('span');
  chip.className = 'gs-chip gs-on';
  chip.dataset.val = val;
  chip.textContent = val;
  chip.onclick = () => gsTogChip(chip);
  group.appendChild(chip);
  input.value = '';
  gsUpdateCombo();
}

function gsChipValues(groupId) {
  const chips = document.querySelectorAll(`#${groupId} .gs-chip.gs-on`);
  return [...chips].map(c => c.dataset.val).filter(Boolean);
}

function gsSliderVal(id) {
  return parseFloat(document.getElementById(id)?.value || 0);
}
function gsSelectVal(id) {
  return document.getElementById(id)?.value || '';
}

// ── Build fixed cfg from chip groups (first selected value) ────────
function gsFirstChipVal(groupId, fallback) {
  const v = gsChipValues(groupId)?.[0];
  return v !== undefined ? parseFloat(v) : fallback;
}

function gsBuildBaseCfg() {
  return {
    th1_pct:    gsSliderVal('gs-th1-pct') || 99.0,
    th1_mode:   gsSelectVal('gs-th1-mode'),
    th1_win:    gsFirstChipVal('gs-th1-win-group', 100),
    th1_recalc: gsFirstChipVal('gs-th1-recalc-group', 10),
    th2_alpha:  3.5,
    th2_win:    gsFirstChipVal('gs-th2-win-group', 80),
    th2_recalc: gsFirstChipVal('gs-th2-recalc-group', 10),
    th3_zmin:   2.0,
    th3_zmax:   gsFirstChipVal('gs-th3-zmax-group', 10.0),
    th3_win:    gsFirstChipVal('gs-th3-win-group', 80),
    th3_recalc: gsFirstChipVal('gs-th3-recalc-group', 10),
    th4_alpha:  3.5,
    th4_win:    gsFirstChipVal('gs-th4-win-group', 150),
    th4_cons:   gsFirstChipVal('gs-th4-cons-group', 5),
    th4_eth:    gsFirstChipVal('gs-th4-eth-group', 0.95),
    th4_recalc: gsFirstChipVal('gs-th4-recalc-group', 10),
  };
}

// ── Build th_configs: sweep chips × fixed base ─────────────────────
function gsBuildThConfigs() {
  const configs = [];

  // Per-threshold recalc arrays — each sweeps independently
  const r1s = gsChipValues('gs-th1-recalc-group') || ['10'];
  const r2s = gsChipValues('gs-th2-recalc-group') || ['10'];
  const r3s = gsChipValues('gs-th3-recalc-group') || ['10'];
  const r4s = gsChipValues('gs-th4-recalc-group') || ['10'];

  if (document.getElementById('gs-en1')?.checked) {
    const pcts = gsChipValues('gs-th1-pct-group') || ['99'];
    const wins = gsChipValues('gs-th1-win-group') || ['100'];
    const mode = gsSelectVal('gs-th1-mode');
    for (const pct of pcts) {
      for (const win of wins) {
        for (const r1 of r1s) {
          configs.push({
            name: `TH1 P${parseFloat(pct)} w${win} r${parseFloat(r1)}`,
            th_type: 'P99 Static',
            cfg: { ...gsBuildBaseCfg(),
              th1_pct: parseFloat(pct), th1_mode: mode,
              th1_win: parseFloat(win), th1_recalc: parseFloat(r1) },
          });
        }
      }
    }
  }

  if (document.getElementById('gs-en2')?.checked) {
    const alphas = gsChipValues('gs-th2-alpha-group') || ['3.5'];
    const wins   = gsChipValues('gs-th2-win-group')   || ['80'];
    for (const a of alphas) {
      for (const win of wins) {
        for (const r2 of r2s) {
          configs.push({
            name: `TH2 α${parseFloat(a)} w${win} r${parseFloat(r2)}`,
            th_type: 'Sliding Mu+αStd',
            cfg: { ...gsBuildBaseCfg(),
              th2_alpha: parseFloat(a),
              th2_win: parseFloat(win), th2_recalc: parseFloat(r2) },
          });
        }
      }
    }
  }

  if (document.getElementById('gs-en3')?.checked) {
    const zmins = gsChipValues('gs-th3-zmin-group') || ['2.0'];
    const zmaxs = gsChipValues('gs-th3-zmax-group') || ['10.0'];
    const wins  = gsChipValues('gs-th3-win-group')  || ['80'];
    for (const zmin of zmins) {
      for (const zmax of zmaxs) {
        for (const win of wins) {
          for (const r3 of r3s) {
            configs.push({
              name: `TH3 z${parseFloat(zmin)}-${parseFloat(zmax)} w${win} r${parseFloat(r3)}`,
              th_type: 'Adaptive-z',
              cfg: { ...gsBuildBaseCfg(),
                th3_zmin: parseFloat(zmin), th3_zmax: parseFloat(zmax),
                th3_win: parseFloat(win),   th3_recalc: parseFloat(r3) },
            });
          }
        }
      }
    }
  }

  if (document.getElementById('gs-en4')?.checked) {
    const alphas = gsChipValues('gs-th4-alpha-group') || ['3.5'];
    const wins   = gsChipValues('gs-th4-win-group')   || ['150'];
    const conss  = gsChipValues('gs-th4-cons-group')  || ['5'];
    const eths   = gsChipValues('gs-th4-eth-group')   || ['0.95'];
    for (const a of alphas) {
      for (const win of wins) {
        for (const cons of conss) {
          for (const eth of eths) {
            for (const r4 of r4s) {
              configs.push({
                name: `TH4 α${parseFloat(a)} w${win} c${parseFloat(cons)} r${parseFloat(r4)}`,
                th_type: 'Entropy-lock',
                cfg: { ...gsBuildBaseCfg(),
                  th4_alpha: parseFloat(a), th4_win: parseFloat(win),
                  th4_cons: parseFloat(cons), th4_eth: parseFloat(eth),
                  th4_recalc: parseFloat(r4) },
              });
            }
          }
        }
      }
    }
  }

  return configs.length ? configs : [{name:'P99 Static', th_type:'P99 Static', cfg:gsBuildBaseCfg()}];
}

// ── Mode change: show/hide n_trials ───────────────────────────────
function gsOnModeChange() {
  const mode = document.getElementById('gs-search-mode')?.value;
  const wrap = document.getElementById('gs-n-trials-wrap');
  if (wrap) wrap.style.display = (mode === 'random' || mode === 'optuna') ? 'block' : 'none';
  gsUpdateCombo();
}

// ── Combination counter ────────────────────────────────────────────
function gsUpdateCombo() {
  const get = id => Math.max((gsChipValues(id) || []).length, 1);
  const model = get('gs-arch-group') * get('gs-hidden-group') * get('gs-seq-group') *
                get('gs-epochs-group') * get('gs-lr-group') * get('gs-ewma-group');
  const thConfigs = gsBuildThConfigs();
  const total = model * thConfigs.length;

  const el = document.getElementById('gs-combo-count');
  if (el) el.textContent = total.toLocaleString();
  const warn = document.getElementById('gs-warn');
  if (warn) {
    warn.style.display = total > 200 ? 'block' : 'none';
    if (warn.style.display === 'block')
      warn.textContent = `${total.toLocaleString()} combos — เปลี่ยน mode เป็น Random/Optuna`;
  }
}

// ── Sync slider ↔ number input ─────────────────────────────────────
function gsSv(id, val, fmt) {
  const d = document.getElementById('gs-v-'+id);
  if (d) d.textContent = fmt ? parseFloat(val).toFixed(fmt) : val;
}
function gsSyncNum(id, val) {
  const n = document.getElementById('gs-n-'+id);
  if (n) n.value = val;
  gsUpdateCombo();
}
function gsSyncRng(id, val) {
  const sliders = document.querySelectorAll(`input[type=range][data-gssync="${id}"]`);
  sliders.forEach(s => { s.value = val; });
  gsUpdateCombo();
}

// ── Start ──────────────────────────────────────────────────────────
async function gsStart() {
  const btn = document.getElementById('gs-run-btn');
  btn.disabled = true; btn.textContent = '⏳ Starting…';

  const getChipVals = (id, parser) => (gsChipValues(id) || []).map(parser);
  const mode    = gsSelectVal('gs-search-mode');
  const nTrials = Math.max(parseInt(document.getElementById('gs-n-trials')?.value || 30), 1);

  const body = {
    archs:       getChipVals('gs-arch-group',   v => v),
    hiddens:     getChipVals('gs-hidden-group',  v => parseInt(v)),
    seq_lens:    getChipVals('gs-seq-group',     v => parseInt(v)),
    epochs_list: getChipVals('gs-epochs-group',  v => parseInt(v)),
    lrs:         getChipVals('gs-lr-group',      v => parseFloat(v)),
    ewmas:       getChipVals('gs-ewma-group',    v => parseFloat(v)),
    batch_size:  64, layers: 1,
    agg:         gsSelectVal('gs-agg'),
    scoring:     gsSelectVal('gs-scoring'),
    search_mode: mode,
    n_trials:    nTrials,
    th_configs:  gsBuildThConfigs(),
  };

  if (!body.archs.length)       { alert('เลือก Architecture อย่างน้อย 1'); btn.disabled=false; btn.textContent='Run grid search'; return; }
  if (!body.epochs_list.length) body.epochs_list = [30];
  if (!body.lrs.length)         body.lrs = [0.005];
  if (!body.ewmas.length)       body.ewmas = [0.3];

  try {
    const res  = await fetch('/api/grid-search/start/', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    GRID_JOB_ID = data.job_id;
    const modeLabel = {grid:'Grid',random:'Random',optuna:'Optuna Bayesian'}[data.mode] || data.mode;
    const gsStopBtn = document.getElementById('gs-stop-btn');
    if (gsStopBtn) gsStopBtn.style.display = 'inline-flex';
    gsInitProgressTable(data.total, modeLabel);
    if (GRID_POLL) clearInterval(GRID_POLL);
    GRID_POLL = setInterval(gsPoll, 1200);
  } catch(err) {
    alert('Error: ' + err.message);
    btn.disabled = false; btn.textContent = 'Run grid search';
  }
}

// ── Init progress table ────────────────────────────────────────────
function gsInitProgressTable(total, modeLabel) {
  document.getElementById('gs-prog-panel').style.display = 'block';
  document.getElementById('gs-results-panel').style.display = 'none';
  const modeStr = modeLabel ? ` — ${modeLabel}` : '';
  document.getElementById('gs-prog-count').textContent = `0 / ${total}${modeStr}`;
  document.getElementById('gs-prog-tbody').innerHTML =
    `<tr><td colspan="10" style="padding:20px;text-align:center;color:var(--text-3);font-family:'JetBrains Mono',monospace;font-size:0.78rem;">⏳ Initialising…</td></tr>`;
}

// ── Poll ───────────────────────────────────────────────────────────
async function gsPoll() {
  if (!GRID_JOB_ID) return;
  try {
    const res  = await fetch(`/api/grid-search/status/${GRID_JOB_ID}/`);
    const data = await res.json();
    gsRenderProgress(data);
    if (data.status === 'done') {
      clearInterval(GRID_POLL);
      const sb = document.getElementById('gs-stop-btn'); if (sb) sb.style.display='none';
      gsRenderResults(data);
      gsLoadBestCharts();
      document.getElementById('gs-run-btn').disabled = false;
      document.getElementById('gs-run-btn').textContent = 'Run grid search';
    } else if (data.status === 'stopped') {
      clearInterval(GRID_POLL);
      const sbS = document.getElementById('gs-stop-btn'); if (sbS) sbS.style.display='none';
      gsRenderResults(data);
      document.getElementById('gs-run-btn').disabled = false;
      document.getElementById('gs-run-btn').textContent = 'Run grid search';
    } else if (data.status === 'error') {
      clearInterval(GRID_POLL);
      const sb2 = document.getElementById('gs-stop-btn'); if (sb2) sb2.style.display='none';
      document.getElementById('gs-prog-tbody').innerHTML =
        `<tr><td colspan="10" style="padding:20px;color:var(--red);font-family:'JetBrains Mono',monospace;">❌ ${data.error}</td></tr>`;
      document.getElementById('gs-run-btn').disabled = false;
      document.getElementById('gs-run-btn').textContent = 'Run grid search';
    }
  } catch(e) { console.error('poll error', e); }
}

// ── Render progress table ──────────────────────────────────────────
function gsRenderProgress(data) {
  document.getElementById('gs-prog-count').textContent = `${data.done} / ${data.total}`;
  const tbody = document.getElementById('gs-prog-tbody');
  if (!data.combos?.length) return;

  tbody.innerHTML = data.combos.map(c => {
    const pct  = c.epoch_progress || 0;
    const dot  = c.status==='done'?'status-done':c.status==='training'?'status-training':'status-idle';
    const f1   = c.f1 !== null ? (c.f1*100).toFixed(1)+'%' : '—';
    const rec  = c.recall !== null ? (c.recall*100).toFixed(0)+'%' : '—';
    const isBest = data.best_idx !== null && c.id === data.best_idx;
    const bg   = isBest ? 'background:#eff6ff;' : '';
    return `<tr style="${bg}">
      <td style="padding:4px 7px;color:var(--text-3);font-family:'JetBrains Mono',monospace;">${c.id+1}</td>
      <td style="padding:4px 7px;">${isBest?'🏆 ':''}${c.arch}</td>
      <td style="padding:4px 7px;">${c.hidden}</td>
      <td style="padding:4px 7px;">${c.seq_len}</td>
      <td style="padding:4px 7px;font-size:0.7rem;">${c.lr}</td>
      <td style="padding:4px 7px;font-size:0.7rem;max-width:110px;overflow:hidden;text-overflow:ellipsis;">${c.th_name}</td>
      <td style="padding:4px 7px;min-width:80px;">
        <div style="height:4px;background:var(--border);border-radius:2px;">
          <div style="width:${pct}%;height:100%;background:var(--accent);border-radius:2px;transition:width .3s;"></div>
        </div>
      </td>
      <td style="padding:4px 7px;font-weight:${c.f1!==null?'600':'400'};color:${c.f1>=0.8?'var(--green)':c.f1>=0.6?'var(--amber)':'var(--text)'};">${f1}</td>
      <td style="padding:4px 7px;">${rec}</td>
      <td style="padding:4px 7px;"><span class="status-dot ${dot}"></span>${c.status}</td>
    </tr>`;
  }).join('');
}

// ── Render results ─────────────────────────────────────────────────
function gsRenderResults(data) {
  const panel = document.getElementById('gs-results-panel');
  panel.style.display = 'block';
  const done = data.combos.filter(c => c.status === 'done').sort((a,b) => (b.score||0)-(a.score||0));
  if (!done.length) return;

  // ── Ranking bar chart ────────────────────────────────────────────
  const top = done.slice(0, Math.min(20, done.length));
  const labels = top.map(c => `${c.arch.replace('-AE','')}\nh=${c.hidden} s=${c.seq_len}\n${c.th_name}`);
  const f1s  = top.map(c => parseFloat(((c.f1||0)*100).toFixed(1)));
  const recs = top.map(c => parseFloat(((c.recall||0)*100).toFixed(1)));
  const bestId = data.best_idx;
  const colors = top.map(c => c.id===bestId ? '#2563eb' : '#94a3b8');

  Plotly.newPlot('gs-rank-chart', [
    {type:'bar', orientation:'h', y:labels, x:f1s,
     name:'F1', marker:{color:colors}, text:f1s.map(v=>v+'%'), textposition:'outside',
     hovertemplate:'F1=%{x}%<extra></extra>'},
    {type:'bar', orientation:'h', y:labels, x:recs,
     name:'Recall', marker:{color:'rgba(16,185,129,0.4)'}, opacity:0.6,
     hovertemplate:'Recall=%{x}%<extra></extra>'},
  ], {
    barmode:'overlay',
    height: Math.max(260, top.length * 32),
    margin:{l:140,r:60,t:30,b:30},
    paper_bgcolor:'#ffffff', plot_bgcolor:'#f8fafc',
    font:{family:'Inter,sans-serif',size:10,color:'#94a3b8'},
    xaxis:{range:[0,110],gridcolor:'#e2e8f0',zeroline:false,title:'Score (%)'},
    yaxis:{gridcolor:'#e2e8f0',autorange:'reversed'},
    legend:{orientation:'h',y:1.08},
  }, {displayModeBar:false, responsive:true});

  // ── Full results table ───────────────────────────────────────────
  const medals = ['🥇','🥈','🥉'];
  document.getElementById('gs-result-tbody').innerHTML = done.map((c,i) => {
    const isBest = c.id === bestId;
    // Extract recalc from th_name (format: "... rN")
    const recalcMatch = c.th_name.match(/r(\d+(?:\.\d+)?)$/);
    const recalcN = recalcMatch ? recalcMatch[1] : '—';
    return `<tr style="${isBest?'background:#eff6ff;':''}">
      <td style="padding:5px 8px;color:var(--text-3);">${medals[i]||i+1}</td>
      <td style="padding:5px 8px;font-weight:${isBest?600:400};">${isBest?'🏆 ':''}${c.arch}</td>
      <td style="padding:5px 8px;">${c.hidden}</td>
      <td style="padding:5px 8px;">${c.seq_len}</td>
      <td style="padding:5px 8px;">${c.epochs}</td>
      <td style="padding:5px 8px;">${c.lr}</td>
      <td style="padding:5px 8px;">${c.ewma}</td>
      <td style="padding:5px 8px;font-size:0.7rem;max-width:130px;overflow:hidden;text-overflow:ellipsis;" title="${c.th_name}">${c.th_name}</td>
      <td style="padding:5px 8px;text-align:center;color:var(--text-3);">${recalcN}</td>
      <td style="padding:5px 8px;">${((c.precision||0)*100).toFixed(1)}%</td>
      <td style="padding:5px 8px;">${((c.recall||0)*100).toFixed(1)}%</td>
      <td style="padding:5px 8px;font-weight:600;color:${(c.f1||0)>=0.8?'#16a34a':(c.f1||0)>=0.6?'#d97706':'#dc2626'};">${((c.f1||0)*100).toFixed(1)}%</td>
      <td style="padding:5px 8px;">${((c.accuracy||0)*100).toFixed(1)}%</td>
      <td style="padding:5px 8px;">${c.n_detected_segs||0}/${c.n_segments||0}</td>
      <td style="padding:5px 8px;font-weight:600;">${((c.score||0)*100).toFixed(1)}%</td>
    </tr>`;
  }).join('');

  panel.scrollIntoView({behavior:'smooth'});
}

// ── Load best charts ───────────────────────────────────────────────
async function gsLoadBestCharts() {
  if (!GRID_JOB_ID) return;
  const run = document.getElementById('gs-best-run')?.value || '';
  const sl  = getShowLabels();
  const res = await fetch(`/api/grid-search/best-charts/${GRID_JOB_ID}/?run=${run}&show_labels=${sl}`);
  if (!res.ok) return;
  const data = await res.json();

  const panel = document.getElementById('gs-best-chart-panel');
  if (panel) panel.style.display = 'block';

  Plotly.newPlot('gs-best-raw',  data.raw.data,       data.raw.layout,       {displayModeBar:false, responsive:true});
  Plotly.newPlot('gs-best-th',   data.threshold.data, data.threshold.layout, {displayModeBar:false, responsive:true});

  if (data.best) {
    const b = data.best;
    document.getElementById('gs-best-summary').innerHTML = `
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
        <span style="font-size:0.72rem;font-weight:700;color:var(--accent);">🏆 Best combo:</span>
        ${[b.arch,`h=${b.hidden}`,`seq=${b.seq_len}`,`ep=${b.epochs}`,`lr=${b.lr}`,`ewma=${b.ewma}`,b.th_name]
          .map(t=>`<span style="font-size:0.7rem;padding:2px 7px;border-radius:12px;background:var(--bg-1);border:1px solid var(--border);">${t}</span>`).join('')}
        <span style="margin-left:auto;font-size:0.85rem;font-weight:700;color:var(--green);">F1 ${((b.f1||0)*100).toFixed(1)}%</span>
        <span style="font-size:0.8rem;color:var(--text-2);">Recall ${((b.recall||0)*100).toFixed(0)}%</span>
        <span style="font-size:0.8rem;color:var(--purple);">Seg ${b.n_detected_segs||0}/${b.n_segments||0}</span>
      </div>`;
  }
}

// ── Save CSV ───────────────────────────────────────────────────────
function gsCSV() {
  if (!GRID_JOB_ID) return;
  fetch(`/api/grid-search/status/${GRID_JOB_ID}/`)
    .then(r => r.json())
    .then(data => {
      const done = data.combos.filter(c => c.status==='done').sort((a,b)=>(b.score||0)-(a.score||0));
      const hdr  = ['rank','arch','hidden','seq_len','epochs','lr','ewma','th_name','recalc_n','precision','recall','f1','accuracy','seg_det','seg_total','score'];
      const rows = [hdr.join(','), ...done.map((c,i) => [
        i+1, `"${c.arch}"`, c.hidden, c.seq_len, c.epochs, c.lr, c.ewma,
        `"${c.th_name}"`,
        (c.th_name.match(/r(\d+(?:\.\d+)?)$/)||['','—'])[1],
        ((c.precision||0)*100).toFixed(1), ((c.recall||0)*100).toFixed(1),
        ((c.f1||0)*100).toFixed(1), ((c.accuracy||0)*100).toFixed(1),
        c.n_detected_segs||0, c.n_segments||0, ((c.score||0)*100).toFixed(1),
      ].join(','))];
      const blob = new Blob([rows.join('\n')], {type:'text/csv'});
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href=url; a.download='grid_search_results.csv'; a.click();
      URL.revokeObjectURL(url);
    });
}


// ── Grid Search: Stop ─────────────────────────────────────────────
async function gsStop() {
  if (!GRID_JOB_ID) return;
  if (GRID_POLL) { clearInterval(GRID_POLL); GRID_POLL = null; }
  try {
    await fetch('/api/grid-search/stop/', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({job_id: GRID_JOB_ID}),
    });
  } catch(e) {}
  // Still show partial results
  gsPoll();
  document.getElementById('gs-run-btn').disabled = false;
  document.getElementById('gs-run-btn').textContent = 'Run grid search';
  const sb = document.getElementById('gs-stop-btn');
  if (sb) sb.style.display = 'none';
}

// ── Grid Search: Download partial/full CSV ────────────────────────
function gsDownloadCSV() {
  if (!GRID_JOB_ID) return;
  window.location.href = `/api/grid-search/export/${GRID_JOB_ID}/`;
}