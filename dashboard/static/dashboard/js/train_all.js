// static/dashboard/js/train_all.js

async function startTrainAll() {
  const btn = document.getElementById('ta-btn-train');
  btn.disabled = true; btn.textContent = '⏳ Training…';
  document.getElementById('ta-msg').textContent = 'Starting…';

  const tbody = document.getElementById('ta-progress-tbody');
  tbody.innerHTML = MODEL_LIST.map(m => {
    const k = m.replace(/[^a-z0-9]/gi, '_');
    return `<tr id="ta-row-${k}">
      <td style="padding:6px 8px;font-family:'JetBrains Mono',monospace;">${m}</td>
      <td style="padding:6px 8px;text-align:center;"><span class="status-dot status-idle"></span>pending</td>
      <td style="padding:6px 8px;"><div class="progress-wrap" style="margin:0"><div class="progress-bar" id="ta-bar-${k}" style="width:0%"></div></div></td>
      <td style="padding:6px 8px;font-family:'JetBrains Mono',monospace;color:var(--text-3);" id="ta-loss-${k}">—</td>
    </tr>`;
  }).join('');

  const body = {
    seq_len:    document.getElementById('ta-seq-len').value,
    epochs:     document.getElementById('ta-epochs').value,
    hidden:     document.getElementById('ta-hidden').value,
    layers:     document.getElementById('ta-layers').value,
    lr:         document.getElementById('ta-lr').value,
    batch_size: document.getElementById('ta-batch').value,
    ewma:       document.getElementById('ta-ewma').value,
  };
  const res  = await fetch('/api/train-all/', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await res.json();
  BATCH_JOB_ID = data.job_id;
  RESULTS_CACHE = null; RESULTS_CACHE_KEY = null;
  pollTrainAll();
}

function pollTrainAll() {
  BATCH_POLL = setInterval(async () => {
    const res  = await fetch(`/api/status-all/${BATCH_JOB_ID}/`);
    const data = await res.json();
    for (const [model, prog] of Object.entries(data.progress)) {
      const k   = model.replace(/[^a-z0-9]/gi, '_');
      const row = document.getElementById(`ta-row-${k}`);
      if (!row) continue;
      const pct = data.epochs > 0 ? Math.round(100 * prog.epoch / data.epochs) : 0;
      const dot = prog.status==='done' ? 'status-done' : prog.status==='training' ? 'status-training' : 'status-idle';
      row.cells[1].innerHTML = `<span class="status-dot ${dot}"></span>${prog.status}`;
      document.getElementById(`ta-bar-${k}`).style.width = pct + '%';
      if (prog.loss !== null) document.getElementById(`ta-loss-${k}`).textContent = prog.loss.toFixed(5);
    }
    if (data.status === 'done') {
      clearInterval(BATCH_POLL);
      document.getElementById('ta-btn-train').disabled = false;
      document.getElementById('ta-btn-train').textContent = '🏋️ Train All Models';
      document.getElementById('ta-msg').innerHTML = '<span class="status-dot status-done"></span>All models trained! Go to 📊 Results to evaluate.';
      const sel = document.getElementById('ta-inspect-model');
      if (sel) sel.innerHTML = MODEL_LIST.map(m => `<option value="${m}">${m}</option>`).join('');
      document.getElementById('ta-pca-section').style.display = 'block';
      loadAllPcas();
      loadModelError();
    } else if (data.status === 'error') {
      clearInterval(BATCH_POLL);
      document.getElementById('ta-msg').innerHTML = `<span class="status-dot status-error"></span>Error: ${data.error}`;
      document.getElementById('ta-btn-train').disabled = false;
      document.getElementById('ta-btn-train').textContent = '🏋️ Train All Models';
    }
  }, 800);
}

async function loadAllPcas() {
  if (!BATCH_JOB_ID) return;
  const grid = document.getElementById('ta-pca-grid');
  const cb   = document.getElementById('ta-color')?.value || 'run';
  if (!grid) return;
  grid.innerHTML = MODEL_LIST.map(m =>
    `<div id="ta-pca-${m.replace(/[^a-z0-9]/gi,'_')}" style="min-height:300px;"></div>`).join('');
  for (const model of MODEL_LIST) {
    const key = model.replace(/[^a-z0-9]/gi, '_');
    const res = await fetch(`/api/charts/model-umap/?job_id=${BATCH_JOB_ID}&model=${encodeURIComponent(model)}&color_by=${cb}`);
    if (!res.ok) continue;
    const fig = await res.json();
    Plotly.newPlot(`ta-pca-${key}`, fig.data, fig.layout, {displayModeBar:false, responsive:true});
  }
}

async function loadModelError() {
  if (!BATCH_JOB_ID) return;
  const model = document.getElementById('ta-inspect-model')?.value;
  const run   = document.getElementById('ta-inspect-run')?.value;
  if (!model || !run) return;
  const agg = getAgg(), sl = getShowLabels();

  const [rawRes, errRes] = await Promise.all([
    fetch(`/api/charts/raw/?run=${run}&show_labels=${sl}`),
    fetch(`/api/charts/model-error/?job_id=${BATCH_JOB_ID}&model=${encodeURIComponent(model)}&run=${run}&agg=${agg}&show_labels=${sl}`),
  ]);
  const [rawFig, errFig] = await Promise.all([rawRes.json(), errRes.json()]);
  Plotly.newPlot('ta-inspect-raw', rawFig.data, rawFig.layout, {displayModeBar:false, responsive:true});
  Plotly.newPlot('ta-inspect-err', errFig.data, errFig.layout, {displayModeBar:false, responsive:true});
}