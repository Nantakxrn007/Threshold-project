// static/dashboard/js/training.js

async function startTraining() {
  const btn = document.getElementById('btn-train');
  btn.disabled = true;
  btn.textContent = '⏳ Training…';
  setStatus('training', 'Starting…');
  document.getElementById('progress-section').style.display = 'block';
  document.getElementById('progress-bar').style.width = '0%';

  const body = {
    model_type: document.getElementById('model-type').value,
    seq_len:    document.getElementById('seq-len').value,
    epochs:     document.getElementById('epochs').value,
    hidden:     document.getElementById('hidden').value,
    layers:     document.getElementById('layers').value,
    lr:         document.getElementById('lr').value,
    batch_size: document.getElementById('batch').value,
    ewma:       document.getElementById('ewma').value,
  };
  const res  = await fetch('/api/train/', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const data = await res.json();
  JOB_ID = data.job_id;
  pollStatus();
}

function pollStatus() {
  POLL_TIMER = setInterval(async () => {
    const res  = await fetch(`/api/status/${JOB_ID}/`);
    const data = await res.json();

    if (data.loss && data.loss.length) {
      const pct = Math.round(100 * data.loss.length / data.epochs);
      document.getElementById('progress-bar').style.width = pct + '%';
      document.getElementById('progress-label').textContent =
        `Epoch ${data.loss.length}/${data.epochs}  loss=${data.loss.at(-1).toFixed(5)}`;
      renderLossChart(data.loss);
    }
    if (data.status === 'done') {
      clearInterval(POLL_TIMER);
      document.getElementById('progress-bar').style.width = '100%';
      setStatus('done', `✅ Done · loss=${data.loss.at(-1).toFixed(5)}`);
      document.getElementById('btn-train').disabled = false;
      document.getElementById('btn-train').textContent = '🔥 Train Model';
      loadUmap();
    } else if (data.status === 'error') {
      clearInterval(POLL_TIMER);
      setStatus('error', '❌ ' + data.error);
      document.getElementById('btn-train').disabled = false;
      document.getElementById('btn-train').textContent = '🔥 Train Model';
    }
  }, 800);
}

function setStatus(type, msg) {
  document.getElementById('train-msg').innerHTML =
    `<span class="status-dot status-${type}"></span>${msg}`;
}

function renderLossChart(history) {
  Plotly.react('loss-chart', [{
    y: history, mode: 'lines',
    line: {color: '#2563eb', width: 2},
    fill: 'tozeroy', fillcolor: 'rgba(37,99,235,0.07)',
    hovertemplate: 'Epoch %{x}<br>Loss=%{y:.5f}<extra></extra>',
  }], {
    height: 140, margin: {l:40,r:8,t:20,b:28},
    paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
    font: {family:'Inter,sans-serif', size:10, color:'#94a3b8'},
    xaxis: {gridcolor:'#e2e8f0', color:'#94a3b8', zeroline:false},
    yaxis: {gridcolor:'#e2e8f0', color:'#94a3b8', zeroline:false},
    title: {text:'Training Loss', font:{size:11,color:'#475569'}},
  }, {displayModeBar:false, responsive:true});
}