// static/dashboard/js/animation.js
// ── Animation Tab — full playback engine ──────────────────────────

(function () {

let _data      = null;   
let _frame     = 0;      
let _timer     = null;   
let _playing   = false;

const SENSOR_ORDER  = ['conductivity', 'pH', 'temperature', 'voltage'];
const SENSOR_COLORS = { conductivity:'#3b82f6', pH:'#10b981', temperature:'#f59e0b', voltage:'#ec4899' };
const SENSOR_UNITS  = { conductivity:'μS/cm', pH:'pH', temperature:'°C', voltage:'V' };
const TH_COLORS     = {
  'P99 Static':      '#ca8a04',
  'Sliding Mu+αStd': '#3b82f6',
  'Adaptive-z':      '#10b981',
  'Entropy-lock':    '#7c3aed',
};
const TH_FILLS = {
  'P99 Static':      'rgba(202,138,4,0.18)',
  'Sliding Mu+αStd': 'rgba(59,130,246,0.18)',
  'Adaptive-z':      'rgba(16,185,129,0.18)',
  'Entropy-lock':    'rgba(124,58,237,0.18)',
};
const CHART_IDS = ['anim-c1','anim-c2','anim-c3','anim-c4','anim-c5'];
const PLOT_CFG  = { displayModeBar: false, responsive: true };

// ── Layout factory ─────────────────────────────────────────────────
function _baseLayout(title, yTitle, height, xMax, showXAxis = true) {
  return {
    height,
    paper_bgcolor: '#ffffff',
    plot_bgcolor:  '#f8fafc',
    margin:        { l:45, r:10, t:20, b: showXAxis ? 25 : 5 }, // ลดขอบบน/ล่าง
    font:  { family: 'Inter, system-ui, sans-serif', color: '#64748b', size: 10 },
    title: { text: title, font: { size: 11, color: '#0f172a' }, x: 0.01, y: 0.95 },
    xaxis: { 
      gridcolor:'#e2e8f0', linecolor:'#e2e8f0', color:'#94a3b8', zeroline:false,
      range: xMax !== null ? [0, xMax] : undefined,
      showticklabels: showXAxis // ปิดเลขแกน X ถ้าเป็นกราฟบนๆ จะได้แนบติดกัน
    },
    yaxis: {
      gridcolor:'#e2e8f0', linecolor:'#e2e8f0', color:'#94a3b8', zeroline:false,
      title: { text: yTitle, font: { size: 9 }, standoff: 2 },
      autorange: true, // เปิดให้ยืดหยุ่น Auto Scale เต็มที่
    },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#ffffff', bordercolor: '#e2e8f0',
      font: { color: '#0f172a', family: 'Inter, system-ui, sans-serif' },
    },
    legend: { orientation:'h', yanchor:'bottom', y:1.02, xanchor:'right', x:1,
              bgcolor:'rgba(255,255,255,0.9)', bordercolor:'#e2e8f0', borderwidth:1,
              font:{ size:9 } },
    shapes: [],
  };
}

// ── Init empty charts ──────────────────────────────
function _initEmptyCharts() {
  CHART_IDS.forEach((id, i) => {
    const el = document.getElementById(id);
    if (!el) return;
    const isError = i === 4;
    const title = isError ? '📈 Reconstruction Error' : `📡 ${SENSOR_ORDER[i]}`;
    const h = isError ? 280 : 70; // Error สูงๆ(280), เซ็นเซอร์แบนๆ(70)
    Plotly.newPlot(id, [], _baseLayout(title, '', h, null, isError), PLOT_CFG);
  });
}

async function loadAnimationData() {
  if (!JOB_ID) { _setStatus('⚠ Train a model first.', 'amber'); return; }
  const run    = document.getElementById('anim-run').value;
  const agg    = document.getElementById('anim-agg').value;
  const params = _buildThParams();

  _setStatus('⏳ Loading data…', 'blue');
  _stopPlayback();

  try {
    const url = `/api/animation/data/?job_id=${JOB_ID}&run=${encodeURIComponent(run)}&agg=${agg}&${params}`;
    const res  = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    _data  = await res.json();
    _frame = 0;
    _updateProgressUI();
    _renderFrame(0);
    _setStatus(`✅ Loaded — ${_data.n} frames · ${run}`, 'green');
    document.getElementById('anim-play-btn').disabled  = false;
    document.getElementById('anim-reset-btn').disabled = false;
  } catch(e) {
    _setStatus(`❌ ${e.message}`, 'red');
  }
}

function _buildThParams() {
  const ids = [
    'th1_pct','th1_mode','th1_win','th1_recalc',
    'th2_alpha','th2_win','th2_recalc',
    'th3_zmin','th3_zmax','th3_win','th3_recalc',
    'th4_alpha','th4_win','th4_cons','th4_eth','th4_recalc',
  ];
  return ids.map(k => {
    const dashId = k.replace(/_/g, '-');
    const el = document.getElementById(dashId);
    return el ? `${k}=${encodeURIComponent(el.value)}` : '';
  }).filter(Boolean).join('&');
}

function _renderFrame(endIdx) {
  if (!_data) return;
  const currentN = endIdx + 1; 
  const idx = Array.from({length: currentN}, (_, i) => i);
  const xMax = _data.n > 1 ? _data.n - 1 : 1; 

  const showTH    = _getChecked('anim-th');
  const showLbl   = document.getElementById('anim-show-labels')?.checked;
  const showBands = document.getElementById('anim-show-bands')?.checked;

  // ── Charts 1-4: Raw sensors ────────────────────────────────────
  SENSOR_ORDER.forEach((sensor, ci) => {
    const chartId  = CHART_IDS[ci];
    const yArr     = _data.raw[sensor]?.slice(0, currentN) ?? [];
    
    const traces   = [{
      x: idx, y: yArr,
      mode: 'lines', name: sensor,
      line: { color: SENSOR_COLORS[sensor], width: 1.8 },
    }];

    if (showLbl && _data.has_labels) {
      const lbl    = _data.labels.per_sensor?.[sensor] ?? [];
      const aMask  = lbl.slice(0, currentN).map((v,i) => v ? i : -1).filter(i => i >= 0);
      if (aMask.length) {
        traces.push({
          x: aMask,
          y: aMask.map(i => yArr[i]),
          mode: 'markers', name: `⚠ label`,
          marker: { color: '#dc2626', size: 5, symbol: 'circle-open',
                    line: { width: 1.5, color: '#dc2626' } },
          showlegend: true,
          hoverinfo: 'skip',
        });
      }
    }

    const layout = _baseLayout(`📡 ${sensor}`, SENSOR_UNITS[sensor] || '', 70, xMax, false);
    
    Plotly.react(chartId, traces, layout, PLOT_CFG);
  });

  // ── Chart 5: Error + thresholds ───────────────────────────────
  const errSlice = _data.error.slice(0, currentN);
  const traces5  = [{
    x: idx, y: errSlice,
    mode: 'lines', name: 'Error',
    line: { color: '#0f172a', width: 2 },
    fill: 'tozeroy', fillcolor: 'rgba(15,23,42,0.04)',
  }];

  const layout5 = _baseLayout('📈 Reconstruction Error', 'error', 280, xMax, true);
  layout5.shapes = []; // ใช้ Shapes สร้าง Background แทน Traces เพื่อไม่ให้กวน Auto-Scale แกน Y
  

  showTH.forEach(thName => {
    const thVals = _data.thresholds[thName]?.slice(0, currentN);
    if (!thVals) return;
    const color = TH_COLORS[thName];

    traces5.push({
      x: idx, y: thVals,
      mode: 'lines', name: thName,
      line: { color, width: 1.6,
              dash: { 'P99 Static':'dot','Sliding Mu+αStd':'dash','Adaptive-z':'longdash','Entropy-lock':'solid' }[thName] },
      legendgroup: thName,
    });

    if (showBands) {
      const exceeded = errSlice.map((e, i) => e > thVals[i] ? i : -1).filter(i => i >= 0);
      if (exceeded.length) {
        const segs = [];
        let start  = exceeded[0];
        let prev   = exceeded[0];
        for (let k = 1; k < exceeded.length; k++) {
          if (exceeded[k] !== prev + 1) { segs.push([start, prev]); start = exceeded[k]; }
          prev = exceeded[k];
        }
        segs.push([start, prev]);

        segs.forEach((seg, si) => {
          if (si === 0) {
            // Dummy trace เอาไว้โชว์แค่ใน Legend เท่านั้น
            traces5.push({
              x: [null], y: [null], mode: 'lines',
              line: { width: 10, color: TH_FILLS[thName].replace(/0\.18\)/, '0.5)') },
              name: `${thName} band`, legendgroup: `band_${thName}`, showlegend: true
            });
          }
          layout5.shapes.push({
            type: 'rect', xref: 'x', yref: 'paper',
            x0: seg[0] - 0.5, x1: seg[1] + 0.5,
            y0: 0, y1: 1, fillcolor: TH_FILLS[thName],
            line: { width: 0 }, layer: 'below'
          });
        });
      }
    }
  });

  if (showLbl && _data.has_labels) {
    const overall = _data.labels.overall.slice(0, currentN);
    const asegs = [];
    let inSeg = false, ss = 0;
    overall.forEach((v, i) => {
      if (v && !inSeg)  { ss = i; inSeg = true; }
      if (!v && inSeg)  { asegs.push([ss, i-1]); inSeg = false; }
    });
    if (inSeg) asegs.push([ss, currentN-1]);

    if (asegs.length > 0) {
      traces5.push({
        x: [null], y: [null], mode: 'lines',
        line: { width: 10, color: 'rgba(220,38,38,0.3)' },
        name: '⚠ Anomaly Label', legendgroup: 'anom_lbl', showlegend: true
      });
      asegs.forEach((seg) => {
        layout5.shapes.push({
          type: 'rect', xref: 'x', yref: 'paper',
          x0: seg[0] - 0.5, x1: seg[1] + 0.5,
          y0: 0, y1: 1, fillcolor: 'rgba(220,38,38,0.10)',
          line: { width: 0 }, layer: 'below'
        });
      });
    }
  }

  Plotly.react(CHART_IDS[4], traces5, layout5, PLOT_CFG);
}

// ── Playback controls ─────────────────────────────────────────────
function _getSpeed() {
  const s = parseInt(document.getElementById('anim-speed')?.value);
  return isNaN(s) || s < 1 ? 30 : s; 
}

function _getStep() {
  const s = parseInt(document.getElementById('anim-step')?.value);
  return isNaN(s) || s < 1 ? 1 : s; 
}

function animPlay() {
  if (!_data) return;
  if (_playing) { _pausePlayback(); return; }
  _playing = true;
  document.getElementById('anim-play-btn').innerHTML = '⏸ Pause';
  const step = _getStep();

  _timer = setInterval(() => {
    if (_frame >= _data.n - 1) {
      _pausePlayback();
      document.getElementById('anim-play-btn').innerHTML = '▶ Play';
      return;
    }
    _frame = Math.min(_frame + step, _data.n - 1);
    _renderFrame(_frame);
    _updateProgressUI();
  }, _getSpeed());
}

function _pausePlayback() {
  _playing = false;
  clearInterval(_timer);
  _timer = null;
  const btn = document.getElementById('anim-play-btn');
  if (btn) btn.innerHTML = '▶ Play';
}

function _stopPlayback() {
  _pausePlayback();
  _frame = 0;
  _updateProgressUI();
}

function animReset() {
  _stopPlayback();
  if (_data) { _renderFrame(0); }
}

function animStop() {
  _stopPlayback();
  if (_data) { _renderFrame(0); }
}

function animScrub(e) {
  if (!_data) return;
  const bar  = document.getElementById('anim-progress-wrap');
  const rect = bar.getBoundingClientRect();
  const pct  = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  _frame     = Math.floor(pct * (_data.n - 1));
  _renderFrame(_frame);
  _updateProgressUI();
}

function animSeek(val) {
  if (!_data) return;
  _frame = Math.max(0, Math.min(_data.n - 1, parseInt(val)));
  _renderFrame(_frame);
  _updateProgressUI();
}

function _updateProgressUI() {
  const n   = _data?.n ?? 1;
  const pct = n > 1 ? (_frame / (n - 1)) * 100 : 0;
  const bar = document.getElementById('anim-progress-bar');
  const lbl = document.getElementById('anim-frame-lbl');
  const sldr= document.getElementById('anim-seek');
  if (bar)  bar.style.width  = pct + '%';
  if (lbl)  lbl.textContent  = `${_frame + 1} / ${n}`;
  if (sldr) { sldr.max = n - 1; sldr.value = _frame; }
}

function _setStatus(msg, color) {
  const el = document.getElementById('anim-status');
  if (!el) return;
  const clr = { green:'#16a34a', red:'#dc2626', amber:'#d97706', blue:'#2563eb' }[color] || '#64748b';
  el.style.color   = clr;
  el.textContent   = msg;
}

function _getChecked(name) {
  return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map(el => el.value);
}

window.loadAnimationData = loadAnimationData;
window.animPlay          = animPlay;
window.animReset         = animReset;
window.animStop          = animStop;
window.animScrub         = animScrub;
window.animSeek          = animSeek;

document.addEventListener('DOMContentLoaded', () => {
  const observer = new MutationObserver(() => {
    const panel = document.getElementById('tab-animation');
    if (panel && panel.classList.contains('active') && !panel.dataset.initialized) {
      panel.dataset.initialized = '1';
      _initEmptyCharts();
    }
  });
  const main = document.querySelector('.main');
  if (main) observer.observe(main, { subtree: true, attributes: true, attributeFilter: ['class'] });
});

})();