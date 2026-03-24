// static/dashboard/js/tabs.js

function switchTab(name, btn) {
  // Mount template-based tabs BEFORE querySelectorAll (element must be in DOM first)
  if (name === 'trainall')   mountTabIfNeeded('trainall',   'tpl-trainall');
  if (name === 'results')    mountTabIfNeeded('results',    'tpl-results');
  if (name === 'gridsearch') mountTabIfNeeded('gridsearch', 'tpl-gridsearch');
  if (name === 'rawdata')     mountTabIfNeeded('rawdata',     'tpl-rawdata');

  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');

  // Lazy-load on first visit
  if (name === 'umap')       loadUmap();
  if (name === 'inspector')  loadInspector();
  if (name === 'anomaly')    loadAnomalyCharts();
  if (name === 'tuner')      loadTuner();
  if (name === 'results')    { loadResultsTable(); loadPrCurve(); }
  if (name === 'gridsearch') gsUpdateCombo();
  if (name === 'rawdata')     rdLoad();
  if (name === 'trainall')  taLoadDevice();   // uses cache if available
}

function mountTabIfNeeded(name, tplId) {
  if (!document.getElementById('tab-' + name)) {
    const tpl = document.getElementById(tplId);
    const div = tpl.content.firstElementChild.cloneNode(true);
    div.id    = 'tab-' + name;
    div.className = 'tab-panel';
    div.removeAttribute('style');  // strip any leftover display:none from template
    document.querySelector('.main').appendChild(div);
  }
}