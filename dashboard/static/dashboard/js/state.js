// static/dashboard/js/state.js
// ── Global state shared across all modules ────────────────────────

// Single-model training
let JOB_ID      = null;
let POLL_TIMER  = null;
let TUNER_TIMER = null;

// Train-All
let BATCH_JOB_ID = null;
let BATCH_POLL   = null;

// Grid Search
let GRID_JOB_ID  = null;
let GRID_POLL    = null;

// Results cache — avoids recomputing on every tab switch
// Invalidated when: new Train All completes, or user clicks ↻ Recompute
let RESULTS_CACHE     = null;   // { rows, has_labels, runs, normal_runs }
let RESULTS_CACHE_KEY = null;   // "batchJobId|th_hash"

// Injected by Django template
// ANOMALY_RUNS, ALL_RUNS, MODEL_LIST are defined inline in index.html