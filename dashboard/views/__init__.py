# dashboard/views/__init__.py
"""Re-export all views so urls.py stays unchanged."""
from .pages       import index
from .single      import (api_train, api_status,
                           api_chart_umap, api_chart_raw, api_chart_error,
                           api_chart_inspector_umap, api_chart_threshold)
from .batch       import (api_train_all, api_status_all,
                           api_chart_model_umap, api_chart_model_error,
                           api_clear_batch_cache, api_device_info)
from .results_api import api_results_table, api_results_abort
from .grid_search import (api_grid_search_start, api_grid_search_status,
                           api_grid_search_best_charts)