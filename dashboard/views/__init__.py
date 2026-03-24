# dashboard/views/__init__.py
"""Re-export all views so urls.py stays unchanged."""
from .animation  import api_animation_data
from .pages       import index
from .single      import (api_train, api_status,
                           api_chart_umap, api_chart_raw, api_chart_error,
                           api_chart_inspector_umap, api_chart_threshold)
from .batch       import (api_train_all, api_status_all,
                           api_chart_model_umap, api_chart_model_error,
                           api_clear_batch_cache, api_device_info,
                           api_stop_batch)
from .results_api import api_results_table, api_results_abort
from .pr_curve    import api_pr_curve
from .rawdata     import (api_rawdata_run, api_rawdata_patch,
                           api_rawdata_undo, api_rawdata_reset, api_rawdata_export)
from .grid_search import (api_grid_search_start, api_grid_search_status,
                           api_grid_search_best_charts, api_grid_search_stop,
                           api_grid_search_export)