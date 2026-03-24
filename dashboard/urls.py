# dashboard/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),

    # Single-model training
    path("api/train/",                   views.api_train,              name="api_train"),
    path("api/status/<str:job_id>/",     views.api_status,             name="api_status"),

    # Single-model charts
    path("api/charts/umap/",             views.api_chart_umap,         name="chart_umap"),
    path("api/charts/raw/",              views.api_chart_raw,          name="chart_raw"),
    path("api/charts/error/",            views.api_chart_error,        name="chart_error"),
    path("api/charts/inspector-umap/",   views.api_chart_inspector_umap, name="chart_inspector_umap"),
    path("api/charts/threshold/",        views.api_chart_threshold,    name="chart_threshold"),

    # Train-All
    path("api/train-all/",                    views.api_train_all,           name="api_train_all"),
    path("api/status-all/<str:job_id>/",      views.api_status_all,          name="api_status_all"),
    path("api/train-all/clear-cache/",        views.api_clear_batch_cache,   name="api_clear_batch_cache"),
    path("api/train-all/stop/",               views.api_stop_batch,          name="api_stop_batch"),
    path("api/device-info/",                  views.api_device_info,         name="api_device_info"),
    path("api/charts/model-umap/",            views.api_chart_model_umap,    name="chart_model_umap"),
    path("api/charts/model-error/",           views.api_chart_model_error,   name="chart_model_error"),
    path("api/results-table/",                views.api_results_table,       name="api_results_table"),
    path("api/results-abort/",                views.api_results_abort,       name="api_results_abort"),
    path("api/pr-curve/",                     views.api_pr_curve,            name="api_pr_curve"),

    # Animation
    path("api/animation/data/",  views.api_animation_data, name="api_animation_data"),

    # Raw Data Editor
    path("api/rawdata/run/",    views.api_rawdata_run,    name="api_rawdata_run"),
    path("api/rawdata/patch/",  views.api_rawdata_patch,  name="api_rawdata_patch"),
    path("api/rawdata/undo/",   views.api_rawdata_undo,   name="api_rawdata_undo"),
    path("api/rawdata/reset/",  views.api_rawdata_reset,  name="api_rawdata_reset"),
    path("api/rawdata/export/", views.api_rawdata_export, name="api_rawdata_export"),

    # Grid Search
    path("api/grid-search/start/",                    views.api_grid_search_start,       name="api_grid_search_start"),
    path("api/grid-search/status/<str:job_id>/",      views.api_grid_search_status,      name="api_grid_search_status"),
    path("api/grid-search/best-charts/<str:job_id>/", views.api_grid_search_best_charts, name="api_grid_search_best_charts"),
    path("api/grid-search/stop/",                     views.api_grid_search_stop,        name="api_grid_search_stop"),
    path("api/grid-search/export/<str:job_id>/",      views.api_grid_search_export,      name="api_grid_search_export"),
]