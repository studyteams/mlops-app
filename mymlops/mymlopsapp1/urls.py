from django.urls import path
from . import views

app_name = "mymlopsapp1"  # <-- Make sure this line exists and is uncommented

urlpatterns = [
    # 설정 생성 및 목록 보기 URL 추가
    path(
        "configs/datasets/create/",
        views.create_dataset_config_view,
        name="create_dataset_config",
    ),
    path(
        "configs/datasets/",
        views.list_dataset_configs_view,
        name="list_dataset_configs",
    ),
    path(
        "configs/features/create/",
        views.create_feature_config_view,
        name="create_feature_config",
    ),
    path(
        "configs/features/",
        views.list_feature_configs_view,
        name="list_feature_configs",
    ),
    path(
        "configs/models/create/",
        views.create_model_config_view,
        name="create_model_config",
    ),
    path("configs/models/", views.list_model_configs_view, name="list_model_configs"),
    path("configs/datasets/upload/", views.dataset_upload_view, name="dataset_upload"),
    # 자원 할당 관리 URL
    path(
        "resources/create/",
        views.create_resource_allocation_view,
        name="create_resource_allocation",
    ),
    path(
        "resources/",
        views.list_resource_allocations_view,
        name="list_resource_allocations",
    ),
    path(
        "resources/<int:pk>/",
        views.resource_allocation_detail_view,
        name="resource_allocation_detail",
    ),
    # 실험 관리 URL
    path(
        "experiments/create/",
        views.create_experiment_view,
        name="create_experiment",
    ),
    path(
        "experiments/",
        views.list_experiments_view,
        name="list_experiments",
    ),
    path(
        "experiments/<int:pk>/",
        views.experiment_detail_view,
        name="experiment_detail",
    ),
    path(
        "experiments/<int:pk>/start/",
        views.start_experiment_view,
        name="start_experiment",
    ),
    path(
        "experiments/<int:experiment_pk>/runs/<int:run_pk>/",
        views.experiment_run_detail_view,
        name="experiment_run_detail",
    ),
    path(
        "experiments/compare/",
        views.compare_experiments_view,
        name="compare_experiments",
    ),
    # 모델 학습 관련 URL
    path("train/", views.train_model_view, name="train_model"),
    # 학습 결과 관련 URL
    path("results/", views.model_results_view, name="model_results"),
    path("compare_results/", views.compare_results_view, name="compare_results"),
    path("training_results/", views.model_results_view, name="training_results"),
    path("", views.home, name="home"),  # 기본 페이지
]
