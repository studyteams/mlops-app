# mymlopsapp1/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django import forms  # Import forms
from django.forms import (
    MultipleChoiceField,
    CheckboxSelectMultiple,
)  # 결과 선택을 위한 폼 필드 임포트
from django.utils import timezone
from django.contrib import messages
import uuid

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)
import joblib

import json  # json 모듈 임포트

# 폼 임포트
from .forms import (
    DatasetConfigForm,
    FeatureConfigForm,
    ModelConfigForm,
    ResourceAllocationForm,
    ExperimentForm,
    ExperimentRunForm,
)
from .models import (
    DatasetConfig,
    FeatureConfig,
    ModelConfig,
    ResourceAllocation,
    Experiment,
    ExperimentRun,
)

# 데이터셋 루트 경로 (settings.py의 BASE_DIR 기준)
DATA_ROOT = os.path.join(settings.BASE_DIR, "data")
# 모델 저장 경로 (settings.MEDIA_ROOT 사용)
MODEL_DIR = os.path.join(settings.MEDIA_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)  # 모델 디렉토리 생성


# --- 자원 할당 관리 뷰 ---
def create_resource_allocation_view(request):
    if request.method == "POST":
        form = ResourceAllocationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "자원 할당 설정이 성공적으로 생성되었습니다.")
            return redirect("mymlopsapp1:list_resource_allocations")
    else:
        form = ResourceAllocationForm()

    return render(
        request,
        "mymlopsapp1/create_config.html",
        {"form": form, "config_type": "Resource Allocation"},
    )


def list_resource_allocations_view(request):
    allocations = ResourceAllocation.objects.all()
    return render(
        request,
        "mymlopsapp1/list_configs.html",
        {
            "configs": allocations,
            "config_type": "Resource Allocation",
            "create_url_name": "mymlopsapp1:create_resource_allocation",
        },
    )


def resource_allocation_detail_view(request, pk):
    allocation = get_object_or_404(ResourceAllocation, pk=pk)
    return render(
        request,
        "mymlopsapp1/resource_allocation_detail.html",
        {"allocation": allocation},
    )


# --- 실험 관리 뷰 ---
def create_experiment_view(request):
    if request.method == "POST":
        form = ExperimentForm(request.POST)
        if form.is_valid():
            experiment = form.save()
            messages.success(
                request, f"실험 '{experiment.name}'이 성공적으로 생성되었습니다."
            )
            return redirect("mymlopsapp1:list_experiments")
    else:
        form = ExperimentForm()

    return render(
        request,
        "mymlopsapp1/create_experiment.html",
        {"form": form},
    )


def list_experiments_view(request):
    experiments = Experiment.objects.all()
    return render(
        request,
        "mymlopsapp1/list_experiments.html",
        {"experiments": experiments},
    )


def experiment_detail_view(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    runs = experiment.runs.all()
    return render(
        request,
        "mymlopsapp1/experiment_detail.html",
        {"experiment": experiment, "runs": runs},
    )


def start_experiment_view(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)

    if experiment.status == "planned":
        # 실험 상태를 실행 중으로 변경
        experiment.status = "running"
        experiment.started_at = timezone.now()
        experiment.save()

        # 새로운 실행 생성
        run_number = experiment.runs.count() + 1
        run_id = f"{experiment.name}_run_{run_number}_{uuid.uuid4().hex[:8]}"

        run = ExperimentRun.objects.create(
            experiment=experiment, run_id=run_id, run_number=run_number, status="queued"
        )

        # 실제 훈련은 여기서 시뮬레이션 (컨셉 앱이므로)
        # 실제 구현에서는 비동기 작업으로 처리
        run.status = "running"
        run.started_at = timezone.now()
        run.save()

        # 시뮬레이션된 훈련 결과 생성
        from .models import TrainingResult, Dataset, MLModel

        # 임시 데이터 생성 (실제로는 모델 훈련 결과)
        training_result = TrainingResult.objects.create(
            dataset=Dataset.objects.first() if Dataset.objects.exists() else None,
            ml_model_config=(
                MLModel.objects.first() if MLModel.objects.exists() else None
            ),
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            training_duration_seconds=120.5,
            version=f"v{experiment.version}",
        )

        run.training_result = training_result
        run.status = "completed"
        run.completed_at = timezone.now()
        run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
        run.save()

        # 실험 완료
        experiment.status = "completed"
        experiment.completed_at = timezone.now()
        experiment.save()

        messages.success(
            request, f"실험 '{experiment.name}'이 성공적으로 완료되었습니다."
        )
    else:
        messages.warning(
            request,
            f"실험 '{experiment.name}'은 이미 {experiment.get_status_display()} 상태입니다.",
        )

    return redirect("mymlopsapp1:experiment_detail", pk=pk)


def experiment_run_detail_view(request, experiment_pk, run_pk):
    experiment = get_object_or_404(Experiment, pk=experiment_pk)
    run = get_object_or_404(ExperimentRun, pk=run_pk, experiment=experiment)

    return render(
        request,
        "mymlopsapp1/experiment_run_detail.html",
        {"experiment": experiment, "run": run},
    )


# --- 실험 비교 뷰 ---
def compare_experiments_view(request):
    if request.method == "POST":
        experiment_ids = request.POST.getlist("experiments")
        if len(experiment_ids) >= 2:
            experiments = Experiment.objects.filter(id__in=experiment_ids)
            return render(
                request,
                "mymlopsapp1/compare_experiments.html",
                {"experiments": experiments},
            )
        else:
            messages.error(request, "비교할 실험을 2개 이상 선택해주세요.")

    experiments = Experiment.objects.filter(status="completed")
    return render(
        request,
        "mymlopsapp1/select_experiments_for_comparison.html",
        {"experiments": experiments},
    )


# --- 데이터셋 업로드 및 설정 폼 (단일 단계) ---
class DatasetUploadAndConfigForm(forms.Form):
    file = forms.FileField(
        label="CSV 파일 선택", help_text="업로드할 데이터셋 CSV 파일을 선택하세요."
    )
    dataset_name = forms.CharField(
        max_length=100,
        label="데이터셋 이름",
        help_text="데이터셋의 고유한 이름을 입력하세요 (예: WineQualityRed_New).",
    )
    # target_column = forms.CharField(
    #     max_length=100,
    #     label="타겟 컬럼 이름",
    #     help_text="예측하거나 분류할 타겟 변수의 정확한 컬럼 이름을 입력하세요.",
    # )
    description = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 3}),
        required=False,
        label="설명",
        help_text="데이터셋에 대한 간단한 설명을 입력하세요.",
    )


# --- 임시 파일 업로드 폼 (초기 파일 업로드) ---
class InitialUploadFileForm(forms.Form):
    file = forms.FileField(
        label="CSV 파일 선택", help_text="업로드할 데이터셋 CSV 파일을 선택하세요."
    )
    dataset_name = forms.CharField(
        max_length=100,
        label="데이터셋 이름",
        help_text="데이터셋의 고유한 이름을 입력하세요 (예: WineQualityRed_New).",
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 3}),
        required=False,
        label="설명",
        help_text="데이터셋에 대한 간단한 설명을 입력하세요.",
    )


# --- 타겟 컬럼 선택 폼 (컬럼 목록 동적 생성) ---
class SelectTargetColumnForm(forms.Form):
    # 이 필드는 뷰에서 동적으로 choices를 채울 것입니다.
    target_column = forms.ChoiceField(
        label="타겟 컬럼 선택",
        help_text="예측하거나 분류할 타겟 변수 컬럼을 선택하세요.",
    )
    # dataset_name, description은 hidden 필드로 전달받아 DatasetConfig 생성에 사용
    dataset_name = forms.CharField(widget=forms.HiddenInput())
    uploaded_filename = forms.CharField(widget=forms.HiddenInput())
    description = forms.CharField(widget=forms.HiddenInput(), required=False)


# --- File Upload Form (Define this directly in views.py or in forms.py) ---
# For simplicity, we'll define it here. For more complex apps, put in forms.py
class UploadFileForm(forms.Form):
    # This field will be used to upload the actual dataset file
    file = forms.FileField(label="Select a CSV file to upload")
    # This field will be used to create a DatasetConfig record for the uploaded file
    dataset_name = forms.CharField(
        max_length=100, label="Dataset Name (e.g., MyNewData)"
    )
    target_column = forms.CharField(
        max_length=50, label="Target Column Name (e.g., price)"
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 3}), required=False
    )


# --- 홈 페이지 ---
def home(request):
    # 대시보드 정보 추가
    context = {
        "message": "Welcome to MLOps All in One App!",
        "total_experiments": Experiment.objects.count(),
        "running_experiments": Experiment.objects.filter(status="running").count(),
        "completed_experiments": Experiment.objects.filter(status="completed").count(),
        "total_models": ModelConfig.objects.count(),
        "total_datasets": DatasetConfig.objects.count(),
        "recent_experiments": Experiment.objects.order_by("-created_at")[:5],
    }
    return render(request, "mymlopsapp1/home.html", context)


# --- DatasetConfig 생성 뷰 ---
def create_dataset_config_view(request):
    if request.method == "POST":
        form = DatasetConfigForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect(
                "mymlopsapp1:list_dataset_configs"
            )  # 목록 페이지로 리다이렉트
    else:
        form = DatasetConfigForm()
    return render(
        request,
        "mymlopsapp1/create_config.html",
        {"form": form, "config_type": "Dataset"},
    )


def list_dataset_configs_view(request):
    configs = DatasetConfig.objects.all()
    return render(
        request,
        "mymlopsapp1/list_configs.html",
        {
            "configs": configs,
            "config_type": "Dataset",
            "create_url_name": "mymlopsapp1:create_dataset_config",
        },
    )


# --- FeatureConfig 생성 뷰 ---
def create_feature_config_view(request):
    # GET 요청 시 DatasetConfig ID를 쿼리 파라미터로 받을 수 있음 (선택 사항)
    dataset_id = request.GET.get("dataset_id")

    if request.method == "POST":
        # POST 요청 시에도 dataset_id를 폼 초기화에 넘겨줘야 합니다.
        form = FeatureConfigForm(request.POST, dataset_id=request.POST.get("dataset"))
        if form.is_valid():
            form.save()
            from django.contrib import messages

            messages.success(
                request,
                f"피처 구성 '{form.cleaned_data['name']}'이(가) 성공적으로 생성되었습니다.",
            )
            return redirect("mymlopsapp1:list_feature_configs")
    else:
        # GET 요청 시, 폼 초기화에 dataset_id를 넘겨서 해당 데이터셋의 컬럼을 미리 로드
        form = FeatureConfigForm(dataset_id=dataset_id)

    return render(
        request,
        "mymlopsapp1/create_config.html",
        {"form": form, "config_type": "Feature"},
    )


def list_feature_configs_view(request):
    configs = FeatureConfig.objects.all()
    return render(
        request,
        "mymlopsapp1/list_configs.html",
        {
            "configs": configs,
            "config_type": "Feature",
            "create_url_name": "mymlopsapp1:create_feature_config",
        },
    )


# --- ModelConfig 생성 뷰 ---
def create_model_config_view(request):
    if request.method == "POST":
        form = ModelConfigForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("mymlopsapp1:list_model_configs")
    else:
        form = ModelConfigForm()
    return render(
        request,
        "mymlopsapp1/create_config.html",
        {"form": form, "config_type": "Model"},
    )


def list_model_configs_view(request):
    configs = ModelConfig.objects.all()
    return render(
        request,
        "mymlopsapp1/list_configs.html",
        {
            "configs": configs,
            "config_type": "Model",
            "create_url_name": "mymlopsapp1:create_model_config",
        },
    )


# --- 기존 훈련 관련 뷰들 ---
def train_model_view(request):
    if request.method == "POST":
        # 폼 데이터 처리
        model_config_id = request.POST.get("model_config")
        dataset_config_id = request.POST.get("dataset_config")
        feature_config_id = request.POST.get("feature_config")

        if not all([model_config_id, dataset_config_id, feature_config_id]):
            return HttpResponse("모든 설정을 선택해주세요.", status=400)

        try:
            model_config = ModelConfig.objects.get(id=model_config_id)
            dataset_config = DatasetConfig.objects.get(id=dataset_config_id)
            feature_config = FeatureConfig.objects.get(id=feature_config_id)
        except (
            ModelConfig.DoesNotExist,
            DatasetConfig.DoesNotExist,
            FeatureConfig.DoesNotExist,
        ):
            return HttpResponse("선택한 설정을 찾을 수 없습니다.", status=404)

        # 데이터 로드
        file_path = os.path.join(DATA_ROOT, dataset_config.file_path)
        if not os.path.exists(file_path):
            return HttpResponse(
                f"데이터셋 파일을 찾을 수 없습니다: {file_path}", status=404
            )

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return HttpResponse(f"데이터셋 로드 중 오류 발생: {str(e)}", status=500)

        # 피처 선택
        features = (
            json.loads(feature_config.features) if feature_config.features else []
        )
        if not features:
            # 피처가 지정되지 않은 경우, 타겟을 제외한 모든 컬럼 사용
            features = [
                col for col in df.columns if col != dataset_config.target_column
            ]

        # 데이터 준비
        X = df[features]
        y = df[dataset_config.target_column]

        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 모델 생성 및 훈련
        model = None
        start_time = timezone.now()

        if model_config.model_type == "LogisticRegression":
            params = (
                json.loads(model_config.parameters) if model_config.parameters else {}
            )
            model = LogisticRegression(**params)
        elif model_config.model_type == "LinearRegression":
            params = (
                json.loads(model_config.parameters) if model_config.parameters else {}
            )
            model = LinearRegression(**params)
        elif model_config.model_type == "RandomForestClassifier":
            params = (
                json.loads(model_config.parameters) if model_config.parameters else {}
            )
            model = RandomForestClassifier(**params)
        elif model_config.model_type == "RandomForestRegressor":
            params = (
                json.loads(model_config.parameters) if model_config.parameters else {}
            )
            model = RandomForestRegressor(**params)
        else:
            return HttpResponse(
                f"지원하지 않는 모델 타입: {model_config.model_type}", status=400
            )

        # 모델 훈련
        model.fit(X_train, y_train)

        # 예측
        y_pred = model.predict(X_test)

        # 평가
        training_duration = (timezone.now() - start_time).total_seconds()

        # 모델 저장
        model_filename = (
            f"{model_config.name}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        )
        model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(model, model_path)

        # 결과 계산
        if model_config.model_type in ["LogisticRegression", "RandomForestClassifier"]:
            accuracy = accuracy_score(y_test, y_pred)
            from sklearn.metrics import precision_score, recall_score, f1_score

            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")
        else:
            accuracy = None
            precision = None
            recall = None
            f1 = r2_score(y_test, y_pred)

        # TrainingResult 저장
        from .models import TrainingResult, Dataset, MLModel

        # Dataset과 MLModel 인스턴스 생성 또는 가져오기
        dataset, created = Dataset.objects.get_or_create(
            name=dataset_config.name,
            defaults={
                "description": dataset_config.description or "",
                "file": dataset_config.file_path,
            },
        )

        ml_model, created = MLModel.objects.get_or_create(
            name=model_config.name,
            defaults={
                "description": model_config.description or "",
                "algorithm": model_config.model_type,
            },
        )

        training_result = TrainingResult.objects.create(
            dataset=dataset,
            ml_model_config=ml_model,
            params=(
                json.loads(model_config.parameters) if model_config.parameters else {}
            ),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_duration_seconds=training_duration,
            model_file_path=model_path,
            version=f"v1.0_{timezone.now().strftime('%Y%m%d_%H%M%S')}",
        )

        return redirect("mymlopsapp1:training_results")

    # GET 요청 처리
    model_configs = ModelConfig.objects.all()
    dataset_configs = DatasetConfig.objects.all()
    feature_configs = FeatureConfig.objects.all()

    return render(
        request,
        "mymlopsapp1/train_model.html",
        {
            "model_configs": model_configs,
            "dataset_configs": dataset_configs,
            "feature_configs": feature_configs,
        },
    )


def model_results_view(request):
    from .models import TrainingResult

    results = TrainingResult.objects.all().order_by("-trained_at")
    return render(request, "mymlopsapp1/model_results.html", {"results": results})


class CompareResultsForm(forms.Form):
    """
    비교할 모델 학습 결과를 선택하기 위한 폼
    """

    results_to_compare = MultipleChoiceField(
        widget=CheckboxSelectMultiple,
        choices=[],
        label="비교할 결과 선택",
        help_text="비교할 모델 학습 결과를 선택하세요 (2개 이상)",
    )


def compare_results_view(request):
    from .models import TrainingResult

    if request.method == "POST":
        form = CompareResultsForm(request.POST)
        if form.is_valid():
            selected_ids = form.cleaned_data["results_to_compare"]
            if len(selected_ids) >= 2:
                results = TrainingResult.objects.filter(id__in=selected_ids)
                return render(
                    request,
                    "mymlopsapp1/compare_results.html",
                    {"results": results},
                )
            else:
                return HttpResponse("비교할 결과를 2개 이상 선택해주세요.", status=400)
    else:
        # GET 요청 시 모든 결과를 선택지로 제공
        all_results = TrainingResult.objects.all()
        choices = [
            (result.id, f"{result.ml_model_config.name} - {result.dataset.name}")
            for result in all_results
        ]
        form = CompareResultsForm()
        form.fields["results_to_compare"].choices = choices

    return render(request, "mymlopsapp1/compare_results.html", {"form": form})


def dataset_upload_view(request):
    if request.method == "POST":
        form = DatasetUploadAndConfigForm(request.POST, request.FILES)
        if form.is_valid():
            # 파일 저장
            uploaded_file = form.cleaned_data["file"]
            filename = uploaded_file.name

            # data/ 디렉토리에 파일 저장
            os.makedirs(DATA_ROOT, exist_ok=True)
            file_path = os.path.join(DATA_ROOT, filename)

            with open(file_path, "wb+") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # DatasetConfig 생성
            dataset_config = DatasetConfig.objects.create(
                name=form.cleaned_data["dataset_name"],
                file_path=filename,
                target_column="",  # 나중에 설정
                description=form.cleaned_data.get("description", ""),
            )

            return redirect("mymlopsapp1:list_dataset_configs")

    else:
        form = DatasetUploadAndConfigForm()

    return render(request, "mymlopsapp1/dataset_upload.html", {"form": form})
