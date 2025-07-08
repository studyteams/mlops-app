# mymlopsapp1/forms.py
from django import forms
from .models import (
    DatasetConfig,
    FeatureConfig,
    ModelConfig,
    ResourceAllocation,
    Experiment,
    ExperimentRun,
)
import json
import pandas as pd  # CSV 파일 파싱을 위해 필요
import os  # 파일 경로 처리를 위해 필요
from django.conf import settings  # settings.BASE_DIR을 사용하기 위해 필요


# --- 자원 할당 설정 폼 ---
class ResourceAllocationForm(forms.ModelForm):
    class Meta:
        model = ResourceAllocation
        fields = [
            "name",
            "cpu_cores",
            "memory_gb",
            "gpu_count",
            "gpu_memory_gb",
            "max_training_time_hours",
            "description",
        ]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 3}),
            "cpu_cores": forms.NumberInput(attrs={"min": 1, "max": 32}),
            "memory_gb": forms.NumberInput(attrs={"min": 1, "max": 128}),
            "gpu_count": forms.NumberInput(attrs={"min": 0, "max": 8}),
            "gpu_memory_gb": forms.NumberInput(attrs={"min": 0, "max": 32}),
            "max_training_time_hours": forms.NumberInput(
                attrs={"min": 0.1, "max": 168.0, "step": 0.1}
            ),
        }


# --- 실험 생성 폼 ---
class ExperimentForm(forms.ModelForm):
    tags_json = forms.CharField(
        widget=forms.Textarea(
            attrs={"rows": 2, "placeholder": '["tag1", "tag2", "tag3"]'}
        ),
        required=False,
        label="태그",
        help_text='실험 태그를 JSON 배열 형식으로 입력하세요 (예: ["wine", "classification", "baseline"])',
    )

    class Meta:
        model = Experiment
        fields = [
            "name",
            "description",
            "version",
            "dataset_config",
            "feature_config",
            "model_config",
            "resource_allocation",
            "created_by",
        ]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 3}),
            "version": forms.TextInput(attrs={"placeholder": "1.0.0"}),
            "created_by": forms.TextInput(attrs={"placeholder": "your_name"}),
        }

    def clean_tags_json(self):
        tags_json = self.cleaned_data.get("tags_json", "")
        if tags_json.strip():
            try:
                tags = json.loads(tags_json)
                if not isinstance(tags, list):
                    raise forms.ValidationError("태그는 배열 형식이어야 합니다.")
                return json.dumps(tags)
            except json.JSONDecodeError:
                raise forms.ValidationError("올바른 JSON 형식으로 입력해주세요.")
        return "[]"

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.tags = self.cleaned_data.get("tags_json", "[]")
        if commit:
            instance.save()
        return instance


# --- 실험 실행 폼 ---
class ExperimentRunForm(forms.ModelForm):
    class Meta:
        model = ExperimentRun
        fields = ["experiment", "run_id", "run_number"]
        widgets = {
            "run_id": forms.TextInput(attrs={"placeholder": "auto-generated"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # run_id는 자동 생성되므로 읽기 전용으로 설정
        self.fields["run_id"].widget.attrs["readonly"] = True
        self.fields["run_id"].required = False


# --- 데이터셋 업로드 및 설정 폼 (단일 단계) ---
# dataset_upload_view에서 사용됨
class DatasetUploadAndConfigForm(forms.Form):
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


# --- DatasetConfig 수동 생성 폼 ---
# create_dataset_config_view에서 사용됨
class DatasetConfigForm(forms.ModelForm):
    class Meta:
        model = DatasetConfig
        fields = ["name", "file_path", "target_column", "description"]
        widgets = {
            "file_path": forms.TextInput(
                attrs={"placeholder": "예: winequality-red.csv (data/ 폴더 기준)"}
            ),
            "description": forms.Textarea(attrs={"rows": 3}),
        }


# --- FeatureConfig 생성/수정 폼 ---
# create_feature_config_view에서 사용됨
class FeatureConfigForm(forms.ModelForm):
    # 어떤 데이터셋에 대한 피처인지 선택하는 필드 추가
    # ModelChoiceField는 Django 모델 객체를 드롭다운으로 선택하게 해줍니다.
    dataset = forms.ModelChoiceField(
        queryset=DatasetConfig.objects.all(),
        label="피처를 적용할 데이터셋",
        help_text="이 피처 조합을 사용할 데이터셋을 선택하세요.",
        required=True,
    )

    # 이 필드는 동적으로 컬럼 목록을 채울 것이므로, 초기에는 choices를 비워둡니다.
    # 사용자가 데이터셋을 선택하면, 해당 데이터셋의 컬럼들을 체크박스로 제공합니다.
    features_selection = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,  # 여러 개를 선택할 수 있도록 체크박스 위젯 사용
        label="선택할 피처들",
        required=False,  # 피처를 선택하지 않으면 (e.g., 모든 피처), 뒤에서 자동으로 처리
        help_text="데이터셋에서 학습에 사용할 피처들을 선택하세요. (아무것도 선택하지 않으면 타겟을 제외한 모든 컬럼이 피처로 간주됩니다.)",
    )

    class Meta:
        model = FeatureConfig
        # 'features' 필드는 forms.MultipleChoiceField인 'features_selection'으로 대체되므로 제외
        # 'dataset_config' 필드는 forms.ModelChoiceField인 'dataset'으로 대체되므로 제외
        fields = ["name", "description"]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 3}),
        }

    # 폼 초기화 시 (GET 요청 시) 동적으로 features_selection 필드의 choices를 설정합니다.
    # 또한, 폼을 생성할 때 뷰에서 dataset_id를 넘겨줄 수 있도록 kwargs를 처리합니다.
    def __init__(self, *args, **kwargs):
        # 뷰에서 전달된 dataset_id를 kwargs에서 추출 (팝하여 제거)
        dataset_id = kwargs.pop("dataset_id", None)
        super().__init__(*args, **kwargs)  # 반드시 super().__init__을 먼저 호출해야 함

        all_columns = []
        # dataset_id가 주어졌을 때만 해당 데이터셋의 컬럼을 로드
        if dataset_id:
            try:
                dataset_config = DatasetConfig.objects.get(id=dataset_id)
                # settings.BASE_DIR과 dataset_config.file_path를 조합하여 실제 파일 경로 생성
                file_path = os.path.join(
                    settings.BASE_DIR, "data", dataset_config.file_path
                )

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    # 타겟 컬럼을 제외한 모든 컬럼을 피처 선택지로 제공
                    all_columns = [
                        col
                        for col in df.columns.tolist()
                        if col != dataset_config.target_column
                    ]
                else:
                    print(
                        f"Warning: Dataset file not found at {file_path} for FeatureConfigForm."
                    )
            except DatasetConfig.DoesNotExist:
                print(
                    f"Warning: DatasetConfig with ID {dataset_id} does not exist for FeatureConfigForm."
                )
            except Exception as e:
                print(
                    f"Error loading columns for FeatureConfigForm initialization: {e}"
                )

        # 모든 컬럼을 체크박스 선택지로 설정 (value, label)
        self.fields["features_selection"].choices = [(col, col) for col in all_columns]

        # 인스턴스가 주어졌을 경우 (수정 모드), 기존 피처 선택 값을 초기화합니다.
        if (
            self.instance and self.instance.pk
        ):  # 인스턴스가 이미 DB에 저장된 객체인 경우 (PK가 있음)
            # models.TextField에 저장된 JSON 문자열을 파이썬 리스트로 변환하여 초기값 설정
            # self.instance.features는 이미 JSON 문자열 형태
            if (
                self.instance.features and self.instance.features != "[]"
            ):  # 빈 리스트가 아닌 경우만 로드
                self.initial["features_selection"] = json.loads(self.instance.features)
            # dataset 필드도 초기화
            if self.instance.dataset_config:
                self.initial["dataset"] = self.instance.dataset_config.id

    # 폼 유효성 검사 후, features_selection의 데이터를 model.features에 맞게 가공합니다.
    def clean(self):
        cleaned_data = super().clean()  # 부모 클래스의 clean 메서드 호출

        selected_features = cleaned_data.get("features_selection")
        dataset_config_obj = cleaned_data.get(
            "dataset"
        )  # dataset 필드에서 DatasetConfig 객체를 가져옴

        if not dataset_config_obj:
            # dataset 필드가 필수이므로 여기에 도달할 일은 거의 없지만, 방어 코드
            self.add_error("dataset", "피처를 적용할 데이터셋을 선택해야 합니다.")
            return cleaned_data  # 에러가 발생했으므로 여기서 반환

        # 사용자가 피처를 아무것도 선택하지 않았을 경우
        if not selected_features:
            # 타겟 컬럼을 제외한 모든 컬럼을 피처로 자동 설정
            file_path = os.path.join(
                settings.BASE_DIR, "data", dataset_config_obj.file_path
            )
            try:
                df = pd.read_csv(file_path)
                # 타겟 컬럼을 제외한 모든 컬럼을 피처로 사용
                all_columns_except_target = [
                    col
                    for col in df.columns.tolist()
                    if col != dataset_config_obj.target_column
                ]
                # 모델의 features 필드는 TextField이므로, JSON 문자열로 저장해야 합니다.
                cleaned_data["features"] = json.dumps(all_columns_except_target)
            except FileNotFoundError:
                self.add_error(
                    "dataset",
                    "선택된 데이터셋 파일이 'data/' 폴더에 존재하지 않습니다.",
                )
            except pd.errors.EmptyDataError:
                self.add_error("dataset", "선택된 데이터셋 파일이 비어 있습니다.")
            except pd.errors.ParserError:
                self.add_error(
                    "dataset",
                    "선택된 데이터셋 파일을 파싱하는 데 실패했습니다. 형식을 확인하세요.",
                )
            except Exception as e:
                self.add_error(None, f"데이터셋 컬럼을 읽는 중 오류 발생: {e}")
        else:
            # 사용자가 피처를 명시적으로 선택한 경우, 선택된 피처들을 JSON 문자열로 저장
            cleaned_data["features"] = json.dumps(selected_features)

        return cleaned_data

    # save 메서드를 오버라이드하여 폼 필드와 모델 필드의 불일치 문제를 해결합니다.
    def save(self, commit=True):
        instance = super().save(
            commit=False
        )  # 모델 인스턴스를 가져오지만 아직 DB에 저장하지 않음

        # clean 메서드에서 가공된 'features' 데이터를 모델의 features 필드에 할당
        instance.features = self.cleaned_data["features"]

        # forms.ModelChoiceField인 'dataset'에서 선택된 DatasetConfig 객체를
        # 모델의 ForeignKey 필드인 'dataset_config'에 할당
        instance.dataset_config = self.cleaned_data["dataset"]

        if commit:  # commit=True일 경우에만 DB에 저장
            instance.save()
        return instance


# --- ModelConfig 생성/수정 폼 ---
# create_model_config_view에서 사용됨
class ModelConfigForm(forms.ModelForm):
    # 모델 타입 드롭다운 목록을 직접 정의 (향후 확장 가능)
    MODEL_TYPE_CHOICES = [
        ("LogisticRegression", "Logistic Regression (Classification)"),
        ("LinearRegression", "Linear Regression (Regression)"),
        ("RandomForestClassifier", "Random Forest Classifier (Classification)"),
        ("RandomForestRegressor", "Random Forest Regressor (Regression)"),
        # 다른 모델 타입 추가 가능
    ]
    model_type = forms.ChoiceField(choices=MODEL_TYPE_CHOICES, label="모델 타입")

    # parameters 필드는 TextField이므로, 사용자가 JSON 문자열로 입력하도록 합니다.
    parameters_json = forms.CharField(
        widget=forms.Textarea(
            attrs={
                "rows": 5,
                "placeholder": '{"C": 1.0, "max_iter": 1000}',
            }
        ),
        required=False,
        label="하이퍼파라미터",
        help_text='모델 하이퍼파라미터를 JSON 형식으로 입력하세요 (예: {"C": 1.0, "max_iter": 1000})',
    )

    class Meta:
        model = ModelConfig
        # 'model_type'은 forms.ChoiceField인 'model_type'으로 대체되므로 제외
        # 'parameters'는 'parameters_json'으로 대체되므로 제외
        fields = ["name", "dataset_config", "feature_config", "description"]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 3}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 인스턴스가 주어졌을 경우 (수정 모드), 기존 파라미터 값을 초기화합니다.
        if self.instance and self.instance.pk:
            # models.TextField에 저장된 JSON 문자열을 그대로 표시
            if self.instance.parameters and self.instance.parameters != "{}":
                self.initial["parameters_json"] = self.instance.parameters

    def clean_parameters_json(self):
        parameters_json = self.cleaned_data.get("parameters_json", "")
        if parameters_json.strip():
            try:
                parameters = json.loads(parameters_json)
                if not isinstance(parameters, dict):
                    raise forms.ValidationError(
                        "하이퍼파라미터는 객체 형식이어야 합니다."
                    )
                return parameters_json
            except json.JSONDecodeError:
                raise forms.ValidationError("올바른 JSON 형식으로 입력해주세요.")
        return "{}"

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.model_type = self.cleaned_data["model_type"]
        instance.parameters = self.cleaned_data.get("parameters_json", "{}")
        if commit:
            instance.save()
        return instance
