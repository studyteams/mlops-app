# MLOps 원앱 솔루션

Django 기반의 통합 MLOps 플랫폼으로, 머신러닝 워크플로우의 전체 라이프사이클을 관리할 수 있습니다.
![CleanShot 2025-07-09 at 00 57 16](https://github.com/user-attachments/assets/2140b6eb-5573-40bc-b80b-2f3c6ed33432)

![CleanShot 2025-07-09 at 00 57 38](https://github.com/user-attachments/assets/0f93dc22-56c9-4f2c-99aa-c9268c4eda7b)

## 🚀 주요 기능

### 1. 데이터 관리
- **데이터셋 업로드**: CSV 파일 업로드 및 관리
- **데이터셋 설정**: 타겟 변수 지정 및 메타데이터 관리
- **데이터 검증**: 파일 형식 및 컬럼 검증

### 2. 피처 엔지니어링
- **피처 선택**: 데이터셋에서 학습에 사용할 피처 선택
- **피처 구성 관리**: 다양한 피처 조합을 설정으로 저장
- **동적 피처 로딩**: 데이터셋 변경 시 자동으로 피처 목록 업데이트

### 3. 모델 관리
- **모델 설정**: 다양한 알고리즘과 하이퍼파라미터 설정
- **모델 버저닝**: 모델 설정의 버전 관리
- **지원 모델**: LogisticRegression, LinearRegression, RandomForest 등

### 4. 자원 할당 관리 ⭐ **NEW**
- **CPU 할당**: 코어 수 설정 (1-32개)
- **메모리 할당**: RAM 용량 설정 (1-128GB)
- **GPU 할당**: GPU 개수 및 메모리 설정 (0-8개, 0-32GB)
- **시간 제한**: 최대 훈련 시간 설정 (0.1-168시간)
- **자원 프로필**: 자주 사용하는 자원 설정을 프로필로 저장

### 5. 실험 관리 및 버저닝 ⭐ **NEW**
- **실험 생성**: 데이터셋, 피처, 모델, 자원을 조합한 실험 설정
- **실험 버저닝**: 실험의 버전 관리 (예: v1.0.0, v1.1.0)
- **실험 태그**: 실험 분류를 위한 태그 시스템
- **실험 상태 추적**: 계획됨 → 실행 중 → 완료/실패 상태 관리
- **실행 기록**: 각 실험의 실행 이력 및 결과 저장

### 6. 실험 실행 및 모니터링 ⭐ **NEW**
- **실험 실행**: 계획된 실험을 실제로 실행
- **실시간 모니터링**: 실행 상태 및 진행 상황 추적
- **자원 사용량 추적**: 실제 사용된 CPU, 메모리, GPU 사용량 기록
- **실행 로그**: 상세한 실행 로그 및 오류 메시지 저장

### 7. 결과 분석 및 비교
- **성능 지표**: 정확도, 정밀도, 재현율, F1 스코어 등
- **결과 비교**: 여러 모델/실험 결과를 한 번에 비교
- **시각화**: 결과를 차트로 시각화 (향후 구현 예정)

## 🏗️ 아키텍처

### 데이터 모델

#### 기존 모델
- `DatasetConfig`: 데이터셋 설정 관리
- `FeatureConfig`: 피처 구성 관리
- `ModelConfig`: 모델 설정 관리
- `TrainingResult`: 훈련 결과 저장

#### 새로운 모델 ⭐
- `ResourceAllocation`: 자원 할당 설정
- `Experiment`: 실험 관리 및 버저닝
- `ExperimentRun`: 실험 실행 기록

### 기술 스택
- **Backend**: Django 4.x
- **Database**: SQLite (개발용) / PostgreSQL (운영용)
- **ML Framework**: Scikit-learn
- **Frontend**: Bootstrap 5
- **Job Queue**: Django (향후 Celery 연동 예정)

## 📦 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성
conda create -n mlops-env python=3.9
conda activate mlops-env

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터베이스 설정
```bash
cd mymlops
python manage.py makemigrations
python manage.py migrate
```

### 3. 개발 서버 실행
```bash
python manage.py runserver
```

### 4. 관리자 계정 생성
```bash
python manage.py createsuperuser
```

## 🎯 사용 가이드

### 1. 자원 할당 설정 생성
1. **자원 할당 관리** → **새 자원 설정**
2. CPU, 메모리, GPU, 시간 제한 설정
3. 설정 저장

### 2. 실험 생성 및 실행
1. **실험 관리** → **새 실험 생성**
2. 데이터셋, 피처, 모델, 자원 할당 선택
3. 실험 버전 및 태그 설정
4. **실험 실행** 버튼으로 훈련 시작

### 3. 실험 모니터링
1. **실험 목록**에서 실험 상태 확인
2. **실험 상세보기**에서 실행 기록 확인
3. **실행 상세보기**에서 로그 및 결과 확인

### 4. 결과 비교
1. **실험 비교**에서 비교할 실험 선택
2. 성능 지표 및 자원 사용량 비교
3. 최적 모델 선택

## 🔧 설정

### 환경 변수
```bash
# settings.py에서 설정
DEBUG = True
SECRET_KEY = 'your-secret-key'
DATABASE_URL = 'sqlite:///db.sqlite3'  # 개발용
```

### 자원 할당 제한
- **CPU**: 1-32 코어
- **메모리**: 1-128GB
- **GPU**: 0-8개
- **GPU 메모리**: 0-32GB
- **최대 시간**: 0.1-168시간

## 📊 대시보드

홈 페이지에서 다음 정보를 확인할 수 있습니다:
- 총 실험 수
- 실행 중인 실험 수
- 완료된 실험 수
- 모델 설정 수
- 최근 실험 목록

## 🔮 향후 계획

### 단기 계획
- [ ] 실험 결과 시각화 (차트, 그래프)
- [ ] 실험 템플릿 기능
- [ ] 자동 하이퍼파라미터 튜닝
- [ ] 모델 배포 기능

### 중기 계획
- [ ] Celery를 통한 비동기 작업 처리
- [ ] Docker 컨테이너 지원
- [ ] Kubernetes 오케스트레이션
- [ ] 실시간 알림 시스템

### 장기 계획
- [ ] A/B 테스트 지원
- [ ] 모델 성능 드리프트 감지
- [ ] 자동 재훈련 파이프라인
- [ ] MLOps 베스트 프랙티스 가이드

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 라이선스

MIT License

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해주세요.

---

**Note**: 이 프로젝트는 컨셉 앱으로, 실제 프로덕션 환경에서 사용하기 전에 보안, 성능, 확장성 등을 고려하여 추가 개발이 필요합니다.
