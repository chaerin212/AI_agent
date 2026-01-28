# AI Agent for Transistor Analysis

트랜지스터 이미지를 분석하고 품질을 판단하는 AI 에이전트입니다.

## 주요 기능

- **트랜지스터 이미지 분석**: 트랜지스터의 molding(몰딩) 상태를 분석하고 설명 생성
- **품질 판단**: Molding 설명을 기반으로 정상/비정상 여부 자동 판단
- **리드 삽입 검사**: 리드(lead)가 홀에 제대로 삽입되었는지 판단
- **통합 파이프라인**: 전체 분석 프로세스를 자동화

## 파일 구조

- `main.py`: 리드 삽입 여부 판단 메인 스크립트
- `transi_analysis_1140.py`: 트랜지스터 이미지 분석 및 묘사 생성
- `judge_molding.py`: Molding 설명을 기반으로 품질 판단
- `integrated_pipeline.py`: 전체 분석 파이프라인 통합 스크립트
- `run_model.py`: 모델 실행 스크립트

## 환경 설정

### 필수 패키지

```bash
pip install requests pandas numpy opencv-python pillow
```

### 환경 변수

`.env` 파일을 생성하거나 환경 변수로 설정:

```bash
export LUXIA_API_KEY="your_api_key_here"
```

Windows PowerShell:
```powershell
$env:LUXIA_API_KEY="your_api_key_here"
```

## 사용 방법

### 1. 트랜지스터 분석

```bash
python transi_analysis_1140.py
```

### 2. Molding 판단

```bash
python judge_molding.py
```

### 3. 리드 삽입 검사

```bash
python main.py
```

### 4. 통합 파이프라인

```bash
python integrated_pipeline.py
```

## API 설정

이 프로젝트는 Luxia API를 사용합니다. API 키는 환경 변수로 설정해야 합니다.

## 라이선스

MIT License
