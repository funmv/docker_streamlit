# 통합 데이터 분석 시스템 (Refactored)

Excel/CSV 파일을 YAML 설정으로 자동 처리하고 시각화하는 Streamlit 기반 애플리케이션

## 프로젝트 구조

```
app/
├── main.py                    # Streamlit 메인 앱
├── backend/                   # 백엔드 로직 (FastAPI 스타일)
│   ├── __init__.py
│   ├── data_service.py       # 데이터 로딩/처리 서비스
│   └── file_service.py       # 파일 저장 서비스 (HDF5만)
├── frontend/                  # 프론트엔드 UI
│   ├── __init__.py
│   ├── config_ui.py          # YAML 설정 탭
│   ├── loading_ui.py         # 데이터 로딩 탭
│   └── viz_ui.py             # 데이터 시각화 탭 (시계열만)
└── utils/                     # 유틸리티
    ├── __init__.py
    └── yaml_utils.py         # YAML 설정 유틸리티
```

## 주요 특징

### 백엔드 (로직)
- **DataService**: 데이터 로딩 및 처리
  - Excel/CSV 파일 로드
  - 헤더 추출
  - 타임스탬프 처리
  - 샘플링
  - 데이터 타입 변환
  - 후처리

- **FileService**: 파일 저장 (HDF5만 지원)
  - 메타데이터 포함 HDF5 저장
  - HDF5에서 메타데이터 로드
  - 날짜 범위 추출

### 프론트엔드 (UI)
- **YAML 설정 탭**: 설정 관리
  - YAML 파일 불러오기/저장
  - 파일 정보, CSV 옵션, 시트 설정
  - 헤더 구조, 타임스탬프 처리
  - 샘플링, 컬럼명 정규화
  - 데이터 타입, 후처리, 에러 처리

- **데이터 로딩 탭**: 데이터 로드 및 저장
  - Excel/CSV 파일 업로드
  - 데이터 미리보기 및 통계
  - HDF5 저장 (단일/다중 시트)

- **데이터 시각화 탭**: 시계열 플롯
  - HDF5 파일 업로드
  - 시계열 그래프 (선/점/선+점)
  - 메타데이터 표시

## 단순화된 부분

1. **파일 저장**: Parquet, CSV, Excel 제거 → HDF5만 지원
2. **시각화**: Scatter, Histogram, Boxplot, Heatmap 제거 → 시계열 플롯만 유지
3. **코드 구조**: FastAPI 스타일로 백엔드/프론트엔드 분리

## 실행 방법

```bash
cd /home/claude/app
streamlit run main.py
```

## 설치 필요 패키지

```bash
pip install streamlit pandas openpyxl plotly pyyaml tables --break-system-packages
```

## 사용 흐름

1. **YAML 설정** 탭에서 데이터 처리 설정
2. **데이터 로딩** 탭에서 파일 업로드 및 처리
3. 처리된 데이터를 HDF5로 저장
4. **데이터 시각화** 탭에서 시계열 플롯 생성

## 메타데이터 지원

- Description, Unit, Tag_name, ID 등 헤더 메타데이터
- HDF5 파일에 메타데이터 저장/로드
- 시각화 시 메타데이터 표시

## 향후 확장

- niceGUI 또는 Reflex로 프론트엔드 마이그레이션 가능
- 백엔드 로직은 그대로 사용 가능