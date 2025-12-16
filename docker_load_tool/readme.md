G:\2025\reflex\app에 계층화시킨 리펙토링 코드
>streamlit run main.py --server.maxUploadSize 1000

```
app/
├── main.py                    # Streamlit 메인 앱
├── backend/                   # 백엔드 로직  
│   ├── __init__.py
│   ├── data_service.py       # 데이터 로딩/처리 서비스
│   └── file_service.py       # 파일 저장 서비스 
├── frontend/                  # 프론트엔드 UI
│   ├── __init__.py
│   ├── config_ui.py          # YAML 설정 탭
│   ├── loading_ui.py         # 데이터 로딩 탭
│   └── viz_ui.py             # 데이터 시각화 탭 
└── utils/                     # 유틸리티
    ├── __init__.py
    └── yaml_utils.py         # YAML 설정 유틸리티
```




