# 베이스 이미지
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    fonts-nanum \
    libhdf5-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Streamlit 앱 복사
COPY . .

# 임시 파일 및 출력 디렉토리 생성
RUN mkdir -p /tmp/streamlit /app/output && \
    chmod 777 /tmp/streamlit /app/output

# .streamlit 디렉토리 복사 (있는 경우)
RUN if [ -d ".streamlit" ]; then mkdir -p /root/.streamlit && cp -r .streamlit/* /root/.streamlit/; fi

# Streamlit 기본 설정 (ENV 형식 개선)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TMPDIR=/tmp/streamlit

# 포트 노출
EXPOSE 8501

# Health check 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit 실행 명령
CMD ["streamlit", "run", "step1_crop_v3o.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=1000", \
     "--server.maxMessageSize=1000", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]