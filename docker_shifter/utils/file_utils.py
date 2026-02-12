"""
파일 입출력 유틸리티
"""
import streamlit as st
import pandas as pd
import io
import zipfile
import tempfile
import os
from typing import List, Dict, Optional


def load_data_file(uploaded_file) -> pd.DataFrame:
    """Feather 또는 H5 파일을 로드하는 함수"""
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith('.feather') or file_name.endswith('.ftr'):
            df = pd.read_feather(uploaded_file)
        elif file_name.endswith('.h5') or file_name.endswith('.hdf5'):
            # H5 파일은 임시 파일로 저장한 후 읽어야 함
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # 임시 파일에서 H5 읽기
                df = pd.read_hdf(tmp_path)
            finally:
                # 임시 파일 삭제
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            st.error(f"지원하지 않는 파일 형식입니다: {file_name}")
            return None

        return df
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {str(e)}")
        return None


def load_feather_file(uploaded_file) -> pd.DataFrame:
    """하위 호환성을 위한 래퍼 함수"""
    return load_data_file(uploaded_file)


def handle_file_upload(uploaded_files) -> None:
    """파일 업로드를 처리하는 함수 (탭1용)"""
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.current_file_index = 0
        st.success(f"✅ {len(uploaded_files)}개 파일이 업로드되었습니다!")


def handle_batch_file_upload(uploaded_files) -> None:
    """배치 파일 업로드를 처리하는 함수 (탭2용)"""
    if uploaded_files:
        st.session_state.batch_uploaded_files = uploaded_files
        st.success(f"✅ {len(uploaded_files)}개 파일이 배치 업로드되었습니다!")


def handle_multi_file_upload(uploaded_files) -> None:
    """다중 파일 업로드를 처리하는 함수 (탭3용)"""
    if uploaded_files:
        st.session_state.multi_uploaded_files = uploaded_files
        st.success(f"✅ {len(uploaded_files)}개 파일이 다중 업로드되었습니다!")


def save_dataframe_to_buffer(df: pd.DataFrame, file_format: str = 'feather') -> bytes:
    """DataFrame을 지정된 형식의 버퍼로 저장"""
    if file_format.lower() in ['feather', 'ftr']:
        buffer = io.BytesIO()
        df.reset_index(drop=True).to_feather(buffer)
        buffer.seek(0)
        return buffer.getvalue()
    elif file_format.lower() in ['h5', 'hdf5']:
        # H5 파일은 임시 파일로 저장한 후 읽어야 함
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_path = tmp_file.name

        try:
            # 임시 파일에 H5로 저장
            df.to_hdf(tmp_path, key='data', mode='w', format='table')
            # 임시 파일을 읽어서 바이트로 반환
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_format}")


def create_zip_download(processed_files: List[Dict], zip_filename: str, file_format: str = 'feather') -> bytes:
    """처리된 파일들을 ZIP으로 압축하여 다운로드 가능한 형태로 만드는 함수"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_info in processed_files:
            # DataFrame을 지정된 형식으로 변환
            file_buffer = save_dataframe_to_buffer(file_info['dataframe'], file_format)

            # ZIP에 파일 추가
            zip_file.writestr(file_info['processed_name'], file_buffer)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
