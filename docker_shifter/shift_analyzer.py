import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Optional, Tuple  
import matplotlib.pyplot as plt
import io
import zipfile
import json                    
from datetime import datetime, timedelta  

# =================================================================================
# Utils 모듈에서 함수 import
# =================================================================================
from utils.font_utils import setup_korean_font
from utils.file_utils import (
    load_data_file,
    load_feather_file,
    handle_file_upload,
    handle_batch_file_upload,
    handle_multi_file_upload,
    save_dataframe_to_buffer,
    create_zip_download
)
from utils.data_utils import (
    apply_time_delay,
    get_data_segment
)
from utils.plot_utils import (
    create_multivariate_plot,
    create_combined_plot,
    create_multi_file_plot
)
from utils.batch_utils import (
    process_batch_files,
    split_files_train_val
)
from utils.dnn_utils import (
    create_positional_encoding,
    extract_time_features,
    extract_dnn_samples_optimized,
    extract_time_features_vectorized,
    create_positional_encoding_vectorized,
    extract_dnn_samples,
    process_all_files_for_dnn,
    save_dnn_dataset
)

# matplotlib 경고 제거를 위한 설정
plt.rcParams['figure.max_open_warning'] = 50

# =================================================================================
# 페이지 설정 및 초기화
# =================================================================================
st.set_page_config(page_title="다변량 시계열 데이터 분석", layout="wide")
setup_korean_font()


# =================================================================================
# 메인 애플리케이션
# =================================================================================
def main():
    st.title("📈 학습용 시계열 데이터 추출 툴")
    
    # 탭 생성 - 추후 확장을 위한 구조
    # tab1, tab2, tab3 = st.tabs(["🔍 신호 관찰", "📊 이동 실행", "📦 데이터 추출"])
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 신호 관찰", "📊 이동 실행", "📦 데이터 추출", "🎯 유사 기동 검색"])


    # =================================================================================
    # 탭 1: 신호 분석 (메인 기능)
    # =================================================================================
    with tab1:
        st.header("🚀 다변량 시계열 신호 관찰 및 분석")
        
        # 파일 업로드 섹션
        st.subheader("📁 파일 업로드")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**FTR/Feather 또는 H5 파일을 직접 업로드하세요:**")
            uploaded_files = st.file_uploader(
                "FTR/Feather 또는 H5 파일들을 선택하세요",
                type=['ftr', 'feather', 'h5', 'hdf5'],
                accept_multiple_files=True
            )
        
        with col2:
            if uploaded_files:
                if st.button("📤 파일 업로드 처리", key="upload_btn"):
                    handle_file_upload(uploaded_files)
        
        # 파일이 업로드된 경우 분석 시작
        if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            files = st.session_state.uploaded_files
            
            # 파일 선택 (기본값: 첫 번째 파일)
            st.subheader("📂 분석할 파일 선택")
            file_names = [f.name for f in files]
            selected_file_index = st.selectbox(
                "분석할 파일을 선택하세요:",
                range(len(files)),
                format_func=lambda x: file_names[x],
                index=0
            )
            
            # 선택된 파일 로드
            selected_file = files[selected_file_index]
            df = load_data_file(selected_file)
            
            if df is not None:
                st.success(f"✅ {selected_file.name} 로딩 완료! Shape: {df.shape}")
                
                # 데이터 미리보기
                with st.expander("📋 데이터 미리보기"):
                    st.dataframe(df.head())
                    st.write(f"**컬럼 정보:** {list(df.columns)}")
                    st.write(f"**데이터 타입:** {df.dtypes.to_dict()}")
                
                # 기본 신호 관찰
                st.subheader("📈 기본 신호 관찰")
                
                # 컬럼 선택
                selected_cols = st.multiselect(
                    "📊 Plot할 컬럼을 선택하세요",
                    df.columns.tolist(),
                    default=df.columns.tolist()[:3] if len(df.columns) >= 3 else df.columns.tolist()
                )
                
                if selected_cols:
                    # 기본 설정
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        downsample_rate = st.slider(
                            "📉 다운샘플 비율 (1/N)", 
                            min_value=1, max_value=100, value=10
                        )
                    with col2:
                        num_segments = st.selectbox(
                            "📊 데이터 분할 수",
                            options=[1, 2, 3, 4, 5],
                            index=2,  # 기본값: 3등분
                            help="전체 데이터를 몇 등분할지 선택"
                        )
                    with col3:
                        selected_segment = st.selectbox(
                            "🎯 분석 구간 선택",
                            options=list(range(num_segments)),
                            format_func=lambda x: f"구간 {x+1}",
                            index=0,  # 기본값: 첫 번째 구간
                            help="분석할 구간을 선택"
                        )
                    
                    # 데이터 구간 정보 표시
                    total_length = len(df)
                    segment_length = total_length // num_segments
                    start_idx = selected_segment * segment_length
                    end_idx = start_idx + segment_length if selected_segment < num_segments - 1 else total_length
                    
                    st.info(f"📊 **선택된 구간**: {start_idx:,} ~ {end_idx:,} (총 {end_idx - start_idx:,}개 포인트, 전체의 {((end_idx - start_idx) / total_length * 100):.1f}%)")
                    
                    crosshair = st.checkbox("▶️ 십자선 Hover 활성화", value=True)
                    
                    # 기본 플롯 생성
                    basic_delays = {col: 0 for col in selected_cols}
                    fig_basic = create_multivariate_plot(
                        df, selected_cols, basic_delays, downsample_rate, crosshair,
                        num_segments, selected_segment
                    )
                    fig_basic.update_layout(title="📊 기본 다변량 시계열 신호")
                    st.plotly_chart(fig_basic, use_container_width=True)
                
                # 시간 지연 분석
                st.subheader("⏱️ 시간 지연 분석")
                st.markdown("선택된 속성에 시간 지연을 적용하여 신호의 상호관계를 분석할 수 있습니다.")
                
                # 지연 분석용 컬럼 선택
                delay_cols = st.multiselect(
                    "🎯 지연 분석할 컬럼을 선택하세요",
                    df.columns.tolist(),
                    key="delay_analysis_cols"
                )
                
                if delay_cols:                    
                    # 각 컬럼별 지연값 입력
                    delays = {}
                    cols_per_row = 3
                    
                    for i in range(0, len(delay_cols), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col_name in enumerate(delay_cols[i:i+cols_per_row]):
                            with cols[j]:
                                delays[col_name] = st.number_input(
                                    f"🔄 {col_name}",
                                    min_value=-1000,
                                    max_value=1000,
                                    value=0,
                                    step=1,
                                    key=f"delay_{col_name}"
                                )
                    
                    # 적용 버튼과 플롯
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🚀 시간 지연 적용 및 플롯 생성", key="apply_delays_btn"):
                            st.session_state.delays_applied = True
                            st.session_state.current_delays = delays.copy()
                            st.session_state.current_delay_cols = delay_cols.copy()
                    
                    # 지연 적용된 플롯 표시
                    if (hasattr(st.session_state, 'delays_applied') and 
                        st.session_state.delays_applied and 
                        hasattr(st.session_state, 'current_delays')):
                        
                        st.markdown("---")
                        st.subheader("📊 시간 지연 적용 결과")
                        
                        # 적용된 지연값 정보 표시
                        delay_info = []
                        for col, delay in st.session_state.current_delays.items():
                            if delay != 0:
                                delay_info.append(f"**{col}**: {delay:+d}")
                        
                        if delay_info:
                            st.info(f"적용된 지연값: {', '.join(delay_info)}")
                        
                        # 함께 표시할 기준 컬럼 선택 (결과 확인 후 선택 가능)
                        available_reference_cols = [col for col in df.columns.tolist() 
                                                  if col not in st.session_state.current_delay_cols]
                        
                        reference_cols = st.multiselect(
                            "📊 함께 비교할 기준 컬럼을 선택하세요 (지연 적용 안됨)",
                            available_reference_cols,
                            key="reference_cols_result"
                        )
                        
                        if reference_cols:
                            st.info(f"기준 신호 (점선): {', '.join(reference_cols)}")
                        
                        # 지연 적용된 플롯 생성 (기준 컬럼과 함께)
                        fig_delayed = create_combined_plot(
                            df, 
                            st.session_state.current_delay_cols,
                            st.session_state.current_delays,
                            reference_cols,
                            downsample_rate,
                            crosshair,
                            num_segments,
                            selected_segment
                        )
                        st.plotly_chart(fig_delayed, use_container_width=True)
                        
                        # 지연 적용된 데이터 저장/다운로드 섹션 추가
                        st.markdown("---")
                        st.subheader("💾 지연 적용 데이터 저장")
                        st.caption("원본에서 shift 선택된 특징을 제외하고, shift 처리된 특징을 포함하여 저장합니다.")
                        
                        # 파일명과 형식 선택
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            default_filename = f"{selected_file.name.split('.')[0]}_shifted"
                            save_filename = st.text_input(
                                "저장할 파일명 (확장자 제외)",
                                value=default_filename
                            )
                        with col2:
                            save_format = st.selectbox(
                                "파일 형식",
                                options=['feather', 'h5'],
                                index=0,
                                help="저장할 파일 형식을 선택하세요"
                            )

                        # 데이터 생성 및 다운로드 버튼
                        if st.button(f"🔄 지연 적용 데이터 생성 및 다운로드 ({save_format.upper()})", key="generate_shifted_data"):
                            try:
                                # 원본 데이터 복사
                                shifted_df = df.copy()
                                
                                # shift 선택된 특징들을 지연 처리된 버전으로 교체
                                for col in st.session_state.current_delay_cols:
                                    delay = st.session_state.current_delays[col]
                                    shifted_series = apply_time_delay(df, col, delay)
                                    
                                    # 원본 컬럼을 지연 적용된 데이터로 교체
                                    shifted_df[col] = shifted_series
                                
                                # 결측값 정보 표시
                                total_na = shifted_df.isna().sum().sum()
                                if total_na > 0:
                                    st.warning(f"⚠️ 시간 지연으로 인해 {total_na:,}개의 결측값이 생성되었습니다.")
                                
                                # 데이터 미리보기
                                with st.expander("📋 생성된 데이터 미리보기"):
                                    st.write(f"**Shape**: {shifted_df.shape}")
                                    st.write(f"**컬럼**: {list(shifted_df.columns)}")
                                    st.dataframe(shifted_df.head(10))
                                    
                                    # 지연 적용 정보 요약
                                    st.write("**지연 적용된 특징:**")
                                    for col, delay in st.session_state.current_delays.items():
                                        st.write(f"- {col}: {delay:+d}틱 지연 적용")
                                    
                                    # 변경되지 않은 특징들
                                    unchanged_cols = [col for col in df.columns if col not in st.session_state.current_delay_cols]
                                    if unchanged_cols:
                                        st.write("**원본 유지된 특징:**")
                                        st.write(f"- {', '.join(unchanged_cols)}")
                                
                                # 선택된 형식으로 저장
                                file_data = save_dataframe_to_buffer(shifted_df, save_format)
                                file_extension = save_format if save_format != 'feather' else 'feather'

                                # 다운로드 버튼
                                st.download_button(
                                    label=f"💾 {save_format.upper()} 파일 다운로드",
                                    data=file_data,
                                    file_name=f"{save_filename}.{file_extension}",
                                    mime="application/octet-stream",
                                    help=f"지연이 적용된 데이터를 {save_format} 형식으로 다운로드"
                                )
                                
                                st.success(f"✅ 지연 적용된 데이터가 성공적으로 생성되었습니다!")
                                
                            except Exception as e:
                                st.error(f"❌ 데이터 생성 중 오류 발생: {str(e)}")
                        
                        # 추가 저장 옵션 (CSV)
                        with st.expander("📄 추가 저장 옵션"):
                            st.markdown("**CSV 형식으로도 저장 가능:**")
                            if st.button("📊 CSV 형식으로 생성", key="generate_csv"):
                                try:
                                    # 동일한 로직으로 데이터 생성
                                    shifted_df = df.copy()
                                    for col in st.session_state.current_delay_cols:
                                        delay = st.session_state.current_delays[col]
                                        shifted_series = apply_time_delay(df, col, delay)
                                        shifted_df[col] = shifted_series
                                    
                                    # CSV로 변환
                                    csv_buffer = io.StringIO()
                                    shifted_df.to_csv(csv_buffer, index=True)
                                    csv_data = csv_buffer.getvalue()
                                    
                                    # CSV 다운로드 버튼
                                    st.download_button(
                                        label="📄 CSV 파일 다운로드",
                                        data=csv_data,
                                        file_name=f"{save_filename}.csv",
                                        mime="text/csv",
                                        help="지연이 적용된 데이터를 CSV 형식으로 다운로드"
                                    )
                                    
                                except Exception as e:
                                    st.error(f"❌ CSV 생성 중 오류 발생: {str(e)}")
    
    # =================================================================================
    # 탭 2: 배치 지연 처리 (새로운 기능)
    # =================================================================================
    with tab2:
        st.header("🔄 배치 지연 처리")
        st.markdown("여러 개의 FTR/Feather 또는 H5 파일에 동일한 지연 설정을 일괄 적용하여 처리합니다.")

        # 배치 파일 업로드 섹션
        st.subheader("📁 배치 파일 업로드")
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("**여러 개의 FTR/Feather 또는 H5 파일을 업로드하세요:**")
            batch_uploaded_files = st.file_uploader(
                "배치 처리할 FTR/Feather 또는 H5 파일들을 선택하세요",
                type=['ftr', 'feather', 'h5', 'hdf5'],
                accept_multiple_files=True,
                key="batch_file_uploader"
            )
        
        with col2:
            if batch_uploaded_files:
                if st.button("📤 배치 파일 업로드 처리", key="batch_upload_btn"):
                    handle_batch_file_upload(batch_uploaded_files)
        
        # 배치 파일이 업로드된 경우 처리 시작
        if 'batch_uploaded_files' in st.session_state and st.session_state.batch_uploaded_files:
            batch_files = st.session_state.batch_uploaded_files
            
            st.success(f"✅ {len(batch_files)}개 파일이 배치 업로드되었습니다!")
            
            # 첫 번째 파일을 기준으로 특징 목록 확인
            first_file = batch_files[0]
            reference_df = load_feather_file(first_file)
            
            if reference_df is not None:
                st.subheader("📊 기준 파일 정보")
                st.info(f"**기준 파일**: {first_file.name} (Shape: {reference_df.shape})")
                
                # 데이터 미리보기
                with st.expander("📋 기준 파일 데이터 미리보기"):
                    st.dataframe(reference_df.head())
                    st.write(f"**사용 가능한 특징**: {list(reference_df.columns)}")
                
                # 특징 선택
                st.subheader("🎯 지연 적용할 특징 선택")
                selected_features = st.multiselect(
                    "배치 처리에 적용할 특징들을 선택하세요",
                    reference_df.columns.tolist(),
                    default=[reference_df.columns[0]] if len(reference_df.columns) > 0 else [],
                    key="batch_feature_selection"
                )
                
                if selected_features:
                    # 지연값 설정
                    st.subheader("⏱️ 지연값 설정")
                    st.caption("모든 파일에 동일한 지연값이 적용됩니다. 양수: 미래→현재, 음수: 과거→현재")
                    
                    batch_delays = {}
                    cols_per_row = 3
                    
                    for i in range(0, len(selected_features), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, feature_name in enumerate(selected_features[i:i+cols_per_row]):
                            with cols[j]:
                                batch_delays[feature_name] = st.number_input(
                                    f"🔄 {feature_name}",
                                    min_value=-1000,
                                    max_value=1000,
                                    value=0,
                                    step=1,
                                    key=f"batch_delay_{feature_name}"
                                )
                    
                    # 배치 처리 설정 요약
                    st.subheader("📋 배치 처리 설정 요약")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**처리 대상 파일:**")
                        for i, file in enumerate(batch_files):
                            st.write(f"{i+1}. {file.name}")
                    
                    with col2:
                        st.markdown("**적용할 지연 설정:**")
                        for feature, delay in batch_delays.items():
                            if delay != 0:
                                st.write(f"• {feature}: {delay:+d}틱")
                            else:
                                st.write(f"• {feature}: 지연 없음")
                    
                    # 배치 처리 실행
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        if st.button("🚀 배치 지연 처리 시작", key="start_batch_processing"):
                            st.session_state.batch_processing_done = False
                            
                            with st.spinner("🔄 배치 처리 중... 잠시만 기다려주세요."):
                                # 배치 처리 실행
                                processed_files = process_batch_files(batch_files, selected_features, batch_delays)
                                
                                if processed_files:
                                    st.session_state.processed_batch_files = processed_files
                                    st.session_state.batch_processing_done = True
                                    st.session_state.batch_selected_features = selected_features
                                    st.session_state.batch_delays = batch_delays
                    
                    # 배치 처리 결과 표시
                    if (hasattr(st.session_state, 'batch_processing_done') and 
                        st.session_state.batch_processing_done and 
                        hasattr(st.session_state, 'processed_batch_files')):
                        
                        st.markdown("---")
                        st.subheader("✅ 배치 처리 완료")
                        
                        processed_files = st.session_state.processed_batch_files
                        
                        # 처리 결과 요약
                        st.success(f"🎉 {len(processed_files)}개 파일이 성공적으로 처리되었습니다!")
                        
                        # 처리된 파일 정보 표시
                        with st.expander("📊 처리된 파일 상세 정보"):
                            for i, file_info in enumerate(processed_files):
                                st.markdown(f"**{i+1}. {file_info['original_name']}**")
                                st.write(f"   • 새 파일명: {file_info['processed_name']}")
                                st.write(f"   • 데이터 크기: {file_info['shape']}")
                                if file_info['applied_delays']:
                                    st.write(f"   • 적용된 지연: {file_info['applied_delays']}")
                                else:
                                    st.write(f"   • 적용된 지연: 없음")
                                st.write("")
                        
                        # 통계 정보
                        total_features_processed = sum(len(f['applied_delays']) for f in processed_files)
                        st.info(f"📈 **처리 통계**: {len(processed_files)}개 파일, {total_features_processed}개 특징에 지연 적용")
                        
                        # 다운로드 섹션
                        st.subheader("💾 배치 처리 결과 다운로드")
                        
                        # ZIP 파일명과 형식 설정
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            default_zip_name = f"batch_shifted_files_{len(processed_files)}files"
                            zip_filename = st.text_input(
                                "ZIP 파일명 (확장자 제외)",
                                value=default_zip_name,
                                key="zip_filename_input"
                            )
                        with col2:
                            batch_save_format = st.selectbox(
                                "파일 형식",
                                options=['feather', 'h5'],
                                index=0,
                                key="batch_save_format",
                                help="ZIP 내부 파일의 저장 형식"
                            )

                        # ZIP 다운로드 버튼
                        if st.button(f"📦 ZIP 파일로 일괄 다운로드 ({batch_save_format.upper()})", key="download_batch_zip"):
                            try:
                                with st.spinner(f"📦 {batch_save_format.upper()} 형식으로 ZIP 파일 생성 중..."):
                                    # 파일명 확장자 업데이트
                                    for file_info in processed_files:
                                        original_name = file_info['original_name'].split('.')[0]
                                        file_info['processed_name'] = f"{original_name}_batch_shifted.{batch_save_format}"

                                    zip_data = create_zip_download(processed_files, f"{zip_filename}.zip", batch_save_format)

                                st.download_button(
                                    label=f"💾 ZIP 파일 다운로드 ({batch_save_format.upper()})",
                                    data=zip_data,
                                    file_name=f"{zip_filename}.zip",
                                    mime="application/zip",
                                    help=f"모든 처리된 파일을 {batch_save_format} 형식으로 ZIP에 압축하여 다운로드"
                                )

                                st.success("✅ ZIP 파일이 생성되었습니다! 다운로드 버튼을 클릭하세요.")

                            except Exception as e:
                                st.error(f"❌ ZIP 파일 생성 중 오류: {str(e)}")
                        
                        # 개별 파일 다운로드 옵션
                        with st.expander("📄 개별 파일 다운로드"):
                            st.markdown("**개별 파일을 따로 다운로드할 수도 있습니다:**")
                            
                            for i, file_info in enumerate(processed_files):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.write(f"**{file_info['processed_name']}** ({file_info['shape'][0]:,} × {file_info['shape'][1]})")
                                
                                with col2:
                                    # 개별 파일 다운로드 (배치와 동일한 형식 사용)
                                    file_data = save_dataframe_to_buffer(file_info['dataframe'], batch_save_format)

                                    st.download_button(
                                        label="💾 다운로드",
                                        data=file_data,
                                        file_name=file_info['processed_name'],
                                        mime="application/octet-stream",
                                        key=f"individual_download_{i}"
                                    )
    


    # =================================================================================
    # 탭 3: 다중 파일 시각화 (새로운 기능)
    # =================================================================================
    with tab3:
        st.header("📊 다중 파일 시각화")
        st.markdown("여러 개의 FTR/Feather 또는 H5 파일을 로드하여 동일한 특징들을 비교 시각화합니다.")

        # 다중 파일 업로드 섹션
        st.subheader("📁 다중 파일 업로드")
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("**여러 개의 FTR/Feather 또는 H5 파일을 업로드하세요:**")
            multi_uploaded_files = st.file_uploader(
                "시각화할 FTR/Feather 또는 H5 파일들을 선택하세요",
                type=['ftr', 'feather', 'h5', 'hdf5'],
                accept_multiple_files=True,
                key="multi_file_uploader"
            )
        
        with col2:
            if multi_uploaded_files:
                if st.button("📤 다중 파일 업로드 처리", key="multi_upload_btn"):
                    handle_multi_file_upload(multi_uploaded_files)
        
        # 다중 파일이 업로드된 경우 시각화 시작
        if 'multi_uploaded_files' in st.session_state and st.session_state.multi_uploaded_files:
            multi_files = st.session_state.multi_uploaded_files
            
            st.success(f"✅ {len(multi_files)}개 파일이 다중 업로드되었습니다!")
            
            # 첫 번째 파일을 기준으로 특징 목록 확인
            first_file = multi_files[0]
            reference_df = load_feather_file(first_file)
            
            if reference_df is not None:
                st.subheader("📊 기준 파일 정보")
                st.info(f"**기준 파일**: {first_file.name} (Shape: {reference_df.shape})")
                
                # 업로드된 파일 목록 표시
                with st.expander("📋 업로드된 파일 목록"):
                    for i, file in enumerate(multi_files):
                        try:
                            temp_df = load_feather_file(file)
                            if temp_df is not None:
                                st.write(f"{i+1}. **{file.name}** - Shape: {temp_df.shape}")
                            else:
                                st.write(f"{i+1}. **{file.name}** - ❌ 로드 실패")
                        except:
                            st.write(f"{i+1}. **{file.name}** - ❌ 로드 실패")
                
                # 특징 선택
                st.subheader("🎯 시각화할 특징 선택")
                multi_selected_features = st.multiselect(
                    "비교할 특징들을 선택하세요",
                    reference_df.columns.tolist(),
                    default=[reference_df.columns[0]] if len(reference_df.columns) > 0 else [],
                    key="multi_feature_selection",
                    help="선택된 특징들이 선택된 파일들에서 비교 시각화됩니다."
                )
                
                # 플롯할 파일 선택 추가
                st.subheader("📂 플롯할 파일 선택")
                file_names = [f.name for f in multi_files]
                selected_file_indices = st.multiselect(
                    "플롯에 포함할 파일들을 선택하세요",
                    range(len(multi_files)),
                    format_func=lambda x: file_names[x],
                    default=[0] if len(multi_files) > 0 else [], 
                    key="multi_file_selection",
                    help="선택된 파일들만 플롯에 표시됩니다."
                )
                
                # 선택된 파일들 가져오기
                selected_files = [multi_files[i] for i in selected_file_indices]
                
                if multi_selected_features and selected_files:
                    # 시각화 설정 (탭1과 동일한 구조)
                    st.subheader("⚙️ 시각화 설정")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        multi_downsample_rate = st.slider(
                            "📉 다운샘플 비율 (1/N)", 
                            min_value=1, max_value=100, value=10,
                            key="multi_downsample"
                        )
                    with col2:
                        multi_num_segments = st.selectbox(
                            "📊 데이터 분할 수",
                            options=[1, 2, 3, 4, 5],
                            index=2,  # 기본값: 3등분
                            help="전체 데이터를 몇 등분할지 선택",
                            key="multi_segments"
                        )
                    with col3:
                        multi_selected_segment = st.selectbox(
                            "🎯 분석 구간 선택",
                            options=list(range(multi_num_segments)),
                            format_func=lambda x: f"구간 {x+1}",
                            index=0,  # 기본값: 첫 번째 구간
                            help="분석할 구간을 선택",
                            key="multi_segment_select"
                        )
                    
                    # 데이터 구간 정보 표시 (기준 파일 기준)
                    total_length = len(reference_df)
                    segment_length = total_length // multi_num_segments
                    start_idx = multi_selected_segment * segment_length
                    end_idx = start_idx + segment_length if multi_selected_segment < multi_num_segments - 1 else total_length
                    
                    st.info(f"📊 **선택된 구간**: {start_idx:,} ~ {end_idx:,} (총 {end_idx - start_idx:,}개 포인트, 전체의 {((end_idx - start_idx) / total_length * 100):.1f}%)")
                    
                    multi_crosshair = st.checkbox("▶️ 십자선 Hover 활성화", value=True, key="multi_crosshair")
                    
                    # 다중 파일 시각화 생성
                    st.subheader("📈 다중 파일 특징 비교")
                    
                    try:
                        # 다중 파일 플롯 생성 (선택된 파일들만)
                        multi_fig = create_multi_file_plot(
                            selected_files,
                            multi_selected_features,
                            multi_downsample_rate,
                            multi_crosshair,
                            multi_num_segments,
                            multi_selected_segment
                        )
                        
                        st.plotly_chart(multi_fig, use_container_width=True)
                        
                        # 추가 정보 표시
                        st.subheader("📋 시각화 요약")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**시각화된 파일:**")
                            for i, file in enumerate(selected_files):
                                st.write(f"{i+1}. {file.name}")
                        
                        with col2:
                            st.markdown("**시각화된 특징:**")
                            for feature in multi_selected_features:
                                st.write(f"• {feature}")
                        
                        # 데이터 특성 분석 (선택된 파일들만)
                        with st.expander("📊 파일별 데이터 특성 비교"):
                            comparison_data = []
                            
                            for file in selected_files:
                                try:
                                    df = load_feather_file(file)
                                    if df is not None:
                                        # 선택된 구간에서 통계 계산
                                        df_segment = get_data_segment(df, multi_num_segments, multi_selected_segment)
                                        
                                        for feature in multi_selected_features:
                                            if feature in df.columns:
                                                feature_data = df_segment[feature]
                                                comparison_data.append({
                                                    '파일명': file.name,
                                                    '특징': feature,
                                                    '평균': f"{feature_data.mean():.4f}",
                                                    '표준편차': f"{feature_data.std():.4f}",
                                                    '최소값': f"{feature_data.min():.4f}",
                                                    '최대값': f"{feature_data.max():.4f}",
                                                    '데이터 포인트': f"{len(feature_data):,}",
                                                    '결측값': int(feature_data.isna().sum())
                                                })
                                except Exception as e:
                                    st.warning(f"⚠️ {file.name} 통계 계산 중 오류: {str(e)}")
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True)
                            else:
                                st.warning("⚠️ 비교할 데이터가 없습니다.")
                        
                    except Exception as e:
                        st.error(f"❌ 다중 파일 시각화 생성 중 오류: {str(e)}")
                        
                elif not multi_selected_features:
                    st.info("🎯 시각화할 특징을 선택해주세요.")
                elif not selected_files:
                    st.info("📂 플롯할 파일을 선택해주세요.")
            else:
                st.error("❌ 기준 파일을 로드할 수 없습니다.")
        else:
            st.info("📁 다중 파일을 업로드하여 시각화를 시작하세요.")
        


        # =================================================================================
        # DNN 학습 데이터 추출 기능 추가
        # =================================================================================
        if 'multi_uploaded_files' in st.session_state and st.session_state.multi_uploaded_files:
            st.markdown("---")
            st.header("🤖 DNN 학습 데이터 추출")
            st.markdown("업로드된 FTR 파일들로부터 DNN 학습용 시계열 데이터를 추출합니다.")
            
            # 데이터 추출 파라미터 설정
            st.subheader("⚙️ 데이터 추출 설정")
            
            # 기본 파라미터
            col1, col2 = st.columns(2)
            with col1:
                start_position = st.number_input(
                    "🎯 시작 위치 (틱)",
                    min_value=0,
                    max_value=100000,
                    value=300,
                    step=1,
                    help="데이터 추출을 시작할 위치 (0부터 시작)",
                    key="dnn_start_pos"
                )
                
                lookback_length = st.number_input(
                    "📈 과거 참조 길이 (틱)",
                    min_value=1,
                    max_value=1000,
                    value=60,
                    step=1,
                    help="각 시점에서 과거 몇 틱을 입력으로 사용할지",
                    key="dnn_lookback"
                )
            
            with col2:
                end_position = st.number_input(
                    "🏁 종료 위치 (틱)",
                    min_value=start_position + 100,
                    max_value=100000,
                    value=start_position + 1700,
                    step=1,
                    help="데이터 추출을 종료할 위치",
                    key="dnn_end_pos"
                )
                
                horizon_length = st.number_input(
                    "🔮 예측 구간 길이 (틱)",
                    min_value=1,
                    max_value=100,
                    value=24,
                    step=1,
                    help="미래 몇 틱을 예측 대상으로 할지",
                    key="dnn_horizon"
                )
            
            # 추가 파라미터
            col3, col4 = st.columns(2)
            with col3:
                step_gap = st.number_input(
                    "⏭️ 스텝 간격",
                    min_value=1,
                    max_value=50,
                    value=2,
                    step=1,
                    help="샘플 추출 시 몇 틱씩 건너뛸지",
                    key="dnn_step_gap"
                )
            
            with col4:
                train_ratio = st.slider(
                    "🎓 훈련/검증 비율",
                    min_value=0.5,
                    max_value=0.95,
                    value=0.8,
                    step=0.05,
                    help="훈련용 파일의 비율 (나머지는 검증용)",
                    key="dnn_train_ratio"
                )
            
            # 시간 정보 설정
            st.subheader("🕐 시간 특징 설정")
            col5, col6 = st.columns(2)
            with col5:
                use_positional_encoding = st.checkbox(
                    "Positional Encoding 사용",
                    value=True,
                    help="시간 정보에 positional encoding 추가",
                    key="dnn_pos_encoding"
                )
            
            with col6:
                tick_interval = st.number_input(
                    "틱 간격 (초)",
                    min_value=1,
                    max_value=60,
                    value=5,
                    step=1,
                    help="각 틱 간의 시간 간격",
                    key="dnn_tick_interval"
                )
            
            # 파라미터 요약 표시
            st.subheader("📋 추출 설정 요약")
            with st.expander("🔍 상세 설정 확인"):
                summary_data = {
                    '파라미터': [
                        '시작 위치', '종료 위치', '과거 참조 길이', '예측 구간 길이',
                        '스텝 간격', '훈련 비율', '검증 비율', 'Positional Encoding',
                        '틱 간격', '총 업로드 파일 수'
                    ],
                    '값': [
                        f"{start_position:,}",
                        f"{end_position:,}",
                        f"{lookback_length}",
                        f"{horizon_length}",
                        f"{step_gap}",
                        f"{train_ratio:.1%}",
                        f"{1-train_ratio:.1%}",
                        "사용" if use_positional_encoding else "미사용",
                        f"{tick_interval}초",
                        f"{len(st.session_state.multi_uploaded_files)}개"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # 예상 샘플 수 계산
                total_samples_per_file = (end_position - start_position - lookback_length - horizon_length) // step_gap
                if total_samples_per_file > 0:
                    estimated_train_samples = total_samples_per_file * int(len(st.session_state.multi_uploaded_files) * train_ratio)
                    estimated_val_samples = total_samples_per_file * (len(st.session_state.multi_uploaded_files) - int(len(st.session_state.multi_uploaded_files) * train_ratio))
                    
                    st.info(f"📊 **예상 샘플 수**: 훈련용 ~{estimated_train_samples:,}개, 검증용 ~{estimated_val_samples:,}개")
                else:
                    st.warning("⚠️ 현재 설정으로는 샘플을 추출할 수 없습니다. 파라미터를 조정해주세요.")
            
            # DNN 데이터 추출 실행
            st.subheader("🚀 데이터 추출 실행")
            
            # 파일명 설정
            default_dataset_name = f"dnn_dataset_{lookback_length}to{horizon_length}_{len(st.session_state.multi_uploaded_files)}files"
            dataset_filename = st.text_input(
                "데이터셋 파일명 (확장자 제외)",
                value=default_dataset_name,
                help="생성될 데이터셋 파일의 이름",
                key="dnn_dataset_filename"
            )
            
            # 추출 버튼
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🤖 DNN 데이터 추출 시작", key="start_dnn_extraction"):
                    if total_samples_per_file <= 0:
                        st.error("❌ 현재 설정으로는 샘플을 추출할 수 없습니다.")
                    else:
                        st.session_state.dnn_extraction_done = False
                        
                        with st.spinner("🔄 DNN 학습 데이터 추출 중... 잠시만 기다려주세요."):
                            try:
                                # 파일들을 훈련/검증용으로 분할
                                train_files, val_files = split_files_train_val(
                                    st.session_state.multi_uploaded_files, train_ratio
                                )
                                
                                st.write(f"📂 **파일 분할 완료**: 훈련용 {len(train_files)}개, 검증용 {len(val_files)}개")
                                
                                # 모든 파일에서 데이터 추출
                                dataset = process_all_files_for_dnn(
                                    train_files, val_files,
                                    start_position, end_position,
                                    lookback_length, horizon_length, step_gap
                                )
                                
                                if len(dataset['train_inputs']) > 0 or len(dataset['val_inputs']) > 0:
                                    # 메타데이터 생성
                                    metadata = {
                                        'extraction_params': {
                                            'start_position': start_position,
                                            'end_position': end_position,
                                            'lookback_length': str(lookback_length),
                                            'horizon_length': str(horizon_length),
                                            'step_gap': str(step_gap),
                                            'train_ratio': train_ratio,
                                            'use_positional_encoding': use_positional_encoding,
                                            'tick_interval': tick_interval
                                        },
                                        'data_info': {
                                            'train_samples': len(dataset['train_inputs']),
                                            'val_samples': len(dataset['val_inputs']),
                                            'input_shape': dataset['train_inputs'].shape if len(dataset['train_inputs']) > 0 else None,
                                            'output_shape': dataset['train_outputs'].shape if len(dataset['train_outputs']) > 0 else None,
                                            'train_files': [f.name for f in train_files],
                                            'val_files': [f.name for f in val_files],
                                            'total_files': len(st.session_state.multi_uploaded_files)
                                        },
                                        'creation_time': datetime.now().isoformat(),
                                        'feature_info': {
                                            'time_features': 3 + (8 if use_positional_encoding else 0),
                                            'time_feature_names': ['hour_norm', 'minute_norm', 'second_norm'] + 
                                                                ([f'pos_enc_{i}' for i in range(8)] if use_positional_encoding else []),
                                            'data_features': len(reference_df.columns) - 1,  # timestamp 제외
                                            'data_feature_names': [col for col in reference_df.columns if col != 
                                                                (reference_df.columns[0] if 'time' not in reference_df.columns[0].lower() 
                                                                and 'timestamp' not in reference_df.columns[0].lower() 
                                                                else next((col for col in reference_df.columns 
                                                                            if 'time' in col.lower() or 'timestamp' in col.lower()), 
                                                                        reference_df.columns[0]))],
                                            'total_features': len(dataset['train_inputs'].shape) > 2 and dataset['train_inputs'].shape[2] or 0,
                                            'feature_order': 'time_features_first_then_data_features'
                                        }
                                    }

                                    # 세션에 저장 (수정된 부분)
                                    st.session_state.dnn_dataset = dataset
                                    st.session_state.dnn_metadata = metadata
                                    st.session_state.dnn_extraction_done = True
                                    st.session_state.dnn_dataset_name = dataset_filename  # filename을 name으로 변경
                                    
                                else:
                                    st.error("❌ 추출된 데이터가 없습니다. 파라미터를 확인해주세요.")
                                    
                            except Exception as e:
                                st.error(f"❌ 데이터 추출 중 오류 발생: {str(e)}")
            
            # DNN 데이터 추출 결과 표시
            if (hasattr(st.session_state, 'dnn_extraction_done') and 
                st.session_state.dnn_extraction_done and 
                hasattr(st.session_state, 'dnn_dataset')):
                
                st.markdown("---")
                st.subheader("✅ DNN 데이터 추출 완료")
                
                dataset = st.session_state.dnn_dataset
                metadata = st.session_state.dnn_metadata
                
                # 추출 결과 요약
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎓 훈련 샘플", f"{len(dataset['train_inputs']):,}")
                with col2:
                    st.metric("🔬 검증 샘플", f"{len(dataset['val_inputs']):,}")
                with col3:
                    st.metric("📊 총 샘플", f"{len(dataset['train_inputs']) + len(dataset['val_inputs']):,}")
                
                # 데이터 형태 정보
                with st.expander("📊 데이터 형태 정보"):
                    if len(dataset['train_inputs']) > 0:
                        st.write(f"**훈련 입력 형태**: {dataset['train_inputs'].shape}")
                        st.write(f"**훈련 출력 형태**: {dataset['train_outputs'].shape}")
                        
                    if len(dataset['val_inputs']) > 0:
                        st.write(f"**검증 입력 형태**: {dataset['val_inputs'].shape}")
                        st.write(f"**검증 출력 형태**: {dataset['val_outputs'].shape}")
                    
                    st.write(f"**시간 특징 수**: {metadata['feature_info']['time_features']}")
                    st.write(f"**전체 특징 수**: {dataset['train_inputs'].shape[-1] if len(dataset['train_inputs']) > 0 else 'N/A'}")
                
                # 파일별 샘플 수 정보
                with st.expander("📁 파일별 샘플 정보"):
                    # 훈련 파일 정보
                    st.markdown("**훈련용 파일:**")
                    train_file_counts = {}
                    for info in dataset['train_info']:
                        file_name = info['file_name']
                        train_file_counts[file_name] = train_file_counts.get(file_name, 0) + 1
                    
                    for file_name, count in train_file_counts.items():
                        st.write(f"  • {file_name}: {count:,}개 샘플")
                    
                    # 검증 파일 정보
                    st.markdown("**검증용 파일:**")
                    val_file_counts = {}
                    for info in dataset['val_info']:
                        file_name = info['file_name']
                        val_file_counts[file_name] = val_file_counts.get(file_name, 0) + 1
                    
                    for file_name, count in val_file_counts.items():
                        st.write(f"  • {file_name}: {count:,}개 샘플")
                

                # 데이터셋 다운로드 (수정된 부분)
                st.subheader("💾 DNN 데이터셋 다운로드")

                if st.button("📦 데이터셋 파일 생성", key="generate_dnn_dataset"):
                    try:
                        with st.spinner("📦 데이터셋 파일 생성 중..."):
                            # 위젯에서 현재 값 가져오기 (수정된 부분)
                            current_filename = st.session_state.get('dnn_dataset_filename', 'dnn_dataset')
                            dataset_data = save_dnn_dataset(
                                dataset, metadata, current_filename
                            )
                        
                        st.download_button(
                            label="💾 DNN 데이터셋 다운로드",
                            data=dataset_data,
                            file_name=f"{current_filename}.npy",  # 수정된 부분
                            mime="application/octet-stream",
                            help="DNN 학습용 데이터셋을 numpy 형식으로 다운로드"
                        )
                        
                        st.success("✅ 데이터셋 파일이 생성되었습니다! 다운로드 버튼을 클릭하세요.")
                        
                        # 사용 예시 코드 표시 (NPY 형식에 맞게 수정)
                        with st.expander("🐍 Python 사용 예시 코드"):
                            st.code(f"""
                import numpy as np

                # 데이터셋 로드
                dataset = np.load('{current_filename}.npy', allow_pickle=True).item()  # 수정된 부분

                # 데이터 접근
                train_inputs = dataset['train_inputs']    # Shape: (samples, lookback, features)
                train_outputs = dataset['train_outputs']  # Shape: (samples, horizon, features)
                val_inputs = dataset['val_inputs']        # Shape: (samples, lookback, features)
                val_outputs = dataset['val_outputs']      # Shape: (samples, horizon, features)

                # 메타데이터 확인
                metadata = dataset['metadata']
                print("추출 파라미터:", metadata['extraction_params'])
                print("데이터 정보:", metadata['data_info'])

                # 샘플 정보
                train_info = dataset['train_info']  # 각 샘플의 상세 정보
                val_info = dataset['val_info']      # 각 샘플의 상세 정보

                print(f"훈련 샘플: {{train_inputs.shape[0]:,}}개")
                print(f"검증 샘플: {{val_inputs.shape[0]:,}}개")
                print(f"입력 형태: {{train_inputs.shape}}")
                print(f"출력 형태: {{train_outputs.shape}}")

                # PyTorch에서 사용 예시
                # import torch
                # train_dataset = torch.utils.data.TensorDataset(
                #     torch.FloatTensor(train_inputs), 
                #     torch.FloatTensor(train_outputs)
                # )

                # 개별 파일로 저장하고 싶은 경우
                # np.save('train_inputs.npy', train_inputs)
                # np.save('train_outputs.npy', train_outputs)
                # np.save('val_inputs.npy', val_inputs)
                # np.save('val_outputs.npy', val_outputs)
                # np.save('metadata.npy', metadata)
                """, language="python")
                        
                    except Exception as e:
                        st.error(f"❌ 데이터셋 생성 중 오류: {str(e)}")
                

                # 데이터 시각화 옵션 (수정된 부분)
                with st.expander("📈 샘플 데이터 미리보기"):
                    if len(dataset['train_inputs']) > 0:
                        sample_idx = st.selectbox(
                            "미리볼 샘플 선택",
                            range(min(10, len(dataset['train_inputs']))),
                            key="sample_preview_idx"
                        )
                        
                        # 전체 특징 수 확인
                        total_features = dataset['train_inputs'].shape[2]
                        
                        # 메타데이터에서 특징 이름 가져오기
                        feature_names = []
                        if 'feature_info' in metadata:
                            time_feature_names = metadata['feature_info'].get('time_feature_names', [])
                            data_feature_names = metadata['feature_info'].get('data_feature_names', [])
                            feature_names = time_feature_names + data_feature_names
                        
                        # 특징 이름이 없으면 기본 이름 사용
                        if len(feature_names) != total_features:
                            feature_names = [f"Feature {i+1}" for i in range(total_features)]
                        
                        # 시각화할 특징 선택 (최대 10개)
                        max_features_to_show = min(10, total_features)
                        selected_feature_indices = st.multiselect(
                            f"시각화할 특징 선택 (전체 {total_features}개 중 최대 {max_features_to_show}개)",
                            range(total_features),
                            default=list(range(min(5, total_features))),  # 기본값: 처음 5개 특징
                            format_func=lambda x: f"{x+1}: {feature_names[x]}" if x < len(feature_names) else f"Feature {x+1}",
                            key="preview_feature_selection"
                        )
                        
                        if selected_feature_indices:
                            # 선택된 특징이 최대 개수를 초과하지 않도록 제한
                            if len(selected_feature_indices) > max_features_to_show:
                                st.warning(f"⚠️ 최대 {max_features_to_show}개 특징만 선택 가능합니다. 처음 {max_features_to_show}개만 표시됩니다.")
                                selected_feature_indices = selected_feature_indices[:max_features_to_show]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**입력 시퀀스 (Input)**")
                                input_sample = dataset['train_inputs'][sample_idx]
                                st.write(f"형태: {input_sample.shape}")
                                
                                # 입력 데이터 시각화 (선택된 특징만)
                                fig_input = go.Figure()
                                for i, feature_idx in enumerate(selected_feature_indices):
                                    feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature {feature_idx+1}'
                                    fig_input.add_trace(go.Scatter(
                                        y=input_sample[:, feature_idx],
                                        mode='lines+markers',
                                        name=feature_name,
                                        line=dict(width=2)
                                    ))
                                
                                fig_input.update_layout(
                                    title=f"입력 시퀀스 (선택된 {len(selected_feature_indices)}개 특징)",
                                    xaxis_title="Time Steps",
                                    yaxis_title="Feature Values",
                                    height=300
                                )
                                st.plotly_chart(fig_input, use_container_width=True)
                            
                            with col2:
                                st.markdown("**출력 시퀀스 (Target)**")
                                output_sample = dataset['train_outputs'][sample_idx]
                                st.write(f"형태: {output_sample.shape}")
                                
                                # 출력 데이터 시각화 (선택된 특징만)
                                fig_output = go.Figure()
                                for i, feature_idx in enumerate(selected_feature_indices):
                                    feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature {feature_idx+1}'
                                    fig_output.add_trace(go.Scatter(
                                        y=output_sample[:, feature_idx],
                                        mode='lines+markers',
                                        name=feature_name,
                                        line=dict(width=2)
                                    ))
                                
                                fig_output.update_layout(
                                    title=f"출력 시퀀스 (선택된 {len(selected_feature_indices)}개 특징)",
                                    xaxis_title="Time Steps", 
                                    yaxis_title="Feature Values",
                                    height=300
                                )
                                st.plotly_chart(fig_output, use_container_width=True)
                            
                            # 샘플 정보 표시
                            sample_info = dataset['train_info'][sample_idx]
                            st.json(sample_info)
                            
                            # 선택된 특징들의 통계 정보 (expander 제거)
                            st.markdown("**📊 선택된 특징들의 통계 정보**")
                            stats_data = []
                            for feature_idx in selected_feature_indices:
                                input_feature_data = input_sample[:, feature_idx]
                                output_feature_data = output_sample[:, feature_idx]
                                feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature {feature_idx+1}'
                                
                                stats_data.append({
                                    '특징': feature_name,
                                    '입력 평균': f"{input_feature_data.mean():.4f}",
                                    '입력 표준편차': f"{input_feature_data.std():.4f}",
                                    '출력 평균': f"{output_feature_data.mean():.4f}",
                                    '출력 표준편차': f"{output_feature_data.std():.4f}",
                                    '입력 범위': f"{input_feature_data.min():.4f} ~ {input_feature_data.max():.4f}",
                                    '출력 범위': f"{output_feature_data.min():.4f} ~ {output_feature_data.max():.4f}"
                                })
                            
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df, use_container_width=True)
                            
                            # 메타데이터 정보 표시 (새로 추가된 부분)
                            st.markdown("---")
                            st.markdown("**📋 데이터셋 메타데이터 정보**")
                            
                            # 메타데이터를 보기 좋게 정리
                            meta_col1, meta_col2 = st.columns(2)
                            
                            with meta_col1:
                                st.markdown("**🔧 추출 파라미터**")
                                extraction_params = metadata.get('extraction_params', {})
                                param_df = pd.DataFrame([
                                    {'파라미터': '시작 위치', '값': f"{extraction_params.get('start_position', 'N/A'):,}"},
                                    {'파라미터': '종료 위치', '값': f"{extraction_params.get('end_position', 'N/A'):,}"},
                                    {'파라미터': '과거 참조 길이', '값': str(extraction_params.get('lookback_length', 'N/A'))},
                                    {'파라미터': '예측 구간 길이', '값': str(extraction_params.get('horizon_length', 'N/A'))},
                                    {'파라미터': '스텝 간격', '값': str(extraction_params.get('step_gap', 'N/A'))},
                                    {'파라미터': '훈련 비율', '값': f"{extraction_params.get('train_ratio', 0):.1%}"},
                                    {'파라미터': 'Positional Encoding', '값': '사용' if extraction_params.get('use_positional_encoding', False) else '미사용'},
                                    {'파라미터': '틱 간격', '값': f"{extraction_params.get('tick_interval', 'N/A')}초"}
                                ])
                                st.dataframe(param_df, use_container_width=True, hide_index=True)
                            
                            with meta_col2:
                                st.markdown("**📊 데이터 정보**")
                                data_info = metadata.get('data_info', {})
                                feature_info = metadata.get('feature_info', {})
                                info_df = pd.DataFrame([
                                    {'항목': '훈련 샘플 수', '값': f"{data_info.get('train_samples', 0):,}"},
                                    {'항목': '검증 샘플 수', '값': f"{data_info.get('val_samples', 0):,}"},
                                    {'항목': '입력 형태', '값': str(data_info.get('input_shape', 'N/A'))},
                                    {'항목': '출력 형태', '값': str(data_info.get('output_shape', 'N/A'))},
                                    {'항목': '시간 특징 수', '값': str(feature_info.get('time_features', 'N/A'))},
                                    {'항목': '데이터 특징 수', '값': str(feature_info.get('data_features', 'N/A'))},
                                    {'항목': '전체 특징 수', '값': str(feature_info.get('total_features', 'N/A'))},
                                    {'항목': '생성 시간', '값': metadata.get('creation_time', 'N/A')[:19] if metadata.get('creation_time') else 'N/A'}
                                ])
                                st.dataframe(info_df, use_container_width=True, hide_index=True)
                            
                            # 특징 이름 매핑 표시
                            if 'feature_info' in metadata and len(feature_names) == total_features:
                                st.markdown("**🏷️ 특징 이름 매핑**")
                                
                                # 시간 특징과 데이터 특징을 분리하여 표시
                                time_features_count = metadata['feature_info'].get('time_features', 0)
                                
                                feature_mapping = []
                                for i, name in enumerate(feature_names):
                                    feature_type = "시간 특징" if i < time_features_count else "데이터 특징"
                                    feature_mapping.append({
                                        '인덱스': i,
                                        '특징명': name,
                                        '타입': feature_type
                                    })
                                
                                mapping_df = pd.DataFrame(feature_mapping)
                                st.dataframe(mapping_df, use_container_width=True, hide_index=True)
                            
                            # 파일 정보
                            if 'data_info' in metadata:
                                train_files = metadata['data_info'].get('train_files', [])
                                val_files = metadata['data_info'].get('val_files', [])
                                
                                if train_files or val_files:
                                    st.markdown("**📁 사용된 파일 정보**")
                                    file_col1, file_col2 = st.columns(2)
                                    
                                    with file_col1:
                                        if train_files:
                                            st.markdown("*훈련용 파일:*")
                                            for i, file_name in enumerate(train_files, 1):
                                                st.write(f"{i}. {file_name}")
                                    
                                    with file_col2:
                                        if val_files:
                                            st.markdown("*검증용 파일:*")
                                            for i, file_name in enumerate(val_files, 1):
                                                st.write(f"{i}. {file_name}")
                        
                        else:
                            st.info("🎯 시각화할 특징을 선택해주세요.")
                    else:
                        st.info("📊 추출된 훈련 데이터가 없습니다.")


        

    # =================================================================================
    # 탭 4: 유사 기동 검색 (새로운 기능)
    # =================================================================================
    with tab4:
        st.header("🎯 유사 기동 검색")
        st.markdown("기준 파일의 특정 온도 조건과 유사한 기동 패턴을 다른 파일들에서 검색합니다.")

        # 다중 파일 업로드 섹션 (탭3과 동일)
        st.subheader("📁 다중 파일 업로드")
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("**여러 개의 FTR/Feather 또는 H5 파일을 업로드하세요:**")
            search_uploaded_files = st.file_uploader(
                "유사 기동 검색할 FTR/Feather 또는 H5 파일들을 선택하세요",
                type=['ftr', 'feather', 'h5', 'hdf5'],
                accept_multiple_files=True,
                key="search_file_uploader"
            )
        
        with col2:
            if search_uploaded_files:
                if st.button("📤 검색용 파일 업로드 처리", key="search_upload_btn"):
                    st.session_state.search_uploaded_files = search_uploaded_files
                    st.success(f"✅ {len(search_uploaded_files)}개 파일이 검색용으로 업로드되었습니다!")
        
        # 검색용 파일이 업로드된 경우 검색 시작
        if 'search_uploaded_files' in st.session_state and st.session_state.search_uploaded_files:
            search_files = st.session_state.search_uploaded_files
            
            st.success(f"✅ {len(search_files)}개 파일이 검색용으로 업로드되었습니다!")
            
            # 첫 번째 파일을 기준으로 특징 목록 확인
            first_file = search_files[0]
            reference_df = load_feather_file(first_file)
            
            if reference_df is not None:
                st.subheader("📊 기준 파일 정보")
                st.info(f"**기준 파일**: {first_file.name} (Shape: {reference_df.shape})")
                
                # 업로드된 파일 목록 표시
                with st.expander("📋 업로드된 파일 목록"):
                    for i, file in enumerate(search_files):
                        try:
                            temp_df = load_feather_file(file)
                            if temp_df is not None:
                                st.write(f"{i+1}. **{file.name}** - Shape: {temp_df.shape}")
                            else:
                                st.write(f"{i+1}. **{file.name}** - ❌ 로드 실패")
                        except:
                            st.write(f"{i+1}. **{file.name}** - ❌ 로드 실패")
                
                # 기준 파일 선택
                st.subheader("📂 기준 파일 선택")
                file_names = [f.name for f in search_files]
                selected_reference_file_index = st.selectbox(
                    "기준이 될 파일을 선택하세요",
                    range(len(search_files)),
                    format_func=lambda x: file_names[x],
                    index=0,  # 기본값: 첫 번째 파일
                    key="reference_file_selection",
                    help="선택된 파일의 tic=80 온도값을 기준으로 다른 모든 파일과 비교합니다."
                )
                
                # 기준 파일과 검색 대상 파일들 설정
                reference_file = search_files[selected_reference_file_index]
                search_target_files = [f for i, f in enumerate(search_files) if i != selected_reference_file_index]
                
                if len(search_target_files) >= 1:  # 최소 1개 이상의 검색 대상 파일 필요
                    
                    st.subheader("🎯 기준 온도 조건 설정")
                    st.info(f"🎯 **기준 파일**: {reference_file.name}")
                    st.info(f"🔍 **검색 대상**: {len(search_target_files)}개 파일 (기준 파일 제외한 모든 파일)")
                    
                    # 기준 파일 로드
                    ref_df = load_feather_file(reference_file)
                    
                    if ref_df is not None:
                        # 필요한 온도 특징들이 존재하는지 확인
                        required_features = ['metal_temp_1st', 'scr_outlet_temp', 'exhaust_gas_temperature']
                        missing_features = [feat for feat in required_features if feat not in ref_df.columns]
                        
                        if missing_features:
                            st.error(f"❌ 기준 파일에서 필요한 특징이 누락되었습니다: {missing_features}")
                            st.write(f"**사용 가능한 컬럼**: {list(ref_df.columns)}")
                        else:
                            # tic=80에서 온도값 추출
                            if len(ref_df) > 80:
                                reference_temps = {
                                    'metal_temp_1st': ref_df.loc[80, 'metal_temp_1st'],
                                    'scr_outlet_temp': ref_df.loc[80, 'scr_outlet_temp'],
                                    'exhaust_gas_temperature': ref_df.loc[80, 'exhaust_gas_temperature']
                                }
                                
                                # 기준 온도값 표시
                                st.subheader("🌡️ 기준 온도값 (tic=80)")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Metal Temp 1st", f"{reference_temps['metal_temp_1st']:.2f}°C")
                                with col2:
                                    st.metric("SCR Outlet Temp", f"{reference_temps['scr_outlet_temp']:.2f}°C")
                                with col3:
                                    st.metric("Exhaust Gas Temp", f"{reference_temps['exhaust_gas_temperature']:.2f}°C")
                                
                                # 가중치 설정
                                st.subheader("⚖️ 온도별 가중치 설정")
                                st.markdown("각 온도 특징의 중요도를 설정하세요. 높은 값일수록 해당 온도의 유사성이 더 중요하게 고려됩니다.")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    weight_metal = st.slider(
                                        "Metal Temp 1st 가중치",
                                        min_value=0.0,
                                        max_value=2.0,
                                        value=1.0,
                                        step=0.1,
                                        key="weight_metal"
                                    )
                                with col2:
                                    weight_scr = st.slider(
                                        "SCR Outlet Temp 가중치",
                                        min_value=0.0,
                                        max_value=2.0,
                                        value=1.0,
                                        step=0.1,
                                        key="weight_scr"
                                    )
                                with col3:
                                    weight_exhaust = st.slider(
                                        "Exhaust Gas Temp 가중치",
                                        min_value=0.0,
                                        max_value=2.0,
                                        value=1.0,
                                        step=0.1,
                                        key="weight_exhaust"
                                    )
                                
                                # 가중치 정규화 옵션
                                normalize_weights = st.checkbox(
                                    "가중치 정규화",
                                    value=True,
                                    help="가중치의 합이 1이 되도록 정규화합니다."
                                )
                                
                                # 유사 기동 검색 실행
                                st.subheader("🔍 유사 기동 검색")
                                
                                if st.button("🚀 유사 기동 검색 시작", key="start_similarity_search"):
                                    with st.spinner("🔄 유사 기동 검색 중... 잠시만 기다려주세요."):
                                        try:
                                            # 가중치 설정
                                            weights = np.array([weight_metal, weight_scr, weight_exhaust])
                                            if normalize_weights and weights.sum() > 0:
                                                weights = weights / weights.sum()
                                            
                                            # 기준 온도 벡터
                                            reference_vector = np.array([
                                                reference_temps['metal_temp_1st'],
                                                reference_temps['scr_outlet_temp'],
                                                reference_temps['exhaust_gas_temperature']
                                            ])
                                            
                                            # 각 파일에서 tic=80의 온도값 추출 및 거리 계산
                                            similarity_results = []
                                            
                                            for target_file in search_target_files:
                                                try:
                                                    target_df = load_feather_file(target_file)
                                                    if target_df is not None and len(target_df) > 80:
                                                        # 필요한 특징들이 존재하는지 확인
                                                        target_missing = [feat for feat in required_features if feat not in target_df.columns]
                                                        if not target_missing:
                                                            # tic=80에서 온도값 추출
                                                            target_temps = np.array([
                                                                target_df.loc[80, 'metal_temp_1st'],
                                                                target_df.loc[80, 'scr_outlet_temp'],
                                                                target_df.loc[80, 'exhaust_gas_temperature']
                                                            ])
                                                            
                                                            # 가중 유클리드 거리 계산
                                                            weighted_diff = weights * (reference_vector - target_temps)
                                                            euclidean_distance = np.sqrt(np.sum(weighted_diff ** 2))
                                                            
                                                            similarity_results.append({
                                                                'file_name': target_file.name,
                                                                'distance': euclidean_distance,
                                                                'metal_temp_1st': target_temps[0],
                                                                'scr_outlet_temp': target_temps[1],
                                                                'exhaust_gas_temperature': target_temps[2],
                                                                'file_object': target_file
                                                            })
                                                        else:
                                                            st.warning(f"⚠️ {target_file.name}에서 누락된 특징: {target_missing}")
                                                    else:
                                                        st.warning(f"⚠️ {target_file.name}: 데이터가 부족합니다 (tic=80 이상 필요)")
                                                        
                                                except Exception as e:
                                                    st.warning(f"⚠️ {target_file.name} 처리 중 오류: {str(e)}")
                                            
                                            # 거리순으로 정렬하여 상위 5개 선택
                                            if similarity_results:
                                                similarity_results.sort(key=lambda x: x['distance'])
                                                top_5_similar = similarity_results[:5]
                                                
                                                # 결과 저장
                                                st.session_state.similarity_results = similarity_results
                                                st.session_state.top_5_similar = top_5_similar
                                                st.session_state.reference_temps = reference_temps
                                                st.session_state.search_weights = weights
                                                st.session_state.search_completed = True
                                                
                                            else:
                                                st.error("❌ 검색 가능한 파일이 없습니다.")
                                                
                                        except Exception as e:
                                            st.error(f"❌ 유사 기동 검색 중 오류 발생: {str(e)}")
                                
                                # 검색 결과 표시
                                if (hasattr(st.session_state, 'search_completed') and 
                                    st.session_state.search_completed and 
                                    hasattr(st.session_state, 'top_5_similar')):
                                    
                                    st.markdown("---")
                                    st.subheader("🏆 유사 기동 검색 결과")
                                    
                                    top_5_similar = st.session_state.top_5_similar
                                    reference_temps = st.session_state.reference_temps
                                    search_weights = st.session_state.search_weights
                                    
                                    if top_5_similar:
                                        # 검색 설정 요약
                                        st.info(f"🎯 **기준 파일**: {reference_file.name} | **가중치**: Metal({search_weights[0]:.1f}), SCR({search_weights[1]:.1f}), Exhaust({search_weights[2]:.1f})")
                                        
                                        # 상위 5개 결과 표시
                                        st.markdown("### 🥇 가장 유사한 기동 TOP 5")
                                        
                                        for i, result in enumerate(top_5_similar, 1):
                                            with st.expander(f"🏅 {i}위: {result['file_name']} (거리: {result['distance']:.4f})"):
                                                col1, col2 = st.columns(2)
                                                
                                                with col1:
                                                    st.markdown("**🌡️ 온도 비교**")
                                                    comparison_data = {
                                                        '특징': ['Metal Temp 1st', 'SCR Outlet Temp', 'Exhaust Gas Temp'],
                                                        '기준값': [
                                                            f"{reference_temps['metal_temp_1st']:.2f}°C",
                                                            f"{reference_temps['scr_outlet_temp']:.2f}°C",
                                                            f"{reference_temps['exhaust_gas_temperature']:.2f}°C"
                                                        ],
                                                        '비교값': [
                                                            f"{result['metal_temp_1st']:.2f}°C",
                                                            f"{result['scr_outlet_temp']:.2f}°C",
                                                            f"{result['exhaust_gas_temperature']:.2f}°C"
                                                        ],
                                                        '차이': [
                                                            f"{abs(reference_temps['metal_temp_1st'] - result['metal_temp_1st']):.2f}°C",
                                                            f"{abs(reference_temps['scr_outlet_temp'] - result['scr_outlet_temp']):.2f}°C",
                                                            f"{abs(reference_temps['exhaust_gas_temperature'] - result['exhaust_gas_temperature']):.2f}°C"
                                                        ]
                                                    }
                                                    
                                                    comparison_df = pd.DataFrame(comparison_data)
                                                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                                                
                                                with col2:
                                                    st.markdown("**📊 상세 정보**")
                                                    st.write(f"**파일명**: {result['file_name']}")
                                                    st.write(f"**유클리드 거리**: {result['distance']:.6f}")
                                                    st.write(f"**순위**: {i}/5")
                                                    
                                                    # 각 온도별 가중 기여도
                                                    metal_contrib = search_weights[0] * abs(reference_temps['metal_temp_1st'] - result['metal_temp_1st'])
                                                    scr_contrib = search_weights[1] * abs(reference_temps['scr_outlet_temp'] - result['scr_outlet_temp'])
                                                    exhaust_contrib = search_weights[2] * abs(reference_temps['exhaust_gas_temperature'] - result['exhaust_gas_temperature'])
                                                    
                                                    st.write("**가중 기여도**:")
                                                    st.write(f"- Metal: {metal_contrib:.4f}")
                                                    st.write(f"- SCR: {scr_contrib:.4f}")
                                                    st.write(f"- Exhaust: {exhaust_contrib:.4f}")
                                        
                                        # 전체 결과 요약 테이블
                                        st.markdown("### 📋 검색 결과 요약")
                                        
                                        summary_data = []
                                        for i, result in enumerate(top_5_similar, 1):
                                            summary_data.append({
                                                '순위': i,
                                                '파일명': result['file_name'],
                                                '유클리드 거리': f"{result['distance']:.6f}",
                                                'Metal Temp': f"{result['metal_temp_1st']:.2f}°C",
                                                'SCR Temp': f"{result['scr_outlet_temp']:.2f}°C",
                                                'Exhaust Temp': f"{result['exhaust_gas_temperature']:.2f}°C"
                                            })
                                        
                                        summary_df = pd.DataFrame(summary_data)
                                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                                        
                                        # 시각화 비교 기능 추가
                                        st.markdown("### 📈 시계열 데이터 비교 시각화")
                                        
                                        # 기준 파일과 비교할 파일들 선택
                                        st.markdown("**기준 파일과 검색 결과 파일들의 시계열 데이터를 비교해보세요:**")
                                        
                                        # 비교할 파일들 선택 (기준 파일 + TOP 5 결과)
                                        available_files_for_plot = [reference_file] + [r['file_object'] for r in top_5_similar]
                                        available_file_names = [f"🎯 {reference_file.name} (기준)"] + [f"🏅 {i+1}위: {r['file_name']}" for i, r in enumerate(top_5_similar)]
                                        
                                        selected_plot_file_indices = st.multiselect(
                                            "비교 시각화할 파일들을 선택하세요",
                                            range(len(available_files_for_plot)),
                                            format_func=lambda x: available_file_names[x],
                                            default=[0, 1] if len(available_files_for_plot) > 1 else [0],  # 기본값: 기준 파일 + 1위
                                            key="similarity_plot_file_selection",
                                            help="선택된 파일들의 시계열 데이터를 함께 비교할 수 있습니다."
                                        )
                                        
                                        selected_plot_files = [available_files_for_plot[i] for i in selected_plot_file_indices]
                                        
                                        if len(selected_plot_files) >= 1:
                                            # 특징 선택 (기준 파일 기준)
                                            ref_df_for_plot = load_feather_file(reference_file)
                                            if ref_df_for_plot is not None:
                                                selected_plot_features = st.multiselect(
                                                    "비교할 특징들을 선택하세요",
                                                    ref_df_for_plot.columns.tolist(),
                                                    default=['metal_temp_1st', 'scr_outlet_temp', 'exhaust_gas_temperature'] if all(feat in ref_df_for_plot.columns for feat in ['metal_temp_1st', 'scr_outlet_temp', 'exhaust_gas_temperature']) else ref_df_for_plot.columns.tolist()[:3],
                                                    key="similarity_feature_selection",
                                                    help="선택된 특징들을 각 파일별로 비교합니다."
                                                )
                                                
                                                if selected_plot_features:
                                                    # 시각화 설정 (탭3과 동일한 구조)
                                                    st.markdown("**⚙️ 시각화 설정**")
                                                    
                                                    col1, col2, col3 = st.columns(3)
                                                    with col1:
                                                        similarity_downsample_rate = st.slider(
                                                            "📉 다운샘플 비율 (1/N)", 
                                                            min_value=1, max_value=100, value=10,
                                                            key="similarity_downsample"
                                                        )
                                                    with col2:
                                                        similarity_num_segments = st.selectbox(
                                                            "📊 데이터 분할 수",
                                                            options=[1, 2, 3, 4, 5],
                                                            index=2,  # 기본값: 3등분
                                                            help="전체 데이터를 몇 등분할지 선택",
                                                            key="similarity_segments"
                                                        )
                                                    with col3:
                                                        similarity_selected_segment = st.selectbox(
                                                            "🎯 분석 구간 선택",
                                                            options=list(range(similarity_num_segments)),
                                                            format_func=lambda x: f"구간 {x+1}",
                                                            index=0,  # 기본값: 첫 번째 구간
                                                            help="분석할 구간을 선택",
                                                            key="similarity_segment_select"
                                                        )
                                                    
                                                    similarity_crosshair = st.checkbox("▶️ 십자선 Hover 활성화", value=True, key="similarity_crosshair")
                                                    
                                                    # 시계열 비교 플롯 생성
                                                    try:
                                                        fig_timeseries = create_multi_file_plot(
                                                            selected_plot_files,
                                                            selected_plot_features,
                                                            similarity_downsample_rate,
                                                            similarity_crosshair,
                                                            similarity_num_segments,
                                                            similarity_selected_segment
                                                        )
                                                        
                                                        # 제목 수정
                                                        segment_info = f"구간 {similarity_selected_segment + 1}/{similarity_num_segments}"
                                                        fig_timeseries.update_layout(title=f"📊 유사 기동 비교 분석 ({segment_info})")
                                                        
                                                        st.plotly_chart(fig_timeseries, use_container_width=True)
                                                        
                                                        # 비교 정보 표시
                                                        st.markdown("**📋 비교 중인 파일:**")
                                                        for i, idx in enumerate(selected_plot_file_indices):
                                                            if idx == 0:
                                                                st.write(f"🎯 **기준**: {reference_file.name}")
                                                            else:
                                                                rank = idx  # 1위부터 시작
                                                                result = top_5_similar[rank-1]
                                                                st.write(f"🏅 **{rank}위**: {result['file_name']} (거리: {result['distance']:.4f})")
                                                        
                                                    except Exception as e:
                                                        st.error(f"❌ 시계열 비교 플롯 생성 중 오류: {str(e)}")
                                                else:
                                                    st.info("🎯 비교할 특징을 선택해주세요.")
                                            else:
                                                st.error("❌ 기준 파일을 로드할 수 없습니다.")
                                        else:
                                            st.info("📂 비교할 파일을 선택해주세요.")
                                        
                                        # 상세 분석 정보
                                        with st.expander("📊 상세 분석 정보"):
                                            st.markdown("**🔍 검색 통계**")
                                            all_results = st.session_state.similarity_results
                                            
                                            stats_col1, stats_col2, stats_col3 = st.columns(3)
                                            with stats_col1:
                                                st.metric("검색된 파일 수", len(all_results))
                                            with stats_col2:
                                                min_distance = min([r['distance'] for r in all_results])
                                                st.metric("최소 거리", f"{min_distance:.6f}")
                                            with stats_col3:
                                                max_distance = max([r['distance'] for r in all_results])
                                                st.metric("최대 거리", f"{max_distance:.6f}")
                                            
                                            st.markdown("**📋 전체 검색 결과**")
                                            full_results_data = []
                                            for i, result in enumerate(all_results, 1):
                                                full_results_data.append({
                                                    '순위': i,
                                                    '파일명': result['file_name'],
                                                    '유클리드 거리': f"{result['distance']:.6f}",
                                                    'Metal Temp': f"{result['metal_temp_1st']:.2f}°C",
                                                    'SCR Temp': f"{result['scr_outlet_temp']:.2f}°C",
                                                    'Exhaust Temp': f"{result['exhaust_gas_temperature']:.2f}°C"
                                                })
                                            
                                            full_results_df = pd.DataFrame(full_results_data)
                                            st.dataframe(full_results_df, use_container_width=True, hide_index=True)
                                    
                                    else:
                                        st.warning("⚠️ 유사한 기동을 찾을 수 없습니다.")
                            else:
                                st.error("❌ 기준 파일의 데이터가 부족합니다. tic=80 이상의 데이터가 필요합니다.")
                    else:
                        st.error("❌ 기준 파일을 로드할 수 없습니다.")
                        
                else:
                    st.warning("⚠️ 유사 기동 검색을 위해서는 최소 2개 이상의 파일이 필요합니다. (기준 파일 1개 + 검색 대상 파일 1개 이상)")
                    st.info("현재 업로드된 파일이 1개뿐입니다. 추가 파일을 업로드해주세요.")
            else:
                st.error("❌ 기준 파일을 로드할 수 없습니다.")
        else:
            st.info("📁 유사 기동 검색을 위해 다중 파일을 업로드하세요.")
    



# =================================================================================
# 애플리케이션 실행
# =================================================================================
if __name__ == "__main__":
    main()






