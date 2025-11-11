"""
HDF5 데이터 분석 프론트엔드
Streamlit 기반 사용자 인터페이스
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
import json

from hdf5_backend import HDF5Backend
from hdf5_visualization import HDF5Visualizer


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 페이지 설정
st.set_page_config(
    page_title="HDF5 데이터 분석 도구",
    page_icon="📊",
    layout="wide"
)

st.title("📊 HDF5 데이터 분석 및 시각화 도구")


# 백엔드 및 시각화 객체 생성
@st.cache_resource
def get_backend():
    return HDF5Backend()

@st.cache_resource
def get_visualizer():
    return HDF5Visualizer()

backend = get_backend()
visualizer = get_visualizer()


def render_formula_builder(df: pd.DataFrame):
    """수식 빌더 UI 렌더링"""
    
    st.subheader("🔢 수식 빌더")
    
    # 세션 상태 초기화
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
        logger.debug("selected_features 초기화")
    
    if 'expression_text' not in st.session_state:
        st.session_state.expression_text = ""
        logger.debug("expression_text 초기화")
    
    if 'feature_shifts' not in st.session_state:
        st.session_state.feature_shifts = {}
        logger.debug("feature_shifts 초기화")
    
    # --- [ ⭐ 수정된 부분 (feature_thresholds 초기화 제거) ⭐ ] ---
    # if 'feature_thresholds' not in st.session_state:
    #     st.session_state.feature_thresholds = {}
    #     logger.debug("feature_thresholds 초기화")
    # --- [ ⭐ 수정된 부분 종료 ⭐ ] ---
    
    # 3단 레이아웃
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        st.markdown("**1️⃣ 특징 선택**")
        
        # 특징 목록
        features = [col for col in df.columns if col.lower() not in ['timestamp', 'datetime', 'date', 'time', 'index']]
        
        # 특징 선택
        selected = st.selectbox(
            "특징 선택",
            ["선택하세요..."] + features,
            key="feature_selector"
        )
        
        # 추가 버튼
        if st.button("➕ 변수에 추가", key="add_feature"):
            if selected != "선택하세요..." and selected not in st.session_state.selected_features:
                if len(st.session_state.selected_features) < 26:  # A-Z
                    st.session_state.selected_features.append(selected)
                    
                    # Shift 초기값 설정
                    var_name = chr(65 + len(st.session_state.selected_features) - 1)
                    st.session_state.feature_shifts[var_name] = 0
                    # st.session_state.feature_thresholds[var_name] = np.nan # 제거
                    
                    logger.info(f"✓ 특징 추가: {var_name} = {selected}")
                    st.rerun()
                else:
                    st.error("최대 26개 변수까지 추가 가능합니다.")
        
        # 선택된 특징 목록 표시
        st.markdown("---")
        st.markdown("**선택된 특징:**")
        
        if st.session_state.selected_features:
            for idx, feat in enumerate(st.session_state.selected_features):
                var_name = chr(65 + idx)
                
                # 특징명, Shift, 삭제 버튼을 같은 행에 표시
                # --- [ ⭐ 수정된 부분 (col_thresh 제거) ⭐ ] ---
                col_feat, col_shift, col_del = st.columns([3, 2.5, 0.5]) 
                # --- [ ⭐ 수정된 부분 종료 ⭐ ] ---
                
                with col_feat:
                    st.text(f"{var_name} = {feat}")
                
                with col_shift:
                    # Shift 값 입력
                    shift_val = st.number_input(
                        "Shift",
                        value=st.session_state.feature_shifts.get(var_name, 0),
                        step=1,
                        key=f"shift_{var_name}",
                        label_visibility="collapsed",
                        help=f"{var_name}의 Shift 값 (양수: 우측 이동, 음수: 좌측 이동)"
                    )
                    st.session_state.feature_shifts[var_name] = shift_val
                
                # --- [ ⭐ 수정된 부분 (임계값 UI 제거) ⭐ ] ---
                # with col_thresh:
                #     current_thresh = st.session_state.feature_thresholds.get(var_name, np.nan)
                #     thresh_val = st.number_input(
                #         "임계값",
                #         value=0.0 if np.isnan(current_thresh) else current_thresh,
                #         step=0.1,
                #         format="%.2f",
                #         key=f"thresh_{var_name}",
                #         label_visibility="collapsed",
                #         help=f"{var_name}의 임계값 (이 값 미만은 수식에서 0으로 처리)"
                #     )
                #     use_thresh = st.checkbox(
                #         "사용",
                #         value=not np.isnan(current_thresh),
                #         key=f"use_thresh_{var_name}",
                #         label_visibility="collapsed"
                #     )
                #     if use_thresh:
                #         st.session_state.feature_thresholds[var_name] = thresh_val
                #     else:
                #         st.session_state.feature_thresholds[var_name] = np.nan
                # --- [ ⭐ 수정된 부분 종료 ⭐ ] ---
                
                with col_del:
                    if st.button("🗑️", key=f"del_{var_name}", help=f"{var_name} 삭제"):
                        st.session_state.selected_features.pop(idx)
                        
                        # Shift 및 임계값도 재정렬
                        new_shifts = {}
                        # new_thresholds = {} # 제거
                        for i, f in enumerate(st.session_state.selected_features):
                            new_var = chr(65 + i)
                            old_var = chr(65 + i if i < idx else i + 1)
                            new_shifts[new_var] = st.session_state.feature_shifts.get(old_var, 0)
                            # new_thresholds[new_var] = st.session_state.feature_thresholds.get(old_var, np.nan) # 제거
                        
                        st.session_state.feature_shifts = new_shifts
                        # st.session_state.feature_thresholds = new_thresholds # 제거
                        
                        logger.info(f"✓ 특징 삭제: {var_name}")
                        st.rerun()
        else:
            st.info("특징을 선택하여 변수에 추가하세요.")
    
    with col2:
        st.markdown("**2️⃣ 수식 입력**")
        
        if st.session_state.selected_features:
            # 변수 안내
            var_info = []
            for idx, feat in enumerate(st.session_state.selected_features):
                var_name = chr(65 + idx)
                info = f"`{var_name}` = {feat}"
                
                # Shift 정보
                shift = st.session_state.feature_shifts.get(var_name, 0)
                if shift != 0:
                    info += f" (Shift: {shift:+d})"
                
                # --- [ ⭐ 수정된 부분 (임계값 정보 제거) ⭐ ] ---
                # thresh = st.session_state.feature_thresholds.get(var_name, np.nan)
                # if not np.isnan(thresh):
                #     info += f" (임계값: {thresh:.1f})"
                # --- [ ⭐ 수정된 부분 종료 ⭐ ] ---
                
                var_info.append(info)
            
            st.markdown("**사용 가능한 변수:**")
            st.markdown("\n".join(f"- {info}" for info in var_info))
            
            st.markdown("---")
            
            # 수식 입력
            expression = st.text_input(
                "수식",
                value=st.session_state.expression_text,
                placeholder="예: (A + B) / C",
                help="변수와 사칙연산(+, -, *, /), 괄호 사용 가능"
            )
            st.session_state.expression_text = expression
            
            # 수식 예시
            st.markdown("**수식 예시:**")
            st.code("A + B\n(A - B) / C\nA * B + C\n(A + B) * (C - D)")
            
        else:
            st.info("먼저 특징을 선택하세요.")
    
    with col3:
        st.markdown("**3️⃣ 계산 실행**")
        
        st.markdown("**범위 제한 (Clipping):**")
        st.caption("계산 결과가 범위를 벗어나면 0으로 처리")
        
        use_clipping = st.checkbox(
            "결과 범위 제한 활성화", 
            key="use_clipping",
            value=st.session_state.get('use_clipping_val', False)
        )
        st.session_state.use_clipping_val = use_clipping

        min_range = None
        max_range = None

        if use_clipping:
            c1, c2 = st.columns(2)
            with c1:
                min_range = st.number_input(
                    "최소값", 
                    value=st.session_state.get('clip_min_val', -1000.0), 
                    format="%.2f", 
                    key="clip_min"
                )
            with c2:
                max_range = st.number_input(
                    "최대값", 
                    value=st.session_state.get('clip_max_val', 1000.0), 
                    format="%.2f", 
                    key="clip_max"
                )
            # 세션에 저장
            st.session_state.clip_min_val = min_range
            st.session_state.clip_max_val = max_range
        
        st.markdown("---")
        
        if st.session_state.selected_features and st.session_state.expression_text:
            if st.button("🚀 계산 실행", type="primary", use_container_width=True):
                # 변수 매핑 생성
                feature_map = {}
                for idx, feat in enumerate(st.session_state.selected_features):
                    var_name = chr(65 + idx)
                    feature_map[var_name] = feat
                
                logger.info(f"=== 계산 실행 ===")
                logger.info(f"수식: {st.session_state.expression_text}")
                logger.info(f"변수 매핑: {feature_map}")
                logger.info(f"Shift: {st.session_state.feature_shifts}")
                # logger.info(f"임계값: {st.session_state.feature_thresholds}") # 제거
                
                clip_min_to_pass = None
                clip_max_to_pass = None
                if st.session_state.get('use_clipping_val', False):
                     clip_min_to_pass = st.session_state.get('clip_min_val')
                     clip_max_to_pass = st.session_state.get('clip_max_val')
                     logger.info(f"Clipping: [{clip_min_to_pass}, {clip_max_to_pass}]")

                # 계산
                result, success, error = backend.calculate_expression(
                    df,
                    st.session_state.expression_text,
                    feature_map,
                    st.session_state.feature_shifts,
                    None, # 임계값 제거됨
                    clip_min_to_pass,
                    clip_max_to_pass
                )
                
                if success:
                    st.session_state.calculation_result = result
                    st.session_state.feature_map = feature_map
                    st.success("✅ 계산 완료!")
                    logger.info("✅ 계산 성공")
                else:
                    st.error(f"❌ 계산 오류: {error}")
                    logger.error(f"❌ 계산 실패: {error}")
        else:
            st.info("특징과 수식을 입력하세요.")


def render_results(df: pd.DataFrame):
    """결과 표시"""
    
    if 'calculation_result' not in st.session_state:
        st.info("수식을 입력하고 계산을 실행하세요.")
        return
    
    result = st.session_state.calculation_result
    feature_map = st.session_state.feature_map
    
    st.subheader("📊 분석 결과")
    
    # 통계
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    stats = backend.get_statistics(result)
    
    col1.metric("평균", f"{stats['평균']:.2f}")
    #col2.metric("표준편차", f"{stats['표준편f차']:.2f}")
    col2.metric("표준편차", f"{stats['표준편차']:.2f}")
    col3.metric("최소값", f"{stats['최소값']:.2f}")
    col4.metric("최대값", f"{stats['최대값']:.2f}")
    col5.metric("중앙값", f"{stats['중앙값']:.2f}")
    col6.metric("데이터 개수", f"{int(stats['데이터 개수'])}")
    
    # Combined 플롯
    selected_range = st.session_state.get('selected_range', None)
    
    clip_min_to_pass = None
    clip_max_to_pass = None
    if st.session_state.get('use_clipping_val', False):
         clip_min_to_pass = st.session_state.get('clip_min_val')
         clip_max_to_pass = st.session_state.get('clip_max_val')
    
    fig = visualizer.create_combined_plot(
        df,
        result,
        st.session_state.expression_text,
        feature_map,
        st.session_state.feature_shifts,
        None, # 임계값 제거됨
        selected_range,
        clip_min_to_pass,
        clip_max_to_pass
    )
    
    # Plotly 이벤트 감지
    selected_data = st.plotly_chart(
        fig,
        use_container_width=True,
        key="combined_plot",
        on_select="rerun"
    )
    
    # Box selection 처리
    if selected_data and selected_data.selection and selected_data.selection.box:
        logger.info("=" * 50)
        logger.info("✓ Box selection 감지됨")
        
        boxes = selected_data.selection.box
        logger.info(f"Box 개수: {len(boxes)}")
        
        if len(boxes) > 0:
            box = boxes[0]
            logger.info(f"Box 내용: {box}")
            
            # Box에서 x 범위 추출
            if 'x' in box:
                x_range = box['x']
                logger.info(f"X 범위: {x_range}")
                
                x_min = min(x_range)
                x_max = max(x_range)
                
                logger.info(f"X 최소/최대: {x_min} ~ {x_max}")
                
                # Index로 변환 (이미 index이므로 직접 사용)
                start_idx = int(np.floor(x_min))
                end_idx = int(np.ceil(x_max))
                
                # 범위 제한
                start_idx = max(0, start_idx)
                end_idx = min(len(df) - 1, end_idx)
                
                logger.info(f"선택 인덱스: {start_idx} ~ {end_idx} (총 {end_idx - start_idx + 1}개)")
                
                if end_idx > start_idx:
                    st.session_state.selected_range = (start_idx, end_idx)
                    st.session_state.range_start = start_idx
                    st.session_state.range_end = end_idx
                    
                    logger.info("✅ 구간 선택 완료")
                    st.rerun()
                else:
                    logger.warning(f"⚠️ 선택된 인덱스: {end_idx - start_idx + 1}개 (2개 이상 필요)")
            else:
                logger.warning("⚠️ Box에 'x' 키가 없습니다")
        
        logger.info("=" * 50)
    
    # 수동 구간 입력
    st.markdown("---")
    st.markdown("**📍 구간 선택 (수동 입력)**")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        range_start = st.number_input(
            "시작 인덱스",
            min_value=0,
            max_value=len(df)-1,
            value=st.session_state.get('range_start', 0),
            key="manual_start"
        )
    
    with col2:
        range_end = st.number_input(
            "종료 인덱스",
            min_value=0,
            max_value=len(df)-1,
            value=st.session_state.get('range_end', len(df)-1),
            key="manual_end"
        )
    
    with col3:
        if st.button("구간 적용", use_container_width=True):
            if range_end > range_start:
                st.session_state.selected_range = (range_start, range_end)
                st.session_state.range_start = range_start
                st.session_state.range_end = range_end
                st.rerun()
    
    # 선택 구간 통계
    if selected_range:
        st.markdown("---")
        st.markdown(f"**📊 선택 구간 통계 (Index {selected_range[0]} ~ {selected_range[1]})**")
        
        segment_stats = backend.get_statistics(result, selected_range[0], selected_range[1])
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        col1.metric("평균", f"{segment_stats['평균']:.2f}")
        col2.metric("표준편차", f"{segment_stats['표준편차']:.2f}")
        col3.metric("최소값", f"{segment_stats['최소값']:.2f}")
        col4.metric("최대값", f"{segment_stats['최대값']:.2f}")
        col5.metric("중앙값", f"{segment_stats['중앙값']:.2f}")
        col6.metric("데이터 개수", f"{int(segment_stats['데이터 개수'])}")


def main():
    """메인 함수"""
    
    # 파일 업로드
    st.sidebar.header("📁 파일 업로드")
    
    uploaded_files = st.sidebar.file_uploader(
        "HDF5 파일 선택 (여러 개 가능)",
        type=['h5', 'hdf5'],
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.info("👈 사이드바에서 HDF5 파일을 업로드하세요.")
        return
    
    # 파일 로드
    logger.info(f"=== 파일 업로드: {len(uploaded_files)}개 ===")
    
    dataframes = []
    
    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.read()
            df = backend.load_hdf5(file_bytes, uploaded_file.name)
            dataframes.append(df)
            st.sidebar.success(f"✅ {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"❌ {uploaded_file.name}: {str(e)}")
            logger.error(f"파일 로드 실패 {uploaded_file.name}: {e}")
    
    if not dataframes:
        st.error("파일을 로드할 수 없습니다.")
        return
    
    # 첫 번째 파일을 기준으로 사용
    df = dataframes[0]
    
    st.sidebar.info(f"총 {len(dataframes)}개 파일 로드 완료\n데이터: {len(df)}행 × {len(df.columns)}열")
    
    # 수식 빌더
    render_formula_builder(df)
    
    st.markdown("---")
    
    # 결과 표시
    render_results(df)
    
    # 다중 파일 처리
    if len(dataframes) > 1 and 'selected_range' in st.session_state:
        st.markdown("---")
        st.subheader("📊 다중 파일 분석")
        
        if st.button("🔄 다중 파일 평균 계산"):
            start_idx, end_idx = st.session_state.selected_range
            
            clip_min_to_pass = None
            clip_max_to_pass = None
            if st.session_state.get('use_clipping_val', False):
                 clip_min_to_pass = st.session_state.get('clip_min_val')
                 clip_max_to_pass = st.session_state.get('clip_max_val')

            avg_result, avg_stats = backend.merge_multiple_files(
                dataframes,
                st.session_state.expression_text,
                st.session_state.feature_map,
                start_idx,
                end_idx,
                st.session_state.feature_shifts,
                None, # 임계값 제거됨
                clip_min_to_pass,
                clip_max_to_pass
            )
            
            if avg_result is not None:
                st.success(f"✅ {len(dataframes)}개 파일 평균 계산 완료!")
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                col1.metric("평균", f"{avg_stats['평균']:.2f}")
                col2.metric("표준편차", f"{avg_stats['표준편차']:.2f}")
                col3.metric("최소값", f"{avg_stats['최소값']:.2f}")
                col4.metric("최대값", f"{avg_stats['최대값']:.2f}")
                col5.metric("중앙값", f"{avg_stats['중앙값']:.2f}")
                col6.metric("데이터 개수", f"{int(avg_stats['데이터 개수'])}")


if __name__ == "__main__":
    main()