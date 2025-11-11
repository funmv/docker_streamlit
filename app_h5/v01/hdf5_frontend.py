"""
HDF5 데이터 분석 및 가시화 도구 - Frontend
A, B, C 변수 기반 수식 계산
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import List

# Backend 모듈 임포트
from hdf5_backend import HDF5Analyzer, ExpressionCalculator, RangeSelector, StatisticsCalculator
from hdf5_visualization import PlotlyVisualizer

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 한글 폰트 설정
try:
    from matplotlib import font_manager, rc
    import matplotlib.pyplot as plt
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        from matplotlib import font_manager, rc
        import matplotlib.pyplot as plt
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)  
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass


def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 파일 업로드
        uploaded_files = st.file_uploader(
            "HDF5 파일 선택",
            type=['h5', 'hdf5'],
            accept_multiple_files=True,
            help="하나 또는 여러 개의 HDF5 파일을 선택하세요"
        )
        
        if uploaded_files:
            if st.button("📂 파일 로드", type="primary"):
                logger.info(f"=== 파일 로드 시작: {len(uploaded_files)}개 ===")
                
                st.session_state.dataframes = []
                st.session_state.file_names = []
                
                progress_bar = st.progress(0)
                for idx, file in enumerate(uploaded_files):
                    try:
                        logger.info(f"파일 {idx+1} 로드 중: {file.name}")
                        df = HDF5Analyzer.load_hdf5(file.read())
                        st.session_state.dataframes.append(df)
                        st.session_state.file_names.append(file.name)
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        logger.info(f"✅ 파일 {idx+1} 로드 완료")
                    except Exception as e:
                        logger.error(f"❌ 파일 로드 실패 ({file.name}): {e}")
                        st.error(f"파일 로드 실패 ({file.name}): {str(e)}")
                
                st.success(f"✅ {len(st.session_state.dataframes)}개 파일 로드 완료")
                logger.info("=== 전체 파일 로드 완료 ===")
        
        st.markdown("---")
        
        # 파일 정보
        if st.session_state.get('dataframes'):
            st.subheader("📁 로드된 파일")
            for idx, name in enumerate(st.session_state.file_names):
                df = st.session_state.dataframes[idx]
                st.text(f"{idx+1}. {name}")
                st.caption(f"   크기: {df.shape[0]} × {df.shape[1]}")


def render_expression_builder(numeric_cols: List[str]) -> tuple:
    """
    A, B, C 변수 기반 수식 빌더
    
    Returns:
        (selected_features, expression): 선택된 특징 리스트, 수식
    """
    st.subheader("🧮 특징 연산 수식")
    
    # 세션 상태 초기화
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    
    if 'expression_text' not in st.session_state:
        st.session_state.expression_text = ""
    
    if 'feature_shifts' not in st.session_state:
        st.session_state.feature_shifts = {}
    
    # 3단 레이아웃
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        st.markdown("**1️⃣ 특징 선택**")
        st.caption("선택 순서대로 A, B, C, ... 변수로 할당됩니다")
        
        # 특징 선택 드롭다운
        selected_col = st.selectbox(
            "특징 선택",
            options=["선택하세요..."] + numeric_cols,
            key="feature_select"
        )
        
        # 특징 추가 버튼
        if st.button("➕ 특징 추가", type="primary", width='stretch'):
            if selected_col != "선택하세요...":
                if selected_col not in st.session_state.selected_features:
                    st.session_state.selected_features.append(selected_col)
                    var_name = chr(65 + len(st.session_state.selected_features) - 1)
                    logger.info(f"✅ 특징 추가: {var_name} = '{selected_col}'")
                    st.success(f"✅ {var_name} = {selected_col}")
                    st.rerun()
                else:
                    st.warning("이미 선택된 특징입니다")
            else:
                st.warning("특징을 선택해주세요")
        
        # 선택된 특징 목록 표시
        st.markdown("---")
        st.markdown("**선택된 특징:**")
        
        if st.session_state.selected_features:
            for idx, feat in enumerate(st.session_state.selected_features):
                var_name = chr(65 + idx)
                
                # 특징명과 Shift 입력을 같은 행에 표시
                col_feat, col_shift, col_del = st.columns([3, 2, 1])
                
                with col_feat:
                    st.text(f"{var_name} = {feat}")
                
                with col_shift:
                    # Shift 값 입력
                    shift_key = f"shift_{idx}"
                    current_shift = st.session_state.feature_shifts.get(feat, 0)
                    
                    new_shift = st.number_input(
                        "Shift",
                        value=current_shift,
                        step=1,
                        key=shift_key,
                        help="양수: 우측 이동, 음수: 좌측 이동",
                        label_visibility="collapsed"
                    )
                    
                    if new_shift != current_shift:
                        st.session_state.feature_shifts[feat] = new_shift
                        logger.info(f"Shift 업데이트: {var_name} = {new_shift}")
                    
                    if new_shift != 0:
                        st.caption(f"↔️ {new_shift:+d}")
                
                with col_del:
                    if st.button("🗑️", key=f"del_{idx}", help="삭제"):
                        logger.info(f"특징 삭제: {var_name} = '{feat}'")
                        st.session_state.selected_features.pop(idx)
                        if feat in st.session_state.feature_shifts:
                            del st.session_state.feature_shifts[feat]
                        st.rerun()
        else:
            st.info("선택된 특징이 없습니다")
        
        # 전체 초기화
        if st.button("🔄 전체 초기화", width='stretch'):
            logger.info("전체 초기화")
            st.session_state.selected_features = []
            st.session_state.expression_text = ""
            st.session_state.feature_shifts = {}
            st.rerun()
    
    with col2:
        st.markdown("**2️⃣ 수식 입력**")
        st.caption("A, B, C 변수를 사용하여 수식을 입력하세요")
        
        # 수식 입력
        expression = st.text_input(
            "수식",
            value=st.session_state.expression_text,
            placeholder="예: A + B, (A - B) / 2, A * B + C",
            key="expr_input"
        )
        
        # 수식 업데이트
        if expression != st.session_state.expression_text:
            st.session_state.expression_text = expression
        
        # 빠른 연산자 버튼
        st.markdown("**빠른 연산:**")
        
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        
        with btn_col1:
            if st.button("**A + B**", width='stretch'):
                st.session_state.expression_text = "A + B"
                st.rerun()
        with btn_col2:
            if st.button("**A - B**", width='stretch'):
                st.session_state.expression_text = "A - B"
                st.rerun()
        with btn_col3:
            if st.button("**A * B**", width='stretch'):
                st.session_state.expression_text = "A * B"
                st.rerun()
        with btn_col4:
            if st.button("**A / B**", width='stretch'):
                st.session_state.expression_text = "A / B"
                st.rerun()
        
        # 복잡한 수식 예제
        st.markdown("**예제:**")
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            if st.button("(A+B)/2", width='stretch'):
                st.session_state.expression_text = "(A + B) / 2"
                st.rerun()
        with example_col2:
            if st.button("A**2+B**2", width='stretch'):
                st.session_state.expression_text = "A**2 + B**2"
                st.rerun()
    
    with col3:
        st.markdown("**3️⃣ 함수 & 상수**")
        
        # 수학 함수
        math_func = st.selectbox(
            "수학 함수",
            options=["선택...", "sqrt", "sin", "cos", "tan", "log", "log10", "exp", "abs"]
        )
        
        if st.button("함수 추가", width='stretch'):
            if math_func != "선택...":
                st.session_state.expression_text += f"{math_func}(A)"
                st.rerun()
        
        # 상수
        st.markdown("**상수:**")
        const_col1, const_col2 = st.columns(2)
        
        with const_col1:
            if st.button("**π**", width='stretch'):
                st.session_state.expression_text += "pi"
                st.rerun()
        with const_col2:
            if st.button("**e**", width='stretch'):
                st.session_state.expression_text += "e"
                st.rerun()
        
        # 현재 수식 표시
        st.markdown("---")
        st.markdown("**현재 수식:**")
        if st.session_state.expression_text:
            st.code(st.session_state.expression_text, language="python")
        else:
            st.info("수식을 입력해주세요")
    
    return st.session_state.selected_features, st.session_state.expression_text, st.session_state.feature_shifts


def main():
    st.set_page_config(
        page_title="HDF5 데이터 분석기",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 HDF5 데이터 분석 도구 (A, B, C 변수 방식)")
    st.markdown("---")
    
    logger.info("=== 애플리케이션 시작 ===")
    
    # 세션 스테이트 초기화
    if 'dataframes' not in st.session_state:
        st.session_state.dataframes = []
    if 'file_names' not in st.session_state:
        st.session_state.file_names = []
    if 'selected_range' not in st.session_state:
        st.session_state.selected_range = None
    
    # 사이드바 렌더링
    render_sidebar()
    
    # 메인 영역
    if not st.session_state.dataframes:
        st.info("👈 사이드바에서 HDF5 파일을 업로드해주세요.")
        return
    
    # 데이터 분석 시작
    df_main = st.session_state.dataframes[0]
    numeric_cols = HDF5Analyzer.get_numeric_columns(df_main)
    time_col = HDF5Analyzer.get_time_column(df_main)
    
    logger.info(f"메인 DataFrame: shape={df_main.shape}")
    logger.info(f"숫자형 컬럼: {len(numeric_cols)}개")
    logger.info(f"시간 컬럼: {time_col}")
    
    # 컬럼 선택 영역
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 원본 데이터 시각화")
        
        # Timestamp 정보 표시
        if time_col:
            st.info(f"✓ 시간 컬럼: {time_col} (hover에 표시)")
            st.caption("X축은 Index(0, 1, 2, ...)를 사용합니다")
        else:
            st.info("📊 X축: 데이터 인덱스 (0, 1, 2, ...)")
        
        # Y축 선택
        y_axis = st.selectbox("Y축 (분석 특징)", numeric_cols, key="y_axis")
    
    with col2:
        st.subheader("🎯 구간 설정")
        
        # 구간 선택 안내
        st.info("💡 그래프에서 마우스로 드래그하여 구간 선택")
        
        # 구간 조작 버튼
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 구간 초기화", width='stretch'):
                logger.info("구간 초기화")
                st.session_state.selected_range = None
                st.rerun()
        
        with col_b:
            if st.button("📏 전체 선택", width='stretch'):
                logger.info(f"전체 선택: 0 ~ {len(df_main)-1}")
                st.session_state.selected_range = (0, len(df_main) - 1)
                st.rerun()
        
        # 현재 선택 상태 표시
        if st.session_state.selected_range:
            start_idx, end_idx = st.session_state.selected_range
            st.success(f"✓ 구간 설정 완료!")
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("시작", f"{int(start_idx)}")
            with col_info2:
                st.metric("종료", f"{int(end_idx)}")
            with col_info3:
                st.metric("개수", f"{int(end_idx - start_idx + 1)}")
            
            # 시간 정보 표시
            if time_col and time_col in df_main.columns:
                try:
                    start_time = df_main[time_col].iloc[int(start_idx)]
                    end_time = df_main[time_col].iloc[int(end_idx)]
                    st.caption(f"📅 {start_time} ~ {end_time}")
                except:
                    pass
        else:
            st.warning("구간이 선택되지 않았습니다")
    
    # 원본 데이터 플롯
    st.markdown("---")
    
    fig_original = PlotlyVisualizer.create_interactive_plot(
        df_main,
        time_col,
        y_axis,
        f"원본 데이터: {y_axis}",
        st.session_state.selected_range
    )
    
    # Plotly 이벤트 처리
    event = st.plotly_chart(
        fig_original, 
        use_container_width=True,
        key="original_plot",
        on_select="rerun"
    )
    
    # Box Selection 처리 (딕셔너리 접근으로 수정)
    if event:
        logger.debug(f"🔍 event 객체 수신: {type(event)}")
        
        if hasattr(event, 'selection'):
            logger.debug(f"✅ selection 속성 존재")
            selection = event.selection
            logger.debug(f"🔍 selection 내용: {selection}")
            
            if selection and hasattr(selection, 'box') and selection.box:
                logger.debug(f"✅ box 데이터 존재: {len(selection.box)}개")
                
                if len(selection.box) > 0:
                    box_item = selection.box[0]
                    logger.debug(f"🔍 box_item 타입: {type(box_item)}")
                    logger.debug(f"🔍 box_item 내용: {box_item}")
                    
                    # 딕셔너리 접근
                    if 'x' in box_item:
                        logger.debug(f"✅ x 키 존재")
                        x_data = box_item['x']
                        logger.debug(f"🔍 x 내용: {x_data}")
                        
                        if len(x_data) == 2:
                            try:
                                # X값은 이미 index
                                x_min = float(x_data[0])
                                x_max = float(x_data[1])
                                
                                logger.info(f"📍 Box Selection 감지: x_min={x_min}, x_max={x_max}")
                                
                                # 정수 인덱스로 변환
                                start_idx = int(max(0, x_min))
                                end_idx = int(min(len(df_main) - 1, x_max))
                                
                                logger.info(f"📍 인덱스 변환: start={start_idx}, end={end_idx}")
                                
                                if start_idx <= end_idx:
                                    st.session_state.selected_range = (start_idx, end_idx)
                                    
                                    logger.info(f"✅✅✅ 새 구간 선택 완료: [{start_idx}, {end_idx}] ✅✅✅")
                                    
                                    st.success(f"🎯 새 구간 선택: Index {start_idx} ~ {end_idx} ({end_idx - start_idx + 1}개)")
                                    
                                    # 짧은 대기 후 새로고침
                                    import time
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    logger.warning(f"⚠️ 유효하지 않은 범위: {start_idx} >= {end_idx}")
                                    
                            except Exception as e:
                                logger.error(f"❌ Box selection 처리 오류: {e}", exc_info=True)
                                st.error(f"구간 선택 오류: {e}")
                    else:
                        logger.debug(f"❌ x 키 없음")
    
    # 수식 빌더
    st.markdown("---")
    selected_features, expression, feature_shifts = render_expression_builder(numeric_cols)
    
    # 계산 실행
    if expression and selected_features:
        logger.info(f"=== 계산 실행 시작 ===")
        logger.info(f"선택된 특징: {selected_features}")
        logger.info(f"수식: {expression}")
        
        try:
            # 계산 수행
            if len(st.session_state.dataframes) == 1:
                # 단일 파일
                result = ExpressionCalculator.calculate_custom(
                    df_main, 
                    selected_features, 
                    expression
                )
                result_label = "계산 결과"
            else:
                # 다중 파일 - 평균 계산
                results = []
                for df in st.session_state.dataframes:
                    try:
                        r = ExpressionCalculator.calculate_custom(
                            df, 
                            selected_features, 
                            expression
                        )
                        results.append(r)
                    except Exception as e:
                        logger.warning(f"일부 파일 계산 실패: {e}")
                        st.warning(f"일부 파일 계산 실패: {str(e)}")
                
                if results:
                    result = pd.concat(results, axis=1).mean(axis=1)
                    result_label = f"계산 결과 (평균, {len(results)}개 파일)"
                else:
                    raise ValueError("모든 파일에서 계산 실패")
            
            # 결과 플롯
            st.markdown("---")
            st.subheader("📊 연산 결과")
            
            # 특징 매핑 표시
            mapping_text = " | ".join([f"{chr(65+i)}={feat}" for i, feat in enumerate(selected_features)])
            st.caption(f"🔤 변수 매핑: {mapping_text}")
            
            # 결과 플롯 생성 (연산 결과 + 모든 사용된 특징)
            fig_result = PlotlyVisualizer.create_combined_result_plot(
                df_main,
                time_col,
                selected_features,
                result,
                expression,
                st.session_state.selected_range,
                feature_shifts  # Shift 정보 전달
            )
            
            st.plotly_chart(fig_result, use_container_width=True, key="result_plot")
            
            # 통계 분석
            st.markdown("---")
            st.subheader("📈 통계 분석")
            
            if st.session_state.selected_range:
                mask = RangeSelector.create_range_mask(df_main, st.session_state.selected_range)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**원본 데이터 (선택 구간)**")
                    stats_original = StatisticsCalculator.calculate_statistics(
                        df_main.loc[mask, y_axis], 
                        y_axis
                    )
                    # st.dataframe(pd.DataFrame([stats_original]).T, use_container_width=True)
                    st.dataframe(pd.DataFrame([stats_original]).T, width='stretch')
                
                with col2:
                    st.markdown("**계산 결과 (선택 구간)**")
                    stats_result = StatisticsCalculator.calculate_statistics(
                        result[mask],
                        "계산 결과"
                    )
                    # st.dataframe(pd.DataFrame([stats_result]).T, use_container_width=True)
                    st.dataframe(pd.DataFrame([stats_result]).T, width='stretch')
            else:
                st.info("구간을 선택하면 해당 구간의 통계값을 확인할 수 있습니다.")
            
            # 데이터 다운로드
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV 다운로드
                download_df = pd.DataFrame({'Index': df_main.index})
                if time_col and time_col in df_main.columns:
                    download_df[time_col] = df_main[time_col]
                download_df['계산결과'] = result
                for feat in selected_features:
                    download_df[feat] = df_main[feat]
                
                csv_data = download_df.to_csv(index=False)
                st.download_button(
                    label="📥 결과 다운로드 (CSV)",
                    data=csv_data,
                    file_name="analysis_result.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            logger.error(f"❌ 계산 오류: {e}", exc_info=True)
            st.error(f"❌ 계산 오류: {str(e)}")
    
    elif expression and not selected_features:
        st.warning("⚠️ 특징을 선택해주세요")
    elif selected_features and not expression:
        st.info("💡 수식을 입력하거나 빠른 연산 버튼을 사용하세요")


if __name__ == "__main__":
    main()