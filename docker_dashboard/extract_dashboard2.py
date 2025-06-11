import streamlit as st

# =====================================
# 반드시 첫 번째 Streamlit 명령어!
# =====================================
st.set_page_config(
    page_title="다변량 시계열 데이터 분석", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 이후에 다른 라이브러리들 import
import pandas as pd
import numpy as np
import os
import zipfile
import io
import matplotlib.pyplot as plt
from scipy.signal import correlate
import plotly.graph_objects as go

# =====================================
# 한글 폰트 설정
# =====================================
def setup_korean_font():
    """한글 폰트 설정 함수"""
    try:
        from matplotlib import font_manager, rc
        # Windows 환경
        try:
            font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
            rc('font', family=font_name)
            plt.rcParams['axes.unicode_minus'] = False
            return "Windows 폰트 로드 성공"
        except:
            # Linux 환경
            try:
                font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                rc('font', family=font_name)  
                plt.rcParams['axes.unicode_minus'] = False
                return "Linux 폰트 로드 성공"
            except:
                # 폰트 로드 실패 시 기본 폰트 사용
                plt.rcParams['axes.unicode_minus'] = False
                return "기본 폰트 사용"
    except Exception as e:
        return f"폰트 설정 오류: {e}"

# 폰트 설정 실행
setup_korean_font()

# matplotlib 경고 제거를 위한 설정
plt.rcParams['figure.max_open_warning'] = 50

# =====================================
# 유틸리티 함수들
# =====================================
def normalized_cross_correlation(data, template):
    """정규화된 교차 상관 계산"""
    data_mean = np.mean(data)
    template_mean = np.mean(template)
    data_normalized = data - data_mean
    template_normalized = template - template_mean
    correlation = correlate(data_normalized, template_normalized, mode='valid')
    data_std = np.std(data)
    template_std = np.std(template)
    ncc = correlation / (data_std * template_std * len(template))
    return ncc

def group_consecutive(values, max_diff=50):
    """연속된 값들을 그룹화"""
    if len(values) == 0:
        return []
    values = sorted(values)
    groups = []
    group = [values[0]]
    for current, next_ in zip(values, values[1:]):
        if next_ <= current + max_diff:            
            group.append(next_)
        else:
            groups.append(group)
            group = [next_]
    groups.append(group)
    return groups

def preprocess_signal(signal2):
    """데이터 전처리 - NaN 및 Inf 값 처리"""
    nan_mask = np.isnan(signal2)
    inf_mask = np.isinf(signal2)
    
    if np.any(nan_mask) or np.any(inf_mask):
        # NaN/Inf 값 제거를 위한 복사본 생성
        clean_signal = signal2.copy()
        
        # 단순한 방법: NaN 및 Inf 값을 이웃 값의 평균으로 대체
        bad_indices = np.where(nan_mask | inf_mask)[0]
        for idx in bad_indices:
            # 좌우 10개 샘플 내에서 유효한 값을 찾아 평균 계산
            window_start = max(0, idx - 10)
            window_end = min(len(signal2), idx + 11)
            window = signal2[window_start:window_end]
            valid_values = window[~(np.isnan(window) | np.isinf(window))]
            
            if len(valid_values) > 0:
                clean_signal[idx] = np.mean(valid_values)
            else:
                # 주변에 유효한 값이 없으면 0으로 대체
                clean_signal[idx] = 0
        
        return clean_signal
    
    return signal2

def get_feather_files_in_data_dir():
    """저장된 feather 파일들 찾기"""
    default_root = "/app/data" if os.path.exists("/app/data") else os.getcwd()
    data_dir = os.path.join(default_root, 'saved_crops')
    feather_files = []
    
    # 디렉토리가 존재하지 않으면 빈 리스트 반환
    if not os.path.exists(data_dir):
        return feather_files
    
    # 디렉토리 내의 모든 .feather 파일 찾기
    try:
        for file in os.listdir(data_dir):
            if file.endswith(".feather"):
                feather_files.append(os.path.join(data_dir, file))
    except OSError:
        pass
    
    return feather_files

def create_download_link_for_all_files(feather_files):
    """모든 feather 파일을 zip으로 압축하여 다운로드"""
    if not feather_files:
        st.warning("다운로드할 파일이 없습니다.")
        return
        
    # 메모리에 ZIP 파일 생성
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in feather_files:
            file_name = os.path.basename(file_path)
            # 각 .feather 파일을 zip에 추가
            zip_file.write(file_path, file_name)
    
    zip_buffer.seek(0)
    
    # ZIP 파일 다운로드 버튼 생성
    st.download_button(
        label="📦 모든 Feather 파일 다운로드",
        data=zip_buffer,
        file_name="all_feather_files.zip",
        mime="application/zip"
    )

def downsample(data, rate):
    """신호 다운샘플링"""
    return data[::rate]

# =====================================
# 세션 상태 초기화
# =====================================
def initialize_session_state():
    """세션 상태 초기화"""
    if 'matching_completed' not in st.session_state:
        st.session_state.matching_completed = False
    if 'st_groups' not in st.session_state:
        st.session_state.st_groups = []
    if 'et_groups' not in st.session_state:
        st.session_state.et_groups = []
    if 'signal' not in st.session_state:
        st.session_state.signal = None
    if 'cp_df' not in st.session_state:
        st.session_state.cp_df = None
    if 'offset_1' not in st.session_state:
        st.session_state.offset_1 = 500
    if 'offset_2' not in st.session_state:
        st.session_state.offset_2 = 500

# =====================================
# 메인 애플리케이션
# =====================================
def main():
    # 세션 상태 초기화
    initialize_session_state()
    
    st.title("🚀 신호 매치 및 추출 앱")
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📊 신호 매칭 및 추출", "💾 파일 다운로드"])
    
    with tab1:
        signal_matching_tab()
    
    with tab2:
        file_download_tab()

def signal_matching_tab():
    """탭 1: 신호 매칭 및 추출"""
    
    # 파일 업로드
    uploaded_file = st.file_uploader("📂 Feather (.ftr) 파일을 선택하세요", type=["ftr"])

    if uploaded_file is None:
        st.warning("⏳ Feather 파일을 업로드해주세요.")
        return

    try:
        # 업로드된 feather 파일 읽기
        first_df = pd.read_feather(uploaded_file)
        cp_df = first_df.copy()
        
        # 사용자가 데이터 속성을 선택할 수 있는 드롭다운 메뉴 추가
        selected_column = st.selectbox(
            "매칭할 신호 데이터 컬럼을 선택하세요:",
            options=cp_df.columns.tolist(),
            index=cp_df.columns.tolist().index('GT FUEL CONSUMPTION') if 'GT FUEL CONSUMPTION' in cp_df.columns else 0
        )
        
        # 선택한 컬럼이 존재하는지 확인
        if selected_column in cp_df.columns:
            signal = cp_df[selected_column].values
            st.success(f"✅ '{selected_column}' 컬럼이 성공적으로 로드되었습니다.")
        else:
            st.error(f"❗ '{selected_column}' 컬럼이 존재하지 않습니다.")
            return

    except Exception as e:
        st.error(f"❗ 파일을 읽는 도중 오류 발생: {e}")
        return

    # 사이드바 설정
    template_1, template_2, templates_loaded = setup_sidebar()
    
    if not templates_loaded:
        st.warning("⚠️ 템플릿 파일들을 먼저 업로드해주세요.")
        return

    # 전처리 적용
    signal = preprocess_signal(signal)
    template_1 = preprocess_signal(template_1)
    template_2 = preprocess_signal(template_2)

    # 매칭 설정 폼
    matching_params = setup_matching_form()

    if matching_params['submitted'] and templates_loaded:
        # 매칭 수행
        perform_matching(signal, template_1, template_2, matching_params, cp_df)

    # 매칭이 완료된 경우에만 추출 버튼 표시
    if st.session_state.matching_completed:
        st.markdown("---")
        st.subheader("🔧 신호 추출")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("✂️ 기동 신호 추출 및 시각화", type="primary"):
                with st.spinner("신호 추출 중..."):
                    perform_extraction()
        with col2:
            st.info(f"추출 준비 완료: 시작 그룹 {len(st.session_state.st_groups)}개, 종료 그룹 {len(st.session_state.et_groups)}개")

def setup_sidebar():
    """사이드바 설정"""
    with st.sidebar:
        st.markdown("🧬 **Template 파일 업로드 (.npy)**")

        uploaded_template_1 = st.file_uploader("📂 기동 시작 템플릿 (Template 1)", type=["npy"], key="t1")
        uploaded_template_2 = st.file_uploader("📂 기동 종료 템플릿 (Template 2)", type=["npy"], key="t2")

        # 기본 템플릿 로딩
        template_1 = None
        template_2 = None

        # 기본값으로 로드 (파일이 존재할 경우에만)
        try:
            if os.path.exists('template_ignit_st.npy'):
                template_1 = np.load('template_ignit_st.npy')
                st.info("✅ 기본 Template 1 로드됨")
            else:
                st.warning("⚠️ 기본 Template 1 파일 없음")
        except Exception as e:
            st.error(f"❗ 기본 Template 1 로드 실패: {e}")

        try:
            if os.path.exists('template_ignit_et.npy'):
                template_2 = np.load('template_ignit_et.npy')
                st.info("✅ 기본 Template 2 로드됨")
            else:
                st.warning("⚠️ 기본 Template 2 파일 없음")
        except Exception as e:
            st.error(f"❗ 기본 Template 2 로드 실패: {e}")

        # 업로드가 있다면 덮어쓰기
        if uploaded_template_1 is not None:
            try:
                template_1 = np.load(uploaded_template_1)
                st.success("✅ Template 1 업로드 완료")
            except Exception as e:
                st.error(f"❗ Template 1 로드 실패: {e}")

        if uploaded_template_2 is not None:
            try:
                template_2 = np.load(uploaded_template_2)
                st.success("✅ Template 2 업로드 완료")
            except Exception as e:
                st.error(f"❗ Template 2 로드 실패: {e}")

        # 템플릿이 로드되지 않은 경우 경고 표시
        if template_1 is None:
            st.error("❗ Template 1이 로드되지 않았습니다. 파일을 업로드해주세요.")
        if template_2 is None:
            st.error("❗ Template 2가 로드되지 않았습니다. 파일을 업로드해주세요.")

        # 템플릿이 모두 로드되지 않은 경우 매칭 기능 비활성화
        templates_loaded = template_1 is not None and template_2 is not None

        if templates_loaded:
            # 템플릿 시각화
            st.header("🔧 매치 설정")
            
            st.markdown("📉 **Template 1 (기동 시작)**")
            fig_t1, ax1 = plt.subplots(figsize=(3, 1.5))
            ax1.plot(template_1, linewidth=0.8)
            ax1.set_title("시작 템플릿", fontsize=10)
            st.pyplot(fig_t1)
            plt.close(fig_t1)

            st.markdown("📈 **Template 2 (기동 종료)**")
            fig_t2, ax2 = plt.subplots(figsize=(3, 1.5))
            ax2.plot(template_2, linewidth=0.8, color='orange')
            ax2.set_title("종료 템플릿", fontsize=10)
            st.pyplot(fig_t2)
            plt.close(fig_t2)

            st.markdown("---")

            st.header("🔄 매칭기 설정")
            max_diff = st.selectbox(
                "연속으로 간주할 최대 차이값",
                options=[1, 10, 50, 100, 200, 500, 1000],
                index=2,  # 기본값을 50으로 설정 (index 2)
                help="두 값 사이의 차이가 이 값 이하이면 연속으로 간주합니다."
            )
            st.markdown("---")

        st.markdown(
            """
            <style>
            .bottom-info {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 21rem;
                max-width: 21rem;
                background-color: var(--background-color);
                padding: 1rem;
                border-top: 1px solid var(--border-color);
                z-index: 999;
                box-sizing: border-box;
            }
            .bottom-info hr {
                margin: 0.2rem 0;
                border-color: var(--text-color-light);
                width: 100%;
            }
            </style>
            <div class="bottom-info">
                <hr>
                🧠 <strong>회사명:</strong> ㈜파시디엘<br>
                🏫 <strong>연구실:</strong> visLAB@PNU<br>
                👨‍💻 <strong>제작자:</strong> (C)Dong2<br>
                🛠️ <strong>버전:</strong> V.1.3 (06-03-2025)<br>
                <hr>
            </div>
            """, 
            unsafe_allow_html=True
        )

    return template_1, template_2, templates_loaded

def setup_matching_form():
    """매칭 설정 폼"""
    with st.form(key="matching_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 기동 시작 설정")
            st_thres = st.slider("기동 시작 NCC Threshold", 0.0, 1.0, 0.85, 0.01)
            st_low = st.number_input("기동 시작 신호 최소값", value=-1.0)
            st_high = st.number_input("기동 시작 신호 최대값", value=1.0)
            offset_1 = st.number_input("기동 시작 offset", value=0)
            remove_st_idx = st.text_input("기동 시작부 제거할 그룹 인덱스 (쉼표로 구분)", value="0")
        
        with col2:
            st.subheader("🔴 기동 종료 설정")
            et_thres = st.slider("기동 종료 NCC Threshold", 0.0, 1.0, 0.85, 0.01)
            et_low = st.number_input("기동 종료 신호 최소값", value=0.0)
            et_high = st.number_input("기동 종료 신호 최대값", value=2.0)
            offset_2 = st.number_input("기동 종료 offset", value=300)
            remove_et_idx = st.text_input("기동 종료부 제거할 그룹 인덱스 (쉼표로 구분)", value="0")

        submitted = st.form_submit_button("▶️ 매치 수행", type="primary")

    # 사이드바에서 max_diff 가져오기 (템플릿이 로드된 경우)
    max_diff = 50  # 기본값
    if 'max_diff' in st.session_state:
        max_diff = st.session_state.max_diff

    return {
        'submitted': submitted,
        'st_thres': st_thres,
        'st_low': st_low,
        'st_high': st_high,
        'offset_1': offset_1,
        'et_thres': et_thres,
        'et_low': et_low,
        'et_high': et_high,
        'offset_2': offset_2,
        'remove_st_idx': remove_st_idx,
        'remove_et_idx': remove_et_idx,
        'max_diff': max_diff
    }

def perform_matching(signal, template_1, template_2, params, cp_df):
    """매칭 수행 함수"""
    
    # 시작부 매칭
    ncc_start = normalized_cross_correlation(signal, template_1)
    st_ncc_above_threshold = np.where(ncc_start > params['st_thres'])[0]
    true_idx_st = st_ncc_above_threshold[
        (signal[st_ncc_above_threshold] > params['st_low']) & 
        (signal[st_ncc_above_threshold] < params['st_high'])
    ]    
    st_groups = group_consecutive(true_idx_st, params['max_diff'])

    for idx in sorted([int(i) for i in params['remove_st_idx'].split(',') if i.strip().isdigit()], reverse=True):
        if 0 <= idx < len(st_groups):
            del st_groups[idx]

    means_start = [np.mean(signal[grp]) for grp in st_groups]
    st.subheader(f"🟢 기동 시작: 그룹 수 = {len(st_groups)}")
    with st.expander("기동 시작 그룹 평균값 (전체 표시)", expanded=True):
        st.markdown(
            f"<div style='max-height: 300px; overflow-y: auto; border:1px solid #ccc; padding:10px;'>"
            + "<br>".join([f"그룹 {i}: 평균 = {v:.4f}" for i, v in enumerate(means_start)])
            + "</div>",
            unsafe_allow_html=True
        )

    # 종료부 매칭
    ncc_end = normalized_cross_correlation(signal, template_2)
    et_ncc_above_threshold = np.where(ncc_end > params['et_thres'])[0]
    true_idx_et = et_ncc_above_threshold[
        (signal[et_ncc_above_threshold] > params['et_low']) & 
        (signal[et_ncc_above_threshold] < params['et_high'])
    ]    
    et_groups = group_consecutive(true_idx_et, params['max_diff'])

    for idx in sorted([int(i) for i in params['remove_et_idx'].split(',') if i.strip().isdigit()], reverse=True):
        if 0 <= idx < len(et_groups):
            del et_groups[idx]

    means_end = [np.mean(signal[grp]) for grp in et_groups]
    st.subheader(f"🔴 기동 종료: 그룹 수 = {len(et_groups)}")
    with st.expander("기동 종료 그룹 평균값 (전체 표시)", expanded=True):
        st.markdown(
            f"<div style='max-height: 300px; overflow-y: auto; border:1px solid #ccc; padding:10px;'>"
            + "<br>".join([f"그룹 {i}: 평균 = {v:.4f}" for i, v in enumerate(means_end)])
            + "</div>",
            unsafe_allow_html=True
        )

    # 세션 상태에 저장
    st.session_state.st_groups = st_groups
    st.session_state.et_groups = et_groups
    st.session_state.signal = signal
    st.session_state.cp_df = cp_df
    st.session_state.offset_1 = params['offset_1']
    st.session_state.offset_2 = params['offset_2']
    st.session_state.matching_completed = True

    # 시각화
    create_visualization(signal, st_groups, et_groups, params['offset_1'], params['offset_2'])

def create_visualization(signal, st_groups, et_groups, offset_1, offset_2):
    """매칭 결과 시각화"""
    
    # 샘플링 비율 선택 위젯 추가 (기본값: 1)
    sampling_rate = st.slider("샘플링 비율 선택", min_value=1, max_value=50, value=1, step=1)

    # 샘플링된 신호와 인덱스 생성
    sampled_signal = downsample(signal, sampling_rate)
    sampled_indices = list(range(0, len(signal), sampling_rate))

    # 기동 시작 매칭 시각화
    fig1 = go.Figure()

    # 메인 신호 플롯 (샘플링 적용)
    fig1.add_trace(
        go.Scatter(
            x=sampled_indices,
            y=sampled_signal,
            mode='lines',
            name='Signal',
            line=dict(color='blue', width=1)
        )
    )

    # 매칭 위치 표시 (샘플링 적용하지 않음 - 정확한 위치 유지)
    for i, grp in enumerate(st_groups):
        x = grp[0] - offset_1 + len(grp)//2  # Kang 수정 부분
        fig1.add_trace(
            go.Scatter(
                x=[x, x],
                y=[min(sampled_signal), max(sampled_signal)],
                mode='lines',
                name=f'Match {i}',
                line=dict(color='red', width=1, dash='dash')
            )
        )
        # 텍스트 레이블 추가
        fig1.add_annotation(
            x=x,
            y=max(sampled_signal) * 0.9,
            text=f"{i}",
            showarrow=False,
            font=dict(color='red', size=15)
        )

    # 레이아웃 설정
    fig1.update_layout(
        title=f'Template 1 Matching (Start) - 샘플링 비율: 1/{sampling_rate}',
        xaxis_title='Sample Index',
        yaxis_title='Signal Value',
        height=600,
        hovermode='closest',
        showlegend=False
    )

    # 플롯 표시
    st.plotly_chart(fig1, use_container_width=True)

    # 기동 종료 매칭 시각화
    fig2 = go.Figure()

    # 메인 신호 플롯 (샘플링 적용)
    fig2.add_trace(
        go.Scatter(
            x=sampled_indices,
            y=sampled_signal,
            mode='lines',
            name='Signal',
            line=dict(color='blue', width=1)
        )
    )

    # 매칭 위치 표시 (샘플링 적용하지 않음 - 정확한 위치 유지)
    for i, grp in enumerate(et_groups):
        x = grp[0] + offset_2
        fig2.add_trace(
            go.Scatter(
                x=[x, x],
                y=[min(sampled_signal), max(sampled_signal)],
                mode='lines',
                name=f'Match {i}',
                line=dict(color='red', width=1, dash='dash')
            )
        )
        # 텍스트 레이블 추가
        fig2.add_annotation(
            x=x,
            y=max(sampled_signal) * 0.9,
            text=f"{i}",
            showarrow=False,
            font=dict(color='red', size=15)
        )

    # 레이아웃 설정
    fig2.update_layout(
        title=f'Template 2 Matching (End) - 샘플링 비율: 1/{sampling_rate}',
        xaxis_title='Sample Index',
        yaxis_title='Signal Value',
        height=600,
        hovermode='closest',
        showlegend=False
    )

    # 플롯 표시
    st.plotly_chart(fig2, use_container_width=True)

def perform_extraction():
    """신호 추출 및 저장"""
    
    # 세션 상태에서 데이터 가져오기
    signal = st.session_state.signal
    cp_df = st.session_state.cp_df
    st_groups = st.session_state.st_groups
    et_groups = st.session_state.et_groups
    offset_1 = st.session_state.offset_1
    offset_2 = st.session_state.offset_2
    
    if len(st_groups) == 0 or len(et_groups) == 0:
        st.warning("시작 또는 종료 그룹이 비어 있어 추출할 수 없습니다.")
        return

    st.success("✅ 추출 및 시각화 실행 중...")
    pairs = []
    for st_grp, et_grp in zip(st_groups, et_groups):
        st_pt = max(0, st_grp[0] - offset_1 + len(st_grp)//2)  ###### Kang
        et_pt = min(len(signal), et_grp[0] + offset_2)
        if st_pt < et_pt:
            pairs.append((st_pt, et_pt))

    st.write(f"총 추출 구간 수: {len(pairs)}")

    # feather 파일로 저장
    saved_root = "/app/data" if os.path.exists("/app/data") else os.getcwd()
    save_folder = os.path.join(saved_root, 'saved_crops')
    os.makedirs(save_folder, exist_ok=True)

    progress_bar = st.progress(0)
    for i, (st_pt, et_pt) in enumerate(pairs):
        # 1. 다변량 crop (DataFrame 그대로 유지)
        crop_df = cp_df.iloc[st_pt:et_pt].copy()  # 모든 컬럼에 대해 crop

        # 2. 인덱스 리셋 (필요시)
        crop_df.reset_index(drop=True, inplace=True)

        # 3. 저장 파일명 생성 (날짜+시간 기반)
        if 'timestamp' in crop_df.columns:  # 'timestamp' 컬럼이 datetime 형태라면
            # st_pt 위치의 날짜/시간을 기반으로 파일명 생성
            timestamp = pd.to_datetime(crop_df['timestamp'].iloc[0])
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        else:
            timestamp_str = f"crop_{i}"

        save_path = os.path.join(save_folder, f"{timestamp_str}_crop.feather")

        # 4. feather 저장 (시간정보와 특징명 모두 보존)
        crop_df.to_feather(save_path)

        # 5. crop 시각화 (특징 하나만 간단히 예시)
        # if i < 5:  # 처음 5개만 시각화
        fig_crop, ax_crop = plt.subplots(figsize=(8, 3))
        ax_crop.plot(signal[st_pt:et_pt])   
        ax_crop.set_title(f"추출 신호 {i} (len={len(crop_df)})")
        st.pyplot(fig_crop)
        plt.close(fig_crop)

        # 진행률 업데이트
        progress_bar.progress((i + 1) / len(pairs))
        
        # if i < 5:  # 처음 5개만 표시
        st.info(f"✅ Saved: {save_path}")

    st.success("🎉 모든 구간 추출 완료!")

def file_download_tab():
    """탭 2: 파일 다운로드"""
    st.header("💾 Feather 파일 다운로드")

    # 다운로드 버튼 표시
    if st.button("Feather 파일 검색 및 다운로드 준비"):
        # 버튼이 클릭되었을 때만 아래 코드 실행
        
        # .feather 파일 찾기
        feather_files = get_feather_files_in_data_dir()
        
        if not feather_files:
            st.warning("디렉토리에 .feather 파일이 없습니다.")
        else:
            # 파일 목록 표시
            st.write(f"총 {len(feather_files)}개의 .feather 파일을 찾았습니다:")
            
            # 파일 목록을 표시
            for file_path in feather_files:
                file_name = os.path.basename(file_path)
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB 단위로 변환
                    st.write(f"- **{file_name}** ({file_size:.2f} MB)")
                except OSError:
                    st.write(f"- **{file_name}** (크기 정보 없음)")
            
            # 구분선 추가
            st.divider()
            
            # 전체 파일 ZIP으로 다운로드 옵션
            create_download_link_for_all_files(feather_files)


    # 프로그램 기능 설명
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("📋 프로그램 기능 안내")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 주요 기능")
        st.markdown("""
        **Tab 1: 신호 매칭 및 추출**
        - Feather 파일 업로드 및 신호 컬럼 선택
        - 기동 시작/종료 템플릿 매칭 (.npy 파일)
        - 정규화된 교차 상관(NCC) 기반 패턴 탐지
        - 매칭 임계값 및 신호 범위 설정
        - 연속 구간 그룹화 및 오프셋 조정
        
        **Tab 2: 파일 다운로드**
        - 추출된 Feather 파일 관리
        - 전체 파일 ZIP 압축 다운로드
        - 파일 크기 및 정보 확인
        """)
    
    with col2:
        st.subheader("⚙️ 분석 도구")
        st.markdown("""
        **신호 전처리**
        - NaN/Inf 값 자동 처리 및 보간
        - 신호 다운샘플링 표시
        - 사용자 정의 샘플링 비율 설정
        - 연속값 그룹화를 위한 최대 차이값 조정
        
        **시각화 기능**
        - Plotly 기반 인터랙티브 시각화
        - 매칭 위치 수직선 및 레이블 표시
        - 실시간 샘플링 비율 조정
        - 추출된 신호 구간별 개별 플롯
        """)
    
    st.markdown("---")
    st.info("💡 **사용 팁**: 먼저 기동 시작/종료 템플릿을 업로드하고, Feather 파일에서 매칭할 신호를 선택한 후 매칭 수행 → 신호 추출 순서로 진행하세요.")
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <small>이 도구는 엔진 기동 신호 패턴을 탐지하고 해당 구간을 자동으로 추출하기 위해 설계되었습니다.</small>
    </div>
    """, unsafe_allow_html=True)    

# =====================================
# 앱 실행
# =====================================
if __name__ == "__main__":
    main()