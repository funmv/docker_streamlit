import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import zipfile
import matplotlib.pyplot as plt
from scipy.signal import correlate
import plotly.graph_objects as go

# =============================================================================
# 초기 설정
# =============================================================================

# 한글 폰트 설정
try:
    from matplotlib import font_manager, rc
    # Windows 환경
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        # Linux 환경
        from matplotlib import font_manager, rc
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)  
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 폰트 로드 실패 시 기본 폰트 사용
        plt.rcParams['axes.unicode_minus'] = False

# matplotlib 경고 제거를 위한 설정
plt.rcParams['figure.max_open_warning'] = 50

# 페이지 설정
st.set_page_config(page_title="다변량 시계열 데이터 분석", layout="wide")

# =============================================================================
# 유틸리티 함수들
# =============================================================================

def normalized_cross_correlation(data, template):
    """정규화된 교차상관 계산"""
    data = np.array(data, dtype=np.float64)
    template = np.array(template, dtype=np.float64)

    data_mean = np.nanmean(data)
    template_mean = np.nanmean(template)
    data_std = np.nanstd(data)
    template_std = np.nanstd(template)

    data_normalized = data - data_mean
    template_normalized = template - template_mean

    data_normalized = np.nan_to_num(data_normalized)
    template_normalized = np.nan_to_num(template_normalized)

    correlation = correlate(data_normalized, template_normalized, mode='valid')
    denominator = data_std * template_std * len(template)
    if denominator == 0:
        return np.zeros_like(correlation)
    ncc = correlation / denominator
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

def get_npy_files_in_data_dir():
    """서버의 /app/data 디렉토리에서 모든 .npy 파일 찾기"""
    data_dir = "/app/data"
    npy_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".npy"):
                npy_files.append(os.path.join(data_dir, file))
    
    return npy_files

def create_download_link_for_all_files(npy_files):
    """모든 .npy 파일을 zip으로 압축하여 다운로드 링크 생성"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in npy_files:
            file_name = os.path.basename(file_path)
            zip_file.write(file_path, file_name)
    
    zip_buffer.seek(0)
    
    st.download_button(
        label="모든 NPY 파일 다운로드",
        data=zip_buffer,
        file_name="all_npy_files.zip",
        mime="application/zip"
    )

# =============================================================================
# 메인 애플리케이션
# =============================================================================

def main():
    st.title("🔍 다변량 시계열 데이터 분석 도구")
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 템플릿 설계", 
        "🔍 템플릿 매칭", 
        "🚀 다채널 신호 관찰", 
        "📁 파일 관리"
    ])
    
    with tab1:
        template_design_tab()
    
    with tab2:
        template_matching_tab()
    
    with tab3:
        multichannel_observation_tab()
    
    with tab4:
        file_management_tab()
    

    # 사이드바 - 공통 설정 (맨 아래로 이동)
    with st.sidebar:
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


# =============================================================================
# 탭 1: 템플릿 설계
# =============================================================================

def template_design_tab():
    st.header("📊 신호 추출을 위한 템플릿 설계")
    
    # 기존 템플릿 업로드
    with st.expander("📂 기존 템플릿 업로드", expanded=False):
        uploaded_template = st.file_uploader("npy 템플릿 업로드", type=["npy"], key="template_upload")
        
        if uploaded_template:
            try:
                template_array = np.load(uploaded_template)
                st.session_state['uploaded_template'] = template_array
                st.success(f"템플릿 shape: {template_array.shape}")
                
                # 템플릿 시각화
                fig_template = go.Figure()
                fig_template.add_trace(go.Scatter(
                    y=template_array,
                    mode='lines',
                    name='Uploaded Template'
                ))
                fig_template.update_layout(
                    title="📈 업로드된 템플릿 시각화",
                    height=300
                )
                st.plotly_chart(fig_template, use_container_width=True)
            except Exception as e:
                st.error(f"템플릿 파일 로드 실패: {e}")
    
    # 새 템플릿 생성
    st.subheader("🆕 새 템플릿 생성")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("ftr(feather) 파일 업로드", type=["ftr", "feather"], key="template_file")
    
    if uploaded_file:
        df = pd.read_feather(uploaded_file)
        st.success(f"파일 로드 완료! 현재 shape: {df.shape}")
        
        # 컬럼 선택
        selected_col = st.selectbox("그래프를 그릴 컬럼을 선택하세요", df.columns.tolist(), key="template_col")
        
        if selected_col:
            # 다운샘플링 비율 설정
            col1, col2 = st.columns(2)
            with col1:
                downsample_rate = st.slider("다운샘플 비율 (1/N)", 1, 50, 10, key="template_downsample")
            
            # 다운샘플링된 데이터 생성
            display_df = df[selected_col].iloc[::downsample_rate].reset_index(drop=True)
            
            # Plotly 그래프 생성
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=np.arange(len(display_df)),
                y=display_df,
                mode='lines',
                name=f"{selected_col} (1/{downsample_rate} 다운샘플)"
            ))
            
            fig.update_layout(
                title="Plotly WebGL 그래프 (다운샘플 적용, Zoom/Pan 가능)",
                dragmode="zoom",
                xaxis=dict(rangeslider=dict(visible=False)),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 템플릿 추출 설정
            st.subheader("템플릿 추출 설정")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x1 = st.number_input("첫 번째 수직선 x좌표", min_value=0, max_value=len(df)-1, value=100, key="x1")
            with col2:
                x2 = st.number_input("두 번째 수직선 x좌표", min_value=0, max_value=len(df)-1, value=200, key="x2")
            with col3:
                template_filename = st.text_input("저장할 템플릿 파일명 (확장자 제외)", value="template", key="template_name")
            
            # 템플릿 추출 및 저장
            if st.button("수직선 추가 및 템플릿 추출/저장", key="extract_template"):
                # 수직선 추가
                fig.add_shape(
                    type="line",
                    x0=x1, y0=0, x1=x1, y1=1,
                    xref='x', yref='paper',
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=x2, y0=0, x1=x2, y1=1,
                    xref='x', yref='paper',
                    line=dict(color="blue", width=2, dash="dash")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 템플릿 데이터 추출
                start_idx = min(x1, x2)
                end_idx = max(x1, x2)
                
                start_idx_original = int(start_idx * downsample_rate)
                end_idx_original = int(end_idx * downsample_rate)
                template_data = df[selected_col].iloc[start_idx_original:end_idx_original+1].to_numpy()
                
                # 세션 상태에 저장
                st.session_state['created_template'] = template_data
                
                # 파일로 저장
                os.makedirs("/app/data", exist_ok=True)
                temp_path = os.path.join("/app/data", f"{template_filename}.npy")
                np.save(temp_path, template_data)
                
                st.success(f"✅ 템플릿이 {template_filename}.npy 로 저장되었습니다!")

# =============================================================================
# 탭 2: 템플릿 매칭
# =============================================================================

def template_matching_tab():
    st.header("🔍 템플릿 매칭 분석")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("분석할 ftr(feather) 파일 업로드", type=["ftr", "feather"], key="matching_file")
    
    if uploaded_file:
        df = pd.read_feather(uploaded_file)
        st.success(f"파일 로드 완료! Shape: {df.shape}")
        
        # 컬럼 선택
        selected_col = st.selectbox("분석할 컬럼을 선택하세요", df.columns.tolist(), key="matching_col")
        
        if selected_col:
            # 템플릿 선택
            st.subheader("템플릿 선택")
            template_source = st.radio(
                "사용할 템플릿을 선택하세요",
                ["새 템플릿 업로드", "업로드된 템플릿", "생성된 템플릿"],
                key="template_source"
            )
            
            template = None
            
            if template_source == "업로드된 템플릿" and 'uploaded_template' in st.session_state:
                template = st.session_state['uploaded_template']
                st.info(f"업로드된 템플릿 사용 (shape: {template.shape})")
                
                # 템플릿 시각화
                fig_template = go.Figure()
                fig_template.add_trace(go.Scatter(
                    y=template,
                    mode='lines',
                    name='Template',
                    line=dict(color='orange', width=2)
                ))
                fig_template.update_layout(
                    title="📈 선택된 템플릿 시각화",
                    height=300,
                    xaxis_title='Sample Index',
                    yaxis_title='Template Value'
                )
                st.plotly_chart(fig_template, use_container_width=True)
                
            elif template_source == "생성된 템플릿" and 'created_template' in st.session_state:
                template = st.session_state['created_template']
                st.info(f"생성된 템플릿 사용 (shape: {template.shape})")
                
                # 템플릿 시각화
                fig_template = go.Figure()
                fig_template.add_trace(go.Scatter(
                    y=template,
                    mode='lines',
                    name='Template',
                    line=dict(color='green', width=2)
                ))
                fig_template.update_layout(
                    title="📈 선택된 템플릿 시각화",
                    height=300,
                    xaxis_title='Sample Index',
                    yaxis_title='Template Value'
                )
                st.plotly_chart(fig_template, use_container_width=True)
                
            elif template_source == "새 템플릿 업로드":
                new_template = st.file_uploader("새 템플릿 파일 업로드", type=["npy"], key="new_template")
                if new_template:
                    template = np.load(new_template)
                    st.info(f"새 템플릿 사용 (shape: {template.shape})")
                    
                    # 템플릿 시각화
                    fig_template = go.Figure()
                    fig_template.add_trace(go.Scatter(
                        y=template,
                        mode='lines',
                        name='Template',
                        line=dict(color='purple', width=2)
                    ))
                    fig_template.update_layout(
                        title="📈 선택된 템플릿 시각화",
                        height=300,
                        xaxis_title='Sample Index',
                        yaxis_title='Template Value'
                    )
                    st.plotly_chart(fig_template, use_container_width=True)
            
            if template is not None:
                # 매칭 설정
                st.subheader("매칭 설정")
                
                with st.form(key="matching_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st_thres = st.slider("NCC Threshold", 0.0, 1.0, 0.2, 0.01)
                        st_low = st.number_input("매칭부 신호 최소값", value=0.0)
                        st_high = st.number_input("매칭부 신호 최대값", value=1.0)
                    
                    with col2:
                        offset_1 = st.number_input("표시 위치 offset", value=500)
                        max_diff = st.number_input("연속으로 간주할 최대 차이값", value=200, 
                                                 help="두 값 사이의 차이가 이 값 이하이면 연속으로 간주합니다.")
                        sampling_rate = st.slider("시각화 샘플링 비율", min_value=1, max_value=50, value=10, step=1)
                    
                    submitted = st.form_submit_button("▶️ 매치 수행")
                
                if submitted:
                    signal = df[selected_col].to_numpy()
                    
                    # 데이터 타입 확인 및 변환
                    if signal.dtype == 'datetime64[ns]' or 'datetime' in str(signal.dtype):
                        st.error("❌ 선택된 컬럼이 datetime 타입입니다. 숫자 타입의 컬럼을 선택해주세요.")
                        return
                    
                    # 숫자 타입으로 변환 시도
                    try:
                        signal = signal.astype(np.float64)
                    except (ValueError, TypeError):
                        st.error("❌ 선택된 컬럼을 숫자 타입으로 변환할 수 없습니다. 다른 컬럼을 선택해주세요.")
                        return
                    
                    # 매칭 수행
                    ncc_start = normalized_cross_correlation(signal, template)
                    ncc_above_threshold = np.where(ncc_start > st_thres)[0]
                    
                    # 유효한 인덱스만 필터링
                    valid_indices = ncc_above_threshold[ncc_above_threshold < len(signal)]
                    
                    true_idx_st = valid_indices[
                        (signal[valid_indices] > st_low) & 
                        (signal[valid_indices] < st_high)
                    ]
                    
                    st_groups = group_consecutive(true_idx_st, max_diff=max_diff)
                    means_start = [np.mean(signal[grp]) for grp in st_groups]
                    
                    # 결과 표시
                    st.subheader(f"🟢 매칭 그룹 수 = {len(st_groups)}")
                    
                    with st.expander("매칭 그룹 평균값 (전체 표시)", expanded=True):
                        st.markdown(
                            f"<div style='max-height: 300px; overflow-y: auto; border:1px solid #ccc; padding:10px;'>"
                            + "<br>".join([f"그룹 {i}: 평균 = {v:.4f}" for i, v in enumerate(means_start)])
                            + "</div>",
                            unsafe_allow_html=True
                        )
                    
                    # 시각화
                    sampled_indices = list(range(0, len(signal), sampling_rate))
                    sampled_signal = signal[sampled_indices]
                    
                    fig1 = go.Figure()
                    
                    # 메인 신호 플롯
                    fig1.add_trace(
                        go.Scatter(
                            x=sampled_indices,
                            y=sampled_signal,
                            mode='lines',
                            name='Signal',
                            line=dict(color='blue', width=1)
                        )
                    )
                    
                    # 매칭 위치 표시
                    for i, grp in enumerate(st_groups):
                        x = grp[0] - offset_1
                        fig1.add_trace(
                            go.Scatter(
                                x=[x, x],
                                y=[min(sampled_signal), max(sampled_signal)],
                                mode='lines',
                                name=f'Match {i}',
                                line=dict(color='red', width=1, dash='dash')
                            )
                        )
                        fig1.add_annotation(
                            x=x,
                            y=max(sampled_signal) * 0.9,
                            text=f"{i}",
                            showarrow=False,
                            font=dict(color='red', size=15)
                        )
                    
                    fig1.update_layout(
                        title=f'Template Matching (정합 위치) - 샘플링 비율: 1/{sampling_rate}',
                        xaxis_title='Sample Index',
                        yaxis_title='Signal Value',
                        height=600,
                        hovermode='closest',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning("템플릿을 선택하거나 업로드해주세요.")

# =============================================================================
# 탭 3: 다채널 신호 관찰
# =============================================================================

def multichannel_observation_tab():
    st.header("🚀 신호 관찰 및 상호 관계 보기")
    
    # 파일 업로드
    uploaded_files = st.file_uploader(
        "FTR/Feather 파일들을 선택하세요",
        type=['ftr', 'feather'],
        accept_multiple_files=True,
        key="multichannel_files"
    )
    
    if uploaded_files:
        # 파일 선택
        if len(uploaded_files) > 1:
            selected_file = st.selectbox(
                "분석할 파일을 선택하세요",
                options=range(len(uploaded_files)),
                format_func=lambda x: uploaded_files[x].name,
                key="selected_multichannel_file"
            )
            df = pd.read_feather(uploaded_files[selected_file])
        else:
            df = pd.read_feather(uploaded_files[0])
        
        st.success(f"✅ Feather 로딩 완료! Shape: {df.shape}")
        
        # 컬럼 선택
        selected_cols = st.multiselect("Plot할 컬럼을 선택하세요", df.columns.tolist(), key="multichannel_cols")
        
        if selected_cols:
            # 설정
            col1, col2 = st.columns(2)
            
            with col1:
                downsample_rate = st.slider("다운샘플 비율", min_value=1, max_value=100, value=10, key="multichannel_downsample")
            
            with col2:
                crosshair = st.checkbox("▶️ 십자선 Hover 활성화", value=True, key="multichannel_crosshair")
            
            # 그래프 생성
            fig = go.Figure()
            
            for col in selected_cols:
                y = df[col].iloc[::downsample_rate]
                x = df.index[::downsample_rate]
                fig.add_trace(go.Scattergl(
                    x=x,
                    y=y,
                    mode='lines',
                    name=col,
                    showlegend=True,
                    hoverinfo='x',
                    hovertemplate=''
                ))
            
            fig.update_layout(
                title="📊 Plotly 그래프 (다운샘플링 적용)",
                dragmode="zoom",
                xaxis=dict(
                    rangeslider=dict(visible=False)
                ),
                height=600
            )
            
            if crosshair:
                fig.update_layout(
                    hovermode="x",
                    xaxis=dict(
                        showspikes=True,
                        spikemode='across',
                        spikesnap='cursor',
                        spikecolor="red",
                        spikethickness=1
                    ),
                    yaxis=dict(
                        showspikes=True,
                        spikemode='across',
                        spikesnap='cursor',
                        spikecolor="blue",
                        spikethickness=1
                    )
                )
            
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 탭 4: 파일 관리
# =============================================================================

def file_management_tab():
    st.header("📁 NPY 파일 관리")
    
    st.subheader("파일 다운로드")
    
    if st.button("NPY 파일 검색 및 다운로드 준비", key="search_files"):
        npy_files = get_npy_files_in_data_dir()
        
        if not npy_files:
            st.warning("디렉토리에 .npy 파일이 없습니다.")
        else:
            st.write(f"총 {len(npy_files)}개의 .npy 파일을 찾았습니다:")
            
            # 파일 목록 표시
            for file_path in npy_files:
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB 단위로 변환
                st.write(f"- **{file_name}** ({file_size:.2f} MB)")
            
            st.divider()
            
            # 전체 파일 ZIP으로 다운로드
            create_download_link_for_all_files(npy_files)

# =============================================================================
# 애플리케이션 실행
# =============================================================================

if __name__ == "__main__":
    main()