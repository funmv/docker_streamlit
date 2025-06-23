import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import zipfile
import matplotlib.pyplot as plt
from scipy.signal import correlate
import plotly.graph_objects as go
import pdb

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
    """정규화된 교차상관 계산 (개선된 버전)"""
    data = np.array(data, dtype=np.float64)
    template = np.array(template, dtype=np.float64)
    
    # NaN이나 무한대 값이 있는지 확인
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        st.warning("⚠️ 신호 데이터에 NaN 또는 무한대 값이 포함되어 있습니다.")
        # NaN/inf 값을 0으로 대체하지만 이를 기록
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(np.isnan(template)) or np.any(np.isinf(template)):
        st.warning("⚠️ 템플릿 데이터에 NaN 또는 무한대 값이 포함되어 있습니다.")
        template = np.nan_to_num(template, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 데이터가 모두 동일한 값인지 확인 (표준편차가 0인 경우)
    data_std = np.std(data)
    template_std = np.std(template)
    
    if data_std == 0 or template_std == 0:
        st.warning("⚠️ 신호 또는 템플릿의 표준편차가 0입니다. (모든 값이 동일)")
        return np.zeros(len(data) - len(template) + 1)
    
    # 평균과 표준편차 계산
    data_mean = np.mean(data)
    template_mean = np.mean(template)
    
    # 정규화
    data_normalized = (data - data_mean) / data_std
    template_normalized = (template - template_mean) / template_std
    
    # 교차상관 계산
    correlation = correlate(data_normalized, template_normalized, mode='valid')
    
    # 정규화된 교차상관 계산
    ncc = correlation / len(template)
    
    return ncc

def is_valid_signal_segment(signal_segment, min_valid_ratio=0.8):
    """신호 구간이 유효한지 확인하는 함수"""
    if len(signal_segment) == 0:
        return False
    
    # NaN, 무한대 값의 비율 확인
    invalid_count = np.sum(np.isnan(signal_segment) | np.isinf(signal_segment))
    valid_ratio = 1 - (invalid_count / len(signal_segment))
    
    if valid_ratio < min_valid_ratio:
        return False
    
    # 모든 값이 동일한지 확인
    if np.std(signal_segment) == 0:
        return False
    
    # 값의 범위가 너무 작은지 확인 (거의 모든 값이 비슷한 경우)
    signal_range = np.max(signal_segment) - np.min(signal_segment)
    if signal_range < 1e-6:  # 매우 작은 변화만 있는 경우
        return False
    
    return True

def filter_valid_matches(indices, signal, template_length, min_valid_ratio=0.8):
    """매칭된 위치에서 실제 신호가 유효한지 확인하여 필터링"""
    valid_indices = []
    
    for idx in indices:
        # 템플릿 길이만큼의 신호 구간 추출
        if idx + template_length <= len(signal):
            signal_segment = signal[idx:idx + template_length]
            
            # 신호 구간의 유효성 검사
            if is_valid_signal_segment(signal_segment, min_valid_ratio):
                valid_indices.append(idx)
    
    return np.array(valid_indices)


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
    # pdb.set_trace()
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
# 유틸리티 함수 추가 (기존 유틸리티 함수들 뒤에 추가)
# =============================================================================
def get_data_segment(df: pd.DataFrame, num_segments: int = 3, selected_segment: int = 0) -> pd.DataFrame:
    """데이터를 등분하여 선택된 구간만 반환하는 함수"""
    total_length = len(df)
    segment_length = total_length // num_segments
    
    start_idx = selected_segment * segment_length
    
    # 마지막 구간의 경우 남은 모든 데이터 포함
    if selected_segment == num_segments - 1:
        end_idx = total_length
    else:
        end_idx = start_idx + segment_length
    
    return df.iloc[start_idx:end_idx].copy()





# =============================================================================
# 메인 애플리케이션
# =============================================================================

def main():
    st.title("🔍 다변량 시계열 데이터 분석 도구")
    
    # 탭 생성 (탭5 추가)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 템플릿 설계", 
        "🔍 템플릿 매칭", 
        "🚀 다채널 신호 관찰", 
        "📁 파일 관리",
        "🔄 배치 템플릿 매칭"  # 새로 추가된 탭
    ])
    
    with tab1:
        template_design_tab()
    
    with tab2:
        template_matching_tab()
    
    with tab3:
        multichannel_observation_tab()
    
    with tab4:
        file_management_tab()
    
    with tab5:  # 새로 추가된 탭
        batch_template_matching_tab()

    

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
            # 데이터 분할 및 다운샘플링 설정
            st.subheader("⚙️ 그래프 설정")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                downsample_rate = st.slider("다운샘플 비율 (1/N)", 1, 50, 1, key="template_downsample")
            
            with col2:
                num_segments = st.selectbox(
                    "📊 데이터 분할 수",
                    options=[1, 2, 3, 4, 5],
                    index=4,  # 기본값: 5등분
                    help="전체 데이터를 몇 등분할지 선택",
                    key="template_segments"
                )
            
            with col3:
                selected_segment = st.selectbox(
                    "🎯 표시 구간 선택",
                    options=list(range(num_segments)),
                    format_func=lambda x: f"구간 {x+1}",
                    index=0,  # 기본값: 첫 번째 구간
                    help="표시할 구간을 선택",
                    key="template_segment_select"
                )
            
            # 데이터 구간 정보 표시
            total_length = len(df)
            segment_length = total_length // num_segments
            start_idx = selected_segment * segment_length
            end_idx = start_idx + segment_length if selected_segment < num_segments - 1 else total_length
            
            st.info(f"📊 **선택된 구간**: {start_idx:,} ~ {end_idx:,} (총 {end_idx - start_idx:,}개 포인트, 전체의 {((end_idx - start_idx) / total_length * 100):.1f}%)")
            
            # 선택된 구간 데이터 추출
            df_segment = get_data_segment(df, num_segments, selected_segment)
            
            # 다운샘플링된 데이터 생성 (선택된 구간에서)
            display_df = df_segment[selected_col].iloc[::downsample_rate].reset_index(drop=True)
            
            # Plotly 그래프 생성
            fig = go.Figure()
            
            # 선택된 구간의 인덱스 계산 (원본 데이터 기준)
            segment_indices = np.arange(start_idx, end_idx, downsample_rate)[:len(display_df)]
            
            fig.add_trace(go.Scattergl(
                x=segment_indices,  # 원본 데이터 기준 인덱스 사용
                y=display_df,
                mode='lines',
                name=f"{selected_col} (구간 {selected_segment+1}/{num_segments}, 1/{downsample_rate} 다운샘플)"
            ))
            
            fig.update_layout(
                title=f"📊 Plotly WebGL 그래프 - 구간 {selected_segment+1}/{num_segments} (다운샘플 적용, Zoom/Pan 가능)",
                dragmode="zoom",
                xaxis=dict(
                    rangeslider=dict(visible=False),
                    title="원본 데이터 인덱스"
                ),
                yaxis=dict(
                    title="신호 값"
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 템플릿 추출 설정
            st.subheader("🎯 템플릿 추출 설정")
            st.markdown("**주의**: 템플릿 추출 좌표는 원본 데이터 기준입니다.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x1 = st.number_input("첫 번째 수직선 x좌표 (원본 기준)", min_value=0, max_value=len(df)-1, value=max(start_idx, 100), key="x1")
            with col2:
                x2 = st.number_input("두 번째 수직선 x좌표 (원본 기준)", min_value=0, max_value=len(df)-1, value=min(end_idx-1, max(start_idx, 200)), key="x2")
            with col3:
                template_filename = st.text_input("저장할 템플릿 파일명 (확장자 제외)", value="template", key="template_name")
            
            # 좌표 유효성 검사
            if x1 >= x2:
                st.warning("⚠️ 첫 번째 좌표가 두 번째 좌표보다 크거나 같습니다. 좌표를 확인해주세요.")
            elif not (start_idx <= x1 < end_idx and start_idx <= x2 < end_idx):
                st.warning(f"⚠️ 입력된 좌표가 현재 표시 구간({start_idx:,} ~ {end_idx:,}) 밖에 있습니다. 다른 구간을 선택하거나 좌표를 조정해주세요.")
            else:
                # 템플릿 추출 및 저장
                if st.button("수직선 추가 및 템플릿 추출/저장", key="extract_template"):
                    # 템플릿 데이터 추출 (원본 데이터에서)
                    start_template = min(x1, x2)
                    end_template = max(x1, x2)
                    
                    template_data = df[selected_col].iloc[start_template:end_template+1].to_numpy()
                    
                    # 세션 상태에 저장
                    st.session_state['created_template'] = template_data
                    st.session_state['template_extracted'] = True
                    st.session_state['initial_start'] = start_template
                    st.session_state['initial_end'] = end_template
                    st.session_state['template_filename'] = template_filename
                    st.session_state['selected_col'] = selected_col
                    st.session_state['df_data'] = df  # 데이터프레임도 저장
                    
                    # 파일로 저장
                    os.makedirs("/app/data", exist_ok=True)
                    temp_path = os.path.join("/app/data", f"{template_filename}.npy")
                    np.save(temp_path, template_data)
                    
                    st.success(f"✅ 템플릿이 {template_filename}.npy 로 저장되었습니다!")
                
                # 선택된 영역을 현재 그래프에 표시 (좌표가 유효한 경우)
                if start_idx <= x1 < end_idx and start_idx <= x2 < end_idx:
                    # 수직선을 추가한 그래프 다시 생성
                    fig_with_lines = go.Figure()
                    
                    fig_with_lines.add_trace(go.Scattergl(
                        x=segment_indices,
                        y=display_df,
                        mode='lines',
                        name=f"{selected_col} (구간 {selected_segment+1}/{num_segments})",
                        line=dict(color='blue', width=1)
                    ))
                    
                    # 수직선 추가
                    y_min, y_max = display_df.min(), display_df.max()
                    
                    # 첫 번째 수직선 (빨간색)
                    fig_with_lines.add_shape(
                        type="line",
                        x0=x1, y0=y_min, x1=x1, y1=y_max,
                        line=dict(color="red", width=3, dash="dash")
                    )
                    
                    # 두 번째 수직선 (파란색)
                    fig_with_lines.add_shape(
                        type="line",
                        x0=x2, y0=y_min, x1=x2, y1=y_max,
                        line=dict(color="blue", width=3, dash="dash")
                    )
                    
                    # 선택 영역 하이라이트
                    fig_with_lines.add_shape(
                        type="rect",
                        x0=min(x1, x2), y0=y_min,
                        x1=max(x1, x2), y1=y_max,
                        fillcolor="yellow",
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                    )
                    
                    # 수직선 라벨 추가
                    fig_with_lines.add_annotation(
                        x=x1, y=y_max * 0.95,
                        text=f"시작: {x1}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="red",
                        font=dict(color="red", size=12)
                    )
                    
                    fig_with_lines.add_annotation(
                        x=x2, y=y_max * 0.85,
                        text=f"끝: {x2}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="blue",
                        font=dict(color="blue", size=12)
                    )
                    
                    fig_with_lines.update_layout(
                        title=f"📊 템플릿 추출 위치 - 구간 {selected_segment+1}/{num_segments} (좌표: {min(x1,x2)}~{max(x1,x2)})",
                        dragmode="zoom",
                        xaxis=dict(
                            title="원본 데이터 인덱스",
                            rangeslider=dict(visible=False)
                        ),
                        yaxis=dict(title="신호 값"),
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_with_lines, use_container_width=True)
            


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
            # 선택된 컬럼의 데이터 품질 확인
            signal_data = df[selected_col]
            
            # 기본 데이터 품질 정보 표시
            with st.expander("📊 선택된 신호 데이터 품질 정보"):
                total_points = len(signal_data)
                nan_count = signal_data.isna().sum()
                inf_count = np.sum(np.isinf(signal_data.replace([np.inf, -np.inf], np.nan).dropna()))
                valid_points = total_points - nan_count - inf_count
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("전체 데이터 포인트", f"{total_points:,}")
                with col2:
                    st.metric("유효 데이터 포인트", f"{valid_points:,}")
                with col3:
                    st.metric("유효 데이터 비율", f"{(valid_points/total_points*100):.1f}%")
                
                if nan_count > 0:
                    st.warning(f"⚠️ NaN 값: {nan_count:,}개 ({nan_count/total_points*100:.1f}%)")
                if inf_count > 0:
                    st.warning(f"⚠️ 무한대 값: {inf_count:,}개 ({inf_count/total_points*100:.1f}%)")
                
                # 신호의 기본 통계
                try:
                    valid_signal = signal_data.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(valid_signal) > 0:
                        st.write(f"**평균**: {valid_signal.mean():.4f}")
                        st.write(f"**표준편차**: {valid_signal.std():.4f}")
                        st.write(f"**최소값**: {valid_signal.min():.4f}")
                        st.write(f"**최대값**: {valid_signal.max():.4f}")
                    else:
                        st.error("❌ 유효한 데이터가 없습니다.")
                except Exception as e:
                    st.error(f"❌ 통계 계산 중 오류: {str(e)}")
            
            # 템플릿 선택
            st.subheader("📋 템플릿 선택")
            template_source = st.radio(
                "사용할 템플릿을 선택하세요",
                ["새 템플릿 업로드", "업로드된 템플릿", "생성된 템플릿"],
                key="template_source"
            )
            
            template = None
            
            if template_source == "업로드된 템플릿" and 'uploaded_template' in st.session_state:
                template = st.session_state['uploaded_template']
                st.info(f"업로드된 템플릿 사용 (shape: {template.shape})")
                
            elif template_source == "생성된 템플릿" and 'created_template' in st.session_state:
                template = st.session_state['created_template']
                st.info(f"생성된 템플릿 사용 (shape: {template.shape})")
                
            elif template_source == "새 템플릿 업로드":
                new_template = st.file_uploader("새 템플릿 파일 업로드", type=["npy"], key="new_template")
                if new_template:
                    template = np.load(new_template)
                    st.info(f"새 템플릿 사용 (shape: {template.shape})")
            
            # 템플릿이 선택된 경우 품질 확인
            if template is not None:
                # 템플릿 품질 확인
                with st.expander("🔍 템플릿 품질 정보"):
                    template_nan_count = np.sum(np.isnan(template))
                    template_inf_count = np.sum(np.isinf(template))
                    template_valid = len(template) - template_nan_count - template_inf_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("템플릿 길이", len(template))
                    with col2:
                        st.metric("유효 포인트", template_valid)
                    with col3:
                        st.metric("유효 비율", f"{(template_valid/len(template)*100):.1f}%")
                    
                    if template_nan_count > 0:
                        st.warning(f"⚠️ 템플릿에 NaN 값: {template_nan_count}개")
                    if template_inf_count > 0:
                        st.warning(f"⚠️ 템플릿에 무한대 값: {template_inf_count}개")
                    
                    # 템플릿 통계
                    try:
                        clean_template = np.nan_to_num(template, nan=0.0, posinf=0.0, neginf=0.0)
                        if np.std(clean_template) > 0:
                            st.write(f"**템플릿 평균**: {np.mean(clean_template):.4f}")
                            st.write(f"**템플릿 표준편차**: {np.std(clean_template):.4f}")
                        else:
                            st.error("❌ 템플릿의 표준편차가 0입니다. (모든 값이 동일)")
                    except Exception as e:
                        st.error(f"❌ 템플릿 통계 계산 중 오류: {str(e)}")
                
                # 템플릿 시각화
                fig_template = go.Figure()
                fig_template.add_trace(go.Scatter(
                    y=template,
                    mode='lines',
                    name='Selected Template',
                    line=dict(color='orange', width=2)
                ))
                fig_template.update_layout(
                    title="📈 선택된 템플릿 시각화",
                    height=300,
                    xaxis_title='Template Index',
                    yaxis_title='Template Value'
                )
                st.plotly_chart(fig_template, use_container_width=True)
                
                # 매칭 설정
                st.subheader("⚙️ 매칭 설정")
                
                with st.form(key="matching_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st_thres = st.slider("NCC Threshold", 0.0, 1.0, 0.2, 0.01)
                        st_low = st.number_input("매칭부 신호 최소값", value=0.0)
                        st_high = st.number_input("매칭부 신호 최대값", value=1.0)
                        
                        # 추가된 설정: 데이터 품질 관련
                        min_valid_ratio = st.slider(
                            "최소 유효 데이터 비율", 
                            0.5, 1.0, 0.8, 0.05,
                            help="매칭 구간에서 유효한 데이터의 최소 비율"
                        )
                    
                    with col2:
                        offset_1 = st.number_input("표시 위치 offset", value=1000)
                        max_diff = st.number_input("연속으로 간주할 최대 차이값", value=200, 
                                                 help="두 값 사이의 차이가 이 값 이하이면 연속으로 간주합니다.")
                        sampling_rate = st.slider("시각화 샘플링 비율", min_value=1, max_value=50, value=10, step=1)
                        
                        # 추가된 설정: 엄격한 매칭 모드
                        strict_mode = st.checkbox(
                            "엄격한 매칭 모드", 
                            value=True,
                            help="NaN/무한대 값이 있는 구간을 자동으로 제외"
                        )
                    
                    submitted = st.form_submit_button("▶️ 매치 수행")
                
                if submitted:
                    try:
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
                        
                        # 신호 길이 확인
                        if len(signal) < len(template):
                            st.error(f"❌ 신호 길이({len(signal)})가 템플릿 길이({len(template)})보다 짧습니다.")
                            return
                        
                        # 매칭 수행
                        st.write("🔄 NCC 계산 중...")
                        ncc_start = normalized_cross_correlation(signal, template)
                        
                        if len(ncc_start) == 0:
                            st.error("❌ NCC 계산 결과가 없습니다.")
                            return
                        
                        # 1단계: NCC 임계값 필터링
                        ncc_above_threshold = np.where(ncc_start > st_thres)[0]
                        st.write(f"📊 NCC 임계값({st_thres}) 통과: {len(ncc_above_threshold)}개 위치")
                        
                        if len(ncc_above_threshold) == 0:
                            st.warning(f"⚠️ NCC 임계값 {st_thres}을 만족하는 위치가 없습니다. 임계값을 낮춰보세요.")
                            return
                        
                        # 2단계: 유효한 인덱스만 필터링 (신호 범위 내)
                        valid_indices = ncc_above_threshold[ncc_above_threshold < len(signal)]
                        st.write(f"📊 유효 인덱스 범위 내: {len(valid_indices)}개 위치")
                        
                        # 3단계: 엄격한 매칭 모드에서 데이터 품질 검사
                        if strict_mode:
                            st.write("🔍 데이터 품질 검사 중...")
                            quality_valid_indices = filter_valid_matches(
                                valid_indices, signal, len(template), min_valid_ratio
                            )
                            st.write(f"📊 품질 검사 통과: {len(quality_valid_indices)}개 위치")
                            valid_indices = quality_valid_indices
                        
                        # 4단계: 신호값 범위 필터링
                        if len(valid_indices) > 0:
                            signal_values_at_indices = signal[valid_indices]
                            
                            # NaN 값 제거
                            nan_mask = ~np.isnan(signal_values_at_indices)
                            valid_indices = valid_indices[nan_mask]
                            signal_values_at_indices = signal_values_at_indices[nan_mask]
                            
                            # 범위 필터링
                            range_mask = (signal_values_at_indices > st_low) & (signal_values_at_indices < st_high)
                            true_idx_st = valid_indices[range_mask]
                            
                            st.write(f"📊 신호값 범위({st_low}~{st_high}) 필터링 후: {len(true_idx_st)}개 위치")
                        else:
                            true_idx_st = np.array([])
                        
                        # 5단계: 연속 그룹화
                        if len(true_idx_st) > 0:
                            st_groups = group_consecutive(true_idx_st, max_diff=max_diff)
                            means_start = [np.mean(signal[grp]) for grp in st_groups]
                            
                            # 결과 표시
                            st.subheader(f"🟢 매칭 그룹 수 = {len(st_groups)}")
                            
                            # 각 그룹의 품질 정보 표시
                            with st.expander("📊 매칭 그룹 품질 정보"):
                                for i, grp in enumerate(st_groups):
                                    group_signal = signal[grp[0]:grp[-1]+len(template)]
                                    valid_in_group = is_valid_signal_segment(group_signal)
                                    quality_status = "✅ 양호" if valid_in_group else "⚠️ 주의"
                                    
                                    nan_in_group = np.sum(np.isnan(group_signal))
                                    st.write(f"그룹 {i}: 위치 {grp[0]}~{grp[-1]}, 평균 = {means_start[i]:.4f}, "
                                           f"품질: {quality_status}, NaN: {nan_in_group}개")
                            
                            with st.expander("매칭 그룹 평균값 (전체 표시)", expanded=False):
                                st.markdown(
                                    f"<div style='max-height: 300px; overflow-y: auto; border:1px solid #ccc; padding:10px;'>"
                                    + "<br>".join([f"그룹 {i}: 평균 = {v:.4f}" for i, v in enumerate(means_start)])
                                    + "</div>",
                                    unsafe_allow_html=True
                                )
                            
                            # 시각화
                            sampled_indices = list(range(0, len(signal), sampling_rate))
                            sampled_signal = signal[sampled_indices]
                            
                            # NaN 값이 있는 경우 처리
                            nan_mask = ~np.isnan(sampled_signal)
                            clean_sampled_indices = np.array(sampled_indices)[nan_mask]
                            clean_sampled_signal = sampled_signal[nan_mask]
                            
                            fig1 = go.Figure()
                            
                            # 메인 신호 플롯
                            fig1.add_trace(
                                go.Scatter(
                                    x=clean_sampled_indices,
                                    y=clean_sampled_signal,
                                    mode='lines',
                                    name='Signal',
                                    line=dict(color='blue', width=1)
                                )
                            )
                            
                            # NaN 값이 있는 위치 표시 (선택사항)
                            if len(clean_sampled_indices) < len(sampled_indices):
                                nan_indices = np.array(sampled_indices)[~nan_mask]
                                fig1.add_trace(
                                    go.Scatter(
                                        x=nan_indices,
                                        y=[0] * len(nan_indices),  # 0 라인에 표시
                                        mode='markers',
                                        name='NaN Values',
                                        marker=dict(color='red', size=3, symbol='x')
                                    )
                                )
                            
                            # 매칭 위치 표시
                            for i, grp in enumerate(st_groups):
                                x = grp[0] - offset_1 + len(grp)//2
                                fig1.add_trace(
                                    go.Scatter(
                                        x=[x, x],
                                        y=[min(clean_sampled_signal), max(clean_sampled_signal)],
                                        mode='lines',
                                        name=f'Match {i}',
                                        line=dict(color='red', width=1, dash='dash')
                                    )
                                )
                                fig1.add_annotation(
                                    x=x,
                                    y=max(clean_sampled_signal) * 0.9,
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
                            
                            # 매칭 성공률 정보
                            original_candidates = len(ncc_above_threshold)
                            final_matches = len(st_groups)
                            success_rate = (final_matches / original_candidates * 100) if original_candidates > 0 else 0
                            
                            st.info(f"📈 **매칭 성공률**: {success_rate:.1f}% ({final_matches}/{original_candidates})")
                            
                        else:
                            st.warning("⚠️ 모든 필터링 조건을 만족하는 매칭 위치가 없습니다.")
                            st.info("💡 **제안**: NCC 임계값을 낮추거나, 신호값 범위를 조정하거나, 엄격한 매칭 모드를 해제해보세요.")
                            
                    except Exception as e:
                        st.error(f"❌ 매칭 수행 중 오류 발생: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            else:
                st.warning("📋 템플릿을 선택하거나 업로드해주세요.")



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


    # 탭4의 아래쪽에 추가할 프로그램 기능 설명 부분
    # 프로그램 기능 설명
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("📋 프로그램 기능 안내")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 주요 기능")
        st.markdown("""
        **Tab 1: 템플릿 설계**
        - 시계열 패턴 템플릿 생성 및 편집
        - 기준 패턴 정의 및 저장
        - 템플릿 파라미터 설정
        
        **Tab 2: 템플릿 매칭**
        - 설계된 템플릿과 데이터 간 유사도 분석
        - 패턴 매칭 결과 시각화
        - 매칭 임계값 조정 및 결과 필터링
        """)
    
    with col2:
        st.subheader("⚙️ 데이터 분석")
        st.markdown("""
        **Tab 3: 다채널 신호 관찰**
        - 다변량 시계열 데이터 동시 플롯
        - 다운샘플링을 통한 성능 최적화
        - 십자선 Hover 기능으로 정확한 값 확인
        - 인터랙티브 줌/팬 기능
        
        **Tab 4: 파일 관리**
        - FTR/Feather 파일 다운운로드 및 다중 파일 처리
        - 데이터 기본 정보 확인 및 품질 검사
        - 파일 목록 관리 및 선택적 로딩
        """)
    
    st.markdown("---")
    st.info("💡 **사용 팁**: 각 탭은 독립적으로 작동하므로, 데이터를 업로드한 후 원하는 분석 탭으로 자유롭게 이동하여 사용하세요.")
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <small>이 도구는 다변량 시계열 데이터의 탐색적 분석과 시각화를 위해 설계되었습니다.</small>
    </div>
    """, unsafe_allow_html=True)






# =============================================================================
# 추가 유틸리티 함수들 (배치 매칭용)
# =============================================================================

def group_nearby_matches(matches, proximity_threshold=50):
    """인접한 매칭들을 그룹화하고 각 그룹에서 최고 NCC 값을 가진 매칭을 선택"""
    if not matches:
        return []
    
    # center_pos 기준으로 정렬
    sorted_matches = sorted(matches, key=lambda x: x['center_pos'])
    
    grouped_matches = []
    current_group = [sorted_matches[0]]
    
    for i in range(1, len(sorted_matches)):
        current_match = sorted_matches[i]
        last_match = current_group[-1]
        
        # 거리 확인
        if abs(current_match['center_pos'] - last_match['center_pos']) <= proximity_threshold:
            current_group.append(current_match)
        else:
            # 현재 그룹에서 최고 NCC 값을 가진 매칭 선택
            best_match = max(current_group, key=lambda x: x['max_ncc'])
            best_match['is_best_in_group'] = True
            best_match['group_size_nearby'] = len(current_group)
            grouped_matches.append(best_match)
            
            # 새 그룹 시작
            current_group = [current_match]
    
    # 마지막 그룹 처리
    if current_group:
        best_match = max(current_group, key=lambda x: x['max_ncc'])
        best_match['is_best_in_group'] = True
        best_match['group_size_nearby'] = len(current_group)
        grouped_matches.append(best_match)
    
    return grouped_matches

def create_match_visualization(df, selected_col, matches, file_name, proximity_threshold=50):
    """매칭 결과를 시각화하는 함수 - 신호와 매칭 위치 모두 표시"""
    
    # 1. 신호 데이터 안전하게 추출
    if selected_col not in df.columns:
        st.error(f"컬럼 '{selected_col}'이 데이터프레임에 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
        return go.Figure(), []
    
    # pandas Series로 먼저 추출
    signal_series = df[selected_col]
    
    # 데이터 타입 확인 및 변환
    try:
        # pandas Series에서 숫자 변환 및 NaN 처리
        signal_series = pd.to_numeric(signal_series, errors='coerce')  # 숫자로 변환, 오류시 NaN
        signal_series = signal_series.fillna(0)  # NaN을 0으로 대체
        
        # numpy 배열로 변환
        signal = np.array(signal_series, dtype=np.float64)
        
    except Exception as e:
        st.error(f"신호 데이터 변환 오류: {str(e)}")
        return go.Figure(), []
    
    if len(signal) == 0:
        st.warning(f"신호 데이터가 비어있습니다.")
        return go.Figure(), []
    
    # 추가 검증: 모든 값이 0인지 확인
    if np.all(signal == 0):
        st.warning(f"모든 신호 값이 0입니다. 원본 데이터를 확인해주세요.")
        # 그래도 그래프는 생성
    
    # 추가 검증: 무한대 값 확인
    if np.any(np.isinf(signal)):
        st.warning(f"신호에 무한대 값이 포함되어 있습니다.")
        signal = np.nan_to_num(signal, posinf=0.0, neginf=0.0)
    
    # 2. Figure 생성 및 신호 플롯
    fig = go.Figure()
    
    # 전체 신호를 먼저 플롯 - go.Scatter 사용 (렌더링 이슈 해결)
    x_values = np.arange(len(signal))
    
    # 대용량 데이터인 경우 다운샘플링 적용
    if len(signal) > 50000:
        downsample_factor = len(signal) // 25000  # 최대 25,000 포인트로 제한
        x_values = x_values[::downsample_factor]
        signal_downsampled = signal[::downsample_factor]
        st.info(f"📊 대용량 데이터로 인해 {downsample_factor}:1 다운샘플링 적용됨")
    else:
        signal_downsampled = signal
    
    # go.Scatter 사용 (go.Scattergl 대신)
    fig.add_trace(go.Scatter(
        x=x_values,
        y=signal_downsampled,
        mode='lines',
        name=f'{selected_col}',
        line=dict(color='blue', width=2),
        opacity=0.9,
        hovertemplate=f'<b>{selected_col}</b><br>' +
                     'Index: %{x}<br>' +
                     'Value: %{y:.4f}<extra></extra>',
        visible=True  # 명시적으로 visible 설정
    ))
    
    # 3. 인접 매칭 그룹화
    best_matches = group_nearby_matches(matches, proximity_threshold)
    
    # 4. 매칭 위치 표시
    if best_matches:
        signal_min, signal_max = np.min(signal), np.max(signal)
        
        # 신호 범위가 0인 경우 처리
        if signal_min == signal_max:
            signal_min -= 0.1
            signal_max += 0.1
        
        for i, match in enumerate(best_matches):
            center_pos = match['center_pos']
            
            # 유효한 위치인지 확인
            if 0 <= center_pos < len(signal):
                # 수직선을 shape로 추가 (add_vline 대신)
                fig.add_shape(
                    type="line",
                    x0=center_pos,
                    y0=signal_min,
                    x1=center_pos,
                    y1=signal_max,
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash"
                    ),
                    layer="above"
                )
                
                # 매칭 포인트 표시
                match_value = signal[center_pos]
                fig.add_trace(go.Scatter(
                    x=[center_pos],
                    y=[match_value],
                    mode='markers',
                    name=f'M{i} (NCC:{match["max_ncc"]:.3f})',
                    marker=dict(
                        color='red',
                        size=12,
                        symbol='circle',
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate=f'<b>Match {i}</b><br>' +
                                 f'Position: {center_pos}<br>' +
                                 f'NCC: {match["max_ncc"]:.4f}<br>' +
                                 f'Signal Value: {match_value:.4f}<extra></extra>',
                    visible=True  # 명시적으로 visible 설정
                ))
                
                # 텍스트 주석 추가
                fig.add_annotation(
                    x=center_pos,
                    y=signal_max * 0.95,
                    text=f"M{i}",
                    showarrow=False,
                    font=dict(color="red", size=12, family="Arial Black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="red",
                    borderwidth=1
                )
    
    # 5. 레이아웃 설정 - 렌더링 최적화
    fig.update_layout(
        title=dict(
            text=f"📊 {file_name} - {selected_col} 신호 및 매칭 위치",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Sample Index',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True
        ),
        yaxis=dict(
            title=f'{selected_col} Value',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True
        ),
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        ),
        # 렌더링 최적화 설정
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=100, t=50, b=50),
        # 초기 렌더링 강제
        autosize=True,
        # 반응형 설정
        #responsive=True
    )
    
    # 축 범위 명시적 설정 (렌더링 이슈 방지)
    if len(signal) > 0:
        fig.update_xaxes(range=[0, len(signal)-1])
        fig.update_yaxes(range=[signal_min * 1.05, signal_max * 1.05])
    
    return fig, best_matches

def process_single_file_matching(file, template, selected_col, st_thres, st_low, st_high, 
                                max_diff, min_valid_ratio, strict_mode, offset_1):
    """단일 파일에 대한 템플릿 매칭 처리"""
    try:
        # 파일 로드
        if hasattr(file, 'read'):
            df = pd.read_feather(file)
        else:
            df = file  # 이미 DataFrame인 경우
        
        if selected_col not in df.columns:
            return {
                'file_name': file.name if hasattr(file, 'name') else 'Unknown',
                'status': 'error',
                'error': f"컬럼 '{selected_col}'이 존재하지 않습니다.",
                'matches': []
            }
        
        signal = df[selected_col].to_numpy().astype(np.float64)
        
        # 신호 길이 확인
        if len(signal) < len(template):
            return {
                'file_name': file.name if hasattr(file, 'name') else 'Unknown',
                'status': 'error',
                'error': f"신호 길이({len(signal)})가 템플릿 길이({len(template)})보다 짧습니다.",
                'matches': []
            }
        
        # 매칭 수행
        ncc_start = normalized_cross_correlation(signal, template)
        
        if len(ncc_start) == 0:
            return {
                'file_name': file.name if hasattr(file, 'name') else 'Unknown',
                'status': 'error',
                'error': "NCC 계산 결과가 없습니다.",
                'matches': []
            }
        
        # 필터링 단계
        ncc_above_threshold = np.where(ncc_start > st_thres)[0]
        valid_indices = ncc_above_threshold[ncc_above_threshold < len(signal)]
        
        # 엄격한 모드에서 품질 검사
        if strict_mode:
            valid_indices = filter_valid_matches(
                valid_indices, signal, len(template), min_valid_ratio
            )
        
        # 신호값 범위 필터링
        if len(valid_indices) > 0:
            signal_values_at_indices = signal[valid_indices]
            nan_mask = ~np.isnan(signal_values_at_indices)
            valid_indices = valid_indices[nan_mask]
            signal_values_at_indices = signal_values_at_indices[nan_mask]
            
            range_mask = (signal_values_at_indices > st_low) & (signal_values_at_indices < st_high)
            true_idx_st = valid_indices[range_mask]
        else:
            true_idx_st = np.array([])
        
        # 연속 그룹화
        if len(true_idx_st) > 0:
            st_groups = group_consecutive(true_idx_st, max_diff=max_diff)
            
            # 매칭 결과 정리
            matches = []
            for i, grp in enumerate(st_groups):
                match_center = grp[0] - offset_1 + len(grp)//2
                match_info = {
                    'group_id': i,
                    'start_pos': int(grp[0]),
                    'end_pos': int(grp[-1]),
                    'center_pos': int(match_center),
                    'group_size': len(grp),
                    'avg_signal': float(np.mean(signal[grp])),
                    'max_ncc': float(np.max(ncc_start[grp])),
                    'excluded': False  # 사용자가 제외시킬 수 있는 플래그
                }
                matches.append(match_info)
            
            return {
                'file_name': file.name if hasattr(file, 'name') else 'Unknown',
                'status': 'success',
                'total_groups': len(st_groups),
                'matches': matches,
                'signal_length': len(signal),
                'template_length': len(template)
            }
        else:
            return {
                'file_name': file.name if hasattr(file, 'name') else 'Unknown',
                'status': 'no_matches',
                'total_groups': 0,
                'matches': [],
                'signal_length': len(signal),
                'template_length': len(template)
            }
            
    except Exception as e:
        return {
            'file_name': file.name if hasattr(file, 'name') else 'Unknown',
            'status': 'error',
            'error': str(e),
            'matches': []
        }

def export_matches_to_csv(batch_results, include_excluded=False):
    """매칭 결과를 CSV 형식으로 변환"""
    csv_data = []
    
    for result in batch_results:
        if result['status'] == 'success':
            for match in result['matches']:
                # 제외된 매칭을 포함할지 결정
                if not include_excluded and match.get('excluded', False):
                    continue
                    
                csv_data.append({
                    'file_name': result['file_name'],
                    'group_id': match['group_id'],
                    'start_pos': match['start_pos'],
                    'end_pos': match['end_pos'],
                    'center_pos': match['center_pos'],
                    'group_size': match['group_size'],
                    'avg_signal': match['avg_signal'],
                    'max_ncc': match['max_ncc'],
                    'excluded': match.get('excluded', False),
                    'status': result['status']
                })
    
    return pd.DataFrame(csv_data)

# =============================================================================
# 탭 5: 배치 템플릿 매칭 (새로 추가)
# =============================================================================

def batch_template_matching_tab():
    st.header("🔄 배치 템플릿 매칭")
    st.markdown("여러 개의 FTR/Feather 파일에 템플릿 매칭을 일괄 적용하고 결과를 CSV로 저장합니다.")
    
    # 배치 파일 업로드
    st.subheader("📁 배치 파일 업로드")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**여러 개의 FTR/Feather 파일을 업로드하세요:**")
        batch_files = st.file_uploader(
            "배치 매칭할 FTR/Feather 파일들을 선택하세요",
            type=['ftr', 'feather'],
            accept_multiple_files=True,
            key="batch_matching_files"
        )
    
    with col2:
        if batch_files:
            st.success(f"✅ {len(batch_files)}개 파일 업로드됨")
    
    if batch_files:
        # 파일 목록 표시
        with st.expander("📋 업로드된 파일 목록"):
            for i, file in enumerate(batch_files):
                st.write(f"{i+1}. {file.name}")
        
        # 첫 번째 파일을 기준으로 컬럼 확인
        try:
            first_df = pd.read_feather(batch_files[0])
            available_columns = first_df.columns.tolist()
            
            # 템플릿 선택
            st.subheader("📋 템플릿 선택")
            template_source = st.radio(
                "사용할 템플릿을 선택하세요",
                ["업로드된 템플릿", "생성된 템플릿", "새 템플릿 업로드"],
                key="batch_template_source"
            )
            
            template = None
            
            if template_source == "업로드된 템플릿" and 'uploaded_template' in st.session_state:
                template = st.session_state['uploaded_template']
                st.info(f"업로드된 템플릿 사용 (길이: {len(template)})")
                
            elif template_source == "생성된 템플릿" and 'created_template' in st.session_state:
                template = st.session_state['created_template']
                st.info(f"생성된 템플릿 사용 (길이: {len(template)})")
                
            elif template_source == "새 템플릿 업로드":
                new_template_file = st.file_uploader("새 템플릿 파일 업로드", type=["npy"], key="batch_new_template")
                if new_template_file:
                    template = np.load(new_template_file)
                    st.info(f"새 템플릿 사용 (길이: {len(template)})")
            
            if template is not None:
                # 템플릿 시각화
                fig_template = go.Figure()
                fig_template.add_trace(go.Scatter(
                    y=template,
                    mode='lines',
                    name='Batch Template',
                    line=dict(color='purple', width=2)
                ))
                fig_template.update_layout(
                    title="📈 배치 매칭용 템플릿",
                    height=250,
                    xaxis_title='Template Index',
                    yaxis_title='Template Value'
                )
                st.plotly_chart(fig_template, use_container_width=True)
                
                # 분석 설정
                st.subheader("⚙️ 배치 매칭 설정")
                
                # 컬럼 선택
                selected_col = st.selectbox(
                    "분석할 컬럼을 선택하세요", 
                    available_columns, 
                    key="batch_matching_col"
                )
                
                # 매칭 파라미터 설정
                with st.form("batch_matching_form"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**기본 매칭 설정**")
                        batch_ncc_thres = st.slider("NCC Threshold", 0.0, 1.0, 0.2, 0.01, key="batch_ncc")
                        batch_low = st.number_input("신호 최소값", value=-1.0, key="batch_low")
                        batch_high = st.number_input("신호 최대값", value=1.0, key="batch_high")
                    
                    with col2:
                        st.markdown("**고급 설정**")
                        batch_max_diff = st.number_input("연속 그룹 최대 차이", value=200, key="batch_max_diff")
                        batch_offset = st.number_input("표시 위치 offset", value=0, key="batch_offset")
                        batch_min_valid_ratio = st.slider("최소 유효 데이터 비율", 0.5, 1.0, 0.8, 0.05, key="batch_valid_ratio")
                    
                    with col3:
                        st.markdown("**품질 제어**")
                        batch_strict_mode = st.checkbox("엄격한 매칭 모드", value=True, key="batch_strict")
                        
                        # CSV 저장 옵션
                        st.markdown("**저장 옵션**")
                        include_excluded_in_csv = st.checkbox("제외된 매칭도 CSV에 포함", value=False, key="batch_include_excluded")
                    
                    submitted = st.form_submit_button("🚀 배치 매칭 시작")
                
                if submitted:
                    # 배치 매칭 실행
                    st.subheader("📊 배치 매칭 진행 상황")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    batch_results = []
                    
                    for i, file in enumerate(batch_files):
                        status_text.text(f"처리 중: {file.name} ({i+1}/{len(batch_files)})")
                        
                        result = process_single_file_matching(
                            file, template, selected_col, batch_ncc_thres, 
                            batch_low, batch_high, batch_max_diff, batch_min_valid_ratio,
                            batch_strict_mode, batch_offset
                        )
                        
                        batch_results.append(result)
                        progress_bar.progress((i + 1) / len(batch_files))
                    
                    status_text.text("✅ 배치 매칭 완료!")
                    
                    # 결과 저장
                    st.session_state.batch_matching_results = batch_results
                    st.session_state.batch_template_used = template.copy()
                    st.session_state.batch_settings = {
                        'column': selected_col,
                        'ncc_threshold': batch_ncc_thres,
                        'signal_range': (batch_low, batch_high),
                        'max_diff': batch_max_diff,
                        'offset': batch_offset,
                        'strict_mode': batch_strict_mode
                    }
        
        except Exception as e:
            st.error(f"❌ 파일 처리 중 오류: {str(e)}")
    
    # 배치 매칭 결과 표시
    if hasattr(st.session_state, 'batch_matching_results'):
        st.markdown("---")
        st.subheader("📈 배치 매칭 결과")
        
        batch_results = st.session_state.batch_matching_results
        
        # 전체 통계
        total_files = len(batch_results)
        successful_files = sum(1 for r in batch_results if r['status'] == 'success')
        total_matches = sum(len(r.get('matches', [])) for r in batch_results if r['status'] == 'success')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("처리된 파일", total_files)
        with col2:
            st.metric("성공한 파일", successful_files)
        with col3:
            st.metric("총 매칭 수", total_matches)
        
        # 파일별 결과 표시
        st.subheader("📋 파일별 매칭 결과")
        
        for i, result in enumerate(batch_results):
            file_name = result['file_name']
            
            if result['status'] == 'success':
                matches = result.get('matches', [])
                active_matches = [m for m in matches if not m.get('excluded', False)]
                
                with st.expander(f"✅ {file_name} - {len(active_matches)}/{len(matches)}개 매칭"):
                    
                    if matches:
                        # 매칭 선택/해제 인터페이스
                        st.markdown("**매칭 위치 관리:**")
                        
                        # 각 매칭에 대한 체크박스
                        for j, match in enumerate(matches):
                            current_excluded = match.get('excluded', False)
                            
                            col_check, col_info = st.columns([1, 4])
                            
                            with col_check:
                                # 체크박스 (체크되면 포함, 체크 해제되면 제외)
                                include_match = st.checkbox(
                                    "포함", 
                                    value=not current_excluded,
                                    key=f"include_{i}_{j}",
                                    help="체크 해제하면 해당 매칭을 제외합니다"
                                )
                                
                                # 상태 업데이트
                                batch_results[i]['matches'][j]['excluded'] = not include_match
                            
                            with col_info:
                                status_icon = "✅" if include_match else "❌"
                                st.write(f"{status_icon} **그룹 {match['group_id']}**: "
                                       f"위치 {match['start_pos']}-{match['end_pos']}, "
                                       f"중심 {match['center_pos']}, "
                                       f"평균 신호 {match['avg_signal']:.4f}, "
                                       f"최대 NCC {match['max_ncc']:.4f}")
                        
                        # 전체 선택/해제 버튼
                        col_select_all, col_deselect_all = st.columns(2)
                        
                        with col_select_all:
                            if st.button(f"전체 선택", key=f"select_all_{i}"):
                                for j in range(len(matches)):
                                    batch_results[i]['matches'][j]['excluded'] = False
                                st.experimental_rerun()
                        
                        with col_deselect_all:
                            if st.button(f"전체 해제", key=f"deselect_all_{i}"):
                                for j in range(len(matches)):
                                    batch_results[i]['matches'][j]['excluded'] = True
                                st.experimental_rerun()
                    else:
                        st.write("매칭된 위치가 없습니다.")
            
            elif result['status'] == 'no_matches':
                with st.expander(f"⚠️ {file_name} - 매칭 없음"):
                    st.write("설정된 조건을 만족하는 매칭이 발견되지 않았습니다.")
            
            else:  # error
                with st.expander(f"❌ {file_name} - 오류"):
                    st.error(f"오류: {result.get('error', '알 수 없는 오류')}")
        
        # 매칭 위치 시각화 섹션 추가
        st.markdown("---")
        st.subheader("📈 파일별 매칭 위치 시각화")
        
        # 배치 설정이 있는지 확인
        if not hasattr(st.session_state, 'batch_settings'):
            st.error("❌ 배치 매칭을 먼저 실행해주세요.")
            return
        
        # 시각화 설정
        proximity_threshold = st.number_input(
            "🎯 인접 그룹화 임계값 (포인트)",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            help="이 거리 내의 매칭들을 하나의 그룹으로 간주하고 최고 NCC 값만 표시"
        )
        
        # 매칭이 있는 모든 파일을 차례로 시각화
        selected_results = [r for r in batch_results if r['status'] == 'success' and r.get('matches')]
        
        if not selected_results:
            st.info("시각화할 매칭 결과가 있는 파일이 없습니다.")
        else:
            st.info(f"📊 총 {len(selected_results)}개 파일의 매칭 결과를 차례로 표시합니다.")
            
            # batch_files 존재 확인
            if 'batch_files' not in locals() and 'batch_files' not in globals():
                st.error("❌ 원본 파일 정보를 찾을 수 없습니다. 파일을 다시 업로드해주세요.")
                return
        
        # 각 파일에 대해 시각화 생성
        for result_idx, result in enumerate(selected_results):
            file_name = result['file_name']
            matches = [m for m in result.get('matches', []) if not m.get('excluded', False)]
            
            if not matches:
                st.info(f"📄 {file_name}: 표시할 활성 매칭이 없습니다.")
                continue
            
            st.markdown(f"### 📊 {file_name}")
            
            try:
                # 파일 다시 로드 (시각화를 위해)
                original_file = next((f for f in batch_files if f.name == file_name), None)
                if original_file is None:
                    st.error(f"원본 파일 {file_name}을 찾을 수 없습니다.")
                    continue
                
                df = pd.read_feather(original_file)
                selected_col = st.session_state.batch_settings['column']
                
                # 데이터 유효성 검사 추가
                if selected_col not in df.columns:
                    st.error(f"컬럼 '{selected_col}'이 파일 {file_name}에 존재하지 않습니다.")
                    st.write(f"사용 가능한 컬럼: {list(df.columns)}")
                    continue
                
                # 신호 데이터 확인
                signal_data = df[selected_col]
                if signal_data.empty:
                    st.warning(f"파일 {file_name}의 '{selected_col}' 컬럼이 비어있습니다.")
                    continue
                
                # 기본 정보 표시
                st.write(f"**파일**: {file_name} | **특징**: {selected_col} | **데이터 길이**: {len(signal_data):,} | **매칭 수**: {len(matches)}")
                
                # 매칭 위치 시각화
                fig, best_matches = create_match_visualization(
                    df, selected_col, matches, file_name, proximity_threshold
                )
                
                # 그래프가 제대로 생성되었는지 확인
                if len(fig.data) == 0:
                    st.error(f"❌ {file_name}: 그래프가 생성되지 않았습니다.")
                    st.write(f"신호 데이터 샘플: {signal_data.head().tolist()}")
                    continue
                
                # 플롯 표시
                st.plotly_chart(fig, use_container_width=True)
                
                # 신호 정보 표시 (디버깅용)
                with st.expander(f"📊 {file_name} 신호 정보"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("신호 길이", f"{len(signal_data):,}")
                    with col2:
                        st.metric("신호 범위", f"{signal_data.min():.3f} ~ {signal_data.max():.3f}")
                    with col3:
                        st.metric("평균값", f"{signal_data.mean():.3f}")
                    
                    st.write(f"**첫 10개 값**: {signal_data.head(10).tolist()}")
                    st.write(f"**NaN 개수**: {signal_data.isna().sum()}")
                    st.write(f"**데이터 타입**: {signal_data.dtype}")
                
                # 그룹화된 매칭 결과 표시
                if best_matches:
                    st.markdown(f"**🎯 인접 그룹화 결과 ({len(best_matches)}개 최종 매칭):**")
                    
                    # 매칭 선택/해제 인터페이스 (그룹화된 결과용)
                    final_matches_key = f"final_matches_{result_idx}"
                    if final_matches_key not in st.session_state:
                        st.session_state[final_matches_key] = {str(match['center_pos']): True for match in best_matches}
                    
                    cols = st.columns(min(3, len(best_matches)))
                    
                    for i, match in enumerate(best_matches):
                        col_idx = i % len(cols)
                        match_key = str(match['center_pos'])
                        
                        with cols[col_idx]:
                            # 매칭 포함/제외 체크박스
                            is_included = st.checkbox(
                                f"M{i} (pos: {match['center_pos']})",
                                value=st.session_state[final_matches_key].get(match_key, True),
                                key=f"final_match_{result_idx}_{i}",
                                help=f"NCC: {match['max_ncc']:.4f}, 그룹크기: {match['group_size_nearby']}"
                            )
                            
                            st.session_state[final_matches_key][match_key] = is_included
                            
                            # 매칭 정보 표시
                            if is_included:
                                st.success(f"✅ 포함")
                            else:
                                st.error(f"❌ 제외")
                            
                            st.caption(f"NCC: {match['max_ncc']:.4f}")
                            st.caption(f"그룹: {match['group_size_nearby']}개")
                    
                    # 전체 선택/해제 버튼 (그룹화된 결과용)
                    col_select, col_deselect = st.columns(2)
                    
                    with col_select:
                        if st.button(f"🔸 전체 선택", key=f"select_all_final_{result_idx}"):
                            for match in best_matches:
                                st.session_state[final_matches_key][str(match['center_pos'])] = True
                            st.experimental_rerun()
                    
                    with col_deselect:
                        if st.button(f"🔹 전체 해제", key=f"deselect_all_final_{result_idx}"):
                            for match in best_matches:
                                st.session_state[final_matches_key][str(match['center_pos'])] = False
                            st.experimental_rerun()
                    
                    # 선택된 매칭 수 표시
                    selected_count = sum(1 for v in st.session_state[final_matches_key].values() if v)
                    st.info(f"📊 선택된 최종 매칭: {selected_count}/{len(best_matches)}개")
                
            except Exception as e:
                st.error(f"❌ {file_name} 시각화 중 오류: {str(e)}")
        
        # CSV 다운로드 섹션 (수정됨 - 최종 플롯 결과만 포함)
        st.markdown("---")
        st.subheader("💾 최종 결과 다운로드")
        st.markdown("**📊 최종 플롯에서 선택된 매칭 위치만 CSV로 저장됩니다.**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_filename = st.text_input(
                "CSV 파일명 (확장자 제외)",
                value="final_matching_results",
                key="final_csv_filename"
            )
        
        with col2:
            export_mode = st.radio(
                "내보내기 모드",
                ["최종 플롯 결과만", "원본 매칭 결과"],
                help="최종 플롯 결과: 그룹화 후 선택된 매칭만 / 원본 매칭 결과: 초기 매칭 결과"
            )
        
        if st.button("📊 최종 CSV 생성 및 다운로드", key="generate_final_csv"):
            try:
                final_csv_data = []
                
                if export_mode == "최종 플롯 결과만":
                    # 시각화에서 선택된 최종 매칭만 포함
                    for result_idx, result in enumerate(batch_results):
                        if result['status'] != 'success':
                            continue
                        
                        file_name = result['file_name']
                        matches = [m for m in result.get('matches', []) if not m.get('excluded', False)]
                        
                        if not matches:
                            continue
                        
                        # 그룹화된 매칭 가져오기
                        best_matches = group_nearby_matches(matches, proximity_threshold)
                        
                        # 세션에서 선택 상태 확인
                        final_matches_key = f"final_matches_{result_idx}"
                        selected_matches_state = st.session_state.get(final_matches_key, {})
                        
                        for i, match in enumerate(best_matches):
                            match_key = str(match['center_pos'])
                            is_selected = selected_matches_state.get(match_key, True)
                            
                            if is_selected:
                                final_csv_data.append({
                                    'file_name': file_name,
                                    'match_id': f"M{i}",
                                    'center_pos': match['center_pos'],
                                    'start_pos': match['start_pos'],
                                    'end_pos': match['end_pos'],
                                    'max_ncc': match['max_ncc'],
                                    'avg_signal': match['avg_signal'],
                                    'group_size_nearby': match['group_size_nearby'],
                                    'proximity_threshold': proximity_threshold,
                                    'export_type': 'final_plot_selected'
                                })
                else:
                    # 원본 매칭 결과 (기존 방식)
                    final_csv_data = []
                    for result in batch_results:
                        if result['status'] == 'success':
                            for match in result['matches']:
                                if not match.get('excluded', False):
                                    final_csv_data.append({
                                        'file_name': result['file_name'],
                                        'group_id': match['group_id'],
                                        'start_pos': match['start_pos'],
                                        'end_pos': match['end_pos'],
                                        'center_pos': match['center_pos'],
                                        'group_size': match['group_size'],
                                        'avg_signal': match['avg_signal'],
                                        'max_ncc': match['max_ncc'],
                                        'excluded': match.get('excluded', False),
                                        'export_type': 'original_matching'
                                    })
                
                if final_csv_data:
                    csv_df = pd.DataFrame(final_csv_data)
                    csv_string = csv_df.to_csv(index=False)
                    
                    # 다운로드 버튼
                    st.download_button(
                        label="💾 최종 CSV 파일 다운로드",
                        data=csv_string,
                        file_name=f"{csv_filename}.csv",
                        mime="text/csv",
                        help="선택된 최종 매칭 결과를 CSV 형식으로 다운로드"
                    )
                    
                    # 미리보기
                    st.subheader("📋 최종 CSV 미리보기")
                    st.dataframe(csv_df.head(20), use_container_width=True)
                    
                    if len(csv_df) > 20:
                        st.info(f"💡 총 {len(csv_df)}개 행 중 상위 20개만 표시됩니다.")
                    
                    # 통계 정보
                    with st.expander("📊 최종 CSV 통계 정보"):
                        st.write(f"**총 최종 매칭 수**: {len(csv_df)}")
                        st.write(f"**파일 수**: {csv_df['file_name'].nunique()}")
                        
                        if export_mode == "최종 플롯 결과만":
                            st.write(f"**인접 그룹화 임계값**: {proximity_threshold}")
                            st.write(f"**내보내기 타입**: 최종 플롯 선택 결과")
                            
                            # 파일별 최종 매칭 수
                            file_match_counts = csv_df.groupby('file_name').size()
                            st.write("**파일별 최종 매칭 수**:")
                            for file_name, count in file_match_counts.items():
                                st.write(f"  - {file_name}: {count}개")
                        else:
                            st.write(f"**내보내기 타입**: 원본 매칭 결과")
                else:
                    st.warning("⚠️ 내보낼 최종 매칭 결과가 없습니다.")
                    
            except Exception as e:
                st.error(f"❌ 최종 CSV 생성 중 오류: {str(e)}")
        
        # 현재 선택 상태 요약
        with st.expander("📊 현재 선택 상태 요약"):
            total_final_matches = 0
            
            st.markdown("**파일별 최종 선택 상태:**")
            for result_idx, result in enumerate(batch_results):
                if result['status'] != 'success':
                    continue
                
                file_name = result['file_name']
                matches = [m for m in result.get('matches', []) if not m.get('excluded', False)]
                
                if not matches:
                    st.write(f"📄 {file_name}: 매칭 없음")
                    continue
                
                best_matches = group_nearby_matches(matches, proximity_threshold)
                final_matches_key = f"final_matches_{result_idx}"
                selected_matches_state = st.session_state.get(final_matches_key, {})
                
                selected_count = sum(1 for match in best_matches 
                                   if selected_matches_state.get(str(match['center_pos']), True))
                total_final_matches += selected_count
                
                st.write(f"📄 {file_name}: {selected_count}/{len(best_matches)}개 선택됨")
            
            st.success(f"🎯 **전체 최종 선택된 매칭**: {total_final_matches}개") 
        
        # 설정 정보 표시
        with st.expander("⚙️ 사용된 매칭 설정"):
            if hasattr(st.session_state, 'batch_settings'):
                settings = st.session_state.batch_settings
                st.json(settings)
    
    else:
        st.info("📁 배치 파일을 업로드하고 템플릿을 선택한 후 매칭을 시작하세요.")

# =============================================================================
# 애플리케이션 실행
# =============================================================================

if __name__ == "__main__":
    main()