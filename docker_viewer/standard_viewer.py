import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import re
from datetime import datetime
import tempfile

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

# 세션 상태 초기화
if 'folder_path' not in st.session_state:
    st.session_state.folder_path = ""
if 'data_dict' not in st.session_state:
    st.session_state.data_dict = {}
if 'digital_signals' not in st.session_state:
    st.session_state.digital_signals = []
if 'analog_signals' not in st.session_state:
    st.session_state.analog_signals = []
if 'sampling_method' not in st.session_state:
    st.session_state.sampling_method = "원본 데이터 (샘플링 없음)"
if 'max_points' not in st.session_state:
    st.session_state.max_points = 500
if 'color_mapping' not in st.session_state:
    st.session_state.color_mapping = {}
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}
if 'selection_order' not in st.session_state:
    st.session_state.selection_order = []
if 'default_signals_applied' not in st.session_state:
    st.session_state.default_signals_applied = False
if 'reset_selections' not in st.session_state:
    st.session_state.reset_selections = False
if 'apply_defaults' not in st.session_state:
    st.session_state.apply_defaults = False

# 디폴트 신호 목록 정의
DEFAULT_SIGNALS = [
    "COP-A Running",
    "COP-B Running", 
    "GT5_Speed",
    "GT6_Speed",
    "GT5 발전기 Breaker Close",
    "GT6 발전기 Breaker Close",
    "ST Ready to Start",
    "S3_L14HM",
    "GT5 Load Setpoint 상승(출력증발)",
    "GT6 Load Setpoint 상승(출력증발)",
    "GT5 MW",
    "GT6 MW",
    "HRSG5 HP BYPASS Vv 개도",
    "HRSG6 HP BYPASS Vv 개도",
    "GT5 AGC On",
    "GT6 AGC On"
]

def handle_file_upload(uploaded_files):
    """업로드된 파일들을 처리하는 함수"""
    if uploaded_files:
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        st.session_state.folder_path = temp_dir
        
        # 업로드된 파일들을 임시 디렉토리에 저장
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # 데이터 초기화
        st.session_state.data_dict = {}
        st.session_state.digital_signals = []
        st.session_state.analog_signals = []
        st.session_state.color_mapping = {}
        st.session_state.selection_order = []
        
        st.success(f"{len(uploaded_files)}개 파일이 업로드되었습니다.")
        st.rerun()

def sample_data(df, method, max_points):
    """데이터 샘플링 함수"""
    if method == "원본 데이터 (샘플링 없음)" or len(df) <= max_points:
        return df
    
    if method == "균등 샘플링":
        step = max(1, len(df) // max_points)
        return df.iloc[::step]
    
    elif method == "랜덤 샘플링":
        n_sample = min(max_points, len(df))
        return df.sample(n=n_sample).sort_index()
    
    elif method == "시작/끝 우선 샘플링":
        start_points = max_points // 4
        end_points = max_points // 4
        middle_points = max_points - start_points - end_points
        
        start_data = df.head(start_points)
        end_data = df.tail(end_points)
        
        if len(df) > start_points + end_points:
            middle_start = start_points
            middle_end = len(df) - end_points
            if middle_end > middle_start:
                middle_step = max(1, (middle_end - middle_start) // middle_points)
                middle_data = df.iloc[middle_start:middle_end:middle_step]
            else:
                middle_data = pd.DataFrame()
        else:
            middle_data = pd.DataFrame()
        
        return pd.concat([start_data, middle_data, end_data]).drop_duplicates().sort_index()
    
    return df

def extract_date_from_filename(filename):
    """파일명에서 날짜 정보를 추출하는 함수"""
    patterns = [
        r'(\d{4})(\d{2})(\d{2})',
        r'(\d{4})-(\d{2})-(\d{2})',
        r'(\d{4})_(\d{2})_(\d{2})',
        r'(\d{2})(\d{2})(\d{4})',
    ]
    
    basename = os.path.basename(filename).split('.')[0]
    
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            groups = match.groups()
            if len(groups[0]) == 4:  # YYYY format
                year, month, day = groups
            else:  # DD format
                day, month, year = groups
            try:
                date_obj = datetime(int(year), int(month), int(day))
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
    
    return basename

def is_digital_signal(series):
    """신호가 디지털인지 판단하는 함수 (고유값이 2개 이하)"""
    unique_values = series.dropna().nunique()
    return unique_values <= 2

def load_ftr_file(filepath):
    """FTR/Feather 파일을 로드하는 함수"""
    try:
        df = pd.read_feather(filepath)
        return df
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {os.path.basename(filepath)} - {str(e)}")
        return None

def find_similar_files_by_temp(data_dict, folder_path, all_files):
    """COP-A/COP-B Running 중 먼저 변화하는 시점의 온도값들 기준으로 유사한 파일들을 찾는 함수"""
    if not data_dict:
        return [], None, None, None, None
    
    # 현재 선택된 파일에서 COP-A/COP-B Running 중 먼저 0에서 1이상으로 바뀌는 시점 찾기
    reference_temps = None
    reference_filename = None
    reference_time = None
    reference_cop_type = None
    
    for filename, df in data_dict.items():
        if ('COP-A Running' in df.columns and 'COP-B Running' in df.columns and 
            '1st metal temp' in df.columns and 'RH Bore temp' in df.columns):
            
            # COP-A Running 변화 시점 찾기
            cop_a_change_idx = None
            cop_a_running = df['COP-A Running']
            for i in range(1, len(cop_a_running)):
                if cop_a_running.iloc[i-1] == 0 and cop_a_running.iloc[i] >= 1:
                    cop_a_change_idx = i
                    break
            
            # COP-B Running 변화 시점 찾기
            cop_b_change_idx = None
            cop_b_running = df['COP-B Running']
            for i in range(1, len(cop_b_running)):
                if cop_b_running.iloc[i-1] == 0 and cop_b_running.iloc[i] >= 1:
                    cop_b_change_idx = i
                    break
            
            # 더 먼저 변화하는 시점 선택
            selected_idx = None
            selected_cop = None
            
            if cop_a_change_idx is not None and cop_b_change_idx is not None:
                if cop_a_change_idx <= cop_b_change_idx:
                    selected_idx = cop_a_change_idx
                    selected_cop = "COP-A"
                else:
                    selected_idx = cop_b_change_idx
                    selected_cop = "COP-B"
            elif cop_a_change_idx is not None:
                selected_idx = cop_a_change_idx
                selected_cop = "COP-A"
            elif cop_b_change_idx is not None:
                selected_idx = cop_b_change_idx
                selected_cop = "COP-B"
            
            if selected_idx is not None:
                # 해당 시점에서의 두 온도값 가져오기
                metal_temp = df['1st metal temp'].iloc[selected_idx]
                bore_temp = df['RH Bore temp'].iloc[selected_idx]
                reference_temps = (metal_temp, bore_temp)
                reference_filename = filename
                reference_time = selected_idx
                reference_cop_type = selected_cop
                break
    
    if reference_temps is None:
        return [], None, None, None, None
    
    # 나머지 파일들에서 유사한 온도 조합 찾기
    temp_similarities = []
    
    for file in all_files:
        file_key = file.split('.')[0]
        
        # 이미 선택된 파일은 제외
        if file_key in data_dict.keys():
            continue
            
        try:
            filepath = os.path.join(folder_path, file)
            df = load_ftr_file(filepath)
            
            if (df is not None and 
                'COP-A Running' in df.columns and 'COP-B Running' in df.columns and
                '1st metal temp' in df.columns and 
                'RH Bore temp' in df.columns):
                
                # COP-A Running 변화 시점 찾기
                cop_a_change_idx = None
                cop_a_running = df['COP-A Running']
                for i in range(1, len(cop_a_running)):
                    if cop_a_running.iloc[i-1] == 0 and cop_a_running.iloc[i] >= 1:
                        cop_a_change_idx = i
                        break
                
                # COP-B Running 변화 시점 찾기
                cop_b_change_idx = None
                cop_b_running = df['COP-B Running']
                for i in range(1, len(cop_b_running)):
                    if cop_b_running.iloc[i-1] == 0 and cop_b_running.iloc[i] >= 1:
                        cop_b_change_idx = i
                        break
                
                # 더 먼저 변화하는 시점 선택
                selected_idx = None
                selected_cop = None
                
                if cop_a_change_idx is not None and cop_b_change_idx is not None:
                    if cop_a_change_idx <= cop_b_change_idx:
                        selected_idx = cop_a_change_idx
                        selected_cop = "COP-A"
                    else:
                        selected_idx = cop_b_change_idx
                        selected_cop = "COP-B"
                elif cop_a_change_idx is not None:
                    selected_idx = cop_a_change_idx
                    selected_cop = "COP-A"
                elif cop_b_change_idx is not None:
                    selected_idx = cop_b_change_idx
                    selected_cop = "COP-B"
                
                if selected_idx is not None:
                    # 해당 시점에서의 두 온도값 가져오기
                    file_metal_temp = df['1st metal temp'].iloc[selected_idx]
                    file_bore_temp = df['RH Bore temp'].iloc[selected_idx]
                    
                    # 유클리디언 거리 계산
                    euclidean_dist = np.sqrt(
                        (reference_temps[0] - file_metal_temp)**2 + 
                        (reference_temps[1] - file_bore_temp)**2
                    )
                    
                    temp_similarities.append({
                        'filename': file,
                        'metal_temp': file_metal_temp,
                        'bore_temp': file_bore_temp,
                        'distance': euclidean_dist,
                        'time_idx': selected_idx,
                        'cop_type': selected_cop
                    })
        except Exception as e:
            continue
    
    # 유클리디언 거리가 가장 작은 순으로 정렬하여 상위 5개 반환
    temp_similarities.sort(key=lambda x: x['distance'])
    return temp_similarities[:5], reference_filename, reference_temps, reference_time, reference_cop_type

def create_color_mapping(data_dict):
    """파일별 색상 매핑 생성"""
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    plotly_colors = px.colors.qualitative.Set1
    
    color_mapping = {}
    for idx, filename in enumerate(data_dict.keys()):
        color_mapping[filename] = {
            'matplotlib': colors[idx % len(colors)],
            'plotly': plotly_colors[idx % len(plotly_colors)]
        }
    
    return color_mapping

def create_signal_plots(data_dict, signal_list, signal_type, sampling_method, max_points, color_mapping, preview_undersampling_level=3):
    """신호 플롯을 생성하는 함수"""
    plot_data = {}
    line_styles = ['-', '--', '-.', ':']
    
    # 사용자 설정에 따른 언더샘플링
    if preview_undersampling_level == 0:
        preview_max_points = max_points  # 언더샘플링 없음
    else:
        preview_max_points = max(50, max_points // preview_undersampling_level)  # 최소 50개 포인트 보장
    
    for signal in signal_list:
        # 기존 figure가 있다면 닫기
        plt.close('all')
        
        fig, ax = plt.subplots(figsize=(5, 0.8))
        
        file_idx = 0
        for filename, df in data_dict.items():
            if signal in df.columns:
                # 기본 샘플링 적용
                sampled_df = sample_data(df, sampling_method, max_points)
                
                # 추가 언더샘플링 적용 (미리보기 성능 향상)
                if preview_undersampling_level > 0 and len(sampled_df) > preview_max_points:
                    step = max(1, len(sampled_df) // preview_max_points)
                    sampled_df = sampled_df.iloc[::step]
                
                color = color_mapping[filename]['matplotlib']
                linestyle = line_styles[file_idx % len(line_styles)]
                linewidth = 1.0 + (file_idx % 3) * 0.15  # 선 두께 줄임
                
                ax.plot(sampled_df.index, sampled_df[signal], 
                       color=color, linestyle=linestyle, linewidth=linewidth,
                       label=f"{filename}")
                file_idx += 1
        
        # Y축 범위를 고정하여 높이 일정하게 유지
        try:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        except:
            pass
        
        # 타이틀을 플롯 내부 좌측 상단에 표시
        ax.text(0.02, 0.95, signal, transform=ax.transAxes, 
                fontsize=5, fontweight='bold', verticalalignment='top',  # 폰트 크기 줄임
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9))  # 패딩 줄임
        
        # 범례를 플롯 내부 우측 하단에 표시
        if len(data_dict) > 1:  # 파일이 여러 개일 때만 범례 표시
            ax.legend(loc='lower right', fontsize=4, framealpha=0.8)  # 범례 폰트도 줄임
        
        # Y축 라벨 제거하고 틱만 표시
        ax.tick_params(labelsize=4, length=1.5)  # 틱 폰트와 길이 줄임
        ax.grid(True, alpha=0.15, linewidth=0.3)  # 그리드 선 두께 줄임
        
        # 여백을 극도로 줄임
        plt.tight_layout(pad=0.1)
        plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05)
        
        plot_data[signal] = fig
    
    return plot_data

def normalize_signal_name(signal_name):
    """신호명을 정규화하는 함수 (공백, 언더스코어, 특수문자 제거, 영문만 소문자 변환)"""
    import re
    # 공백, 언더스코어, 하이픈, 괄호 등 제거
    normalized = re.sub(r'[\s_\-\(\)]', '', signal_name)
    # 영문자만 소문자로 변환 (한글은 그대로 유지)
    result = ''
    for char in normalized:
        if char.isascii() and char.isalpha():
            result += char.lower()
        else:
            result += char
    return result

def find_matching_signal(target_signal, available_signals):
    """타겟 신호와 유사한 신호를 찾는 함수"""
    target_normalized = normalize_signal_name(target_signal)
    
    # 1단계: 정확한 매칭 (정규화 후)
    for signal in available_signals:
        signal_normalized = normalize_signal_name(signal)
        if target_normalized == signal_normalized:
            return signal
    
    # 2단계: 부분 매칭 시도 (포함 관계)
    for signal in available_signals:
        signal_normalized = normalize_signal_name(signal)
        if target_normalized in signal_normalized or signal_normalized in target_normalized:
            return signal
    
    # 3단계: 더 관대한 매칭 (핵심 키워드 기반)
    # 한글이 포함된 경우를 위한 추가 매칭
    target_keywords = extract_keywords(target_signal)
    for signal in available_signals:
        signal_keywords = extract_keywords(signal)
        if len(target_keywords) > 0 and len(signal_keywords) > 0:
            # 키워드의 80% 이상이 일치하면 매칭으로 간주
            common_keywords = set(target_keywords) & set(signal_keywords)
            if len(common_keywords) >= max(1, len(target_keywords) * 0.8):
                return signal
    
    return None

def extract_keywords(signal_name):
    """신호명에서 핵심 키워드를 추출하는 함수"""
    import re
    # 공백, 언더스코어, 하이픈 등으로 분리
    parts = re.split(r'[\s_\-\(\)]+', signal_name)
    keywords = []
    for part in parts:
        if len(part) > 0:
            # 영문은 소문자로, 한글은 그대로
            normalized_part = ''
            for char in part:
                if char.isascii() and char.isalpha():
                    normalized_part += char.lower()
                else:
                    normalized_part += char
            keywords.append(normalized_part)
    return [k for k in keywords if len(k) > 0]

def apply_default_signal_selection():
    """디폴트 신호들을 자동으로 선택하는 함수"""
    if st.session_state.default_signals_applied:
        return
    
    # 모든 신호 목록 생성 (디지털 + 아날로그)
    all_available_signals = st.session_state.digital_signals + st.session_state.analog_signals
    
    # 디폴트 신호 중 실제 존재하는 신호들만 선택 (유사 매칭 포함)
    available_default_signals = []
    for target_signal in DEFAULT_SIGNALS:
        matched_signal = find_matching_signal(target_signal, all_available_signals)
        if matched_signal:
            available_default_signals.append(matched_signal)
    
    # 선택 순서 설정
    st.session_state.selection_order = available_default_signals.copy()
    
    # 체크박스 상태 설정
    for signal in available_default_signals:
        if signal in st.session_state.digital_signals:
            checkbox_key = f"digital_{signal}"
        else:
            checkbox_key = f"analog_{signal}"
        
        st.session_state[checkbox_key] = True
    
    # 디폴트 적용 완료 표시
    st.session_state.default_signals_applied = True
def update_selection_order(signal, is_selected):
    """선택 순서를 업데이트하는 콜백 함수"""
    if is_selected:  # 체크된 경우
        if signal not in st.session_state.selection_order:
            st.session_state.selection_order.append(signal)
    else:  # 체크 해제된 경우
        if signal in st.session_state.selection_order:
            st.session_state.selection_order.remove(signal)

def display_signal_plots_with_checkboxes(plot_data, signal_type):
    """플롯과 체크박스를 표시하는 함수 - 그리드 레이아웃으로 촘촘하게 배치"""
    checkbox_states = {}
    
    # CSS로 플롯 간 간격 최소화 - 더 강력한 스타일 적용
    st.markdown("""
    <style>
    .plot-container {
        margin: -20px 0 !important;
        padding: 0 !important;
    }
    .stCheckbox {
        margin-top: 5px !important;
    }
    /* Streamlit 기본 요소들의 간격 조정 */
    .element-container {
        margin-bottom: -20px !important;
    }
    .stPlotlyChart, .stPyplot {
        margin-bottom: -30px !important;
        margin-top: -10px !important;
    }
    /* 컬럼 간격 조정 */
    [data-testid="column"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 2열로 배치하여 더 촘촘하게 표시
    signals = list(plot_data.keys())
    
    for i in range(0, len(signals), 2):
        cols = st.columns([2, 0.2, 2, 0.2])  # 체크박스 컬럼을 더 줄임
        
        # 첫 번째 신호
        signal1 = signals[i]
        with cols[0]:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(plot_data[signal1], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[1]:
            checkbox_key1 = f"{signal_type}_{signal1}"
            checkbox_states[signal1] = st.checkbox(
                "✓", 
                key=checkbox_key1,
                help=f"Select {signal1}",
                on_change=lambda signal=signal1: update_selection_order(signal, st.session_state.get(checkbox_key1, False))
            )
        
        # 두 번째 신호 (있는 경우)
        if i + 1 < len(signals):
            signal2 = signals[i + 1]
            with cols[2]:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.pyplot(plot_data[signal2], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with cols[3]:
                checkbox_key2 = f"{signal_type}_{signal2}"
                checkbox_states[signal2] = st.checkbox(
                    "✓", 
                    key=checkbox_key2,
                    help=f"Select {signal2}",
                    on_change=lambda signal=signal2: update_selection_order(signal, st.session_state.get(checkbox_key2, False))
                )
        
        # 각 행 사이에 작은 간격 추가 (선택사항)
        # st.markdown("<div style='margin: -15px 0;'></div>", unsafe_allow_html=True)
    
    # 메모리 정리
    for fig in plot_data.values():
        plt.close(fig)
    plt.close('all')
    
    return checkbox_states

def plot_selected_signals_matplotlib(data_dict, selected_signals, sampling_method, max_points, color_mapping, undersampling_level=2):
    """선택된 신호들을 matplotlib로 플롯하는 함수 (빠른 렌더링)"""
    if not selected_signals or not data_dict:
        st.warning("선택된 신호가 없습니다.")
        return
    
    # 기존 figure들 모두 닫기
    plt.close('all')
    
    # 고정된 높이로 서브플롯 생성 - 각 플롯박스 높이를 일정하게 유지
    n_signals = len(selected_signals)
    fixed_height_per_signal = 0.7  # 각 신호당 고정 높이
    total_height = fixed_height_per_signal * n_signals
    
    fig, axes = plt.subplots(n_signals, 1, figsize=(12, total_height), sharex=True)
    
    # 단일 신호인 경우 axes를 리스트로 변환
    if n_signals == 1:
        axes = [axes]
    
    line_styles = ['-', '--', '-.', ':']
    
    # 언더샘플링 레벨에 따른 최대 포인트 계산
    if undersampling_level == 0:
        fast_max_points = max_points  # 언더샘플링 없음
    else:
        fast_max_points = max(50, max_points // undersampling_level)  # 최소 50개 포인트 보장
    
    for idx, signal in enumerate(selected_signals):
        ax = axes[idx]
        file_idx = 0
        
        for filename, df in data_dict.items():
            if signal in df.columns:
                # 기본 샘플링 적용
                sampled_df = sample_data(df, sampling_method, max_points)
                
                # 추가 언더샘플링 적용 (플롯 성능 향상)
                if undersampling_level > 0 and len(sampled_df) > fast_max_points:
                    step = max(1, len(sampled_df) // fast_max_points)
                    sampled_df = sampled_df.iloc[::step]
                
                color = color_mapping[filename]['matplotlib']
                linestyle = line_styles[file_idx % len(line_styles)]
                linewidth = 1.2 + (file_idx % 3) * 0.2  # 선 두께도 줄임
                
                ax.plot(sampled_df.index, sampled_df[signal], 
                       color=color, linestyle=linestyle, linewidth=linewidth,
                       label=f"{filename}", alpha=0.8)
                file_idx += 1
        
        # 플롯박스 경계선 두께 줄이기
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)  # 경계선 두께를 0.5로 설정
        
        # 특징명을 각 플롯박스 내부 좌측 끝 상부에 위치
        ax.text(0.01, 0.95, signal, transform=ax.transAxes, 
                fontsize=8, fontweight='bold',  # 폰트 크기 약간 증가
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, linewidth=0.5))
        
        ax.grid(True, alpha=0.3)
        
        # 틱 설정 - X축과 Y축 모두 라벨 표시, 폰트 크기 줄임
        ax.tick_params(labelsize=7, length=2)  # 라벨 크기를 7로 증가
        
        # Y축 틱 개수 제한
        ax.locator_params(axis='y', nbins=4)
        
        # 범례는 첫 번째 플롯에만 표시
        if idx == 0 and len(data_dict) > 1:
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9)  # 범례 폰트도 증가
        
        # Y축 범위 조정
        try:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
        except:
            pass
    
    # X축 틱 개수 제한 및 라벨 표시
    axes[-1].locator_params(axis='x', nbins=8)  # X축 틱을 8개 정도로 제한
    axes[-1].set_xlabel('Time Index', fontsize=9)  # X축 라벨 폰트 크기 증가
    
    # 전체 제목 - 언더샘플링 정보 포함
    sampling_info = f"(언더샘플링 레벨: {undersampling_level})" if undersampling_level > 0 else "(원본 데이터)"
    fig.suptitle(f'선택된 신호 통합 분석 {sampling_info}', fontsize=13, fontweight='bold', y=0.98)
    
    # 레이아웃 조정 - 고정 높이에 맞춰 간격 조정
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.2)  # 세로 간격을 약간 늘림 (고정 높이에 맞춰)
    
    # Streamlit에 표시
    st.pyplot(fig, use_container_width=True)
    
    # 메모리 정리
    plt.close(fig)

def main():
    st.title("🔍 다변량 시계열 데이터 분석 도구")
    st.markdown("---")
    
    # 선택 초기화 처리 (위젯 생성 전에 실행)
    if st.session_state.reset_selections:
        # 모든 체크박스를 False로 설정 (키 삭제하지 않음)
        all_signals = st.session_state.digital_signals + st.session_state.analog_signals
        for signal in all_signals:
            digital_key = f"digital_{signal}"
            analog_key = f"analog_{signal}"
            # 기존 키가 있으면 False로 설정, 없으면 새로 생성
            st.session_state[digital_key] = False
            st.session_state[analog_key] = False
        
        st.session_state.selection_order = []
        st.session_state.reset_selections = False
        st.rerun()
    
    # 디폴트 선택 적용 처리 (위젯 생성 전에 실행)
    if st.session_state.apply_defaults:
        # 기존 선택 모두 False로 설정
        all_signals = st.session_state.digital_signals + st.session_state.analog_signals
        for signal in all_signals:
            digital_key = f"digital_{signal}"
            analog_key = f"analog_{signal}"
            st.session_state[digital_key] = False
            st.session_state[analog_key] = False
        
        # 디폴트 신호 선택 적용
        st.session_state.default_signals_applied = False
        if st.session_state.digital_signals or st.session_state.analog_signals:
            apply_default_signal_selection()
        
        st.session_state.apply_defaults = False
        st.rerun()
    
    # 1단계: 데이터 입력 방법 선택
    st.header("1단계: 데이터 입력 방법 선택")
    
    input_method = st.radio(
        "데이터 입력 방법을 선택하세요:",
        ["파일 업로드", "폴더 경로 입력"],
        horizontal=True
    )
    
    if input_method == "파일 업로드":
        st.markdown("**FTR/Feather 파일을 직접 업로드하세요:**")
        uploaded_files = st.file_uploader(
            "FTR/Feather 파일들을 선택하세요",
            type=['ftr', 'feather'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("파일 업로드 처리"):
                handle_file_upload(uploaded_files)
    
    else:  # 폴더 경로 입력
        col1, col2 = st.columns([3, 1])
        
        with col1:
            folder_path = st.text_input("FTR/Feather 파일이 있는 폴더 경로:", 
                                       value=st.session_state.folder_path, 
                                       placeholder="예: output, C:/data/ftr_files")
        
        # 세션 상태 업데이트
        if folder_path != st.session_state.folder_path:
            st.session_state.folder_path = folder_path
            st.session_state.data_dict = {}
            st.session_state.digital_signals = []
            st.session_state.analog_signals = []
            st.session_state.color_mapping = {}
            st.session_state.selection_order = []
    
    # 데이터 처리 부분
    if st.session_state.folder_path and os.path.exists(st.session_state.folder_path):
        # 폴더 내 모든 FTR/Feather 파일 찾기
        all_files = sorted([f for f in os.listdir(st.session_state.folder_path) 
                           if os.path.isfile(os.path.join(st.session_state.folder_path, f)) 
                           and (f.endswith('.ftr') or f.endswith('.feather'))])
        
        if all_files:
            st.success(f"총 {len(all_files)}개의 FTR/Feather 파일을 발견했습니다.")
            
            # 사용자에게 멀티 선택 UI 제공
            selected_files = st.multiselect(
                "📂 분석할 파일들을 선택하세요:",
                options=all_files,
                default=all_files[:1] if all_files else []  # 첫 번째 파일만 기본값으로
            )
            
            if selected_files:
                st.write(f"선택된 파일 수: {len(selected_files)}")
                
                # 샘플링 설정
                st.header("2단계: 데이터 샘플링 설정")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    sampling_method = st.selectbox(
                        "📊 데이터 샘플링 방법:",
                        options=[
                            "원본 데이터 (샘플링 없음)",
                            "균등 샘플링", 
                            "랜덤 샘플링",
                            "시작/끝 우선 샘플링"
                        ],
                        index=1
                    )
                
                with col2:
                    max_points = st.number_input(
                        "최대 표시 포인트 수:",
                        min_value=100,
                        max_value=5000,
                        value=500,
                        step=100
                    )
                
                st.session_state.sampling_method = sampling_method
                st.session_state.max_points = max_points
                
                if sampling_method != "원본 데이터 (샘플링 없음)":
                    st.info(f"선택된 샘플링: {sampling_method} (최대 {max_points}개 포인트)")
                
                # 3단계: 데이터 로드
                st.header("3단계: 데이터 로드 및 분석")
                
                # 새로운 파일이 선택되었는지 확인
                current_files = set(selected_files)
                if not st.session_state.data_dict or set(st.session_state.data_dict.keys()) != set([f.split('.')[0] for f in current_files]):
                    with st.spinner("데이터를 로드하고 있습니다..."):
                        st.session_state.data_dict = {}
                        for filename in selected_files:
                            filepath = os.path.join(st.session_state.folder_path, filename)
                            file_key = filename.split('.')[0]
                            df = load_ftr_file(filepath)
                            if df is not None:
                                st.session_state.data_dict[file_key] = df
                    
                    # 색상 매핑 생성
                    if st.session_state.data_dict:
                        st.session_state.color_mapping = create_color_mapping(st.session_state.data_dict)
                    
                    # 신호 분류
                    if st.session_state.data_dict:
                        all_digital = set()
                        all_analog = set()
                        
                        for df in st.session_state.data_dict.values():
                            for col in df.columns:
                                if is_digital_signal(df[col]):
                                    all_digital.add(col)
                                else:
                                    all_analog.add(col)
                        
                        st.session_state.digital_signals = sorted(list(all_digital))
                        st.session_state.analog_signals = sorted(list(all_analog))
                        st.session_state.selection_order = []  # 신호 분류가 바뀌면 선택 순서도 초기화
                        st.session_state.default_signals_applied = False  # 디폴트 적용 상태도 초기화
                
                if st.session_state.data_dict:
                    st.success(f"총 {len(st.session_state.data_dict)}개 파일이 성공적으로 로드되었습니다.")
                    
                    # 디폴트 신호 자동 선택 적용 (데이터 로드 후 한 번만)
                    if not st.session_state.default_signals_applied and (st.session_state.digital_signals or st.session_state.analog_signals):
                        apply_default_signal_selection()
                        matched_count = len([s for s in st.session_state.selection_order if s])
                        st.info(f"💡 디폴트 신호 중 {matched_count}개가 자동으로 선택되었습니다. 필요에 따라 선택을 변경할 수 있습니다.")
                    
                    # 유사한 온도의 파일 추천
                    similar_files, ref_filename, ref_temps, ref_time, ref_cop_type = find_similar_files_by_temp(
                        st.session_state.data_dict, st.session_state.folder_path, all_files
                    )
                    
                    if similar_files:
                        st.subheader("🌡️ 유사한 온도 조건 파일 추천")
                        st.info(f"기준 파일: {ref_filename} ({ref_cop_type} Running 변화 시점 t={ref_time})")
                        st.write(f"**기준 온도**: 1st Metal Temp: {ref_temps[0]:.2f}°C, RH Bore Temp: {ref_temps[1]:.2f}°C")
                        
                        rec_cols = st.columns(min(5, len(similar_files)))
                        for i, file_info in enumerate(similar_files):
                            with rec_cols[i % len(rec_cols)]:
                                st.metric(
                                    f"추천 {i+1}",
                                    file_info['filename'],
                                    f"거리: {file_info['distance']:.2f}"
                                )
                                st.caption(f"Metal: {file_info['metal_temp']:.1f}°C")
                                st.caption(f"Bore: {file_info['bore_temp']:.1f}°C")
                                st.caption(f"{file_info['cop_type']} t={file_info['time_idx']}")
                    else:
                        st.warning("COP-A/COP-B Running 변화 시점을 찾을 수 없거나 필요한 온도 데이터가 없습니다.")
                    
                    # 데이터 정보 표시
                    st.subheader("📊 데이터 정보")
                    info_cols = st.columns(len(st.session_state.data_dict))
                    for i, (filename, df) in enumerate(st.session_state.data_dict.items()):
                        with info_cols[i % len(info_cols)]:
                            digital_count = len([col for col in df.columns if col in st.session_state.digital_signals])
                            analog_count = len([col for col in df.columns if col in st.session_state.analog_signals])
                            st.metric(
                                f"📄 {filename}", 
                                f"{len(df.columns)}개 신호",
                                f"디지털:{digital_count} 아날로그:{analog_count}"
                            )
                    
                    # 4단계: 디지털 신호 플롯
                    if st.session_state.digital_signals:
                        st.header("4단계: 디지털 신호 분석")
                        st.markdown("각 신호를 확인하고 관심 있는 신호를 선택하세요.")
                        
                        # 미리보기 언더샘플링 설정 추가
                        preview_col1, preview_col2 = st.columns([2, 1])
                        with preview_col1:
                            digital_preview_undersampling = st.selectbox(
                                "📊 4단계 미리보기 언더샘플링 레벨:",
                                options=list(range(11)),  # 0~10
                                index=3,  # 기본값 3
                                format_func=lambda x: "언더샘플링 없음" if x == 0 else f"레벨 {x} (1/{x} 샘플링)",
                                key="digital_preview_undersampling"
                            )
                        with preview_col2:
                            st.info("높을수록 빠른 미리보기")
                        
                        # 플롯 생성
                        digital_plots = create_signal_plots(
                            st.session_state.data_dict, 
                            st.session_state.digital_signals, 
                            "digital",
                            st.session_state.sampling_method,
                            st.session_state.max_points,
                            st.session_state.color_mapping,
                            digital_preview_undersampling
                        )
                        
                        digital_checkboxes = display_signal_plots_with_checkboxes(digital_plots, "digital")
                    else:
                        digital_checkboxes = {}
                    
                    # 5단계: 아날로그 신호 플롯
                    if st.session_state.analog_signals:
                        st.header("5단계: 아날로그 신호 분석")
                        
                        # 미리보기 언더샘플링 설정 추가
                        analog_preview_col1, analog_preview_col2 = st.columns([2, 1])
                        with analog_preview_col1:
                            analog_preview_undersampling = st.selectbox(
                                "📊 5단계 미리보기 언더샘플링 레벨:",
                                options=list(range(11)),  # 0~10
                                index=3,  # 기본값 3
                                format_func=lambda x: "언더샘플링 없음" if x == 0 else f"레벨 {x} (1/{x} 샘플링)",
                                key="analog_preview_undersampling"
                            )
                        with analog_preview_col2:
                            st.info("높을수록 빠른 미리보기")
                        
                        analog_plots = create_signal_plots(
                            st.session_state.data_dict, 
                            st.session_state.analog_signals, 
                            "analog",
                            st.session_state.sampling_method,
                            st.session_state.max_points,
                            st.session_state.color_mapping,
                            analog_preview_undersampling
                        )
                        
                        analog_checkboxes = display_signal_plots_with_checkboxes(analog_plots, "analog")
                    else:
                        analog_checkboxes = {}
                    
                    # 6단계: 선택된 신호들을 matplotlib로 통합 플롯
                    st.header("6단계: 선택된 신호 통합 분석")
                    
                    # 디폴트 신호 선택 버튼 추가
                    col_reset, col_default = st.columns([1, 1])
                    with col_reset:
                        if st.button("🔄 모든 선택 해제"):
                            st.session_state.reset_selections = True
                            st.rerun()
                    
                    with col_default:
                        if st.button("⭐ 디폴트 신호 선택"):
                            st.session_state.apply_defaults = True
                            st.rerun()
                    
                    # 언더샘플링 레벨 설정 UI 추가
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        undersampling_level = st.selectbox(
                            "📊 6단계 플롯 언더샘플링 레벨:",
                            options=list(range(11)),  # 0~10
                            index=2,  # 기본값 2
                            format_func=lambda x: "언더샘플링 없음 (원본 데이터)" if x == 0 else f"레벨 {x} (1/{x} 샘플링)"
                        )
                    with col2:
                        st.info("높은 레벨일수록 더 빠른 플롯")
                    
                    # 선택된 신호들 수집 (선택 순서대로)
                    all_selected = []
                    for signal, is_selected in digital_checkboxes.items():
                        if is_selected:
                            all_selected.append(signal)
                    
                    for signal, is_selected in analog_checkboxes.items():
                        if is_selected:
                            all_selected.append(signal)
                    
                    # 선택 순서에 따라 정렬 - 개선된 로직
                    selected_signals = []
                    # 먼저 선택 순서에 있는 신호들을 순서대로 추가
                    for signal in st.session_state.selection_order:
                        if signal in all_selected:
                            selected_signals.append(signal)
                    
                    # 선택 순서에 없지만 현재 선택된 신호들을 마지막에 추가 (혹시 모를 경우를 대비)
                    for signal in all_selected:
                        if signal not in selected_signals:
                            selected_signals.append(signal)
                            # 선택 순서에도 추가
                            if signal not in st.session_state.selection_order:
                                st.session_state.selection_order.append(signal)
                    
                    if selected_signals:
                        st.write(f"**선택된 신호 ({len(selected_signals)}개)**: {', '.join(selected_signals)}")
                        st.info("💡 신호들은 선택한 순서대로 플롯됩니다. 각 플롯박스는 고정된 높이로 표시됩니다.")
                        
                        # 선택 순서 디버깅 정보 (개발용 - 필요시 주석 해제)
                        # st.write(f"**디버그 - 선택 순서**: {st.session_state.selection_order}")
                        
                        # matplotlib로 빠른 렌더링
                        plot_selected_signals_matplotlib(
                            st.session_state.data_dict, 
                            selected_signals,
                            st.session_state.sampling_method,
                            st.session_state.max_points,
                            st.session_state.color_mapping,
                            undersampling_level
                        )
                    else:
                        st.info("분석할 신호를 선택해주세요.")
                
                else:
                    st.error("선택된 파일을 로드할 수 없습니다.")
        else:
            st.warning("선택한 폴더에 FTR/Feather 파일이 없습니다.")
    
    elif st.session_state.folder_path:
        st.error("존재하지 않는 폴더 경로입니다.")

if __name__ == "__main__":
    main()