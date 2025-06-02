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
# 한글 폰트 설정
# =================================================================================
def setup_korean_font():
    """한글 폰트 설정 함수"""
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

# =================================================================================
# 페이지 설정 및 초기화
# =================================================================================
st.set_page_config(page_title="다변량 시계열 데이터 분석", layout="wide")
setup_korean_font()

# =================================================================================
# 유틸리티 함수들
# =================================================================================
def load_feather_file(uploaded_file) -> pd.DataFrame:
    """Feather 파일을 로드하는 함수"""
    try:
        df = pd.read_feather(uploaded_file)
        return df
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {str(e)}")
        return None

def apply_time_delay(df: pd.DataFrame, column: str, delay: int) -> pd.Series:
    """시계열 데이터에 시간 지연을 적용하는 함수"""
    if delay == 0:
        return df[column]
    elif delay > 0:
        # 양수 지연: 미래 값을 현재로 이동 (앞쪽에 NaN 추가)
        delayed_series = df[column].shift(-delay)
    else:
        # 음수 지연: 과거 값을 현재로 이동 (뒤쪽에 NaN 추가)
        delayed_series = df[column].shift(-delay)
    
    return delayed_series

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

def create_multivariate_plot(df: pd.DataFrame, selected_cols: List[str], 
                           delays: Dict[str, int], downsample_rate: int = 1, 
                           crosshair: bool = True, num_segments: int = 3, 
                           selected_segment: int = 0) -> go.Figure:
    """기본 다변량 시계열 플롯을 생성하는 함수"""
    # 데이터 구간 선택
    df_segment = get_data_segment(df, num_segments, selected_segment)
    
    fig = go.Figure()
    
    for col in selected_cols:
        delay = delays.get(col, 0)
        
        # 1단계: 선택된 구간에서 시간 지연 적용
        y_data = apply_time_delay(df_segment, col, delay)
        
        # 2단계: 지연 적용된 데이터에 다운샘플링 적용
        y = y_data.iloc[::downsample_rate]
        x = df_segment.index[::downsample_rate]
        
        # 지연값이 있는 경우 레이블에 표시
        label = f"{col} (delay: {delay})" if delay != 0 else col
        
        fig.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode='lines',
            name=label,
            showlegend=True,
            hoverinfo='x',
            hovertemplate=''
        ))
    
    # 구간 정보를 제목에 추가
    segment_info = f"구간 {selected_segment + 1}/{num_segments}"
    fig.update_layout(
        title=f"📊 다변량 시계열 신호 분석 ({segment_info})",
        dragmode="zoom",
        xaxis=dict(
            rangeslider=dict(visible=False),
            title="시간 인덱스"
        ),
        yaxis=dict(
            title="신호 값"
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
                spikethickness=1,
                title="시간 인덱스"
            ),
            yaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="blue",
                spikethickness=1,
                title="신호 값"
            )
        )
    
    return fig

def create_combined_plot(df: pd.DataFrame, delay_cols: List[str], 
                        delays: Dict[str, int], reference_cols: List[str] = None,
                        downsample_rate: int = 1, crosshair: bool = True,
                        num_segments: int = 3, selected_segment: int = 0) -> go.Figure:
    """지연 적용된 컬럼과 기준 컬럼을 함께 표시하는 플롯을 생성하는 함수"""
    # 데이터 구간 선택
    df_segment = get_data_segment(df, num_segments, selected_segment)
    
    fig = go.Figure()
    
    # 지연 적용된 컬럼들 추가
    for col in delay_cols:
        delay = delays.get(col, 0)
        
        # 1단계: 선택된 구간에서 시간 지연 적용
        y_data = apply_time_delay(df_segment, col, delay)
        
        # 2단계: 지연 적용된 데이터에 다운샘플링 적용
        y = y_data.iloc[::downsample_rate]
        x = df_segment.index[::downsample_rate]
        
        # 지연값이 있는 경우 레이블에 표시
        label = f"{col} (delay: {delay:+d})" if delay != 0 else f"{col} (original)"
        
        fig.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode='lines',
            name=label,
            showlegend=True,
            hoverinfo='x',
            hovertemplate='',
            line=dict(width=2)  # 지연 적용된 신호는 두꺼운 선
        ))
    
    # 기준 컬럼들 추가 (지연 적용 안됨)
    if reference_cols:
        for col in reference_cols:
            # 1단계: 선택된 구간의 원본 데이터 (지연 적용 안함)
            y_data = df_segment[col]
            
            # 2단계: 다운샘플링 적용
            y = y_data.iloc[::downsample_rate]
            x = df_segment.index[::downsample_rate]
            
            fig.add_trace(go.Scattergl(
                x=x,
                y=y,
                mode='lines',
                name=f"{col} (reference)",
                showlegend=True,
                hoverinfo='x',
                hovertemplate='',
                line=dict(width=1, dash='dot')  # 기준 신호는 점선으로 구분
            ))
    
    # 구간 정보를 제목에 추가
    segment_info = f"구간 {selected_segment + 1}/{num_segments}"
    fig.update_layout(
        title=f"📊 시간 지연 적용 신호 vs 기준 신호 비교 ({segment_info})",
        dragmode="zoom",
        xaxis=dict(
            rangeslider=dict(visible=False),
            title="시간 인덱스"
        ),
        yaxis=dict(
            title="신호 값"
        ),
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    if crosshair:
        fig.update_layout(
            hovermode="x",
            xaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="red",
                spikethickness=1,
                title="시간 인덱스"
            ),
            yaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="blue",
                spikethickness=1,
                title="신호 값"
            )
        )
    
    return fig

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

def create_multi_file_plot(selected_files: List, selected_features: List[str], 
                          downsample_rate: int = 1, crosshair: bool = True,
                          num_segments: int = 3, selected_segment: int = 0) -> go.Figure:
    """선택된 파일들의 특징들을 플롯하는 함수 (탭1,2 방식과 동일)"""
    fig = go.Figure()
    
    # 파일별로 처리
    for file in selected_files:
        try:
            df = load_feather_file(file)
            if df is None:
                continue
            
            # 데이터 구간 선택
            df_segment = get_data_segment(df, num_segments, selected_segment)
            
            # 선택된 특징들 처리
            for feature in selected_features:
                if feature in df.columns:
                    # 1단계: 선택된 구간의 원본 데이터
                    y_data = df_segment[feature]
                    
                    # 2단계: 다운샘플링 적용
                    y = y_data.iloc[::downsample_rate]
                    x = df_segment.index[::downsample_rate]
                    
                    # 파일명과 특징명을 포함한 레이블
                    file_name = file.name.split('.')[0]  # 확장자 제거
                    label = f"{file_name}_{feature}"
                    
                    fig.add_trace(go.Scattergl(
                        x=x,
                        y=y,
                        mode='lines',
                        name=label,
                        showlegend=True,
                        hoverinfo='x',
                        hovertemplate=''
                    ))
                    
        except Exception as e:
            st.warning(f"⚠️ {file.name} 플롯 생성 중 오류: {str(e)}")
            continue
    
    # 구간 정보를 제목에 추가
    segment_info = f"구간 {selected_segment + 1}/{num_segments}"
    fig.update_layout(
        title=f"📊 다중 파일 특징 비교 ({segment_info})",
        dragmode="zoom",
        xaxis=dict(
            rangeslider=dict(visible=False),
            title="시간 인덱스"
        ),
        yaxis=dict(
            title="신호 값"
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
                spikethickness=1,
                title="시간 인덱스"
            ),
            yaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="blue",
                spikethickness=1,
                title="신호 값"
            )
        )
    
    return fig

def process_batch_files(files: List, selected_features: List[str], delays: Dict[str, int]) -> List[Dict]:
    """배치로 여러 파일에 지연 처리를 적용하는 함수"""
    processed_files = []
    
    for i, file in enumerate(files):
        try:
            # 파일 로드
            df = load_feather_file(file)
            if df is None:
                continue
            
            # 선택된 특징들이 파일에 존재하는지 확인
            missing_features = [feat for feat in selected_features if feat not in df.columns]
            if missing_features:
                st.warning(f"⚠️ {file.name}에서 누락된 특징: {missing_features}")
                continue
            
            # 지연 처리 적용
            processed_df = df.copy()
            for feature in selected_features:
                delay = delays.get(feature, 0)
                if delay != 0:
                    shifted_series = apply_time_delay(df, feature, delay)
                    processed_df[feature] = shifted_series
            
            # 처리된 데이터 정보 저장
            processed_files.append({
                'original_name': file.name,
                'processed_name': f"{file.name.split('.')[0]}_batch_shifted.feather",
                'dataframe': processed_df,
                'shape': processed_df.shape,
                'applied_delays': {feat: delays[feat] for feat in selected_features if delays.get(feat, 0) != 0}
            })
            
        except Exception as e:
            st.error(f"❌ {file.name} 처리 중 오류: {str(e)}")
            continue
    
    return processed_files

def create_zip_download(processed_files: List[Dict], zip_filename: str) -> bytes:
    """처리된 파일들을 ZIP으로 압축하여 다운로드 가능한 형태로 만드는 함수"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_info in processed_files:
            # DataFrame을 feather 형식으로 변환
            feather_buffer = io.BytesIO()
            file_info['dataframe'].reset_index(drop=True).to_feather(feather_buffer)
            feather_buffer.seek(0)
            
            # ZIP에 파일 추가
            zip_file.writestr(file_info['processed_name'], feather_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()



# DNN 관련 모든 함수들 - create_zip_download 함수 뒤에 추가하세요
def create_positional_encoding(position: int, d_model: int = 8) -> np.ndarray:
    """시간 포지션에 대한 positional encoding 생성"""
    pe = np.zeros(d_model)
    for i in range(0, d_model, 2):
        pe[i] = np.sin(position / (10000 ** (i / d_model)))
        if i + 1 < d_model:
            pe[i + 1] = np.cos(position / (10000 ** (i / d_model)))
    return pe

def extract_time_features(timestamp_value, use_positional_encoding: bool = True) -> np.ndarray:
    """timestamp로부터 시간 특징 추출"""
    
    # Timestamp 타입을 숫자로 변환
    if hasattr(timestamp_value, 'timestamp'):
        # pandas Timestamp 객체인 경우
        timestamp_seconds = timestamp_value.timestamp()
    elif isinstance(timestamp_value, (int, float)):
        # 이미 숫자인 경우
        timestamp_seconds = float(timestamp_value)
    else:
        try:
            # 문자열이나 다른 형태인 경우 pandas로 변환 시도
            timestamp_seconds = pd.to_datetime(timestamp_value).timestamp()
        except:
            # 변환 실패시 기본값 사용
            timestamp_seconds = 0.0
    
    # 기본 시간 특징 (시, 분, 초)
    hours = int((timestamp_seconds // 3600) % 24)
    minutes = int((timestamp_seconds % 3600) // 60) 
    seconds = int(timestamp_seconds % 60)
    
    # 정규화된 시간 특징 (0-1 범위)
    time_features = np.array([
        hours / 23.0,           # 시간 (0-23 -> 0-1)
        minutes / 59.0,         # 분 (0-59 -> 0-1)
        seconds / 59.0          # 초 (0-59 -> 0-1)
    ])
    
    if use_positional_encoding:
        # Positional encoding 추가
        pe = create_positional_encoding(int(timestamp_seconds // 5))  # 5초 단위
        time_features = np.concatenate([time_features, pe])
    
    return time_features

def split_files_train_val(files: List, train_ratio: float = 0.8) -> Tuple[List, List]:
    """파일들을 훈련용과 검증용으로 분할"""
    total_files = len(files)
    train_size = int(total_files * train_ratio)
    
    # 파일들을 섞어서 분할
    import random
    shuffled_files = files.copy()
    random.shuffle(shuffled_files)
    
    train_files = shuffled_files[:train_size]
    val_files = shuffled_files[train_size:]
    
    return train_files, val_files



def extract_dnn_samples_optimized(df: pd.DataFrame, start_pos: int, end_pos: int, 
                                  lookback: int, horizon: int, step_gap: int = 1,
                                  timestamp_col: str = None, use_positional_encoding: bool = True) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """최적화된 단일 파일에서 DNN 학습용 샘플 추출 (벡터화 연산 사용)"""
    
    # timestamp 컬럼 확인
    if timestamp_col is None:
        # timestamp 관련 컬럼 자동 검색
        timestamp_candidates = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
        if timestamp_candidates:
            timestamp_col = timestamp_candidates[0]
        else:
            timestamp_col = df.columns[0]  # 첫 번째 컬럼을 timestamp로 사용
    
    # 특징 컬럼들 (timestamp 제외)
    feature_cols = [col for col in df.columns if col != timestamp_col]
    
    # 데이터를 numpy 배열로 변환 (메모리 효율성과 속도 향상)
    data_features_array = df[feature_cols].values.astype(np.float32)  # float32로 메모리 절약
    
    # 결측값 처리 (한 번에 처리)
    data_features_array = np.nan_to_num(data_features_array, nan=0.0)
    
    # timestamp 배열 준비
    if timestamp_col in df.columns:
        timestamp_array = df[timestamp_col].values
        # timestamp 결측값 처리
        nan_mask = pd.isna(timestamp_array)
        if nan_mask.any():
            # 결측값을 인덱스 * 5초로 대체
            timestamp_array = np.where(nan_mask, np.arange(len(df)) * 5, timestamp_array)
    else:
        # timestamp 컬럼이 없으면 인덱스 * 5초로 생성
        timestamp_array = np.arange(len(df)) * 5
    
    # 시간 특징 배열 미리 계산 (벡터화)
    time_features_array = extract_time_features_vectorized(timestamp_array, use_positional_encoding)
    
    # 데이터와 시간 특징 결합
    combined_features_array = np.concatenate([time_features_array, data_features_array], axis=1)
    
    # 샘플 추출 범위 계산
    max_pos = min(end_pos, len(df) - horizon)
    actual_start = max(start_pos, lookback)
    
    # 유효한 샘플 위치들 계산
    sample_positions = np.arange(actual_start, max_pos, step_gap)
    
    if len(sample_positions) == 0:
        return np.array([]), np.array([]), []
    
    # 입력 시퀀스 인덱스 생성 (벡터화)
    # shape: (num_samples, lookback)
    input_indices = sample_positions[:, np.newaxis] - np.arange(lookback, 0, -1)[np.newaxis, :]
    
    # 출력 시퀀스 인덱스 생성 (벡터화)
    # shape: (num_samples, horizon)
    output_indices = sample_positions[:, np.newaxis] + np.arange(horizon)[np.newaxis, :]
    
    # 유효한 인덱스인지 확인
    valid_input_mask = (input_indices >= 0) & (input_indices < len(combined_features_array))
    valid_output_mask = (output_indices >= 0) & (output_indices < len(combined_features_array))
    valid_samples_mask = valid_input_mask.all(axis=1) & valid_output_mask.all(axis=1)
    
    # 유효한 샘플만 선택
    valid_sample_positions = sample_positions[valid_samples_mask]
    valid_input_indices = input_indices[valid_samples_mask]
    valid_output_indices = output_indices[valid_samples_mask]
    
    if len(valid_sample_positions) == 0:
        return np.array([]), np.array([]), []
    
    # 벡터화된 인덱싱으로 샘플 추출
    # input_samples shape: (num_samples, lookback, features)
    input_samples = combined_features_array[valid_input_indices]
    
    # output_samples shape: (num_samples, horizon, features)
    output_samples = combined_features_array[valid_output_indices]
    
    # 샘플 정보 생성 (벡터화)
    sample_info = []
    for i, pos in enumerate(valid_sample_positions):
        sample_info.append({
            'sample_index': i,
            'input_start': int(pos - lookback),
            'input_end': int(pos),
            'output_start': int(pos),
            'output_end': int(pos + horizon),
            'current_position': int(pos)
        })
    
    return input_samples.astype(np.float32), output_samples.astype(np.float32), sample_info


def extract_time_features_vectorized(timestamp_array: np.ndarray, use_positional_encoding: bool = True) -> np.ndarray:
    """벡터화된 시간 특징 추출"""
    
    # Timestamp 배열을 숫자로 변환
    timestamp_seconds = np.zeros_like(timestamp_array, dtype=np.float64)
    
    for i, timestamp_value in enumerate(timestamp_array):
        if hasattr(timestamp_value, 'timestamp'):
            # pandas Timestamp 객체인 경우
            timestamp_seconds[i] = timestamp_value.timestamp()
        elif isinstance(timestamp_value, (int, float)):
            # 이미 숫자인 경우
            timestamp_seconds[i] = float(timestamp_value)
        else:
            try:
                # 문자열이나 다른 형태인 경우 pandas로 변환 시도
                timestamp_seconds[i] = pd.to_datetime(timestamp_value).timestamp()
            except:
                # 변환 실패시 기본값 사용
                timestamp_seconds[i] = 0.0
    
    # 벡터화된 시간 특징 계산
    hours = ((timestamp_seconds // 3600) % 24) / 23.0
    minutes = ((timestamp_seconds % 3600) // 60) / 59.0
    seconds = (timestamp_seconds % 60) / 59.0
    
    # 기본 시간 특징
    time_features = np.column_stack([hours, minutes, seconds])
    
    if use_positional_encoding:
        # Positional encoding 벡터화
        positions = (timestamp_seconds // 5).astype(int)  # 5초 단위
        pe_array = create_positional_encoding_vectorized(positions, d_model=8)
        time_features = np.concatenate([time_features, pe_array], axis=1)
    
    return time_features.astype(np.float32)


def create_positional_encoding_vectorized(positions: np.ndarray, d_model: int = 8) -> np.ndarray:
    """벡터화된 positional encoding 생성"""
    
    # positions shape: (n,) -> (n, 1)
    pos = positions[:, np.newaxis]
    
    # 인덱스 배열 생성
    i = np.arange(0, d_model, 2)[np.newaxis, :]  # shape: (1, d_model//2)
    
    # 각도 계산 (벡터화)
    angles = pos / (10000 ** (i / d_model))  # shape: (n, d_model//2)
    
    # PE 배열 초기화
    pe = np.zeros((len(positions), d_model), dtype=np.float32)
    
    # sin과 cos 계산 (벡터화)
    pe[:, 0::2] = np.sin(angles)  # 짝수 인덱스
    if d_model % 2 == 1:
        pe[:, 1::2] = np.cos(angles[:, :-1])  # 홀수 인덱스 (마지막 제외)
    else:
        pe[:, 1::2] = np.cos(angles)  # 홀수 인덱스
    
    return pe


# 기존 함수를 최적화된 버전으로 대체하는 래퍼 함수
def extract_dnn_samples(df: pd.DataFrame, start_pos: int, end_pos: int, 
                       lookback: int, horizon: int, step_gap: int = 1,
                       timestamp_col: str = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """기존 함수 인터페이스를 유지하면서 최적화된 버전 호출"""
    
    # use_positional_encoding은 전역 설정에서 가져오거나 기본값 True 사용
    try:
        # Streamlit 세션 상태에서 설정 가져오기
        import streamlit as st
        use_positional_encoding = st.session_state.get('dnn_pos_encoding', True)
    except:
        use_positional_encoding = True
    
    return extract_dnn_samples_optimized(
        df, start_pos, end_pos, lookback, horizon, step_gap, 
        timestamp_col, use_positional_encoding
    )



def process_all_files_for_dnn(train_files: List, val_files: List, 
                             start_pos: int, end_pos: int, lookback: int, 
                             horizon: int, step_gap: int) -> Dict:
    """모든 파일에서 DNN 데이터 추출"""
    
    train_inputs = []
    train_outputs = []
    train_info = []
    
    val_inputs = []
    val_outputs = []
    val_info = []
    
    # Training 파일들 처리
    st.write("🔄 Training 데이터 추출 중...")
    for i, file in enumerate(train_files):
        try:
            df = load_feather_file(file)
            if df is not None:
                input_arr, output_arr, info = extract_dnn_samples(
                    df, start_pos, end_pos, lookback, horizon, step_gap
                )
                
                if len(input_arr) > 0:
                    train_inputs.append(input_arr)
                    train_outputs.append(output_arr)
                    
                    # 파일 정보 추가
                    for sample_info in info:
                        sample_info['file_name'] = file.name
                        sample_info['file_index'] = i
                        sample_info['split'] = 'train'
                    train_info.extend(info)
                    
                st.write(f"   ✅ {file.name}: {len(input_arr)}개 샘플 추출")
        except Exception as e:
            st.error(f"   ❌ {file.name}: 처리 실패 - {str(e)}")
    
    # Validation 파일들 처리
    st.write("🔄 Validation 데이터 추출 중...")
    for i, file in enumerate(val_files):
        try:
            df = load_feather_file(file)
            if df is not None:
                input_arr, output_arr, info = extract_dnn_samples(
                    df, start_pos, end_pos, lookback, horizon, step_gap
                )
                
                if len(input_arr) > 0:
                    val_inputs.append(input_arr)
                    val_outputs.append(output_arr)
                    
                    # 파일 정보 추가
                    for sample_info in info:
                        sample_info['file_name'] = file.name
                        sample_info['file_index'] = i
                        sample_info['split'] = 'validation'
                    val_info.extend(info)
                    
                st.write(f"   ✅ {file.name}: {len(input_arr)}개 샘플 추출")
        except Exception as e:
            st.error(f"   ❌ {file.name}: 처리 실패 - {str(e)}")
    
    # 데이터 결합
    final_train_inputs = np.concatenate(train_inputs, axis=0) if train_inputs else np.array([])
    final_train_outputs = np.concatenate(train_outputs, axis=0) if train_outputs else np.array([])
    
    final_val_inputs = np.concatenate(val_inputs, axis=0) if val_inputs else np.array([])
    final_val_outputs = np.concatenate(val_outputs, axis=0) if val_outputs else np.array([])
    
    return {
        'train_inputs': final_train_inputs,
        'train_outputs': final_train_outputs,
        'train_info': train_info,
        'val_inputs': final_val_inputs,
        'val_outputs': final_val_outputs,
        'val_info': val_info
    }

def save_dnn_dataset(dataset: Dict, metadata: Dict, filename: str) -> bytes:
    """DNN 데이터셋을 NPY 형식으로 저장"""
    
    # 전체 데이터 구성
    full_dataset = {
        'metadata': metadata,
        'train_inputs': dataset['train_inputs'],
        'train_outputs': dataset['train_outputs'],
        'train_info': dataset['train_info'],
        'val_inputs': dataset['val_inputs'],
        'val_outputs': dataset['val_outputs'],
        'val_info': dataset['val_info']
    }
    
    # numpy save 형식으로 직렬화
    buffer = io.BytesIO()
    np.save(buffer, full_dataset, allow_pickle=True)
    buffer.seek(0)
    
    return buffer.getvalue()




# =================================================================================
# 메인 애플리케이션
# =================================================================================
def main():
    st.title("📈 학습용 시계열 데이터 추출 툴")
    
    # 탭 생성 - 추후 확장을 위한 구조
    tab1, tab2, tab3 = st.tabs(["🔍 신호 관찰", "📊 이동 실행", "📦 데이터 추출"])
    
    # =================================================================================
    # 탭 1: 신호 분석 (메인 기능)
    # =================================================================================
    with tab1:
        st.header("🚀 다변량 시계열 신호 관찰 및 분석")
        
        # 파일 업로드 섹션
        st.subheader("📁 파일 업로드")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**FTR/Feather 파일을 직접 업로드하세요:**")
            uploaded_files = st.file_uploader(
                "FTR/Feather 파일들을 선택하세요",
                type=['ftr', 'feather'],
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
            df = load_feather_file(selected_file)
            
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
                        
                        # 파일명 입력
                        default_filename = f"{selected_file.name.split('.')[0]}_shifted"
                        save_filename = st.text_input(
                            "저장할 파일명 (확장자 제외)",
                            value=default_filename,
                            help="feather 형식으로 저장됩니다"
                        )
                        
                        # 데이터 생성 및 다운로드 버튼
                        if st.button("🔄 지연 적용 데이터 생성 및 다운로드", key="generate_shifted_data"):
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
                                
                                # feather 형식으로 저장
                                output_buffer = io.BytesIO()
                                shifted_df.reset_index(drop=True).to_feather(output_buffer)
                                output_buffer.seek(0)
                                
                                # 다운로드 버튼
                                st.download_button(
                                    label="💾 Feather 파일 다운로드",
                                    data=output_buffer.getvalue(),
                                    file_name=f"{save_filename}.feather",
                                    mime="application/octet-stream",
                                    help="지연이 적용된 데이터를 feather 형식으로 다운로드"
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
        st.markdown("여러 개의 FTR 파일에 동일한 지연 설정을 일괄 적용하여 처리합니다.")
        
        # 배치 파일 업로드 섹션
        st.subheader("📁 배치 파일 업로드")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**여러 개의 FTR/Feather 파일을 업로드하세요:**")
            batch_uploaded_files = st.file_uploader(
                "배치 처리할 FTR/Feather 파일들을 선택하세요",
                type=['ftr', 'feather'],
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
                        
                        # ZIP 파일명 설정
                        default_zip_name = f"batch_shifted_files_{len(processed_files)}files"
                        zip_filename = st.text_input(
                            "ZIP 파일명 (확장자 제외)",
                            value=default_zip_name,
                            key="zip_filename_input"
                        )
                        
                        # ZIP 다운로드 버튼
                        if st.button("📦 ZIP 파일로 일괄 다운로드", key="download_batch_zip"):
                            try:
                                with st.spinner("📦 ZIP 파일 생성 중..."):
                                    zip_data = create_zip_download(processed_files, f"{zip_filename}.zip")
                                
                                st.download_button(
                                    label="💾 ZIP 파일 다운로드",
                                    data=zip_data,
                                    file_name=f"{zip_filename}.zip",
                                    mime="application/zip",
                                    help="모든 처리된 파일을 ZIP으로 압축하여 다운로드"
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
                                    # 개별 파일 다운로드
                                    feather_buffer = io.BytesIO()
                                    file_info['dataframe'].reset_index(drop=True).to_feather(feather_buffer)
                                    feather_buffer.seek(0)
                                    
                                    st.download_button(
                                        label="💾 다운로드",
                                        data=feather_buffer.getvalue(),
                                        file_name=file_info['processed_name'],
                                        mime="application/octet-stream",
                                        key=f"individual_download_{i}"
                                    )
    


    # =================================================================================
    # 탭 3: 다중 파일 시각화 (새로운 기능)
    # =================================================================================
    with tab3:
        st.header("📊 다중 파일 시각화")
        st.markdown("여러 개의 FTR 파일을 로드하여 동일한 특징들을 비교 시각화합니다.")
        
        # 다중 파일 업로드 섹션
        st.subheader("📁 다중 파일 업로드")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**여러 개의 FTR/Feather 파일을 업로드하세요:**")
            multi_uploaded_files = st.file_uploader(
                "시각화할 FTR/Feather 파일들을 선택하세요",
                type=['ftr', 'feather'],
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
                                                    '결측값': feature_data.isna().sum()
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
                                            'lookback_length': lookback_length,
                                            'horizon_length': horizon_length,
                                            'step_gap': step_gap,
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
                                    {'파라미터': '과거 참조 길이', '값': extraction_params.get('lookback_length', 'N/A')},
                                    {'파라미터': '예측 구간 길이', '값': extraction_params.get('horizon_length', 'N/A')},
                                    {'파라미터': '스텝 간격', '값': extraction_params.get('step_gap', 'N/A')},
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
                                    {'항목': '시간 특징 수', '값': feature_info.get('time_features', 'N/A')},
                                    {'항목': '데이터 특징 수', '값': feature_info.get('data_features', 'N/A')},
                                    {'항목': '전체 특징 수', '값': feature_info.get('total_features', 'N/A')},
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
# 애플리케이션 실행
# =================================================================================
if __name__ == "__main__":
    main()




