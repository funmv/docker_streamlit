"""
DNN 관련 유틸리티
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import List, Dict, Tuple
from .file_utils import load_data_file


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


def extract_dnn_samples(df: pd.DataFrame, start_pos: int, end_pos: int,
                       lookback: int, horizon: int, step_gap: int = 1,
                       timestamp_col: str = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """기존 함수 인터페이스를 유지하면서 최적화된 버전 호출"""

    # use_positional_encoding은 전역 설정에서 가져오거나 기본값 True 사용
    try:
        # Streamlit 세션 상태에서 설정 가져오기
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
            df = load_data_file(file)
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
            df = load_data_file(file)
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
