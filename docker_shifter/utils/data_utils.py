"""
데이터 처리 유틸리티
"""
import pandas as pd


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
