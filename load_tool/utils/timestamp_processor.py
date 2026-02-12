"""
타임스탬프 처리 모듈
타임스탬프 관련 변환 및 처리 함수들
"""
import pandas as pd
import logging
from typing import Dict


def process_timestamp(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """타임스탬프 처리"""
    ts_config = config.get('timestamp', {})

    if not ts_config:
        return df

    target_name = ts_config.get('target_name', 'timestamp')

    # 옵션 1: 분리된 시간 컬럼 합치기
    if ts_config.get('combine_time_columns', False):
        df = combine_time_columns(df, ts_config, logger)
        return df

    # 옵션 2: 기존 timestamp 컬럼 찾기
    ts_col = None

    if ts_config.get('use_first_column', False):
        ts_col = df.columns[0]
    else:
        keywords = ts_config.get('keywords', ['timestamp', 'datetime', 'date', 'time'])
        for col in df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in keywords):
                ts_col = col
                break

    if ts_col:
        df = convert_timestamp(df, ts_col, ts_config, target_name, logger)
    elif ts_config.get('strict', False):
        raise ValueError("Timestamp column not found and strict mode is enabled")

    return df


def combine_time_columns(df: pd.DataFrame, ts_config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """분리된 시간 컬럼 합치기"""
    time_columns = ts_config.get('time_columns', ['year', 'month', 'day', 'hour', 'minute', 'second'])
    defaults = ts_config.get('defaults', {})
    base_year = ts_config.get('base_year', 2000)

    # 시간 컬럼 찾기
    found_cols = {}
    for time_col in time_columns:
        for col in df.columns:
            if time_col in str(col).lower():
                found_cols[time_col] = col
                break

    if not found_cols:
        return df

    # 시간 데이터 생성
    time_data = {}
    for part in ['year', 'month', 'day', 'hour', 'minute', 'second']:
        if part in found_cols:
            time_data[part] = pd.to_numeric(df[found_cols[part]], errors='coerce')
            # 2자리 연도 변환
            if part == 'year':
                time_data[part] = time_data[part].apply(
                    lambda x: x + base_year if x < 100 else x
                )
        else:
            default_val = defaults.get(part, ts_config.get(f'default_{part}',
                                      {'year': 2025, 'month': 1, 'day': 1,
                                       'hour': 0, 'minute': 0, 'second': 0}[part]))
            time_data[part] = default_val

    # timestamp 생성
    try:
        target_name = ts_config.get('target_name', 'timestamp')
        temp_col = pd.to_datetime(time_data)

        # 문자열로 변환 (초 단위까지만) 후 다시 datetime으로
        df[target_name] = pd.to_datetime(
            temp_col.dt.strftime('%Y-%m-%d %H:%M:%S'),
            format='%Y-%m-%d %H:%M:%S'
        )

        # 원본 컬럼 제거
        if ts_config.get('drop_time_columns', True):
            df = df.drop(columns=list(found_cols.values()))

        # timestamp 컬럼을 첫 번째로 이동
        cols = [target_name] + [c for c in df.columns if c != target_name]
        df = df[cols]

    except Exception as e:
        logger.warning(f"Failed to combine time columns: {str(e)}")

    return df


def convert_timestamp(df: pd.DataFrame, ts_col: str, ts_config: Dict, target_name: str, logger: logging.Logger) -> pd.DataFrame:
    """타임스탬프 컬럼 변환"""
    formats = ts_config.get('formats', [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S"
    ])

    # 이미 datetime이면 초 단위로 변환
    if pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        if ts_col != target_name:
            df[target_name] = df[ts_col]
            df = df.drop(columns=[ts_col])
        df[target_name] = pd.to_datetime(
            df[target_name].dt.strftime('%Y-%m-%d %H:%M:%S'),
            format='%Y-%m-%d %H:%M:%S'
        )
        cols = [target_name] + [c for c in df.columns if c != target_name]
        df = df[cols]
        return df

    # object 타입이고 Timestamp 객체가 섞여있는 경우 처리
    if df[ts_col].dtype == 'object':
        try:
            temp_col = pd.to_datetime(df[ts_col], errors='coerce')
            df[target_name] = pd.to_datetime(
                temp_col.dt.strftime('%Y-%m-%d %H:%M:%S'),
                format='%Y-%m-%d %H:%M:%S'
            )

            if df[target_name].notna().sum() > 0:
                if ts_col != target_name:
                    df = df.drop(columns=[ts_col])
                cols = [target_name] + [c for c in df.columns if c != target_name]
                df = df[cols]
                return df
        except Exception as e:
            logger.error(f"object 변환 실패: {e}")

    # 형식 시도
    for fmt in formats:
        try:
            temp_col = pd.to_datetime(df[ts_col], format=fmt, errors='coerce')
            df[target_name] = pd.to_datetime(
                temp_col.dt.strftime('%Y-%m-%d %H:%M:%S'),
                format='%Y-%m-%d %H:%M:%S'
            )

            if df[target_name].notna().sum() > 0:
                if ts_col != target_name:
                    df = df.drop(columns=[ts_col])
                cols = [target_name] + [c for c in df.columns if c != target_name]
                df = df[cols]
                return df
        except:
            continue

    # 자동 파싱 시도
    try:
        temp_col = pd.to_datetime(df[ts_col], errors='coerce')
        df[target_name] = pd.to_datetime(
            temp_col.dt.strftime('%Y-%m-%d %H:%M:%S'),
            format='%Y-%m-%d %H:%M:%S'
        )

        if ts_col != target_name:
            df = df.drop(columns=[ts_col])
        cols = [target_name] + [c for c in df.columns if c != target_name]
        df = df[cols]
    except:
        logger.warning(f"Failed to convert timestamp column: {ts_col}")

    return df


def remove_timestamp_if_needed(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """타임스탬프 제거 (옵션)"""
    ts_config = config.get('timestamp', {})

    if not ts_config.get('exclude_from_output', False):
        return df

    # timestamp 컬럼 찾아서 제거
    target_name = ts_config.get('target_name', 'timestamp')

    if target_name in df.columns:
        logger.info(f"Removing timestamp column '{target_name}' from output")
        df = df.drop(columns=[target_name])
        df = df.reset_index(drop=True)

    return df
