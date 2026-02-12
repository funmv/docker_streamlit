"""
데이터 처리 모듈
샘플링, 정규화, 타입 변환, 후처리 함수들
"""
import pandas as pd
import re
import logging
from typing import Dict, List


def apply_sampling(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """샘플링 적용"""
    sampling_config = config.get('sampling', {})

    if not sampling_config.get('enabled', False):
        return df

    interval = sampling_config.get('interval', 5)
    method = sampling_config.get('method', 'every_n')

    if interval <= 1:
        logger.warning("Sampling interval must be > 1, skipping sampling")
        return df

    try:
        # timestamp 컬럼 찾기
        ts_col = None
        ts_config = config.get('timestamp', {})
        target_name = ts_config.get('target_name', 'timestamp')

        if target_name in df.columns:
            ts_col = target_name

        logger.info(f"Applying sampling: method={method}, interval={interval}")

        if method == 'every_n':
            # 단순 N개마다 1개 선택
            df_sampled = df.iloc[::interval].reset_index(drop=True)

        elif method in ['mean', 'median', 'first', 'last']:
            # 그룹화하여 집계
            # 타임스탬프가 있으면 시간순으로 정렬
            if ts_col and pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                df = df.sort_values(by=ts_col).reset_index(drop=True)

            # interval 크기의 그룹으로 나누기
            df['_group'] = df.index // interval

            # timestamp 컬럼과 다른 컬럼 분리
            if ts_col:
                other_cols = [c for c in df.columns if c not in [ts_col, '_group']]
            else:
                other_cols = [c for c in df.columns if c != '_group']

            # 집계 방법에 따라 처리
            if method == 'mean':
                df_agg = df.groupby('_group')[other_cols].mean()
            elif method == 'median':
                df_agg = df.groupby('_group')[other_cols].median()
            elif method == 'first':
                df_agg = df.groupby('_group')[other_cols].first()
            elif method == 'last':
                df_agg = df.groupby('_group')[other_cols].last()

            # timestamp는 first로 처리
            if ts_col:
                ts_agg = df.groupby('_group')[ts_col].first()
                df_sampled = pd.concat([ts_agg, df_agg], axis=1).reset_index(drop=True)
                # timestamp를 첫 번째 컬럼으로
                cols = [ts_col] + [c for c in df_sampled.columns if c != ts_col]
                df_sampled = df_sampled[cols]
            else:
                df_sampled = df_agg.reset_index(drop=True)

        else:
            logger.warning(f"Unknown sampling method: {method}")
            return df

        original_rows = len(df)
        sampled_rows = len(df_sampled)
        reduction = 100 * (1 - sampled_rows / original_rows)

        logger.info(f"Sampling complete: {original_rows} → {sampled_rows} rows ({reduction:.1f}% reduction)")

        return df_sampled

    except Exception as e:
        logger.error(f"Sampling failed: {str(e)}")
        return df


def normalize_column_names(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """컬럼명 정규화"""
    col_config = config.get('column_names', {})

    if not col_config:
        return df

    new_columns = []
    for col in df.columns:
        col_str = str(col)

        if col_config.get('replace_spaces'):
            col_str = col_str.replace(' ', col_config['replace_spaces'])

        if not col_config.get('keep_special_chars', True):
            col_str = re.sub(r'[^a-zA-Z0-9가-힣_]', '', col_str)

        if col_config.get('lowercase', False):
            col_str = col_str.lower()

        new_columns.append(col_str)

    df.columns = new_columns
    return df


def convert_types(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """데이터 타입 변환"""
    type_config = config.get('data_types', {})

    if not type_config:
        return df

    # Value mapping (먼저 적용)
    value_mapping = type_config.get('value_mapping', {})
    if value_mapping:
        for col in df.columns:
            if col == 'timestamp':
                continue
            try:
                if df[col].dtype == object:
                    df[col] = df[col].map(lambda x: value_mapping.get(x, x))
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass
            except:
                continue

    # Boolean 타입 통일 처리
    for col in df.columns:
        if col == 'timestamp':
            continue
        try:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)

            elif df[col].dtype == object:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2:
                    unique_str = set([str(v).upper() for v in unique_vals])
                    if unique_str.issubset({'TRUE', 'FALSE', '0', '1'}):
                        df[col] = df[col].map({
                            'True': 1, 'true': 1, 'TRUE': 1, True: 1,
                            'False': 0, 'false': 0, 'FALSE': 0, False: 0,
                            '1': 1, '0': 0, 1: 1, 0: 0
                        })
        except Exception as e:
            logger.warning(f"컬럼 '{col}' Boolean 변환 실패: {e}")
            continue

    # Null 값 처리
    null_values = type_config.get('null_values', [])
    if null_values:
        df = df.replace(null_values, pd.NA)
        try:
            df = df.infer_objects(copy=False)
        except:
            pass

    # 자동 타입 추론
    if type_config.get('auto_infer', False):
        sample_rows = type_config.get('sample_rows', 100)
        df = auto_infer_types(df, sample_rows)

    return df


def auto_infer_types(df: pd.DataFrame, sample_rows: int) -> pd.DataFrame:
    """자동 타입 추론"""
    for col in df.columns:
        if col == 'timestamp':
            continue
        try:
            sample = df[col].head(sample_rows).dropna()
            if len(sample) == 0:
                continue
            converted = pd.to_numeric(sample, errors='coerce')
            if converted.notna().sum() / len(sample) > 0.8:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            continue
    return df


def post_process(df: pd.DataFrame, config: Dict, logger: logging.Logger, removed_columns: List[Dict]) -> pd.DataFrame:
    """후처리"""
    post_config = config.get('post_processing', {})

    if not post_config:
        return df

    if post_config.get('remove_empty_rows', False):
        df = df.dropna(how='all')

    threshold = post_config.get('remove_high_null_columns')
    if threshold:
        null_pct = (df.isnull().sum() / len(df)) * 100
        cols_to_drop = null_pct[null_pct > threshold].index.tolist()
        if cols_to_drop:
            # 제거된 컬럼 정보 기록
            for col in cols_to_drop:
                removed_info = {
                    'tag_name': col,
                    'description': col,
                    'unit': None,
                    'reason': f'Null 비율 높음 ({null_pct[col]:.1f}% > {threshold}%)'
                }
                removed_columns.append(removed_info)
                logger.info(f"컬럼 '{col}' 제거됨 (Null {null_pct[col]:.1f}%)")

            df = df.drop(columns=cols_to_drop)

    if post_config.get('remove_duplicates', False):
        df = df.drop_duplicates()

    return df
