"""
데이터 유틸리티 모듈
YAML, Parquet, DataFrame 변환 관련 유틸리티 함수들
"""
import pandas as pd
import yaml
import json
import logging
from typing import Dict
from datetime import datetime, date


def dict_to_yaml_string(data: Dict) -> str:
    """딕셔너리를 YAML 문자열로 변환"""
    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


def yaml_string_to_dict(yaml_str: str) -> Dict:
    """YAML 문자열을 딕셔너리로 변환"""
    return yaml.safe_load(yaml_str)


def extract_date_range_from_df(df: pd.DataFrame) -> str:
    """
    DataFrame에서 timestamp 컬럼의 날짜 범위 추출
    
    Returns:
        날짜 범위 문자열 (예: "20250101_20250131" 또는 "20250115")
    """
    ts_col = None
    for col in df.columns:
        if 'timestamp' in str(col).lower() or 'datetime' in str(col).lower() or 'date' in str(col).lower():
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                ts_col = col
                break
    
    if ts_col is None:
        return ""
    
    try:
        valid_dates = df[ts_col].dropna()
        
        if len(valid_dates) == 0:
            return ""
        
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        
        min_str = pd.to_datetime(min_date).strftime('%Y%m%d')
        max_str = pd.to_datetime(max_date).strftime('%Y%m%d')
        
        if min_str == max_str:
            return min_str
        else:
            return f"{min_str}_{max_str}"
    except:
        return ""


def save_to_parquet_with_metadata(df: pd.DataFrame, buffer, compression='snappy'):
    """
    DataFrame을 메타데이터와 함께 Parquet으로 저장
    
    Args:
        df: 저장할 DataFrame (attrs 포함)
        buffer: 저장할 버퍼
        compression: 압축 방식
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    table = pa.Table.from_pandas(df)
    
    metadata = {}
    if hasattr(df, 'attrs') and df.attrs:
        metadata['pandas_attrs'] = json.dumps(df.attrs)
    
    existing_meta = table.schema.metadata or {}
    existing_meta.update({k.encode(): v.encode() if isinstance(v, str) else v 
                        for k, v in metadata.items()})
    
    new_schema = table.schema.with_metadata(existing_meta)
    table = table.cast(new_schema)
    
    pq.write_table(table, buffer, compression=compression)


def save_to_hdf5_with_metadata(df: pd.DataFrame, buffer, key='data', compression='gzip'):
    """
    DataFrame을 메타데이터와 함께 HDF5로 저장
    
    Args:
        df: 저장할 DataFrame (attrs 포함)
        buffer: 저장할 버퍼 (파일 경로만 지원, BytesIO 불가)
        key: HDF5 내 데이터셋 키
        compression: 압축 방식 ('gzip', 'lzf', 'blosc' 등)
    
    메타데이터 저장 방식:
        fixed format 사용 + 그룹 속성에 메타데이터 저장
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== HDF5 저장 시작 ===")
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"hasattr(df, 'attrs'): {hasattr(df, 'attrs')}")
    
    if hasattr(df, 'attrs'):
        logger.info(f"df.attrs keys: {list(df.attrs.keys())}")
        if 'header_metadata' in df.attrs:
            header_meta = df.attrs['header_metadata']
            logger.info(f"header_metadata keys: {list(header_meta.keys())}")
            for key_name, values in header_meta.items():
                logger.info(f"  {key_name}: {len(values) if isinstance(values, list) else 'not a list'} items")
    
    # BytesIO 체크
    if hasattr(buffer, 'seek'):
        logger.error("❌ HDF5는 파일 경로만 지원합니다. BytesIO는 사용할 수 없습니다.")
        raise ValueError("HDF5 저장은 파일 경로만 지원합니다.")
    
    # fixed format으로 저장 (모든 컬럼 보존)
    df.to_hdf(
        buffer, 
        key=key, 
        mode='w',
        format='fixed',  # fixed는 모든 데이터 타입 지원
        complevel=9 if compression == 'gzip' else 0,
        complib=compression if compression != 'gzip' else 'zlib'
    )
    
    logger.info(f"✅ DataFrame HDF5 저장 완료")
    
    # 메타데이터를 그룹 속성으로 저장
    if hasattr(df, 'attrs'):
        try:
            import tables
            import json
            
            with tables.open_file(buffer, 'r+') as h5file:
                # 그룹 가져오기
                group = h5file.get_node(f'/{key}')
                
                # DataFrame.attrs를 JSON으로 직렬화하여 저장
                attrs_json = json.dumps(df.attrs, default=str)
                group._v_attrs.pandas_attrs = attrs_json
                
                logger.info(f"✅ HDF5 메타데이터 저장 완료 (그룹 속성)")
                
                # 저장된 메타데이터 확인
                if 'header_metadata' in df.attrs:
                    header_meta = df.attrs['header_metadata']
                    saved_count = {}
                    for key_name, values in header_meta.items():
                        if isinstance(values, list):
                            count = len([v for v in values if pd.notna(v) and v is not None])
                            saved_count[key_name] = count
                    logger.info(f"저장된 메타데이터: {saved_count}")
        
        except Exception as e:
            logger.error(f"❌ HDF5 메타데이터 저장 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"✅ HDF5 저장 완료: shape={df.shape}, key='{key}'")


def safe_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame을 st.dataframe()에 안전하게 표시할 수 있도록 변환
    모든 datetime을 string으로 변환하여 PyArrow 오류 방지
    """
    logger = logging.getLogger(__name__)
    
    # attrs 백업
    original_attrs = df.attrs.copy() if hasattr(df, 'attrs') else {}
    
    new_data = {}
    
    for col in df.columns:
        # datetime 타입 → string
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            new_data[col] = df[col].astype(str).replace('NaT', '').replace('None', '')
        
        # object 타입 처리
        elif df[col].dtype == 'object':
            non_null = df[col].dropna()
            if len(non_null) > 0:
                first_val = non_null.iloc[0]
                
                if isinstance(first_val, (pd.Timestamp, type(pd.NaT), datetime, date)):
                    new_data[col] = df[col].astype(str).replace('NaT', '').replace('None', '')
                else:
                    try:
                        new_data[col] = df[col].astype(str)
                    except:
                        new_data[col] = df[col].copy()
            else:
                new_data[col] = df[col].copy()
        else:
            new_data[col] = df[col].copy()
    
    df_safe = pd.DataFrame(new_data, index=df.index)
    
    # attrs 복원
    df_safe.attrs = original_attrs
    
    return df_safe


def load_from_hdf5_with_metadata(file_path, key='data'):
    """
    HDF5에서 DataFrame과 메타데이터를 함께 로드
    
    Args:
        file_path: HDF5 파일 경로
        key: HDF5 내 데이터셋 키
    
    Returns:
        DataFrame (attrs에 메타데이터 포함)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== HDF5 로드 시작: {file_path} ===")
    
    # DataFrame 로드
    df = pd.read_hdf(file_path, key=key)
    logger.info(f"DataFrame 로드 완료: shape={df.shape}")
    
    # 메타데이터 로드 (그룹 속성에서)
    try:
        import tables
        import json
        
        with tables.open_file(file_path, 'r') as h5file:
            group = h5file.get_node(f'/{key}')
            
            # pandas_attrs 속성 확인
            if hasattr(group._v_attrs, 'pandas_attrs'):
                attrs_json = group._v_attrs.pandas_attrs
                df.attrs = json.loads(attrs_json)
                
                logger.info(f"✅ HDF5 메타데이터 로드 완료")
                logger.info(f"df.attrs keys: {list(df.attrs.keys())}")
                
                if 'header_metadata' in df.attrs:
                    header_meta = df.attrs['header_metadata']
                    loaded_count = {}
                    for key_name, values in header_meta.items():
                        if isinstance(values, list):
                            count = len([v for v in values if v is not None and str(v) != 'None'])
                            loaded_count[key_name] = count
                    logger.info(f"로드된 메타데이터: {loaded_count}")
            else:
                logger.warning("⚠️ HDF5 파일에 pandas_attrs가 없습니다.")
    
    except Exception as e:
        logger.error(f"❌ HDF5 메타데이터 로드 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return df


def prepare_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame을 Parquet 저장 가능하도록 데이터 타입 정리
    Mixed object dtype 문제 해결
    """
    logger = logging.getLogger(__name__)
    
    # attrs 백업
    original_attrs = df.attrs.copy() if hasattr(df, 'attrs') else {}
    
    df_copy = df.copy()
    
    logger.info(f"Parquet 준비: {df_copy.shape}")
    
    for col in df_copy.columns:
        # boolean 타입은 그대로 유지
        if df_copy[col].dtype == 'bool':
            continue
        
        # 숫자 타입은 그대로 유지
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
        
        # datetime 타입은 그대로 유지
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            continue
        
        # object 타입 처리 (mixed 타입 포함)
        if df_copy[col].dtype == 'object':
            try:
                # 모든 값을 문자열로 강제 변환
                df_copy[col] = df_copy[col].astype(str)
                
                # 'nan', 'None', 'NaT' 문자열을 실제 NaN으로 변환
                df_copy[col] = df_copy[col].replace(['nan', 'None', 'NaT', ''], pd.NA)
                
                # 샘플 확인 후 boolean 변환 시도
                sample = df_copy[col].dropna().head(100)
                if len(sample) > 0:
                    unique_vals = set(sample.str.lower().unique())
                    
                    # boolean으로 변환 가능한지 확인
                    if unique_vals.issubset({'true', 'false'}):
                        df_copy[col] = df_copy[col].str.lower().map({
                            'true': True, 
                            'false': False
                        })
                    else:
                        # 문자열로 유지 (이미 변환됨)
                        pass
            except Exception as e:
                # 변환 실패 시 강제로 문자열 변환
                logger.warning(f"컬럼 '{col}' 변환 실패, 문자열로 강제 변환: {e}")
                df_copy[col] = df_copy[col].apply(lambda x: str(x) if pd.notna(x) else None)
    
    # attrs 복원
    df_copy.attrs = original_attrs
    
    return df_copy


def prepare_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame을 Streamlit 표시 가능하도록 타입 정리
    """
    # attrs 백업
    original_attrs = df.attrs.copy() if hasattr(df, 'attrs') else {}
    
    df_copy = df.copy(deep=True)
    
    for col in df_copy.columns:
        # datetime 타입 처리
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            try:
                df_copy[col] = pd.to_datetime(
                    df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S'),
                    format='%Y-%m-%d %H:%M:%S',
                    errors='coerce'
                )
            except:
                pass
        
        # object 타입 처리
        elif df_copy[col].dtype == 'object':
            try:
                non_null = df_copy[col].dropna()
                if len(non_null) == 0:
                    continue
                
                sample = non_null.head(100)
                has_timestamp = any(isinstance(x, (pd.Timestamp, type(pd.NaT))) for x in sample)
                
                if has_timestamp:
                    temp_col = pd.to_datetime(df_copy[col], errors='coerce')
                    df_copy[col] = pd.to_datetime(
                        temp_col.dt.strftime('%Y-%m-%d %H:%M:%S'),
                        format='%Y-%m-%d %H:%M:%S',
                        errors='coerce'
                    )
                    continue
                
                if 'timestamp' in str(col).lower() or 'datetime' in str(col).lower() or 'date' in str(col).lower():
                    try:
                        temp_col = pd.to_datetime(df_copy[col], errors='coerce')
                        if temp_col.notna().sum() > 0:
                            df_copy[col] = pd.to_datetime(
                                temp_col.dt.strftime('%Y-%m-%d %H:%M:%S'),
                                format='%Y-%m-%d %H:%M:%S',
                                errors='coerce'
                            )
                    except:
                        pass
            except:
                pass
    
    # attrs 복원
    df_copy.attrs = original_attrs
    
    return df_copy