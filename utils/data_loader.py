"""
데이터 로더 모듈 (v2)
YAML 설정 기반 Excel/CSV 파일 로더
- 타임스탬프 제외 옵션
- 샘플링 기능
"""
import pandas as pd
import logging
import re
from typing import Dict, Optional, Union
from datetime import datetime


class UnifiedDataLoader:
    """YAML 설정 기반 통합 Excel/CSV 데이터 로더"""
    
    def __init__(self, config: Dict, progress_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        self.setup_logging()
        self.metadata = {}
        
    def setup_logging(self):
        """로깅 설정"""
        error_config = self.config.get('error_handling', {})
        log_level = logging.DEBUG if error_config.get('verbose', False) else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load(self, file_obj, file_type: str, sheet_name: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """파일 로드 (Excel 또는 CSV)"""
        if file_type == 'excel':
            return self._load_excel(file_obj, sheet_name)
        else:
            return self._load_csv(file_obj)
    
    def _load_csv(self, file_obj) -> pd.DataFrame:
        """CSV 파일 로드"""
        csv_options = self.config.get('csv_options', {})
        read_options = {
            'encoding': csv_options.get('encoding', 'utf-8'),
            'delimiter': csv_options.get('delimiter', ','),
            'quotechar': csv_options.get('quotechar', '"'),
            'skip_blank_lines': csv_options.get('skip_blank_lines', True),
            'header': None,
            'low_memory': False
        }
        
        if 'comment' in csv_options and csv_options['comment']:
            read_options['comment'] = csv_options['comment']
        
        df_raw = pd.read_csv(file_obj, **read_options)
        return self._process(df_raw, 'csv_data')
    
    def _load_excel(self, file_obj, sheet_name: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Excel 파일 로드"""
        sheets_config = self.config.get('sheets', {})
        mode = sheets_config.get('mode', 'single')
        
        excel_file = pd.ExcelFile(file_obj)
        all_sheets = excel_file.sheet_names
        
        # 처리할 시트 결정
        if sheet_name:
            target_sheets = [sheet_name]
        elif mode == 'single':
            target_sheets = [all_sheets[0]]
        elif mode == 'all':
            exclude = sheets_config.get('exclude', [])
            target_sheets = [s for s in all_sheets if s not in exclude]
        elif mode == 'specific':
            if sheets_config.get('names'):
                target_sheets = sheets_config['names']
            elif sheets_config.get('indices'):
                indices = sheets_config['indices']
                target_sheets = [all_sheets[i] for i in indices if i < len(all_sheets)]
            else:
                target_sheets = [all_sheets[0]]
        else:
            target_sheets = [all_sheets[0]]
        
        # 단일 시트 처리
        if len(target_sheets) == 1:
            return self._load_single_sheet(file_obj, target_sheets[0])
        
        # 다중 시트 처리
        results = {}
        total_sheets = len(target_sheets)
        
        for idx, sheet in enumerate(target_sheets, 1):
            try:
                if self.progress_callback:
                    self.progress_callback(sheet, idx, total_sheets)
                
                df = self._load_single_sheet(file_obj, sheet)
                if df is not None and not df.empty:
                    results[sheet] = df
            except Exception as e:
                self.logger.error(f"Error processing sheet '{sheet}': {str(e)}")
                if self.config.get('error_handling', {}).get('on_parse_error') == 'raise':
                    raise
        
        return results
    
    def _load_single_sheet(self, file_obj, sheet_name: str) -> pd.DataFrame:
        """단일 시트 로드 및 처리"""
        df_raw = pd.read_excel(file_obj, sheet_name=sheet_name, header=None)
        
        if df_raw.empty:
            self.logger.warning(f"Sheet '{sheet_name}' is empty")
            return None
        
        return self._process(df_raw, sheet_name)
    
    def _process(self, df_raw: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """데이터 처리 파이프라인"""
        header_config = self.config.get('header', {})
        
        # 1. skip_rows 적용
        skip_rows = header_config.get('skip_rows', 0)
        if skip_rows > 0:
            df_raw = df_raw.iloc[skip_rows:].reset_index(drop=True)
        
        # 2. 헤더 정보 추출
        header_rows_config = header_config.get('header_rows', {})
        header_info = self._extract_headers(df_raw, header_rows_config)
        
        # 3. 데이터 시작 행
        data_start_row = header_config.get('data_start_row', 1)
        adjusted_start = max(0, data_start_row - 1)
        
        # 4. 데이터 추출
        df_data = df_raw.iloc[adjusted_start:].copy()
        df_data.columns = header_info['columns']
        df_data = df_data.reset_index(drop=True)
        
        # 5. 타임스탬프 처리
        df_data = self._process_timestamp(df_data)
        
        # 6. 컬럼명 정규화
        df_data = self._normalize_column_names(df_data)
        
        # 7. 데이터 타입 변환
        df_data = self._convert_types(df_data)
        
        # 8. 샘플링 (타임스탬프 제거 전에 수행)
        df_data = self._apply_sampling(df_data)
        
        # 9. 타임스탬프 제거 (옵션)
        df_data = self._remove_timestamp_if_needed(df_data)
        
        # 10. 후처리
        df_data = self._post_process(df_data)
        
        # 11. 메타데이터 저장
        self.metadata[source_name] = {
            'total_rows': len(df_data),
            'total_columns': len(df_data.columns),
            'columns': df_data.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df_data.dtypes.items()},
            'header_metadata': header_info.get('metadata', {})
        }
        
        # DataFrame.attrs에도 메타데이터 저장
        df_data.attrs['source_name'] = source_name
        df_data.attrs['header_metadata'] = header_info.get('metadata', {})
        
        return df_data
    
    def _extract_headers(self, df_raw: pd.DataFrame, header_rows_config: Dict) -> Dict:
        """헤더 정보 추출"""
        header_data = {}
        
        for header_type, row_num in header_rows_config.items():
            if row_num is not None:
                idx = row_num - 1
                if 0 <= idx < len(df_raw):
                    header_data[header_type] = df_raw.iloc[idx].tolist()
                else:
                    self.logger.warning(f"Row {row_num} is out of range")
        
        # 컬럼명 결정
        columns = []
        if 'description' in header_data and header_data['description']:
            columns = [str(x) if pd.notna(x) else f'Col_{i}' for i, x in enumerate(header_data['description'])]
        elif 'tag_name' in header_data and header_data['tag_name']:
            columns = [str(x) if pd.notna(x) else f'Col_{i}' for i, x in enumerate(header_data['tag_name'])]
        elif 'id' in header_data and header_data['id']:
            columns = [str(x) if pd.notna(x) else f'Col_{i}' for i, x in enumerate(header_data['id'])]
        else:
            columns = [f'Col_{i}' for i in range(len(df_raw.columns))]
        
        # 중복 컬럼명 처리
        seen = {}
        duplicates_found = []
        for i, col in enumerate(columns):
            if col in seen:
                seen[col] += 1
                duplicates_found.append(col)
                columns[i] = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
        
        if duplicates_found:
            self.logger.warning(f"중복 컬럼명 발견: {len(duplicates_found)}개")
        
        return {
            'columns': columns,
            'metadata': header_data
        }
    
    def _process_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """타임스탬프 처리"""
        ts_config = self.config.get('timestamp', {})
        
        if not ts_config:
            return df
        
        target_name = ts_config.get('target_name', 'timestamp')
        
        # 옵션 1: 분리된 시간 컬럼 합치기
        if ts_config.get('combine_time_columns', False):
            df = self._combine_time_columns(df, ts_config)
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
            df = self._convert_timestamp(df, ts_col, ts_config, target_name)
        elif ts_config.get('strict', False):
            raise ValueError("Timestamp column not found and strict mode is enabled")
        
        return df
    
    def _combine_time_columns(self, df: pd.DataFrame, ts_config: Dict) -> pd.DataFrame:
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
            self.logger.warning(f"Failed to combine time columns: {str(e)}")
        
        return df
    
    def _convert_timestamp(self, df: pd.DataFrame, ts_col: str, ts_config: Dict, target_name: str) -> pd.DataFrame:
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
                self.logger.error(f"object 변환 실패: {e}")
        
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
            self.logger.warning(f"Failed to convert timestamp column: {ts_col}")
        
        return df
    
    def _apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """샘플링 적용 (새로운 기능)"""
        sampling_config = self.config.get('sampling', {})
        
        if not sampling_config.get('enabled', False):
            return df
        
        interval = sampling_config.get('interval', 5)
        method = sampling_config.get('method', 'every_n')
        
        if interval <= 1:
            self.logger.warning("Sampling interval must be > 1, skipping sampling")
            return df
        
        try:
            # timestamp 컬럼 찾기
            ts_col = None
            ts_config = self.config.get('timestamp', {})
            target_name = ts_config.get('target_name', 'timestamp')
            
            if target_name in df.columns:
                ts_col = target_name
            
            self.logger.info(f"Applying sampling: method={method}, interval={interval}")
            
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
                self.logger.warning(f"Unknown sampling method: {method}")
                return df
            
            original_rows = len(df)
            sampled_rows = len(df_sampled)
            reduction = 100 * (1 - sampled_rows / original_rows)
            
            self.logger.info(f"Sampling complete: {original_rows} → {sampled_rows} rows ({reduction:.1f}% reduction)")
            
            return df_sampled
            
        except Exception as e:
            self.logger.error(f"Sampling failed: {str(e)}")
            return df
    
    def _remove_timestamp_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        """타임스탬프 제거 (옵션) (새로운 기능)"""
        ts_config = self.config.get('timestamp', {})
        
        if not ts_config.get('exclude_from_output', False):
            return df
        
        # timestamp 컬럼 찾아서 제거
        target_name = ts_config.get('target_name', 'timestamp')
        
        if target_name in df.columns:
            self.logger.info(f"Removing timestamp column '{target_name}' from output")
            df = df.drop(columns=[target_name])
            df = df.reset_index(drop=True)  # 🔹 이 줄 추가
        
        return df
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """컬럼명 정규화"""
        col_config = self.config.get('column_names', {})
        
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
    
    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 변환"""
        type_config = self.config.get('data_types', {})
        
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
                    # self.logger.info(f"컬럼 '{col}': bool -> int 변환")
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
                            # self.logger.info(f"컬럼 '{col}': Boolean 문자열 -> int 변환")
            except Exception as e:
                self.logger.warning(f"컬럼 '{col}' Boolean 변환 실패: {e}")
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
            df = self._auto_infer_types(df, sample_rows)
        
        return df
    
    def _auto_infer_types(self, df: pd.DataFrame, sample_rows: int) -> pd.DataFrame:
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
    
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """후처리"""
        post_config = self.config.get('post_processing', {})
        
        if not post_config:
            return df
        
        if post_config.get('remove_empty_rows', False):
            df = df.dropna(how='all')
        
        threshold = post_config.get('remove_high_null_columns')
        if threshold:
            null_pct = (df.isnull().sum() / len(df)) * 100
            cols_to_drop = null_pct[null_pct > threshold].index.tolist()
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
        
        if post_config.get('remove_duplicates', False):
            df = df.drop_duplicates()
        
        return df