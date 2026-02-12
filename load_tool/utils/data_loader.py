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
from .timestamp_processor import (
    process_timestamp, remove_timestamp_if_needed
)
from .data_processors import (
    apply_sampling, normalize_column_names, convert_types, post_process
)


class UnifiedDataLoader:
    """YAML 설정 기반 통합 Excel/CSV 데이터 로더"""
    
    def __init__(self, config: Dict, progress_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        self.setup_logging()
        self.metadata = {}
        self.removed_columns = []  # 제거된 컬럼 정보 저장
        
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
        # 제거된 컬럼 정보 초기화
        self.removed_columns = []

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
        # 각 시트마다 제거된 컬럼 정보 초기화
        self.removed_columns = []

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

        # 5. 중복 tag_name 컬럼 제외
        excluded_indices = header_info.get('excluded_indices', set())
        if excluded_indices:
            # 제외할 컬럼을 제거
            cols_to_keep = [i for i in range(len(df_data.columns)) if i not in excluded_indices]
            df_data = df_data.iloc[:, cols_to_keep]
            self.logger.info(f"중복 tag_name으로 {len(excluded_indices)}개 컬럼 제외됨")

        df_data.columns = header_info['columns']
        df_data = df_data.reset_index(drop=True)

        # 6. 베이스 태그명 중복 + 데이터 동일 컬럼 제거
        df_data, header_info = self._remove_duplicate_base_tags(df_data, header_info)

        # 7. 타임스탬프 처리
        df_data = process_timestamp(df_data, self.config, self.logger)

        # 8. 컬럼명 정규화
        df_data = normalize_column_names(df_data, self.config)

        # 9. 데이터 타입 변환
        df_data = convert_types(df_data, self.config, self.logger)

        # 10. 샘플링 (타임스탬프 제거 전에 수행)
        df_data = apply_sampling(df_data, self.config, self.logger)

        # 11. 타임스탬프 제거 (옵션)
        df_data = remove_timestamp_if_needed(df_data, self.config, self.logger)

        # 12. 후처리
        df_data = post_process(df_data, self.config, self.logger, self.removed_columns)

        # 13. 메타데이터 저장
        self.metadata[source_name] = {
            'total_rows': len(df_data),
            'total_columns': len(df_data.columns),
            'columns': df_data.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df_data.dtypes.items()},
            'header_metadata': header_info.get('metadata', {}),
            'removed_columns': self.removed_columns  # 제거된 컬럼 정보 추가
        }
        
        # DataFrame.attrs에도 메타데이터 저장
        df_data.attrs['source_name'] = source_name
        df_data.attrs['header_metadata'] = header_info.get('metadata', {})
        df_data.attrs['removed_columns'] = self.removed_columns  # 제거된 컬럼 정보 추가
        
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

        # tag_name 중복 체크 및 제외할 컬럼 인덱스 찾기
        excluded_indices = set()
        if 'tag_name' in header_data and header_data['tag_name']:
            tag_names = header_data['tag_name']
            tag_count = {}

            # 중복된 tag_name 찾기
            for i, tag in enumerate(tag_names):
                tag_str = str(tag) if pd.notna(tag) else f'_NA_{i}'
                if tag_str in tag_count:
                    tag_count[tag_str].append(i)
                else:
                    tag_count[tag_str] = [i]

            # 중복된 tag_name을 가진 컬럼 인덱스를 excluded_indices에 추가
            # 첫 번째는 남기고 나머지만 제외
            for tag_str, indices in tag_count.items():
                if len(indices) > 1:  # 중복된 경우
                    # 첫 번째는 남기고 나머지만 제외
                    duplicates_to_remove = indices[1:]
                    excluded_indices.update(duplicates_to_remove)
                    self.logger.warning(f"중복 tag_name '{tag_str}' 발견: {len(duplicates_to_remove)}개 컬럼 제외 (첫 번째는 유지)")

                    # 제거된 컬럼 정보 저장 (첫 번째 제외)
                    for idx in duplicates_to_remove:
                        removed_info = {
                            'tag_name': tag_str,
                            'description': header_data.get('description', [None] * len(tag_names))[idx] if 'description' in header_data else None,
                            'unit': header_data.get('unit', [None] * len(tag_names))[idx] if 'unit' in header_data else None,
                            'reason': '완전 중복 (동일한 tag_name)'
                        }
                        self.removed_columns.append(removed_info)

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

        # 제외할 컬럼 필터링
        if excluded_indices:
            columns = [col for i, col in enumerate(columns) if i not in excluded_indices]
            # header_data의 모든 메타데이터도 필터링
            for key in header_data.keys():
                if isinstance(header_data[key], list):
                    header_data[key] = [val for i, val in enumerate(header_data[key]) if i not in excluded_indices]

        # 남은 컬럼명 중복 처리 (description이 중복된 경우 대비)
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
            self.logger.warning(f"중복 컬럼명 발견: {len(duplicates_found)}개 (번호 부여)")

        return {
            'columns': columns,
            'metadata': header_data,
            'excluded_indices': excluded_indices
        }

    def _remove_duplicate_base_tags(self, df_data: pd.DataFrame, header_info: Dict) -> tuple:
        """
        베이스 태그명이 같고 상위 row 데이터가 동일한 컬럼 제거
        예: TEMP.1, TEMP.2가 같은 데이터를 가지면 중복으로 제거
        """
        # tag_name 메타데이터가 없으면 스킵
        if 'metadata' not in header_info or 'tag_name' not in header_info['metadata']:
            return df_data, header_info

        tag_names = header_info['metadata']['tag_name']

        self.logger.info(f"=== 중복 태그 검사 시작 ===")
        self.logger.info(f"tag_names 개수: {len(tag_names)}, df_data 컬럼 개수: {len(df_data.columns)}")

        # 설정에서 비교할 row 수 가져오기 (기본값: 10)
        compare_rows = self.config.get('duplicate_detection', {}).get('compare_rows', 10)
        compare_rows = min(compare_rows, len(df_data))  # 데이터 행 수보다 클 수 없음
        self.logger.info(f"비교할 row 수: {compare_rows}")

        if compare_rows <= 0:
            return df_data, header_info

        # 베이스 태그명 추출 (예: "TEMP.1" -> "TEMP")
        base_tags = {}
        for i, tag in enumerate(tag_names):
            if pd.notna(tag):
                tag_str = str(tag)
                # .숫자 패턴 제거
                base_tag = re.sub(r'\.\d+$', '', tag_str)
                if base_tag not in base_tags:
                    base_tags[base_tag] = []
                base_tags[base_tag].append(i)

        # 중복 가능성이 있는 베이스 태그 출력
        duplicate_candidates = {k: v for k, v in base_tags.items() if len(v) > 1}
        if duplicate_candidates:
            self.logger.info(f"중복 가능성 있는 베이스 태그: {list(duplicate_candidates.keys())}")
            for base_tag, indices in duplicate_candidates.items():
                tag_list = [tag_names[i] for i in indices]
                self.logger.info(f"  {base_tag}: {tag_list}")
        else:
            self.logger.info("중복 가능성 있는 베이스 태그 없음")

        # 중복 제거할 컬럼 인덱스
        cols_to_remove = []

        # 각 베이스 태그별로 처리
        for base_tag, indices in base_tags.items():
            if len(indices) <= 1:
                # 같은 베이스 태그를 가진 컬럼이 1개 이하면 스킵
                continue

            # 컬럼명 가져오기
            col_names = [df_data.columns[i] for i in indices]

            # 상위 N개 row 데이터 비교
            groups = {}  # 데이터가 같은 컬럼들을 그룹화

            for idx, col_name in zip(indices, col_names):
                # 상위 N개 row 데이터 추출
                data_sample = df_data[col_name].head(compare_rows).tolist()
                # 리스트를 튜플로 변환 (해시 가능하게)
                data_key = tuple(data_sample)

                if data_key not in groups:
                    groups[data_key] = []
                groups[data_key].append((idx, col_name, tag_names[idx]))

            # 각 그룹에서 2개 이상이면 중복 (첫 번째만 남기고 제거)
            for data_key, group_items in groups.items():
                if len(group_items) > 1:
                    # 첫 번째는 남기고 나머지 제거
                    for idx, col_name, tag_name in group_items[1:]:
                        cols_to_remove.append(col_name)
                        self.logger.warning(
                            f"베이스 태그 '{base_tag}' 중복: '{tag_name}' (데이터 동일) -> 제거"
                        )

                        # 제거된 컬럼 정보 저장
                        removed_info = {
                            'tag_name': tag_name,
                            'description': col_name,  # 이미 description이 컬럼명으로 사용됨
                            'unit': header_info['metadata'].get('unit', [None] * len(tag_names))[idx] if 'unit' in header_info['metadata'] else None,
                            'reason': f'베이스 태그 중복 ({base_tag}) + 데이터 동일'
                        }
                        self.removed_columns.append(removed_info)

        # 컬럼 제거
        if cols_to_remove:
            # 제거할 컬럼의 원래 인덱스 찾기
            original_cols = df_data.columns.tolist()
            cols_to_remove_indices = [i for i, col in enumerate(original_cols) if col in cols_to_remove]

            df_data = df_data.drop(columns=cols_to_remove)
            self.logger.info(f"베이스 태그 중복으로 {len(cols_to_remove)}개 컬럼 제거됨")

            # header_info의 columns도 업데이트
            header_info['columns'] = df_data.columns.tolist()

            # metadata도 업데이트 (제거된 인덱스를 반영)
            for key in header_info['metadata'].keys():
                if isinstance(header_info['metadata'][key], list):
                    # 제거할 인덱스를 제외한 나머지만 유지
                    old_list = header_info['metadata'][key]
                    new_list = [val for i, val in enumerate(old_list) if i not in cols_to_remove_indices]
                    header_info['metadata'][key] = new_list

        return df_data, header_info
