"""
YAML 설정 유틸리티
"""
import yaml
from typing import Dict
from datetime import datetime


def get_default_config() -> Dict:
    """기본 설정 반환"""
    return {
        'file_info': {'description': '', 'file_type': 'excel'},
        'csv_options': {
            'encoding': 'utf-8', 
            'delimiter': ',', 
            'quotechar': '"', 
            'skip_blank_lines': True, 
            'comment': '#'
        },
        'sheets': {'mode': 'single', 'names': [], 'indices': [], 'exclude': []},
        'header': {'skip_rows': 0, 'header_rows': {}, 'data_start_row': 1},
        'timestamp': {
            'combine_time_columns': False, 
            'keywords': ['timestamp', 'datetime', 'date'],
            'use_first_column': False, 
            'target_name': 'timestamp', 
            'drop_time_columns': True, 
            'strict': False,
            'exclude_from_output': False
        },
        'sampling': {
            'enabled': False,
            'method': 'every_n',
            'interval': 5
        },
        'column_names': {'replace_spaces': '_', 'keep_special_chars': True, 'lowercase': False},
        'data_types': {'auto_infer': True, 'sample_rows': 100, 'value_mapping': {}, 'null_values': []},
        'post_processing': {
            'remove_empty_rows': True, 
            'remove_high_null_columns': None, 
            'remove_duplicates': False
        },
        'output': {'format': 'hdf5', 'compression': 'gzip', 'save_metadata': True},
        'error_handling': {
            'on_parse_error': 'skip_row', 
            'save_log': True, 
            'log_path': 'logs/parser.log', 
            'verbose': False
        }
    }


def dict_to_yaml_string(data: Dict) -> str:
    """딕셔너리를 YAML 문자열로 변환"""
    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


def yaml_string_to_dict(yaml_str: str) -> Dict:
    """YAML 문자열을 딕셔너리로 변환"""
    return yaml.safe_load(yaml_str)


def load_yaml_file(file_obj) -> Dict:
    """YAML 파일 로드"""
    config_data = yaml.safe_load(file_obj)
    
    # 숫자형 값들이 제대로 로딩되도록 보장
    if 'header' in config_data:
        if 'skip_rows' in config_data['header']:
            config_data['header']['skip_rows'] = int(config_data['header']['skip_rows'])
        if 'data_start_row' in config_data['header']:
            config_data['header']['data_start_row'] = int(config_data['header']['data_start_row'])
        if 'header_rows' in config_data['header'] and config_data['header']['header_rows']:
            header_rows = {}
            for key, val in config_data['header']['header_rows'].items():
                if val is not None and val != '':
                    try:
                        header_rows[key] = int(val)
                    except:
                        header_rows[key] = 0
            config_data['header']['header_rows'] = header_rows
    
    return config_data


def save_yaml_file(config: Dict) -> str:
    """YAML 파일명 생성"""
    return f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"