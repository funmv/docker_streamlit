"""
UI 컴포넌트 모듈
Streamlit UI 탭 렌더링 함수들
"""
import streamlit as st
import pandas as pd
import io
import re
import yaml
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
from .data_loader import UnifiedDataLoader
from .data_utils import (
    dict_to_yaml_string, yaml_string_to_dict,
    extract_date_range_from_df, save_to_parquet_with_metadata,
    save_to_hdf5_with_metadata, load_from_hdf5_with_metadata,
    safe_display_df, prepare_df_for_parquet, prepare_df_for_display
)
from .visualization_plots import (
    render_timeseries_plot, render_scatter_plot, render_histogram,
    render_boxplot, render_correlation_heatmap
)


def render_config_tab():
    st.header("📋 YAML 설정")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 기본값으로 초기화", use_container_width=True):
            st.session_state.config = {
                'file_info': {'description': '', 'file_type': 'excel'},
                'csv_options': {'encoding': 'utf-8', 'delimiter': ',', 'quotechar': '"', 
                               'skip_blank_lines': True, 'comment': '#'},
                'sheets': {'mode': 'single', 'names': [], 'indices': [], 'exclude': []},
                'header': {'skip_rows': 0, 'header_rows': {}, 'data_start_row': 1},
                'timestamp': {
                    'combine_time_columns': False, 
                    'keywords': ['timestamp', 'datetime', 'date'],
                    'use_first_column': False, 
                    'target_name': 'timestamp', 
                    'drop_time_columns': True, 
                    'strict': False,
                    'exclude_from_output': False  # 🆕
                },
                'sampling': {  # 🆕
                    'enabled': False,
                    'method': 'every_n',
                    'interval': 5
                },                
                'column_names': {'replace_spaces': '_', 'keep_special_chars': True, 'lowercase': False},
                'data_types': {'auto_infer': True, 'sample_rows': 100, 'value_mapping': {}, 'null_values': []},
                'post_processing': {'remove_empty_rows': True, 'remove_high_null_columns': None, 
                                   'remove_duplicates': False},
                'output': {'format': 'parquet', 'compression': 'snappy', 'save_metadata': True},
                'error_handling': {'on_parse_error': 'skip_row', 'save_log': True, 
                                  'log_path': 'logs/parser.log', 'verbose': False}
            }
            st.session_state['yaml_loaded'] = False
            st.rerun()
    
    with col2:
        uploaded_yaml = st.file_uploader("📥 YAML 파일 불러오기", type=['yaml', 'yml'], key='yaml_upload')
        if uploaded_yaml:
            # YAML이 적용되었는지 확인
            if st.session_state.get('yaml_loaded', False):
                st.success("✅ YAML 설정이 적용되었습니다. 아래에서 확인하세요.")
            
            col2a, col2b = st.columns(2)
            with col2a:
                if st.button("YAML 적용", key='apply_yaml', type='primary'):
                    try:
                        # 파일 포인터를 처음으로 되돌리기
                        uploaded_yaml.seek(0)
                        config_data = yaml.safe_load(uploaded_yaml)
                        
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
                        
                        # session_state에 저장
                        st.session_state.config = config_data
                        st.session_state['yaml_loaded'] = True
                        
                        # 적용 완료 메시지와 함께 새로고침
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ YAML 파일 로드 실패: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            with col2b:
                # 현재 헤더 설정 미리보기
                if st.button("🔍 미리보기", key='preview_yaml'):
                    try:
                        uploaded_yaml.seek(0)
                        config_data = yaml.safe_load(uploaded_yaml)
                        if 'header' in config_data:
                            st.info(f"""
**YAML 파일 내용:**
- skip_rows: {config_data['header'].get('skip_rows', 0)}
- data_start_row: {config_data['header'].get('data_start_row', 1)}
- header_rows: {config_data['header'].get('header_rows', {})}
                            """)
                    except Exception as e:
                        st.error(f"미리보기 실패: {str(e)}")
    
    with col3:
        yaml_str = dict_to_yaml_string(st.session_state.config)
        st.download_button(
            label="💾 YAML 파일 저장",
            data=yaml_str,
            file_name=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
            mime="text/yaml",
            use_container_width=True
        )
    
    st.divider()
    
    # 설정 섹션들
    with st.expander("📁 파일 정보", expanded=True):
        st.session_state.config['file_info']['description'] = st.text_input(
            "설명", 
            value=st.session_state.config['file_info'].get('description', '')
        )
        st.session_state.config['file_info']['file_type'] = st.selectbox(
            "파일 타입",
            options=['excel', 'csv'],
            index=0 if st.session_state.config['file_info'].get('file_type', 'excel') == 'excel' else 1
        )
    
    # CSV 옵션 (file_type이 csv일 때만 표시)
    if st.session_state.config['file_info']['file_type'] == 'csv':
        with st.expander("📄 CSV 옵션"):
            csv_opt = st.session_state.config.get('csv_options', {})
            
            col1, col2 = st.columns(2)
            with col1:
                csv_opt['encoding'] = st.selectbox(
                    "인코딩",
                    options=['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1'],
                    index=['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1'].index(
                        csv_opt.get('encoding', 'utf-8')
                    )
                )
                csv_opt['delimiter'] = st.text_input("구분자", value=csv_opt.get('delimiter', ','))
            
            with col2:
                csv_opt['quotechar'] = st.text_input("따옴표 문자", value=csv_opt.get('quotechar', '"'))
                csv_opt['comment'] = st.text_input("주석 문자 (선택)", value=csv_opt.get('comment', '#'))
            
            csv_opt['skip_blank_lines'] = st.checkbox("빈 줄 건너뛰기", value=csv_opt.get('skip_blank_lines', True))
            
            st.session_state.config['csv_options'] = csv_opt
    
    # 시트 설정 (Excel 전용)
    if st.session_state.config['file_info']['file_type'] == 'excel':
        with st.expander("📊 시트 처리 설정"):
            sheets = st.session_state.config.get('sheets', {})
            
            sheets['mode'] = st.radio(
                "처리 모드",
                options=['single', 'all', 'specific'],
                index=['single', 'all', 'specific'].index(sheets.get('mode', 'single')),
                horizontal=True
            )
            
            if sheets['mode'] == 'specific':
                names_str = st.text_input("시트 이름들 (쉼표로 구분)", 
                                         value=','.join(sheets.get('names', [])))
                sheets['names'] = [n.strip() for n in names_str.split(',') if n.strip()]
                
                indices_str = st.text_input("시트 인덱스들 (쉼표로 구분)", 
                                           value=','.join(map(str, sheets.get('indices', []))))
                try:
                    sheets['indices'] = [int(i.strip()) for i in indices_str.split(',') if i.strip()]
                except:
                    sheets['indices'] = []
            
            if sheets['mode'] == 'all':
                exclude_str = st.text_input("제외할 시트 (쉼표로 구분)", 
                                          value=','.join(sheets.get('exclude', [])))
                sheets['exclude'] = [e.strip() for e in exclude_str.split(',') if e.strip()]
            
            st.session_state.config['sheets'] = sheets
    
    # 헤더 구조
    with st.expander("📑 헤더 구조", expanded=True):
        header = st.session_state.config.get('header', {})
        
        # skip_rows 처리
        current_skip = 0
        if 'skip_rows' in header and header['skip_rows'] is not None:
            try:
                current_skip = int(header['skip_rows'])
            except:
                current_skip = 0
        
        header['skip_rows'] = st.number_input(
            "상단에서 건너뛸 행 수 (1-based)",
            min_value=0, value=current_skip, step=1
        )
        
        st.markdown("**헤더 행 번호 (skip_rows 적용 후 기준, 1-based)**")
        st.caption("0을 입력하면 해당 헤더를 사용하지 않습니다.")
        col1, col2, col3 = st.columns(3)
        
        # header_rows 딕셔너리 가져오기 (없으면 빈 딕셔너리)
        if 'header_rows' not in header:
            header['header_rows'] = {}
        header_rows = header['header_rows']
        
        with col1:
            # 기본값 계산
            current_desc = 0
            if 'description' in header_rows and header_rows['description'] is not None:
                try:
                    current_desc = int(header_rows['description'])
                except:
                    current_desc = 0
            
            desc_row = st.number_input("설명(Description) 행", min_value=0, 
                                       value=current_desc, 
                                       step=1)
            if desc_row > 0:
                header_rows['description'] = int(desc_row)
            elif 'description' in header_rows:
                del header_rows['description']
        
        with col2:
            # 기본값 계산
            current_unit = 0
            if 'unit' in header_rows and header_rows['unit'] is not None:
                try:
                    current_unit = int(header_rows['unit'])
                except:
                    current_unit = 0
            
            unit_row = st.number_input("단위(Unit) 행", min_value=0, 
                                       value=current_unit,
                                       step=1)
            if unit_row > 0:
                header_rows['unit'] = int(unit_row)
            elif 'unit' in header_rows:
                del header_rows['unit']
        
        with col3:
            # 기본값 계산
            current_tag = 0
            if 'tag_name' in header_rows and header_rows['tag_name'] is not None:
                try:
                    current_tag = int(header_rows['tag_name'])
                except:
                    current_tag = 0
            
            tag_row = st.number_input("태그명(Tag) 행", min_value=0, 
                                      value=current_tag,
                                      step=1)
            if tag_row > 0:
                header_rows['tag_name'] = int(tag_row)
            elif 'tag_name' in header_rows:
                del header_rows['tag_name']
        
        header['header_rows'] = header_rows
        
        # data_start_row 처리
        current_data_start = 1
        if 'data_start_row' in header and header['data_start_row'] is not None:
            try:
                current_data_start = int(header['data_start_row'])
            except:
                current_data_start = 1
        
        header['data_start_row'] = st.number_input(
            "데이터 시작 행 (skip_rows 적용 후 기준, 1-based)",
            min_value=1, value=current_data_start, step=1
        )
        
        st.session_state.config['header'] = header
    
    # 타임스탬프 처리
    with st.expander("🕐 타임스탬프 처리"):
        ts = st.session_state.config.get('timestamp', {})
        
        ts['combine_time_columns'] = st.checkbox(
            "분리된 시간 컬럼 합치기 (year, month, day 등)",
            value=ts.get('combine_time_columns', False)
        )
        
        if ts['combine_time_columns']:
            st.markdown("**시간 컬럼 설정**")
            time_cols_str = st.text_input(
                "찾을 시간 컬럼 (쉼표로 구분)",
                value=','.join(ts.get('time_columns', ['year', 'month', 'day', 'hour', 'minute', 'second']))
            )
            ts['time_columns'] = [c.strip() for c in time_cols_str.split(',') if c.strip()]
            
            col1, col2, col3 = st.columns(3)
            defaults = ts.get('defaults', {})
            with col1:
                defaults['year'] = st.number_input("기본 연도", value=defaults.get('year', 2025))
                defaults['month'] = st.number_input("기본 월", min_value=1, max_value=12, 
                                                    value=defaults.get('month', 1))
            with col2:
                defaults['day'] = st.number_input("기본 일", min_value=1, max_value=31, 
                                                  value=defaults.get('day', 1))
                defaults['hour'] = st.number_input("기본 시", min_value=0, max_value=23, 
                                                   value=defaults.get('hour', 0))
            with col3:
                defaults['minute'] = st.number_input("기본 분", min_value=0, max_value=59, 
                                                     value=defaults.get('minute', 0))
                defaults['second'] = st.number_input("기본 초", min_value=0, max_value=59, 
                                                     value=defaults.get('second', 0))
            ts['defaults'] = defaults
            
            ts['base_year'] = st.number_input("2자리 연도 변환 기준년도", 
                                             value=ts.get('base_year', 2000))
        else:
            keywords_str = st.text_input(
                "타임스탬프 키워드 (쉼표로 구분)",
                value=','.join(ts.get('keywords', ['timestamp', 'datetime', 'date']))
            )
            ts['keywords'] = [k.strip() for k in keywords_str.split(',') if k.strip()]
            
            ts['use_first_column'] = st.checkbox(
                "첫 번째 컬럼을 타임스탬프로 사용",
                value=ts.get('use_first_column', False)
            )
        
        ts['target_name'] = st.text_input("생성할 타임스탬프 컬럼명", 
                                         value=ts.get('target_name', 'timestamp'))
        ts['drop_time_columns'] = st.checkbox("원본 시간 컬럼 제거", 
                                               value=ts.get('drop_time_columns', True))
        ts['strict'] = st.checkbox("엄격 모드 (타임스탬프 없으면 에러)", 
                                   value=ts.get('strict', False))
        
        
        # 🆕 새로운 옵션: 타임스탬프 제외
        st.divider()
        st.markdown("#### 🆕 출력 옵션")
        st.caption("저장할 파일에서 타임스탬프 컬럼을 제외할지 선택합니다.")
        
        ts['exclude_from_output'] = st.checkbox(
            "⚠️ 저장 시 타임스탬프 제외 (특징 컬럼만 저장)",
            value=ts.get('exclude_from_output', False),
            help="체크하면 타임스탬프 없이 특징 데이터만 저장됩니다. 시간 정보가 불필요하거나 연속적이지 않은 경우 사용하세요."
        )
        
        if ts['exclude_from_output']:
            st.warning("⚠️ 저장 시 타임스탬프가 제외됩니다. 시각화는 가능하지만 시계열 분석이 제한될 수 있습니다.")
        
        st.session_state.config['timestamp'] = ts
    
    # 🆕 샘플링 설정 (새로운 섹션)
    with st.expander("🎯 샘플링 설정 (신규)"):
        sampling = st.session_state.config.get('sampling', {})
        
        st.markdown("#### 데이터 샘플링")
        st.caption("데이터가 너무 촘촘한 경우 간격을 두고 샘플링하거나 집계할 수 있습니다.")
        
        sampling['enabled'] = st.checkbox(
            "샘플링 활성화",
            value=sampling.get('enabled', False),
            help="데이터 간격을 줄여서 저장합니다."
        )
        
        if sampling['enabled']:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                sampling['interval'] = st.number_input(
                    "샘플링 간격",
                    min_value=2,
                    max_value=1000,
                    value=sampling.get('interval', 5),
                    step=1,
                    help="N개 데이터마다 1개를 선택합니다. 예: 5 → 5개당 1개"
                )
            
            with col2:
                sampling['method'] = st.selectbox(
                    "샘플링 방법",
                    options=['every_n', 'mean', 'median', 'first', 'last'],
                    index=['every_n', 'mean', 'median', 'first', 'last'].index(
                        sampling.get('method', 'every_n')
                    ),
                    help="""
• every_n: N개마다 1개 선택 (단순 샘플링)
• mean: N개씩 그룹화하여 평균값 저장
• median: N개씩 그룹화하여 중앙값 저장
• first: N개씩 그룹화하여 첫 번째 값 저장
• last: N개씩 그룹화하여 마지막 값 저장
                    """
                )
            
            # 예시 표시
            method_examples = {
                'every_n': "예) 1,2,3,4,5,6,7,8,9,10 → 1,6 (5개마다 1개)",
                'mean': "예) [1,2,3,4,5],[6,7,8,9,10] → 3.0, 8.0 (5개씩 평균)",
                'median': "예) [1,2,3,4,5],[6,7,8,9,10] → 3, 8 (5개씩 중앙값)",
                'first': "예) [1,2,3,4,5],[6,7,8,9,10] → 1, 6 (5개씩 첫값)",
                'last': "예) [1,2,3,4,5],[6,7,8,9,10] → 5, 10 (5개씩 끝값)"
            }
            
            st.info(f"**{sampling['method']}** - {method_examples[sampling['method']]}")
            
            # 예상 축소율 계산
            estimated_reduction = (1 - 1/sampling['interval']) * 100
            st.success(f"✅ 예상 데이터 축소율: 약 {estimated_reduction:.1f}%")
        
        st.session_state.config['sampling'] = sampling
    
    # 컬럼명 정규화
    with st.expander("🔤 컬럼명 정규화"):
        col_names = st.session_state.config.get('column_names', {})
        
        col1, col2 = st.columns(2)
        with col1:
            replace_space = st.text_input("공백 치환 문자", 
                                         value=col_names.get('replace_spaces', '_'))
            col_names['replace_spaces'] = replace_space if replace_space else None
        
        with col2:
            col_names['keep_special_chars'] = st.checkbox("특수문자 유지", 
                                                          value=col_names.get('keep_special_chars', True))
            col_names['lowercase'] = st.checkbox("소문자 변환", 
                                                 value=col_names.get('lowercase', False))
        
        st.session_state.config['column_names'] = col_names
    
    # 데이터 타입
    with st.expander("🔢 데이터 타입 변환"):
        dtypes = st.session_state.config.get('data_types', {})
        
        col1, col2 = st.columns(2)
        with col1:
            dtypes['auto_infer'] = st.checkbox("자동 타입 추론", 
                                               value=dtypes.get('auto_infer', True))
            if dtypes['auto_infer']:
                dtypes['sample_rows'] = st.number_input("샘플 행 수", min_value=10, 
                                                       value=dtypes.get('sample_rows', 100))
        
        with col2:
            st.markdown("**값 매핑 (문자열 → 불린/숫자)**")
            mapping_str = st.text_area(
                "형식: KEY=VALUE (한 줄에 하나씩)",
                value='\n'.join([f"{k}={v}" for k, v in dtypes.get('value_mapping', {}).items()]),
                height=100
            )
            value_mapping = {}
            for line in mapping_str.split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip()
                    if v.lower() == 'true':
                        value_mapping[k] = True
                    elif v.lower() == 'false':
                        value_mapping[k] = False
                    else:
                        try:
                            value_mapping[k] = float(v) if '.' in v else int(v)
                        except:
                            value_mapping[k] = v
            dtypes['value_mapping'] = value_mapping
        
        null_str = st.text_input(
            "NULL로 간주할 값들 (쉼표로 구분)",
            value=','.join(dtypes.get('null_values', []))
        )
        dtypes['null_values'] = [n.strip() for n in null_str.split(',') if n.strip()]
        
        st.session_state.config['data_types'] = dtypes
    
    # 후처리
    with st.expander("🔧 후처리"):
        post = st.session_state.config.get('post_processing', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            post['remove_empty_rows'] = st.checkbox("빈 행 제거", 
                                                    value=post.get('remove_empty_rows', True))
        with col2:
            enable_null_threshold = st.checkbox("NULL 비율 높은 컬럼 제거", 
                                               value=post.get('remove_high_null_columns') is not None)
            if enable_null_threshold:
                post['remove_high_null_columns'] = st.slider("NULL 비율 임계값 (%)", 0, 100, 
                                                             value=post.get('remove_high_null_columns', 90))
            else:
                post['remove_high_null_columns'] = None
        with col3:
            post['remove_duplicates'] = st.checkbox("중복 행 제거", 
                                                    value=post.get('remove_duplicates', False))
        
        st.session_state.config['post_processing'] = post
    
    # 출력 설정
    with st.expander("💾 출력 설정"):
        output = st.session_state.config.get('output', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            output['format'] = st.selectbox(
                "출력 형식",
                options=['parquet', 'csv', 'excel'],
                index=['parquet', 'csv', 'excel'].index(output.get('format', 'parquet'))
            )
        with col2:
            if output['format'] == 'parquet':
                output['compression'] = st.selectbox(
                    "압축 방식",
                    options=['snappy', 'gzip', 'brotli'],
                    index=['snappy', 'gzip', 'brotli'].index(output.get('compression', 'snappy'))
                )
        with col3:
            output['save_metadata'] = st.checkbox("메타데이터 저장", 
                                                 value=output.get('save_metadata', True))
        
        st.session_state.config['output'] = output
    
    # 에러 처리
    with st.expander("⚠️ 에러 처리"):
        error = st.session_state.config.get('error_handling', {})
        
        col1, col2 = st.columns(2)
        with col1:
            error['on_parse_error'] = st.selectbox(
                "파싱 에러 발생 시",
                options=['skip_row', 'raise', 'ignore'],
                index=['skip_row', 'raise', 'ignore'].index(error.get('on_parse_error', 'skip_row'))
            )
            error['save_log'] = st.checkbox("로그 저장", value=error.get('save_log', True))
        
        with col2:
            error['log_path'] = st.text_input("로그 경로", value=error.get('log_path', 'logs/parser.log'))
            error['verbose'] = st.checkbox("상세 로그 (DEBUG)", value=error.get('verbose', False))
        
        st.session_state.config['error_handling'] = error
    
    # 현재 설정 미리보기
    with st.expander("👁️ 현재 설정 미리보기"):
        st.code(dict_to_yaml_string(st.session_state.config), language='yaml')




# ============================================
# Tab 2: 데이터 로딩
# ============================================
def render_loading_tab():
    st.header("📂 데이터 로딩")
    
    # 파일 타입에 따라 다른 업로더 표시
    file_type = st.session_state.config['file_info']['file_type']
    
    if file_type == 'csv':
        # CSV는 다중 파일 업로드 가능
        uploaded_files = st.file_uploader(
            "CSV 파일 업로드 (여러 개 선택 가능)",
            type=['csv'],
            accept_multiple_files=True,
            key='csv_upload'
        )
    else:
        # Excel은 단일 파일
        uploaded_files = st.file_uploader(
            "Excel 파일 업로드",
            type=['xlsx', 'xls'],
            key='excel_upload'
        )
        # 리스트로 변환 (통일된 처리를 위해)
        if uploaded_files:
            uploaded_files = [uploaded_files]
    
    if uploaded_files:
        # 파일 정보 표시
        if len(uploaded_files) > 1:
            st.info(f"📁 {len(uploaded_files)}개 파일 선택됨")
            for i, f in enumerate(uploaded_files, 1):
                st.caption(f"{i}. {f.name}")
        
        # 파일 확장자 체크
        file_type = st.session_state.config['file_info']['file_type']
        
        if file_type == 'excel' and uploaded_files[0].name.split('.')[-1].lower() not in ['xlsx', 'xls']:
            st.warning("⚠️ 업로드된 파일은 Excel이 아니지만 설정은 Excel입니다. 설정을 확인해주세요.")
        elif file_type == 'csv' and any(f.name.split('.')[-1].lower() != 'csv' for f in uploaded_files):
            st.warning("⚠️ 일부 파일이 CSV가 아닙니다. 설정을 확인해주세요.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 데이터 로드 시작", type="primary", use_container_width=True):
                progress_text = st.empty()
                progress_bar = st.empty()
                
                with st.spinner("데이터를 로딩하는 중..."):
                    try:
                        # 진행 상황 콜백 함수
                        def update_progress(sheet_name, current, total):
                            progress_text.text(f"처리 중: {current}/{total} - {sheet_name}")
                            progress_bar.progress(current / total)
                        
                        # 로더 생성 (콜백 전달)
                        loader = UnifiedDataLoader(st.session_state.config, progress_callback=update_progress)
                        
                        # logger 선언
                        import logging
                        logger = logging.getLogger(__name__)
                        
                        # 타입 정리 함수 정의 (루프 밖에서)
                        def fix_timestamps_immediately(df):
                            """즉시 모든 타임스탬프 정리 - 완전히 새로운 DataFrame 생성"""
                            logger.info(f"🔧 fix_timestamps_immediately 시작: {df.shape}")
                            
                            # 완전히 새로운 딕셔너리로 컬럼 생성
                            new_data = {}
                            
                            for col in df.columns:
                                try:
                                    # datetime 타입인 경우
                                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                                        logger.info(f"  -> '{col}': datetime 타입 정리 중")
                                        # 새로운 Series 생성
                                        new_data[col] = pd.to_datetime(
                                            df[col].dt.strftime('%Y-%m-%d %H:%M:%S'),
                                            format='%Y-%m-%d %H:%M:%S',
                                            errors='coerce'
                                        )
                                    
                                    # object 타입에 Timestamp 있는 경우
                                    elif df[col].dtype == 'object':
                                        non_null = df[col].dropna()
                                        if len(non_null) > 0:
                                            first_val = non_null.iloc[0]
                                            if isinstance(first_val, (pd.Timestamp, type(pd.NaT))):
                                                logger.warning(f"  -> '{col}': object 타입 Timestamp 발견! 변환 중...")
                                                temp_col = pd.to_datetime(df[col], errors='coerce')
                                                new_data[col] = pd.to_datetime(
                                                    temp_col.dt.strftime('%Y-%m-%d %H:%M:%S'),
                                                    format='%Y-%m-%d %H:%M:%S',
                                                    errors='coerce'
                                                )
                                                logger.info(f"  -> '{col}': 변환 완료 dtype={new_data[col].dtype}")
                                            else:
                                                # 일반 object - 복사
                                                new_data[col] = df[col].copy()
                                        else:
                                            # 빈 컬럼 - 복사
                                            new_data[col] = df[col].copy()
                                    else:
                                        # 다른 타입 - 복사
                                        new_data[col] = df[col].copy()
                                        
                                except Exception as e:
                                    logger.error(f"  -> '{col}': 변환 실패 - {e}")
                                    # 실패 시 원본 복사
                                    new_data[col] = df[col].copy()
                            
                            # 완전히 새로운 DataFrame 생성
                            df_new = pd.DataFrame(new_data, index=df.index)
                            
                            # attrs 복사 (메타데이터 유지)
                            df_new.attrs = df.attrs.copy()
                            
                            logger.info(f"🔧 fix_timestamps_immediately 완료")
                            return df_new
                        
                        # 다중 파일 처리
                        all_data = {}
                        
                        for idx, uploaded_file in enumerate(uploaded_files, 1):
                            if len(uploaded_files) > 1:
                                progress_text.text(f"파일 {idx}/{len(uploaded_files)} 처리 중: {uploaded_file.name}")
                                progress_bar.progress(idx / len(uploaded_files))
                            
                            # 데이터 로드
                            data = loader.load(uploaded_file, file_type)
                            
                            # 파일명에서 확장자 제거
                            file_base_name = uploaded_file.name.rsplit('.', 1)[0]
                            
                            # 결과 저장 (즉시 정리)
                            if isinstance(data, pd.DataFrame):
                                # fix_timestamps_immediately가 새 DataFrame 생성
                                df_clean = fix_timestamps_immediately(data)
                                all_data[file_base_name] = df_clean
                            else:
                                # 다중 시트인 경우
                                for sheet_name, sheet_df in data.items():
                                    combined_name = f"{file_base_name}_{sheet_name}"
                                    # fix_timestamps_immediately가 새 DataFrame 생성
                                    df_clean = fix_timestamps_immediately(sheet_df)
                                    all_data[combined_name] = df_clean
                        
                        # 진행 상황 표시 정리
                        progress_text.empty()
                        progress_bar.empty()
                        
                        logger.info(f"💾 session_state 저장 준비: {len(all_data)}개 파일")
                        
                        # 완전히 새로운 DataFrame으로 저장 (더 이상 수정하지 않음)
                        if len(all_data) == 1:
                            final_df = list(all_data.values())[0]
                            
                            logger.info(f"   단일 DataFrame 저장: shape={final_df.shape}")
                            logger.info(f"   timestamp dtype: {final_df['timestamp'].dtype if 'timestamp' in final_df.columns else 'N/A'}")
                            
                            # timestamp 컬럼의 샘플 값 타입 체크
                            if 'timestamp' in final_df.columns:
                                sample_vals = final_df['timestamp'].dropna().head(3)
                                for idx, val in enumerate(sample_vals):
                                    logger.info(f"   timestamp[{idx}] 타입={type(val).__name__}, 값={val}")
                            
                            # ⚠️ CRITICAL: session_state에 저장하기 전에 완전히 정리
                            # prepare_df_for_display로 한 번 더 정리 (Timestamp 객체 제거)
                            final_df_safe = prepare_df_for_display(final_df)
                            
                            logger.info(f"   prepare_df_for_display 후 timestamp dtype: {final_df_safe['timestamp'].dtype if 'timestamp' in final_df_safe.columns else 'N/A'}")
                            
                            if 'timestamp' in final_df_safe.columns:
                                sample_vals = final_df_safe['timestamp'].dropna().head(3)
                                for idx, val in enumerate(sample_vals):
                                    logger.info(f"   정리 후 timestamp[{idx}] 타입={type(val).__name__}, 값={val}")
                            
                            st.session_state.loaded_data = final_df_safe
                            
                            logger.info(f"   ✅ st.session_state.loaded_data 저장 완료")
                            
                            # 메타데이터는 정리된 DataFrame에서 생성
                            st.session_state.metadata = {
                                'source_name': final_df_safe.attrs.get('source_name', 'unknown'),
                                'header_metadata': final_df_safe.attrs.get('header_metadata', {}),
                                'removed_columns': final_df_safe.attrs.get('removed_columns', []),  # 제거된 컬럼 정보 추가
                                'shape': final_df_safe.shape,
                                'columns': final_df_safe.columns.tolist(),
                                'dtypes': {str(k): str(v) for k, v in final_df_safe.dtypes.items()}
                            }
                        else:
                            logger.info(f"   다중 DataFrame 저장: {len(all_data)}개")
                            
                            # 각 DataFrame을 정리
                            cleaned_data = {}
                            for name, df in all_data.items():
                                logger.info(f"   - {name}: shape={df.shape}")
                                cleaned_data[name] = prepare_df_for_display(df)
                            
                            st.session_state.loaded_data = cleaned_data
                            
                            # 각 시트의 메타데이터 생성
                            st.session_state.metadata = {
                                sheet: {
                                    'source_name': df.attrs.get('source_name', sheet),
                                    'header_metadata': df.attrs.get('header_metadata', {}),
                                    'removed_columns': df.attrs.get('removed_columns', []),  # 제거된 컬럼 정보 추가
                                    'shape': df.shape,
                                    'columns': df.columns.tolist(),
                                    'dtypes': {str(k): str(v) for k, v in df.dtypes.items()}
                                }
                                for sheet, df in cleaned_data.items()
                            }
                        
                        # 중복 컬럼명 체크 및 경고
                        def check_duplicates(df):
                            cols = df.columns.tolist()
                            # _숫자로 끝나는 중복 패턴 찾기
                            import re
                            dup_pattern = {}
                            for col in cols:
                                match = re.match(r'(.+)_(\d+)$', col)
                                if match:
                                    base = match.group(1)
                                    if base not in dup_pattern:
                                        dup_pattern[base] = []
                                    dup_pattern[base].append(col)
                            return dup_pattern
                        
                        duplicate_info = None
                        if len(all_data) == 1:
                            duplicate_info = check_duplicates(list(all_data.values())[0])
                        
                        # 메타데이터 체크
                        metadata_info = []
                        if 'metadata' in st.session_state and st.session_state.metadata:
                            header_meta = st.session_state.metadata.get('header_metadata', {})
                            if header_meta:
                                if 'description' in header_meta:
                                    metadata_info.append(f"✅ Description: {len(header_meta['description'])}개")
                                if 'unit' in header_meta:
                                    metadata_info.append(f"✅ Unit: {len(header_meta['unit'])}개")
                                if 'tag_name' in header_meta:
                                    metadata_info.append(f"✅ Tag_name: {len(header_meta['tag_name'])}개")
                                if 'id' in header_meta:
                                    metadata_info.append(f"✅ ID: {len(header_meta['id'])}개")
                        
                        st.success("✅ 데이터 로딩 완료!")
                        
                        # 메타데이터 정보 표시
                        if metadata_info:
                            st.info("📋 **메타데이터 발견:**\n" + "\n".join(metadata_info))
                        else:
                            st.warning("⚠️ 메타데이터가 없습니다. YAML 설정에서 header_rows를 확인하세요.")
                        
                        # 중복 컬럼명이 있었으면 경고 표시
                        if duplicate_info and any(len(v) > 0 for v in duplicate_info.values()):
                            with st.warning("⚠️ 중복된 컬럼명이 발견되어 자동으로 수정되었습니다."):
                                st.caption("중복된 컬럼명에 '_숫자' 접미사가 추가되었습니다.")
                                dup_count = sum(len(v) for v in duplicate_info.values())
                                st.caption(f"총 {dup_count}개의 중복 발견")
                        
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ 데이터 로딩 실패: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with col2:
            if st.button("🗑️ 초기화", use_container_width=True):
                # 모든 세션 상태 완전 제거
                keys_to_remove = ['loaded_data', 'metadata']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # 강제 재시작
                st.rerun()
    
    # 로드된 데이터 표시
    if st.session_state.loaded_data is not None:
        st.divider()
        st.subheader("📊 로드된 데이터")
        
        import logging
        logger = logging.getLogger(__name__)
        
        # session_state에는 이미 정리된 데이터가 저장됨
        data = st.session_state.loaded_data
        
        # 단일 DataFrame
        if isinstance(data, pd.DataFrame):
            st.markdown(f"**Shape:** {data.shape[0]:,} rows × {data.shape[1]:,} columns")
            
            # 데이터 미리보기
            with st.expander("🔍 데이터 미리보기", expanded=True):
                logger.info(f"📊 [미리보기] 시작: data.shape={data.shape}")
                logger.info(f"   data의 timestamp dtype: {data['timestamp'].dtype if 'timestamp' in data.columns else 'N/A'}")
                
                n_rows = st.slider("표시할 행 수", 5, 100, 10, key='single_preview_rows')
                
                # safe_display_df가 모든 변환 처리
                preview_head = data.head(n_rows)
                
                logger.info(f"   .head() 후 - shape={preview_head.shape}")
                logger.info(f"   safe_display_df 호출 전")
                
                st.dataframe(safe_display_df(preview_head), use_container_width=True)
            
            # 메타데이터
            if st.session_state.metadata:
                with st.expander("ℹ️ 메타데이터"):
                    meta = st.session_state.metadata
                    if 'header_metadata' in meta and meta['header_metadata']:
                        st.json(meta['header_metadata'])

            # 제거된 컬럼 정보
            if st.session_state.metadata and 'removed_columns' in st.session_state.metadata:
                removed_cols = st.session_state.metadata['removed_columns']
                if removed_cols and len(removed_cols) > 0:
                    with st.expander(f"🗑️ 제거된 중복 컬럼 ({len(removed_cols)}개)", expanded=True):
                        st.warning(f"⚠️ 총 {len(removed_cols)}개의 중복 컬럼이 제거되었습니다.")

                        # 테이블 형태로 표시
                        removed_df = pd.DataFrame(removed_cols)

                        # 컬럼 순서 정리
                        cols_order = ['tag_name', 'description', 'unit', 'reason']
                        cols_order = [c for c in cols_order if c in removed_df.columns]
                        removed_df = removed_df[cols_order]

                        # 컬럼명 한글로 변경
                        removed_df.columns = ['태그명', '설명', '단위', '제거 이유'][:len(removed_df.columns)]

                        st.dataframe(removed_df, use_container_width=True, hide_index=True)

                        # 제거 이유별 통계
                        st.caption("**제거 이유별 통계:**")
                        reason_counts = pd.DataFrame(removed_cols)['reason'].value_counts()
                        for reason, count in reason_counts.items():
                            st.caption(f"  - {reason}: {count}개")

            # 데이터 통계 (원본 display_df 사용 - datetime 유지)
            with st.expander("📈 기본 통계"):
                logger.info(f"📊 [기본통계] 시작")
                
                # data는 이미 prepare_df_for_display로 정리됨
                stats_df = data.describe(include='all').reset_index()
                
                logger.info(f"   describe() 후 stats_df.shape={stats_df.shape}")
                # logger.info(f"   stats_df dtypes: {dict(stats_df.dtypes)}")
                logger.info(f"   safe_display_df 호출 전")
                
                st.dataframe(safe_display_df(stats_df), use_container_width=True)
            
            # 저장 옵션
            st.divider()
            st.subheader("💾 데이터 저장")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                save_format = st.selectbox("저장 형식", ['hdf5', 'parquet', 'csv', 'excel'], key='single_save_format')
            
            with col2:
                file_name = st.text_input("파일명 (확장자 제외)", value="output_data", key='single_file_name')
                include_date = st.checkbox("파일명에 날짜 추가", value=False,
                                         help="데이터의 timestamp에서 날짜 범위를 추출하여 파일명에 추가합니다 (예: output_data_20250101_20250131.parquet)",
                                         key='single_include_date')
            
            with col3:
                st.write("")  # 간격
                st.write("")  # 간격
                if st.button("💾 저장", type="primary", use_container_width=True, key='single_save_btn'):
                    try:
                        # timestamp에서 날짜 범위 추출
                        date_str = extract_date_range_from_df(data) if include_date else ''
                        
                        if save_format == 'hdf5':
                            # HDF5는 파일 경로가 필요하므로 임시 파일 사용
                            import tempfile
                            import os
                            
                            # HDF5 저장 전 데이터 타입 정리
                            data_copy = prepare_df_for_parquet(data)
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                                tmp_path = tmp.name
                            
                            try:
                                # HDF5로 저장
                                save_to_hdf5_with_metadata(data_copy, tmp_path, key='data', compression='gzip')
                                
                                # 파일 읽기
                                with open(tmp_path, 'rb') as f:
                                    buffer = io.BytesIO(f.read())
                                
                                mime = 'application/x-hdf5'
                                ext = '.h5'
                            finally:
                                # 임시 파일 삭제
                                if os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                        
                        elif save_format == 'parquet':
                            buffer = io.BytesIO()
                            # Parquet 저장 전 데이터 타입 정리
                            data_copy = prepare_df_for_parquet(data)
                            
                            # 메타데이터와 함께 저장
                            save_to_parquet_with_metadata(data_copy, buffer, compression='snappy')
                            
                            mime = 'application/octet-stream'
                            ext = '.parquet'
                        elif save_format == 'csv':
                            buffer = io.BytesIO()
                            data.to_csv(buffer, index=False, encoding='utf-8-sig')
                            mime = 'text/csv'
                            ext = '.csv'
                        else:  # excel
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                data.to_excel(writer, index=False, sheet_name='Data')
                            mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            ext = '.xlsx'
                        
                        buffer.seek(0)
                        
                        # 파일명 생성
                        if date_str:
                            download_name = f"{file_name}_{date_str}{ext}"
                        else:
                            download_name = f"{file_name}{ext}"
                        
                        st.download_button(
                            label=f"⬇️ {download_name} 다운로드",
                            data=buffer,
                            file_name=download_name,
                            mime=mime,
                            use_container_width=True
                        )
                        
                        # 저장 완료 메시지
                        st.success(f"✅ {save_format.upper()} 파일 준비 완료!")
                        
                        # 메타데이터 저장 확인 (HDF5인 경우만)
                        if save_format == 'hdf5' and hasattr(data, 'attrs') and 'header_metadata' in data.attrs:
                            header_meta = data.attrs['header_metadata']
                            meta_saved = []
                            if 'description' in header_meta:
                                meta_saved.append(f"Description ({len(header_meta['description'])}개)")
                            if 'unit' in header_meta:
                                meta_saved.append(f"Unit ({len(header_meta['unit'])}개)")
                            if 'tag_name' in header_meta:
                                meta_saved.append(f"Tag_name ({len(header_meta['tag_name'])}개)")
                            if 'id' in header_meta:
                                meta_saved.append(f"ID ({len(header_meta['id'])}개)")
                            
                            if meta_saved:
                                st.info(f"💾 **메타데이터 저장됨:** {', '.join(meta_saved)}")
                    
                    except Exception as e:
                        st.error(f"❌ 저장 실패: {str(e)}")
        
        # 다중 DataFrame (시트별)
        else:
            st.markdown(f"**총 {len(data)}개 시트 로드됨**")
            
            selected_sheet = st.selectbox("시트 선택", options=list(data.keys()), key='loading_sheet_select')
            df = data[selected_sheet]
            
            st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]:,} columns")
            
            # 데이터 미리보기
            with st.expander("🔍 데이터 미리보기", expanded=True):
                n_rows = st.slider("표시할 행 수", 5, 100, 10, key='multi_sheet_preview_rows')
                
                # safe_display_df가 모든 변환 처리
                st.dataframe(safe_display_df(df.head(n_rows)), use_container_width=True)
            
            # 메타데이터
            if st.session_state.metadata and selected_sheet in st.session_state.metadata:
                with st.expander("ℹ️ 메타데이터"):
                    meta = st.session_state.metadata[selected_sheet]
                    if 'header_metadata' in meta and meta['header_metadata']:
                        st.json(meta['header_metadata'])

            # 제거된 컬럼 정보 (다중 시트)
            if st.session_state.metadata and selected_sheet in st.session_state.metadata:
                meta = st.session_state.metadata[selected_sheet]
                if 'removed_columns' in meta:
                    removed_cols = meta['removed_columns']
                    if removed_cols and len(removed_cols) > 0:
                        with st.expander(f"🗑️ 제거된 중복 컬럼 ({len(removed_cols)}개)", expanded=True):
                            st.warning(f"⚠️ 총 {len(removed_cols)}개의 중복 컬럼이 제거되었습니다.")

                            # 테이블 형태로 표시
                            removed_df = pd.DataFrame(removed_cols)

                            # 컬럼 순서 정리
                            cols_order = ['tag_name', 'description', 'unit', 'reason']
                            cols_order = [c for c in cols_order if c in removed_df.columns]
                            removed_df = removed_df[cols_order]

                            # 컬럼명 한글로 변경
                            removed_df.columns = ['태그명', '설명', '단위', '제거 이유'][:len(removed_df.columns)]

                            st.dataframe(removed_df, use_container_width=True, hide_index=True)

                            # 제거 이유별 통계
                            st.caption("**제거 이유별 통계:**")
                            reason_counts = pd.DataFrame(removed_cols)['reason'].value_counts()
                            for reason, count in reason_counts.items():
                                st.caption(f"  - {reason}: {count}개")

            # 데이터 통계 (원본 사용 - datetime 유지)
            with st.expander("📈 기본 통계"):
                stats_df = df.describe(include='all')
                st.dataframe(safe_display_df(stats_df), use_container_width=True)
            
            # 저장 옵션
            st.divider()
            st.subheader("💾 데이터 저장")
            
            # 시트 선택 옵션
            st.markdown("**📋 저장할 데이터 선택**")
            
            all_sheet_names = list(data.keys())
            
            col_select1, col_select2 = st.columns([3, 1])
            
            with col_select1:
                save_mode = st.radio(
                    "저장 모드",
                    ["개별 저장", "선택한 시트 병합", "모든 시트 병합"],
                    help="개별 저장: 각 시트를 별도 파일로 저장\n병합: 여러 시트를 하나의 파일로 결합",
                    horizontal=True,
                    key='multi_save_mode'
                )
            
            with col_select2:
                if save_mode != "개별 저장":
                    sort_by_time = st.checkbox(
                        "시간순 정렬",
                        value=True,
                        help="timestamp가 있으면 시간순으로, 없으면 입력 순서대로 병합",
                        key='sort_by_time'
                    )
                else:
                    sort_by_time = False
            
            # 선택한 시트 병합 모드일 때 선택 UI
            selected_sheets = []
            if save_mode == "선택한 시트 병합":
                st.markdown("**병합할 시트 선택** (순서대로 병합됩니다)")
                selected_sheets = st.multiselect(
                    "시트 선택",
                    options=all_sheet_names,
                    default=all_sheet_names[:min(3, len(all_sheet_names))],
                    key='selected_sheets_to_merge'
                )
                if not selected_sheets:
                    st.warning("⚠️ 병합할 시트를 최소 1개 이상 선택해주세요.")
            elif save_mode == "모든 시트 병합":
                selected_sheets = all_sheet_names
                st.info(f"📊 총 {len(selected_sheets)}개 시트를 병합합니다.")
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                save_format = st.selectbox("저장 형식", ['hdf5', 'parquet', 'csv', 'excel'], key='multi_save_format')
                
                # 병합 모드에서는 개별 파일 저장 옵션 숨김
                if save_mode == "개별 저장":
                    save_all = st.checkbox("모든 시트 저장", value=False, key='multi_save_all')
                else:
                    save_all = False
            
            with col2:
                file_name = st.text_input("파일명 (확장자 제외)", value="output_data", key='multi_file_name')
                if save_mode == "개별 저장" and save_all and save_format in ['parquet', 'csv']:
                    include_date = st.checkbox("파일명에 날짜 추가", value=True, 
                                             help="각 시트의 timestamp에서 날짜 범위를 추출하여 파일명에 추가합니다",
                                             key='multi_include_date')
                elif save_mode != "개별 저장":
                    include_date = st.checkbox("파일명에 날짜 추가", value=True,
                                             help="병합된 데이터의 timestamp에서 날짜 범위를 추출하여 파일명에 추가합니다",
                                             key='multi_include_date_merged')
                else:
                    include_date = False
            
            with col3:
                st.write("")
                st.write("")
                if st.button("💾 저장", type="primary", use_container_width=True, key='multi_save_btn'):
                    try:
                        # 병합 모드
                        if save_mode in ["선택한 시트 병합", "모든 시트 병합"]:
                            if not selected_sheets:
                                st.error("❌ 병합할 시트를 선택해주세요.")
                            else:
                                # 선택된 시트들을 하나로 병합
                                dfs_to_merge = [data[sheet] for sheet in selected_sheets]
                                
                                # timestamp 컬럼 찾기
                                has_timestamp = False
                                ts_col = None
                                for df in dfs_to_merge:
                                    for col in df.columns:
                                        if 'timestamp' in str(col).lower() or 'datetime' in str(col).lower():
                                            if pd.api.types.is_datetime64_any_dtype(df[col]):
                                                has_timestamp = True
                                                ts_col = col
                                                break
                                    if has_timestamp:
                                        break
                                
                                # 병합
                                merged_df = pd.concat(dfs_to_merge, ignore_index=True)
                                
                                # 정렬
                                if sort_by_time and has_timestamp and ts_col:
                                    merged_df = merged_df.sort_values(by=ts_col).reset_index(drop=True)
                                    st.success(f"✅ {len(selected_sheets)}개 시트를 시간순으로 병합했습니다. (총 {len(merged_df)}행)")
                                else:
                                    st.success(f"✅ {len(selected_sheets)}개 시트를 입력 순서대로 병합했습니다. (총 {len(merged_df)}행)")
                                
                                # 날짜 범위 추출
                                date_str = extract_date_range_from_df(merged_df) if include_date else ''
                                
                                # 파일 생성
                                if save_format == 'hdf5':
                                    # HDF5는 임시 파일 필요
                                    import tempfile
                                    import os
                                    
                                    # 데이터 타입 정리
                                    merged_df_copy = prepare_df_for_parquet(merged_df)
                                    
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                                        tmp_path = tmp.name
                                    
                                    try:
                                        save_to_hdf5_with_metadata(merged_df_copy, tmp_path, key='data', compression='gzip')
                                        
                                        with open(tmp_path, 'rb') as f:
                                            buffer = io.BytesIO(f.read())
                                        
                                        mime = 'application/x-hdf5'
                                        ext = '.h5'
                                    finally:
                                        if os.path.exists(tmp_path):
                                            os.unlink(tmp_path)
                                
                                elif save_format == 'parquet':
                                    buffer = io.BytesIO()
                                    merged_df_copy = prepare_df_for_parquet(merged_df)
                                    save_to_parquet_with_metadata(merged_df_copy, buffer, compression='snappy')
                                    mime = 'application/octet-stream'
                                    ext = '.parquet'
                                elif save_format == 'csv':
                                    buffer = io.BytesIO()
                                    merged_df.to_csv(buffer, index=False, encoding='utf-8-sig')
                                    mime = 'text/csv'
                                    ext = '.csv'
                                else:  # excel
                                    buffer = io.BytesIO()
                                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                        merged_df.to_excel(writer, index=False, sheet_name='MergedData')
                                    mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                    ext = '.xlsx'
                                
                                buffer.seek(0)
                                
                                # 파일명 생성
                                if date_str:
                                    download_name = f"{file_name}_{date_str}_merged{ext}"
                                else:
                                    download_name = f"{file_name}_merged{ext}"
                                
                                st.download_button(
                                    label=f"⬇️ {download_name} 다운로드",
                                    data=buffer,
                                    file_name=download_name,
                                    mime=mime,
                                    use_container_width=True
                                )
                        
                        # 개별 저장 모드
                        elif save_mode == "개별 저장":
                            if save_all:
                                # 모든 시트를 하나의 Excel 파일로
                                if save_format == 'excel':
                                    buffer = io.BytesIO()
                                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                        for sheet_name, sheet_df in data.items():
                                            safe_name = re.sub(r'[<>:"/\\|?*]', '_', sheet_name)[:31]
                                            sheet_df.to_excel(writer, index=False, sheet_name=safe_name)
                                    buffer.seek(0)
                                    
                                    st.download_button(
                                        label=f"⬇️ {file_name}.xlsx 다운로드 (모든 시트)",
                                        data=buffer,
                                        file_name=f"{file_name}.xlsx",
                                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                        use_container_width=True
                                    )
                                    st.success("✅ Excel 파일 준비 완료!")
                                
                                # 모든 시트를 개별 HDF5 파일로
                                elif save_format == 'hdf5':
                                    st.info(f"총 {len(data)}개의 HDF5 파일을 생성합니다...")
                                    
                                    import zipfile
                                    import tempfile
                                    import os
                                    zip_buffer = io.BytesIO()
                                    
                                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                        for sheet_name, sheet_df in data.items():
                                            safe_name = re.sub(r'[<>:"/\\|?*]', '_', sheet_name)
                                            
                                            date_str = extract_date_range_from_df(sheet_df) if include_date else ''
                                            
                                            # 데이터 타입 정리
                                            sheet_df_copy = prepare_df_for_parquet(sheet_df)
                                            
                                            # 임시 HDF5 파일 생성
                                            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                                                tmp_path = tmp.name
                                            
                                            try:
                                                save_to_hdf5_with_metadata(sheet_df_copy, tmp_path, key='data', compression='gzip')
                                                
                                                with open(tmp_path, 'rb') as f:
                                                    h5_data = f.read()
                                                
                                                # 파일명 생성
                                                if date_str:
                                                    file_name_in_zip = f"{file_name}_{date_str}_{safe_name}.h5"
                                                else:
                                                    file_name_in_zip = f"{file_name}_{safe_name}.h5"
                                                
                                                zip_file.writestr(file_name_in_zip, h5_data)
                                            
                                            finally:
                                                if os.path.exists(tmp_path):
                                                    os.unlink(tmp_path)
                                    
                                    zip_buffer.seek(0)
                                    
                                    if include_date:
                                        first_sheet = list(data.values())[0]
                                        date_str = extract_date_range_from_df(first_sheet)
                                        if date_str:
                                            zip_name = f"{file_name}_{date_str}_all_sheets.zip"
                                        else:
                                            zip_name = f"{file_name}_all_sheets.zip"
                                    else:
                                        zip_name = f"{file_name}_all_sheets.zip"
                                    
                                    st.download_button(
                                        label=f"⬇️ {zip_name} (전체 다운로드)",
                                        data=zip_buffer,
                                        file_name=zip_name,
                                        mime='application/zip',
                                        use_container_width=True
                                    )
                                    st.success(f"✅ {len(data)}개의 HDF5 파일이 ZIP으로 압축되었습니다!")
                                
                                # 모든 시트를 개별 Parquet 파일로
                                elif save_format == 'parquet':
                                    st.info(f"총 {len(data)}개의 파일을 생성합니다...")
                                    
                                    # ZIP 파일 생성
                                    import zipfile
                                    zip_buffer = io.BytesIO()
                                    
                                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                        for sheet_name, sheet_df in data.items():
                                            safe_name = re.sub(r'[<>:"/\\|?*]', '_', sheet_name)
                                            
                                            # 각 시트의 날짜 범위 추출
                                            date_str = extract_date_range_from_df(sheet_df) if include_date else ''
                                            
                                            # Parquet 저장 전 데이터 타입 정리
                                            sheet_df_copy = prepare_df_for_parquet(sheet_df)
                                            
                                            # Parquet 파일 생성
                                            parquet_buffer = io.BytesIO()
                                            save_to_parquet_with_metadata(sheet_df_copy, parquet_buffer, compression='snappy')
                                            parquet_buffer.seek(0)
                                            
                                            # 파일명 생성
                                            if date_str:
                                                file_name_in_zip = f"{file_name}_{date_str}_{safe_name}.parquet"
                                            else:
                                                file_name_in_zip = f"{file_name}_{safe_name}.parquet"
                                            
                                            # ZIP에 추가
                                            zip_file.writestr(file_name_in_zip, parquet_buffer.getvalue())
                                    
                                    zip_buffer.seek(0)
                                    
                                    # ZIP 파일명 생성
                                    if include_date:
                                        # 첫 번째 시트의 날짜 정보 사용
                                        first_sheet = list(data.values())[0]
                                        date_str = extract_date_range_from_df(first_sheet)
                                        if date_str:
                                            zip_name = f"{file_name}_{date_str}_all_sheets.zip"
                                        else:
                                            zip_name = f"{file_name}_all_sheets.zip"
                                    else:
                                        zip_name = f"{file_name}_all_sheets.zip"
                                    
                                    st.download_button(
                                        label=f"⬇️ {zip_name} (전체 다운로드)",
                                        data=zip_buffer,
                                        file_name=zip_name,
                                        mime='application/zip',
                                        use_container_width=True
                                    )
                                    
                                    st.success(f"✅ {len(data)}개 Parquet 파일이 포함된 ZIP 파일 준비 완료!")
                                    
                                    # 개별 파일 다운로드 옵션
                                    with st.expander("📄 개별 파일 다운로드 (선택사항)"):
                                        for sheet_name, sheet_df in data.items():
                                            safe_name = re.sub(r'[<>:"/\\|?*]', '_', sheet_name)
                                            
                                            # 각 시트의 날짜 범위 추출
                                            date_str = extract_date_range_from_df(sheet_df) if include_date else ''
                                            
                                            # Parquet 저장 전 데이터 타입 정리
                                            sheet_df_copy = prepare_df_for_parquet(sheet_df)
                                            
                                            buffer = io.BytesIO()
                                            save_to_parquet_with_metadata(sheet_df_copy, buffer, compression='snappy')
                                            buffer.seek(0)
                                            
                                            # 파일명 생성
                                            if date_str:
                                                download_name = f"{file_name}_{date_str}_{safe_name}.parquet"
                                            else:
                                                download_name = f"{file_name}_{safe_name}.parquet"
                                            
                                            st.download_button(
                                                label=f"⬇️ {download_name}",
                                                data=buffer,
                                                file_name=download_name,
                                                mime='application/octet-stream',
                                                key=f'download_{sheet_name}',
                                                use_container_width=True
                                            )
                                
                                # 모든 시트를 개별 CSV 파일로
                                elif save_format == 'csv':
                                    st.info(f"총 {len(data)}개의 파일을 생성합니다...")
                                    
                                    # ZIP 파일 생성
                                    import zipfile
                                    zip_buffer = io.BytesIO()
                                    
                                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                        for sheet_name, sheet_df in data.items():
                                            safe_name = re.sub(r'[<>:"/\\|?*]', '_', sheet_name)
                                            
                                            # 각 시트의 날짜 범위 추출
                                            date_str = extract_date_range_from_df(sheet_df) if include_date else ''
                                            
                                            # CSV 파일 생성
                                            csv_buffer = io.BytesIO()
                                            sheet_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                                            csv_buffer.seek(0)
                                            
                                            # 파일명 생성
                                            if date_str:
                                                file_name_in_zip = f"{file_name}_{date_str}_{safe_name}.csv"
                                            else:
                                                file_name_in_zip = f"{file_name}_{safe_name}.csv"
                                            
                                            # ZIP에 추가
                                            zip_file.writestr(file_name_in_zip, csv_buffer.getvalue())
                                    
                                    zip_buffer.seek(0)
                                    
                                    # ZIP 파일명 생성
                                    if include_date:
                                        # 첫 번째 시트의 날짜 정보 사용
                                        first_sheet = list(data.values())[0]
                                        date_str = extract_date_range_from_df(first_sheet)
                                        if date_str:
                                            zip_name = f"{file_name}_{date_str}_all_sheets.zip"
                                        else:
                                            zip_name = f"{file_name}_all_sheets.zip"
                                    else:
                                        zip_name = f"{file_name}_all_sheets.zip"
                                    
                                    st.download_button(
                                        label=f"⬇️ {zip_name} (전체 다운로드)",
                                        data=zip_buffer,
                                        file_name=zip_name,
                                        mime='application/zip',
                                        use_container_width=True
                                    )
                                    
                                    st.success(f"✅ {len(data)}개 CSV 파일이 포함된 ZIP 파일 준비 완료!")
                                    
                                    # 개별 파일 다운로드 옵션
                                    with st.expander("📄 개별 파일 다운로드 (선택사항)"):
                                        for sheet_name, sheet_df in data.items():
                                            safe_name = re.sub(r'[<>:"/\\|?*]', '_', sheet_name)
                                            
                                            # 각 시트의 날짜 범위 추출
                                            date_str = extract_date_range_from_df(sheet_df) if include_date else ''
                                            
                                            buffer = io.BytesIO()
                                            sheet_df.to_csv(buffer, index=False, encoding='utf-8-sig')
                                            buffer.seek(0)
                                            
                                            # 파일명 생성
                                            if date_str:
                                                download_name = f"{file_name}_{date_str}_{safe_name}.csv"
                                            else:
                                                download_name = f"{file_name}_{safe_name}.csv"
                                            
                                            st.download_button(
                                                label=f"⬇️ {download_name}",
                                                data=buffer,
                                                file_name=download_name,
                                                mime='text/csv',
                                                key=f'download_csv_{sheet_name}',
                                                use_container_width=True
                                            )
                                else:
                                    st.warning("⚠️ 다중 시트는 Excel 또는 개별 파일 형식으로만 저장 가능합니다.")
                            else:
                                # 선택된 시트만
                                # 날짜 범위 추출
                                date_str = extract_date_range_from_df(df) if include_date else ''
                                
                                if save_format == 'hdf5':
                                    # HDF5는 임시 파일 필요
                                    import tempfile
                                    import os
                                    
                                    # 데이터 타입 정리
                                    df_copy = prepare_df_for_parquet(df)
                                    
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                                        tmp_path = tmp.name
                                    
                                    try:
                                        save_to_hdf5_with_metadata(df_copy, tmp_path, key='data', compression='gzip')
                                        
                                        with open(tmp_path, 'rb') as f:
                                            buffer = io.BytesIO(f.read())
                                        
                                        mime = 'application/x-hdf5'
                                        ext = '.h5'
                                    finally:
                                        if os.path.exists(tmp_path):
                                            os.unlink(tmp_path)
                                
                                elif save_format == 'parquet':
                                    buffer = io.BytesIO()
                                    # Parquet 저장 전 데이터 타입 정리
                                    df_copy = prepare_df_for_parquet(df)
                                    save_to_parquet_with_metadata(df_copy, buffer, compression='snappy')
                                    mime = 'application/octet-stream'
                                    ext = '.parquet'
                                elif save_format == 'csv':
                                    buffer = io.BytesIO()
                                    df.to_csv(buffer, index=False, encoding='utf-8-sig')
                                    mime = 'text/csv'
                                    ext = '.csv'
                                else:  # excel
                                    buffer = io.BytesIO()
                                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                        df.to_excel(writer, index=False, sheet_name='Data')
                                    mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                    ext = '.xlsx'
                                
                                buffer.seek(0)
                                
                                # 파일명 생성
                                if date_str:
                                    download_name = f"{file_name}_{date_str}_{selected_sheet}{ext}"
                                else:
                                    download_name = f"{file_name}_{selected_sheet}{ext}"
                                
                                st.download_button(
                                    label=f"⬇️ {download_name} 다운로드",
                                    data=buffer,
                                    file_name=download_name,
                                    mime=mime,
                                    use_container_width=True
                            )
                            st.success(f"✅ {save_format.upper()} 파일 준비 완료!")
                    
                    except Exception as e:
                        st.error(f"❌ 저장 실패: {str(e)}")


# ============================================
# Tab 3: 데이터 가시화
# ============================================
def render_visualization_tab():
    st.header("📊 데이터 가시화")
    
    # Parquet 파일 업로드 옵션
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_parquet = st.file_uploader(
            "Parquet/HDF5 파일 불러오기 (선택사항)",
            type=['parquet', 'h5', 'hdf5'],
            key='viz_parquet'
        )
    
    with col2:
        if uploaded_parquet:
            file_name = uploaded_parquet.name
            file_ext = file_name.split('.')[-1].lower()
            
            button_label = "📥 HDF5 로드" if file_ext in ['h5', 'hdf5'] else "📥 Parquet 로드"
            
            if st.button(button_label, use_container_width=True):
                try:
                    if file_ext in ['h5', 'hdf5']:
                        # HDF5 파일 로드
                        import tempfile
                        import os
                        
                        # 임시 파일로 저장 (HDF5는 파일 경로 필요)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                            tmp.write(uploaded_parquet.read())
                            tmp_path = tmp.name
                        
                        try:
                            # HDF5에서 메타데이터와 함께 로드
                            df = load_from_hdf5_with_metadata(tmp_path, key='data')
                            
                            # 메타데이터를 session_state에 저장
                            st.session_state.metadata = {
                                'source_name': df.attrs.get('source_name', 'hdf5_file'),
                                'header_metadata': df.attrs.get('header_metadata', {}),
                                'removed_columns': df.attrs.get('removed_columns', []),  # 제거된 컬럼 정보 추가
                                'shape': df.shape,
                                'columns': df.columns.tolist(),
                                'dtypes': {str(k): str(v) for k, v in df.dtypes.items()}
                            }
                            
                            header_meta = df.attrs.get('header_metadata', {})
                            
                            msg_parts = ["✅ HDF5 로드 완료!"]
                            if header_meta:
                                msg_parts.append(f"메타데이터: {len(header_meta)}개 필드")
                            
                            st.success(" | ".join(msg_parts))
                            
                            st.session_state.loaded_data = df
                            st.rerun()
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                    
                    else:
                        # Parquet 파일 로드
                        import pyarrow.parquet as pq
                        import json
                        
                        # PyArrow로 Parquet 읽기 (메타데이터 포함)
                        parquet_file = pq.read_table(uploaded_parquet)
                        
                        # DataFrame으로 변환
                        df = parquet_file.to_pandas()
                        
                        # 메타데이터 복원
                        if parquet_file.schema.metadata:
                            metadata_bytes = parquet_file.schema.metadata
                            if b'pandas_attrs' in metadata_bytes:
                                attrs_json = metadata_bytes[b'pandas_attrs'].decode('utf-8')
                                df.attrs = json.loads(attrs_json)
                        
                        # 메타데이터를 session_state에 저장
                        st.session_state.metadata = {
                            'source_name': df.attrs.get('source_name', 'parquet_file'),
                            'header_metadata': df.attrs.get('header_metadata', {}),
                            'removed_columns': df.attrs.get('removed_columns', []),  # 제거된 컬럼 정보 추가
                            'shape': df.shape,
                            'columns': df.columns.tolist(),
                            'dtypes': {str(k): str(v) for k, v in df.dtypes.items()}
                        }
                        
                        # 데이터 타입 확인 및 표시
                        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
                        header_meta = df.attrs.get('header_metadata', {})
                        
                        msg_parts = ["✅ Parquet 로드 완료!"]
                        if bool_cols:
                            msg_parts.append(f"Boolean 컬럼: {', '.join(bool_cols)}")
                        if header_meta:
                            msg_parts.append(f"메타데이터: {len(header_meta)}개 필드")
                        
                        st.success(" | ".join(msg_parts))
                        
                        st.session_state.loaded_data = df
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ 로드 실패: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # 로드된 데이터 확인
    if st.session_state.loaded_data is None:
        st.info("ℹ️ 먼저 '데이터 로딩' 탭에서 데이터를 로드하거나, Parquet 파일을 업로드해주세요.")
        return
    
    data = st.session_state.loaded_data
    
    # 다중 시트인 경우 선택
    if isinstance(data, dict):
        selected_sheet = st.selectbox("시트 선택", options=list(data.keys()), key='viz_sheet_select')
        df = data[selected_sheet]
    else:
        df = data
    
    st.divider()
    
    # 메타데이터 정보
    if st.session_state.metadata:
        with st.expander("ℹ️ 헤더 메타데이터 확인"):
            if isinstance(st.session_state.metadata, dict):
                if isinstance(data, dict):
                    meta = st.session_state.metadata.get(selected_sheet, {})
                else:
                    meta = st.session_state.metadata
                
                header_meta = meta.get('header_metadata', {})
                if header_meta:
                    st.markdown("**사용 가능한 헤더 정보:**")
                    
                    # 각 헤더 타입별로 표시
                    if 'description' in header_meta:
                        with st.container():
                            st.markdown("##### 📝 설명(Description)")
                            desc_list = header_meta['description'][:10]  # 처음 10개만
                            st.write(", ".join([str(d) for d in desc_list if pd.notna(d)]))
                    
                    if 'unit' in header_meta:
                        with st.container():
                            st.markdown("##### 📏 단위(Unit)")
                            unit_list = header_meta['unit'][:10]  # 처음 10개만
                            st.write(", ".join([str(u) for u in unit_list if pd.notna(u)]))
                    
                    if 'tag_name' in header_meta:
                        with st.container():
                            st.markdown("##### 🏷️ 태그명(Tag)")
                            tag_list = header_meta['tag_name'][:10]  # 처음 10개만
                            st.write(", ".join([str(t) for t in tag_list if pd.notna(t)]))
                    
                    # 전체 메타데이터 JSON으로 표시
                    if st.checkbox("전체 메타데이터 보기", key='show_full_meta'):
                        st.json(header_meta)
                else:
                    st.info("메타데이터가 없습니다.")
            else:
                st.info("메타데이터가 없습니다.")
    
    st.divider()
    
    # 숫자형 컬럼 찾기 (Boolean도 포함)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # 숫자형 컬럼 중에서 0/1만 있는 이진 데이터 찾기 (DIO 신호)
    binary_cols = []
    for col in numeric_cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
            binary_cols.append(col)
    
    # Boolean 컬럼을 숫자형으로 변환하여 추가
    df_plot = df.copy()
    converted_cols = []
    
    if bool_cols:
        for col in bool_cols:
            df_plot[col] = df_plot[col].astype(float)  # True=1.0, False=0.0
            if col not in numeric_cols:
                numeric_cols.append(col)
                converted_cols.append(col)
    
    # 정보 메시지 표시
    if converted_cols or binary_cols:
        info_parts = []
        if converted_cols:
            info_parts.append(f"Boolean: {', '.join(converted_cols)}")
        if binary_cols:
            info_parts.append(f"이진(0/1): {', '.join(binary_cols)}")
        # st.info(f"ℹ️ DIO 신호 감지 → {' | '.join(info_parts)} → 플롯 가능")
    
    if not numeric_cols:
        st.warning("⚠️ 숫자형 또는 Boolean 컬럼이 없습니다.")
        return
    
    # 시각화 타입 선택
    viz_type = st.radio(
        "시각화 타입",
        options=['시계열 그래프', '산점도', '히스토그램', '박스플롯', '상관관계 히트맵'],
        horizontal=True
    )
    
    if viz_type == '시계열 그래프':
        render_timeseries_plot(df_plot, numeric_cols, datetime_cols)
    elif viz_type == '산점도':
        render_scatter_plot(df_plot, numeric_cols)
    elif viz_type == '히스토그램':
        render_histogram(df_plot, numeric_cols)
    elif viz_type == '박스플롯':
        render_boxplot(df_plot, numeric_cols)
    else:  # 상관관계 히트맵
        render_correlation_heatmap(df_plot, numeric_cols)
