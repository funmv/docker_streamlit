"""
YAML 설정 UI 탭
"""
import streamlit as st
from datetime import datetime
from utils.yaml_utils import dict_to_yaml_string, load_yaml_file, get_default_config


def render_config_tab():
    """YAML 설정 탭 렌더링"""
    st.header("📋 YAML 설정")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 기본값으로 초기화", use_container_width=True):
            st.session_state.config = get_default_config()
            st.session_state['yaml_loaded'] = False
            st.rerun()
    
    with col2:
        uploaded_yaml = st.file_uploader("📥 YAML 파일 불러오기", type=['yaml', 'yml'], key='yaml_upload')
        if uploaded_yaml:
            if st.session_state.get('yaml_loaded', False):
                st.success("✅ YAML 설정이 적용되었습니다.")
            
            col2a, col2b = st.columns(2)
            with col2a:
                if st.button("YAML 적용", key='apply_yaml', type='primary'):
                    try:
                        uploaded_yaml.seek(0)
                        config_data = load_yaml_file(uploaded_yaml)
                        
                        st.session_state.config = config_data
                        st.session_state['yaml_loaded'] = True
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ YAML 파일 로드 실패: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            with col2b:
                if st.button("🔍 미리보기", key='preview_yaml'):
                    try:
                        uploaded_yaml.seek(0)
                        config_data = load_yaml_file(uploaded_yaml)
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
    
    # ===== 파일 정보 =====
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
    
    # ===== CSV 옵션 =====
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
    
    # ===== 시트 설정 =====
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
    
    # ===== 헤더 구조 =====
    with st.expander("📑 헤더 구조", expanded=True):
        header = st.session_state.config.get('header', {})
        
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
        
        if 'header_rows' not in header:
            header['header_rows'] = {}
        header_rows = header['header_rows']
        
        with col1:
            current_desc = 0
            if 'description' in header_rows and header_rows['description'] is not None:
                try:
                    current_desc = int(header_rows['description'])
                except:
                    current_desc = 0
            
            desc_row = st.number_input("설명(Description) 행", min_value=0, 
                                       value=current_desc, step=1)
            if desc_row > 0:
                header_rows['description'] = int(desc_row)
            elif 'description' in header_rows:
                del header_rows['description']
        
        with col2:
            current_unit = 0
            if 'unit' in header_rows and header_rows['unit'] is not None:
                try:
                    current_unit = int(header_rows['unit'])
                except:
                    current_unit = 0
            
            unit_row = st.number_input("단위(Unit) 행", min_value=0, 
                                       value=current_unit, step=1)
            if unit_row > 0:
                header_rows['unit'] = int(unit_row)
            elif 'unit' in header_rows:
                del header_rows['unit']
        
        with col3:
            current_tag = 0
            if 'tag_name' in header_rows and header_rows['tag_name'] is not None:
                try:
                    current_tag = int(header_rows['tag_name'])
                except:
                    current_tag = 0
            
            tag_row = st.number_input("태그명(Tag) 행", min_value=0, 
                                      value=current_tag, step=1)
            if tag_row > 0:
                header_rows['tag_name'] = int(tag_row)
            elif 'tag_name' in header_rows:
                del header_rows['tag_name']
        
        header['header_rows'] = header_rows
        
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
    
    # ===== 타임스탬프 처리 =====
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
        
        st.divider()
        st.markdown("#### 🆕 출력 옵션")
        st.caption("저장할 파일에서 타임스탬프 컬럼을 제외할지 선택합니다.")
        
        ts['exclude_from_output'] = st.checkbox(
            "⚠️ 저장 시 타임스탬프 제외 (특징 컬럼만 저장)",
            value=ts.get('exclude_from_output', False),
            help="체크하면 타임스탬프 없이 특징 데이터만 저장됩니다."
        )
        
        if ts['exclude_from_output']:
            st.warning("⚠️ 저장 시 타임스탬프가 제외됩니다.")
        
        st.session_state.config['timestamp'] = ts
    
    # ===== 샘플링 설정 =====
    with st.expander("🎯 샘플링 설정"):
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
                    help="N개 데이터마다 1개를 선택합니다."
                )
            
            with col2:
                sampling['method'] = st.selectbox(
                    "샘플링 방법",
                    options=['every_n', 'mean', 'median', 'first', 'last'],
                    index=['every_n', 'mean', 'median', 'first', 'last'].index(
                        sampling.get('method', 'every_n')
                    ),
                    help="""
• every_n: N개마다 1개 선택
• mean: N개씩 그룹화하여 평균값
• median: N개씩 그룹화하여 중앙값
• first: N개씩 그룹화하여 첫 번째 값
• last: N개씩 그룹화하여 마지막 값
                    """
                )
            
            # 예상 축소율
            estimated_reduction = (1 - 1/sampling['interval']) * 100
            st.success(f"✅ 예상 데이터 축소율: 약 {estimated_reduction:.1f}%")
        
        st.session_state.config['sampling'] = sampling
    
    # ===== 컬럼명 정규화 =====
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
    
    # ===== 데이터 타입 =====
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
    
    # ===== 후처리 =====
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
    
    # ===== 에러 처리 =====
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
    
    # ===== 현재 설정 미리보기 =====
    with st.expander("👁️ 현재 설정 미리보기"):
        st.code(dict_to_yaml_string(st.session_state.config), language='yaml')