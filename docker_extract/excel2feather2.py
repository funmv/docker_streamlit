import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from datetime import datetime
import time
import stat
import shutil
import gc 
import matplotlib.pyplot as plt 
import glob
import io

# =====================================
# 반드시 첫 번째 Streamlit 명령어!
# =====================================
st.set_page_config(
    page_title="데이터 변환 도구", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# 한글 폰트 설정
# =====================================
def setup_korean_font():
    """한글 폰트 설정 함수"""
    try:
        from matplotlib import font_manager, rc
        # Windows 환경
        try:
            font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
            rc('font', family=font_name)
            plt.rcParams['axes.unicode_minus'] = False
            return "Windows 폰트 로드 성공"
        except:
            # Linux 환경
            try:
                font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                rc('font', family=font_name)  
                plt.rcParams['axes.unicode_minus'] = False
                return "Linux 폰트 로드 성공"
            except:
                # 폰트 로드 실패 시 기본 폰트 사용
                plt.rcParams['axes.unicode_minus'] = False
                return "기본 폰트 사용"
    except Exception as e:
        return f"폰트 설정 오류: {e}"

# 폰트 설정 실행
setup_korean_font()

# matplotlib 경고 제거를 위한 설정
plt.rcParams['figure.max_open_warning'] = 50

# ✅ FutureWarning 해결을 위한 설정
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass

# =====================================
# 공통 유틸리티 함수들
# =====================================
def safe_rmtree(path):
    """안전한 폴더 삭제"""
    def handle_readonly(func, path, exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    
    if os.path.exists(path):
        try:
            shutil.rmtree(path, onerror=handle_readonly)
        except OSError:
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        os.chmod(file_path, stat.S_IWRITE)
                        os.unlink(file_path)
                    except:
                        pass

def remove_korean_and_special_chars(text):
    """한글, 특수문자 및 공백 제거 (영문, 숫자, -, _ 만 허용)"""
    import re
    return re.sub(r'[^a-zA-Z0-9\-_]', '', text)

# =====================================
# 사이드바 공통 함수
# =====================================
def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown(
            """
            <style>
            .bottom-info {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 21rem;
                max-width: 21rem;
                background-color: var(--background-color);
                padding: 1rem;
                border-top: 1px solid var(--border-color);
                z-index: 999;
                box-sizing: border-box;
            }
            .bottom-info hr {
                margin: 0.2rem 0;
                border-color: var(--text-color-light);
                width: 100%;
            }
            </style>
            <div class="bottom-info">
                <hr>
                🧠 <strong>회사명:</strong> ㈜파시디엘<br>
                🏫 <strong>연구실:</strong> visLAB@PNU<br>
                👨‍💻 <strong>제작자:</strong> (C)Dong2<br>
                🛠️ <strong>버전:</strong> V.1.3 (06-03-2025)<br>
                <hr>
            </div>
            """, 
            unsafe_allow_html=True
        )

# =====================================
# Excel 관련 함수들
# =====================================
def identify_date_columns(df, date_column_name):
    """날짜 컬럼들을 식별하는 함수"""
    date_columns = set()
    
    if date_column_name and date_column_name in df.columns:
        date_columns.add(date_column_name)
    
    date_keywords = ['date', 'time', 'datetime', 'timestamp', '날짜', '시간', '일시']
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in date_keywords):
            date_columns.add(col)
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.add(col)
    
    return list(date_columns)

def extract_unique_text_values(df, date_columns):
    """DataFrame에서 모든 문자값들과 빈도를 추출하는 함수"""
    text_frequency = {}
    text_columns_dict = {}
    
    columns_to_process = [col for col in df.columns if col not in date_columns]
    
    for column in columns_to_process:
        text_values = df[column].dropna().astype(str)
        
        for value in text_values:
            try:
                float(value)
            except (ValueError, TypeError):
                if value not in ['nan', 'None', '']:
                    if value not in text_frequency:
                        text_frequency[value] = 0
                        text_columns_dict[value] = set()
                    
                    value_count = (text_values == value).sum()
                    text_frequency[value] += value_count
                    text_columns_dict[value].add(column)
    
    return text_frequency, text_columns_dict

def extract_date_info_from_data(df, date_column):
    """데이터에서 날짜 정보를 추출하여 파일명용 문자열 생성"""
    try:
        if date_column in df.columns:
            date_series = pd.to_datetime(df[date_column], errors='coerce')
            date_series = date_series.dropna()
            
            if len(date_series) > 0:
                start_date = date_series.min()
                end_date = date_series.max()
                
                try:
                    start_str = start_date.strftime("%Y%m%d_%H%M")
                    end_str = end_date.strftime("%Y%m%d_%H%M")
                    
                    if start_str == end_str:
                        return f"_{start_str}"
                    else:
                        start_date_only = start_date.strftime("%Y%m%d")
                        end_date_only = end_date.strftime("%Y%m%d")
                        
                        if start_date_only == end_date_only:
                            return f"_{start_date_only}"
                        else:
                            return f"_{start_date_only}_{end_date_only}"
                            
                except Exception:
                    start_str = start_date.strftime("%Y%m%d")
                    end_str = end_date.strftime("%Y%m%d")
                    
                    if start_str == end_str:
                        return f"_{start_str}"
                    else:
                        return f"_{start_str}_{end_str}"
        
        return ""
    except Exception:
        return ""

def apply_text_mapping(df, mapping_dict, date_columns):
    """DataFrame에 문자-숫자 매핑을 적용하는 함수"""
    df_converted = df.copy()
    
    columns_to_process = [col for col in df_converted.columns if col not in date_columns]
    
    for column in columns_to_process:
        for text, number in mapping_dict.items():
            try:
                df_converted[column] = df_converted[column].replace(text, number).infer_objects(copy=False)
            except (AttributeError, TypeError):
                df_converted[column] = df_converted[column].replace(text, number)
        
        try:
            df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
        except:
            pass
    
    for date_col in date_columns:
        if date_col in df_converted.columns:
            try:
                df_converted[date_col] = pd.to_datetime(df_converted[date_col], errors='coerce')
            except:
                pass
    
    return df_converted

@st.cache_data(show_spinner=False, max_entries=3, ttl=300)
def load_excel(file, sheet_name, usecols, nrows, date_column, skiprows, skip_next=0):
    """Excel 파일을 읽는 함수"""
    
    if file is None:
        raise ValueError("파일이 None입니다.")
    
    if sheet_name is None or sheet_name == "":
        raise ValueError("시트명이 유효하지 않습니다.")

    def count_rows():    
        if ':' in usecols:
            first_col = usecols.split(':')[0]
        else:
            first_col = usecols

        range_col = f"{first_col}:{first_col}"
        temp_df = pd.read_excel(
            file,
            sheet_name=sheet_name,
            skiprows=skiprows + 1 + skip_next,
            header=None,
            usecols=range_col,  
            engine='openpyxl',
        )
        mask = temp_df.iloc[:, 0].notna() & (temp_df.iloc[:, 0].astype(str).str.strip() != '')
        temp_count = mask.sum()
        final_nrows = temp_count - 10 if temp_count > 10 else temp_count
        return final_nrows

    try:
        if skip_next > 0:
            header_df = pd.read_excel(
                file,
                sheet_name=sheet_name,
                skiprows=skiprows,
                nrows=1,
                usecols=usecols,
                engine='openpyxl'
            )   
            
            column_names = header_df.columns.tolist()
            
            try:
                final_nrows = count_rows()
            except Exception:
                final_nrows = nrows
            
            data_df = pd.read_excel(
                file,
                sheet_name=sheet_name,
                skiprows=skiprows + 1 + skip_next,
                header=None,
                usecols=usecols,
                nrows=final_nrows,
                engine='openpyxl',
                na_values=['I/O Timeout', 'Configure', 'Not Connect', 'Bad', 'Comm Fail']
            )
            
            data_df.columns = column_names
            
        else:
            try:
                final_nrows = count_rows()
            except Exception:
                final_nrows = nrows

            data_df = pd.read_excel(
                file,
                sheet_name=sheet_name,
                usecols=usecols,
                nrows=final_nrows,
                skiprows=skiprows,
                header=0,
                engine='openpyxl',
                na_values=['I/O Timeout', 'Configure', 'Not Connect', 'Bad', 'Comm Fail']
            )
        
        if date_column and date_column in data_df.columns:
            try:
                data_df[date_column] = pd.to_datetime(data_df[date_column], errors='coerce')
            except Exception:
                pass
        
        gc.collect()
        
        return data_df
        
    except Exception as e:
        gc.collect()
        raise RuntimeError(f"Excel 읽기 실패: {str(e)}")

# =====================================
# CSV 관련 함수들
# =====================================
@st.cache_data(show_spinner=False)
def replace_999_with_neighbors_mean(df):
   """999 값을 인접값의 평균으로 대체"""
   df = df.copy()
   for col in df.columns:
       values = df[col].values
       for i in range(1, len(values) - 1):
           if values[i] == 999:
               if values[i - 1] != 999 and values[i + 1] != 999:
                   values[i] = (values[i - 1] + values[i + 1]) / 2
               else:
                   values[i] = np.nan
       df[col] = values
   return df

def get_feather_files_in_directory():
    """저장 디렉토리에서 모든 .ftr 파일 찾기"""
    default_root = "/app/data" if os.path.exists("/app/data") else os.getcwd()
    feather_files = []
    
    try:
        for file in os.listdir(default_root):
            if file.endswith(".ftr"):
                file_path = os.path.join(default_root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                feather_files.append({
                    "파일명": file,
                    "경로": file_path,
                    "크기(MB)": f"{file_size:.2f}"
                })
    except Exception as e:
        st.error(f"파일 목록을 가져올 수 없습니다: {e}")
    
    return feather_files

def create_zip_download(file_paths):
    """여러 파일을 ZIP으로 압축하여 다운로드 준비"""
    import zipfile
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in file_paths:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                zip_file.write(file_path, file_name)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# =====================================
# 탭 1: Excel → Feather 변환기
# =====================================
def tab_excel_converter():
    """탭 1: Excel → Feather 변환기"""
    st.header("📊 대용량 Excel → Feather 변환기")
    
    # 세션 상태 초기화
    session_keys = ['text_mapping', 'text_frequency', 'text_columns', 'date_columns', 
                   'file_loaded', 'df', 'raw_df']
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = {} if key in ['text_mapping', 'text_frequency', 'text_columns'] else []

    uploaded_file = st.file_uploader("📂 Excel 파일 업로드 (.xlsx)", type=["xlsx", "xls"])

    # Excel 읽기 설정
    with st.expander("🔧 Excel 읽기 설정", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            usecols = st.text_input("읽을 컬럼 범위 (usecols)", value="A:CY")
            skiprows = st.number_input("건너뛸 행 수 (skiprows)", min_value=0, value=3)
        
        with col2:
            date_column = st.text_input("날짜 컬럼명", value="Description")
            skip_next = st.number_input("헤드행 다음 건너뛸 수", min_value=0, value=2)
        
        with col3:
            nrows = st.number_input("읽을 행 수 (nrows)", min_value=1000, max_value=10**7, step=10000, value=3000)

    # 시트 선택
    sheet_name = None
    available_sheets = []

    if uploaded_file is not None:
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            available_sheets = excel_file.sheet_names
            
            if len(available_sheets) == 1:
                sheet_name = available_sheets[0]
                st.success(f"📋 시트 자동 선택: {sheet_name}")
            elif len(available_sheets) > 1:
                sheet_name = st.selectbox("읽을 시트를 선택하세요:", options=available_sheets, index=0)
                st.info(f"총 {len(available_sheets)}개 시트 중 '{sheet_name}' 선택됨")
            else:
                st.error("❌ 시트를 찾을 수 없습니다.")
                
        except Exception as e:
            st.error(f"❌ 시트 정보를 읽을 수 없습니다: {e}")

    # Excel 로딩
    if uploaded_file is not None and sheet_name is not None and st.button("Excel 읽기"):
        progress_container = st.container()
        status_container = st.container()
        
        try:
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🚀 Excel 파일 읽기 시작...")
                progress_bar.progress(10)
                time.sleep(0.1)
                
                status_text.text(f"📖 시트 '{sheet_name}' 처리 중...")
                progress_bar.progress(30)
                
                df = load_excel(
                    file=uploaded_file,
                    sheet_name=sheet_name,
                    usecols=usecols,
                    nrows=nrows,
                    date_column=date_column,
                    skiprows=skiprows,   
                    skip_next=skip_next   
                )
                
                progress_bar.progress(50)
                status_text.text("🗓️ 날짜 컬럼 식별 중...")
                
                date_columns = identify_date_columns(df, date_column)
                st.session_state.date_columns = date_columns
                
                progress_bar.progress(60)
                status_text.text("🔍 데이터 분석 중...")
                
                if df is not None and len(df) > 0:
                    st.session_state.raw_df = df.copy()
                    
                    progress_bar.progress(80)
                    status_text.text("🔤 문자값 추출 중...")
                    
                    text_frequency, text_columns_dict = extract_unique_text_values(df, date_columns)
                    st.session_state.text_frequency = text_frequency
                    st.session_state.text_columns = text_columns_dict
                    
                    st.session_state.df = df
                    st.session_state.file_loaded = True
                    
                    progress_bar.progress(90)
                    status_text.text("📝 파일명 생성 중...")
                    
                    base_filename = os.path.splitext(uploaded_file.name)[0]
                    date_info = extract_date_info_from_data(df, date_column)
                    sheet_info = f"_{sheet_name}" if sheet_name != "Sheet1" else ""
                    enhanced_filename = f"{base_filename}{sheet_info}{date_info}"
                    st.session_state.last_filename = enhanced_filename
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 완료!")
                    
                    time.sleep(0.5)
                    progress_container.empty()
                
            with status_container:
                st.success(f"🎉 Excel 파일 읽기 완료!")
                st.info(f"📊 데이터 크기: {len(df):,}행 × {len(df.columns)}열")
                
                if date_columns:
                    st.success(f"🗓️ 식별된 날짜 컬럼: {date_columns}")
                    st.info("📌 이 컬럼들은 문자-숫자 변환에서 자동으로 제외됩니다.")
                
                if text_frequency:
                    st.warning(f"🔤 발견된 문자값들: {list(text_frequency.keys())}")
                    st.info("👇 아래에서 각 문자값에 대응할 숫자를 설정하세요.")
                else:
                    st.success("✅ 모든 데이터가 이미 숫자 형태이거나 날짜 형태입니다.")
                    
                gc.collect()
                
        except Exception as e:
            progress_container.empty()
            st.error(f"❌ 파일 로딩 중 오류 발생!")
            st.error(f"🔍 오류 상세: {str(e)}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ 캐시 클리어"):
                    st.cache_data.clear()
                    st.rerun()
            with col2:
                if st.button("🔄 페이지 새로고침"):
                    st.rerun()

    # 문자-숫자 매핑 설정
    if st.session_state.get('file_loaded', False) and st.session_state.get('text_frequency', {}):
        st.markdown("---")
        st.subheader("🔢 문자-숫자 변환 설정")
        
        if st.session_state.date_columns:
            st.info(f"🛡️ 보호되는 날짜 컬럼: {st.session_state.date_columns}")
            st.caption("이 컬럼들은 문자-숫자 변환에서 자동으로 제외되어 원본 형태를 유지합니다.")
        
        st.write("**발견된 문자값들과 빈도 정보:**")
        
        frequency_data = []
        for text, freq in st.session_state.text_frequency.items():
            columns_list = list(st.session_state.text_columns[text])
            frequency_data.append({
                "문자값": text,
                "빈도": f"{freq:,}",
                "출현 컬럼": ", ".join(columns_list) if len(columns_list) <= 3 else f"{', '.join(columns_list[:3])}... (총 {len(columns_list)}개)"
            })
        
        frequency_df = pd.DataFrame(frequency_data).sort_values('빈도')
        st.dataframe(frequency_df, use_container_width=True)
        
        number_options = [0, 1, 5, 10, 25, 50, 75, 100]
        
        st.markdown("---")
        st.write("**숫자 매핑 설정:**")
        
        mapping_dict = {}
        sorted_texts = sorted(st.session_state.text_frequency.keys(), 
                             key=lambda x: st.session_state.text_frequency[x], reverse=True)
        
        cols = st.columns(2)
        for i, text in enumerate(sorted_texts):
            with cols[i % 2]:
                if text.upper() in ['OFF', 'STOP', 'FALSE', '0']:
                    default_idx = 0
                elif text.upper() in ['ON', 'RUNNING', 'TRUE', '1']:
                    default_idx = 1
                else:
                    default_idx = 1
                
                freq = st.session_state.text_frequency[text]
                selected_number = st.selectbox(
                    f"{text} ({freq}회) → ",
                    options=number_options,
                    index=default_idx,
                    key=f"mapping_{text}"
                )
                mapping_dict[text] = selected_number
        
        if st.button("🔄 문자-숫자 변환 적용"):
            try:
                converted_df = apply_text_mapping(st.session_state.raw_df, mapping_dict, st.session_state.date_columns)
                st.session_state.df = converted_df
                st.session_state.text_mapping = mapping_dict
                
                st.success("✅ 문자-숫자 변환이 완료되었습니다!")
                
                st.write("**적용된 매핑:**")
                for text, number in mapping_dict.items():
                    st.write(f"• {text} → {number}")
                
                numeric_columns = st.session_state.df.select_dtypes(include=[np.number]).columns
                datetime_columns = st.session_state.df.select_dtypes(include=['datetime64']).columns
                
                st.info(f"✅ 숫자 컬럼 수: {len(numeric_columns)}/{len(st.session_state.df.columns)}")
                st.info(f"🗓️ 날짜 컬럼 수: {len(datetime_columns)}/{len(st.session_state.df.columns)}")
                
            except Exception as e:
                st.error(f"❌ 변환 중 오류: {e}")

    # 메인 데이터 처리 섹션
    if st.session_state.get('file_loaded', False) and st.session_state.get('df') is not None:
        df = st.session_state.df
        st.markdown("---")
        st.success("✅ Excel 로딩 완료!")

        # 컬럼 리스트 표시
        st.subheader("🧾 컬럼 리스트")
        with st.expander(f"전체 컬럼 보기 (총 {len(df.columns)}개)", expanded=True):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            col_info = []
            for i, col in enumerate(df.columns):
                if col in numeric_cols:
                    col_type = "🔢 숫자"
                elif col in datetime_cols:
                    col_type = "🗓️ 날짜"
                elif col in object_cols:
                    col_type = "🔤 텍스트"
                else:
                    col_type = "❓ 기타"
                
                col_info.append(f"{i+1}. {col} ({col_type})")
            
            st.markdown(
                f"""
                <div style='max-height: 300px; overflow-y: scroll; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9'>
                {"<br>".join(col_info)}
                </div>
                """,
                unsafe_allow_html=True
            )

        # 제거할 컬럼 선택
        st.subheader("🗑️ 제거할 컬럼 선택")
        
        removable_columns = [col for col in df.columns.tolist() if col not in st.session_state.date_columns]
        
        if st.session_state.date_columns:
            st.info(f"🛡️ 날짜 컬럼 {st.session_state.date_columns}은(는) 보호되어 제거 옵션에서 제외됩니다.")
        
        cols_to_drop = st.multiselect(
            "데이터프레임에서 제거할 컬럼을 선택하세요",
            removable_columns,
            help="날짜 컬럼은 자동으로 보호되어 선택 목록에서 제외됩니다."
        )

        if st.button("선택한 컬럼 제거하기"):
            if cols_to_drop:
                safe_cols_to_drop = [col for col in cols_to_drop if col not in st.session_state.date_columns]
                
                if safe_cols_to_drop:
                    df = df.drop(columns=safe_cols_to_drop)
                    st.session_state.df = df
                    st.success(f"✅ 선택한 {len(safe_cols_to_drop)}개 컬럼 제거 완료!")
                    st.info(f"남은 컬럼 수: {len(df.columns)}개")
                    
                    if len(safe_cols_to_drop) != len(cols_to_drop):
                        protected_count = len(cols_to_drop) - len(safe_cols_to_drop)
                        st.warning(f"🛡️ {protected_count}개 날짜 컬럼은 보호되어 제거되지 않았습니다.")
                else:
                    st.warning("🛡️ 선택한 모든 컬럼이 보호된 날짜 컬럼입니다.")
            else:
                st.warning("❗ 제거할 컬럼을 선택하세요.")

        # 변환 전/후 비교
        if st.session_state.text_frequency and 'raw_df' in st.session_state:
            st.markdown("---")
            st.subheader("👀 데이터 변환 비교")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**변환 전 (원본):**")
                st.dataframe(st.session_state.raw_df.head(), use_container_width=True)
            
            with col2:
                st.write("**변환 후 (숫자/날짜):**")
                st.dataframe(df.head(), use_container_width=True)
                
            if st.session_state.date_columns:
                st.info("🛡️ 날짜 컬럼들은 원본 형태를 유지하며 변환되지 않았습니다.")

        # 저장 설정
        st.markdown("---")
        st.subheader("💾 Feather 저장 설정")
        default_root = "/app/data" if os.path.exists("/app/data") else os.getcwd()

        default_filename = st.session_state.get('last_filename', 'data')
        save_name = st.text_input(
            "📄 저장 파일명 (확장자 제외)", 
            value=default_filename,
            help="날짜 정보와 시트명이 자동으로 포함됩니다"
        )    

        # FTR 저장 버튼
        if st.button("💾 Feather로 저장하기"):
            save_path = os.path.join(default_root, save_name + ".ftr")
            try:
                if df is None or df.empty:
                    st.error("❌ 저장할 데이터가 비어있습니다.")
                else:
                    df_to_save = df.reset_index(drop=True)
                    df_to_save.to_feather(save_path)
                    st.success(f"✅ Feather 파일로 저장 완료:\n`{save_path}`")
                    st.info(f"📊 저장된 데이터: {len(df_to_save)}행 × {len(df_to_save.columns)}열")
                    
                    if st.session_state.date_columns:
                        preserved_date_cols = [col for col in st.session_state.date_columns if col in df_to_save.columns]
                        if preserved_date_cols:
                            st.success(f"🗓️ 보존된 날짜 컬럼: {preserved_date_cols}")
                            
            except Exception as e:
                st.error(f"❌ 저장 실패: {e}")

        # Feather 다운로드 기능
        st.subheader("💾 Feather 파일 다운로드")

        default_download_filename = st.session_state.get('last_filename', 'ftr_data')
            
        download_name = st.text_input(
            "📄 다운로드할 파일명 (확장자 제외)", 
            value=default_download_filename,
            help="날짜 정보와 시트명이 자동으로 포함됩니다"
        )

        try:
            buffer = io.BytesIO()
            
            if df is None or df.empty:
                st.error("❌ 데이터가 비어있습니다.")
            else:
                df_to_save = df.reset_index(drop=True)
                df_to_save.to_feather(buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="📥 Feather 파일 다운로드",
                    data=buffer.getvalue(),
                    file_name=f"{download_name}.ftr",
                    mime="application/octet-stream"
                )
                
                st.success(f"✅ 다운로드 준비 완료: {download_name}.ftr")
                
                if st.session_state.date_columns:
                    preserved_date_cols = [col for col in st.session_state.date_columns if col in df_to_save.columns]
                    if preserved_date_cols:
                        st.info(f"🗓️ 다운로드 파일에 보존된 날짜 컬럼: {preserved_date_cols}")
            
        except Exception as e:
            st.error(f"❌ 다운로드 준비 실패: {e}")

        # 고속 Plotly 시각화
        st.markdown("---")
        st.subheader("⚡ Plotly 고속 시각화 (WebGL)")
        num_cols = df.select_dtypes(include='number').columns
        
        if len(num_cols) > 0:
            visualizable_columns = [col for col in df.columns.tolist() 
                                   if col not in st.session_state.date_columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if visualizable_columns:
                selected_columns = st.multiselect(
                    '시각화할 숫자 컬럼을 선택하세요',
                    visualizable_columns,
                    default=visualizable_columns[:2] if len(visualizable_columns) >= 2 else visualizable_columns[:1]
                )            
                
                if selected_columns:
                    downsample_rate = st.slider("📉 다운샘플 비율 (1/N)", 1, 50, 10)
                    
                    fig = go.Figure()
                    
                    for col in selected_columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            downsampled_y = df[col][::downsample_rate]
                            fig.add_trace(go.Scattergl(
                                y=downsampled_y,
                                mode='lines',
                                name=str(col)
                            ))
                    
                    fig.update_layout(
                        title=f"선택한 컬럼 시각화 (1/{downsample_rate} 다운샘플링)",
                        xaxis=dict(rangeslider=dict(visible=False)),
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if st.session_state.date_columns:
                        available_date_cols = [col for col in st.session_state.date_columns if col in df.columns]
                        if available_date_cols:
                            st.info("💡 시간축 시각화를 원하면 아래에서 날짜 컬럼을 X축으로 선택할 수 있습니다.")
                            
                            use_date_axis = st.checkbox("🗓️ 날짜를 X축으로 사용")
                            if use_date_axis:
                                date_col_for_x = st.selectbox("X축으로 사용할 날짜 컬럼 선택:", available_date_cols)
                                
                                if date_col_for_x and selected_columns:
                                    fig_time = go.Figure()
                                    
                                    for col in selected_columns:
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            downsampled_df = df.iloc[::downsample_rate]
                                            fig_time.add_trace(go.Scattergl(
                                                x=downsampled_df[date_col_for_x],
                                                y=downsampled_df[col],
                                                mode='lines',
                                                name=str(col)
                                            ))
                                    
                                    fig_time.update_layout(
                                        title=f"시간축 시각화 (X: {date_col_for_x})",
                                        xaxis_title=date_col_for_x,
                                        yaxis_title="값",
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        height=400
                                    )
                                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.info("시각화할 숫자 컬럼을 선택하세요.")
            else:
                st.info("시각화 가능한 숫자 컬럼이 없습니다. (날짜 컬럼은 제외됨)")
        else:
            st.info("숫자형 컬럼이 없습니다.")

        # 모든 시트에 일괄 적용 기능
        st.markdown("---")
        st.header("🔄 모든 시트에 일괄 적용")
        
        if uploaded_file is not None:
            st.subheader("📋 현재 적용할 설정 요약")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔢 문자-숫자 매핑 설정:**")
                if st.session_state.text_mapping:
                    for text, number in st.session_state.text_mapping.items():
                        st.write(f"• {text} → {number}")
                else:
                    st.write("설정된 매핑이 없습니다.")
                    
                st.write("**🛡️ 보호되는 날짜 컬럼:**")
                if st.session_state.date_columns:
                    for col in st.session_state.date_columns:
                        st.write(f"• {col}")
                else:
                    st.write("식별된 날짜 컬럼이 없습니다.")
            
            with col2:
                st.write("**🗑️ 제거할 컬럼 설정:**")
                if 'cols_to_drop' in locals() and cols_to_drop:
                    for col in cols_to_drop:
                        st.write(f"• {col}")
                else:
                    st.write("제거할 컬럼이 없습니다.")
            
            if len(available_sheets) > 0:
                st.write(f"**📋 처리 대상 시트 ({len(available_sheets)}개)**")
            else:
                st.error("❌ 처리할 수 있는 시트가 없습니다.")
                
            if len(available_sheets) > 1:
                st.markdown("---")
                if st.button("🚀 모든 시트에 동일한 처리 적용 및 저장", type="primary"):
                    
                    current_mapping = st.session_state.text_mapping.copy()
                    current_cols_to_drop = cols_to_drop.copy() if 'cols_to_drop' in locals() else []
                    current_date_columns = st.session_state.date_columns.copy()
                    
                    st.info(f"🔄 {len(available_sheets)}개 시트에 일괄 처리를 시작합니다...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    success_files = []
                    error_files = []

                    default_root = "/app/data" if os.path.exists("/app/data") else os.getcwd()
                    for file_path in glob.glob(os.path.join(default_root, "*")):
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print(f'삭제 실패 {file_path}: {e}')

                    
                    for idx, target_sheet in enumerate(available_sheets):
                        try:
                            status_text.text(f"처리 중: {target_sheet} ({idx+1}/{len(available_sheets)})")
                            
                            sheet_df = load_excel(
                                file=uploaded_file,
                                sheet_name=target_sheet,
                                usecols=usecols,
                                nrows=nrows,
                                date_column=date_column,
                                skiprows=skiprows,
                                skip_next=skip_next
                            )
                            
                            sheet_date_columns = identify_date_columns(sheet_df, date_column)
                            
                            if current_mapping:
                                sheet_df = apply_text_mapping(sheet_df, current_mapping, sheet_date_columns)
                            
                            if current_cols_to_drop:
                                cols_to_remove = [col for col in current_cols_to_drop 
                                                if col in sheet_df.columns and col not in sheet_date_columns]
                                if cols_to_remove:
                                    sheet_df = sheet_df.drop(columns=cols_to_remove)

                            base_filename = os.path.splitext(uploaded_file.name)[0]
                            date_info = extract_date_info_from_data(sheet_df, date_column)
                            sheet_info = f"_{target_sheet}" if target_sheet != "Sheet1" else ""

                            safe_base = remove_korean_and_special_chars(base_filename)
                            safe_sheet = remove_korean_and_special_chars(sheet_info)
                            safe_date_info = remove_korean_and_special_chars(date_info)

                            timestamp_suffix = str(int(time.time() * 1000))[-6:]
                            final_filename = f"{safe_date_info}_{timestamp_suffix}"
                            
                            save_path = os.path.join(default_root, final_filename + ".ftr")
                            
                            sheet_df.reset_index(drop=True).to_feather(save_path)
                            success_files.append((target_sheet, save_path, len(sheet_df), sheet_date_columns))
                            
                        except Exception as e:
                            error_files.append((target_sheet, str(e)))
                        
                        progress_bar.progress((idx + 1) / len(available_sheets))
                    
                    status_text.text("처리 완료!")
                    
                    if success_files:
                        st.success(f"✅ {len(success_files)}개 시트 처리 완료!")
                        
                        success_data = []
                        for sheet, path, rows, date_cols in success_files:
                            success_data.append({
                                "시트명": sheet,
                                "행 수": f"{rows:,}",
                                "보호된 날짜 컬럼": ", ".join(date_cols) if date_cols else "없음",
                                "파일 경로": path
                            })
                        
                        st.dataframe(pd.DataFrame(success_data), use_container_width=True)
                    
                    if error_files:
                        st.error(f"❌ {len(error_files)}개 시트 처리 실패:")
                        for sheet, error in error_files:
                            st.write(f"• {sheet}: {error}")
            
            else:
                st.info("💡 시트가 1개뿐이므로 일괄 처리 기능이 필요하지 않습니다.")
        
        # 저장된 모든 Feather 파일 다운로드 기능
        st.markdown("---")
        st.header("📦 저장된 Feather 파일 일괄 다운로드")
        
        if st.button("🔍 저장된 Feather 파일 조회"):
            feather_files = get_feather_files_in_directory()
            
            if feather_files:
                st.write(f"**📁 발견된 Feather 파일 ({len(feather_files)}개):**")
                
                files_df = pd.DataFrame(feather_files)
                st.dataframe(files_df, use_container_width=True)
                
                total_size = sum(float(f["크기(MB)"]) for f in feather_files)
                st.info(f"📊 총 파일 크기: {total_size:.2f} MB")
                
                if len(feather_files) > 0:
                    try:
                        file_paths = [f["경로"] for f in feather_files]
                        zip_data = create_zip_download(file_paths)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        zip_filename = f"feather_files_{timestamp}.zip"
                        
                        st.download_button(
                            label=f"📥 모든 Feather 파일 다운로드 ({len(feather_files)}개 파일)",
                            data=zip_data,
                            file_name=zip_filename,
                            mime="application/zip"
                        )
                        
                        st.success(f"✅ {len(feather_files)}개 파일이 ZIP으로 압축되어 다운로드 준비됨")
                        st.info("🛡️ 모든 파일의 날짜 컬럼이 원본 형태로 보존되었습니다.")
                        
                    except Exception as e:
                        st.error(f"❌ ZIP 생성 실패: {e}")
            else:
                st.warning("📂 저장된 Feather 파일이 없습니다.")
                st.info("💡 먼저 'Feather로 저장하기' 또는 '모든 시트에 동일한 처리 적용'을 실행하세요.")

    else:
        st.info("📂 .xlsx 파일을 먼저 업로드해주세요.")

# =====================================
# 탭 2: CSV → Feather 변환기
# =====================================
def tab_csv_converter():
    """탭 2: CSV → Feather 변환기"""
    st.header("📊 CSV 파일 병합 및 Feather 변환 도구")
    st.write("여러 CSV 파일을 읽어서 시간 인덱스를 조정하고, 노이즈를 제거한 후 Feather 파일로 변환합니다.")

    # 파일 업로드 UI
    st.subheader("📁 CSV 파일 업로드")
    st.write("처리할 CSV 파일들을 선택하여 업로드하세요. (Ctrl 또는 Shift 키를 사용해 여러 파일 선택 가능)")

    uploaded_files = st.file_uploader("CSV 파일 선택", type="csv", accept_multiple_files=True, key="csv_uploader")

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)}개의 CSV 파일이 업로드되었습니다.")
        
        with st.expander("업로드된 파일 목록", expanded=False):
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size} bytes)")
        
        # 임시 폴더 생성
        folder_path = os.path.join(os.getcwd(), "data", "temp_uploads")

        st.session_state['folder_path'] = folder_path
        st.session_state['uploaded_files_count'] = len(uploaded_files)
        st.session_state['uploaded_files_names'] = [file.name for file in uploaded_files]
        st.session_state['files_uploaded'] = True

        if os.path.exists(folder_path):
            safe_rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        
        for uploaded_file in uploaded_files:
            with open(os.path.join(folder_path, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.info(f"업로드된 파일이 임시 폴더에 저장되었습니다: {folder_path}")
        
        csv_files = [file.name for file in uploaded_files]
        st.session_state['csv_files'] = csv_files

        if os.path.exists(folder_path):
            actual_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            st.session_state['actual_csv_files'] = actual_files
            st.write(f"📁 실제 저장된 CSV 파일 수: {len(actual_files)}")

        # CSV 파일 분석
        st.subheader("📊 CSV 파일 분석")
        
        selected_file = st.selectbox(
            "헤더를 확인할 파일 선택",
            csv_files,
            help="선택한 파일의 헤더와 미리보기를 표시합니다",
            key="csv_analysis_selector"
        )
        
        if selected_file:
            try:
                selected_file_path = os.path.join(folder_path, selected_file)
                df_preview = pd.read_csv(selected_file_path, nrows=100)
                
                st.write(f"**선택된 파일: {selected_file}**")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**📋 컬럼 헤더:**")
                    headers_df = pd.DataFrame({
                        "컬럼명": df_preview.columns,
                        "데이터 타입": df_preview.dtypes.astype(str),
                        "샘플 값": [str(df_preview[col].iloc[0]) if not df_preview[col].empty else "N/A" 
                                  for col in df_preview.columns]
                    })
                    st.dataframe(headers_df, use_container_width=True)
                
                with col2:
                    st.write("**📈 파일 정보:**")
                    st.write(f"- 총 컬럼 수: {len(df_preview.columns)}")
                    st.write(f"- 미리보기 행 수: {len(df_preview)}")
                    st.write(f"- 파일 크기: {os.path.getsize(selected_file_path):,} bytes")
                
                st.write("**🔍 데이터 미리보기 (상위 5행):**")
                st.dataframe(df_preview.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ 파일 읽기 오류: {str(e)}")

        # 시간 설정부
        df_head = None
        for csv_file in csv_files:
            try:
                file_path = os.path.join(folder_path, csv_file)
                df_head = pd.read_csv(file_path, nrows=100)
                break
            except Exception as e:
                st.warning(f"⚠️ 파일 읽기 중 오류 발생: {csv_file}\n{str(e)}")

        if df_head is not None:
            st.subheader("시간 컬럼 설정")
            col1, col2, col3 = st.columns(3)

            with col1:
                year_col = st.selectbox("Year 컬럼명", options=[None] + list(df_head.columns), 
                                       index=0 if "year" not in df_head.columns else list(df_head.columns).index("year")+1,
                                       key="year_col_select")
                if year_col is None:
                    year_col = st.text_input("Year 컬럼명 직접 입력", value="year", key="year_col_input")
                month_col = st.selectbox("Month 컬럼명", options=[None] + list(df_head.columns), 
                                        index=0 if "month" not in df_head.columns else list(df_head.columns).index("month")+1,
                                        key="month_col_select")
                if month_col is None:
                    month_col = st.text_input("Month 컬럼명 직접 입력", value="month", key="month_col_input")

            with col2:
                day_col = st.selectbox("Day 컬럼명", options=[None] + list(df_head.columns), 
                                      index=0 if "day" not in df_head.columns else list(df_head.columns).index("day")+1,
                                      key="day_col_select")
                if day_col is None:
                    day_col = st.text_input("Day 컬럼명 직접 입력", value="day", key="day_col_input")
                hour_col = st.selectbox("Hour 컬럼명", options=[None] + list(df_head.columns), 
                                       index=0 if "hour" not in df_head.columns else list(df_head.columns).index("hour")+1,
                                       key="hour_col_select")
                if hour_col is None:
                    hour_col = st.text_input("Hour 컬럼명 직접 입력", value="hour", key="hour_col_input")

            with col3:
                minute_col = st.selectbox("Minute 컬럼명", options=[None] + list(df_head.columns), 
                                         index=0 if "minute" not in df_head.columns else list(df_head.columns).index("minute")+1,
                                         key="minute_col_select")
                if minute_col is None:
                    minute_col = st.text_input("Minute 컬럼명 직접 입력", value="minute", key="minute_col_input")
                second_col = st.selectbox("Second 컬럼명", options=[None] + list(df_head.columns), 
                                         index=0 if "second" not in df_head.columns else list(df_head.columns).index("second")+1,
                                         key="second_col_select")
                if second_col is None:
                    second_col = st.text_input("Second 컬럼명 직접 입력", value="second", key="second_col_input")
                    
            time_columns = {
                'year': year_col,
                'month': month_col, 
                'day': day_col,
                'hour': hour_col,
                'minute': minute_col,
                'second': second_col
            }
            
            st.session_state['time_columns'] = time_columns
        else:
            st.error("읽을 수 있는 CSV 파일이 없습니다.")
            st.stop()

    else:
        st.warning("⚠️ CSV 파일을 업로드해주세요.")

    # 업로드된 파일 처리 옵션
    with st.expander("📁 CSV 데이터 처리 옵션", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            sampling_rate = st.selectbox("리샘플링 주기:", ["1s", "5s", "10s", "30s", "1min"], index=1)
            sampling_method = st.selectbox("리샘플링 방법:", ["median", "mean", "min", "max"], index=0)
        
        with col2:
            remove_spikes = st.checkbox("스파이크 노이즈(999) 제거", value=True)
            timestamp = datetime.now().strftime("%y%m%d%H%M")
            output_filename = st.text_input("출력 파일명:", f"processed_data_{timestamp}.ftr")

    # 탭 분리
    tab1, tab2 = st.tabs(["🔄 데이터 처리", "📊 데이터 시각화"])

    # 탭 1: 데이터 처리
    with tab1:
        if st.session_state.get('processing_complete', False):
            if st.button("🔄 다시 처리", help="새로운 데이터로 다시 처리"):
                keys_to_clear = ['processing_complete', 'resampled_df', 'processing_results', 'folder_path']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            
            st.markdown("---")
            
            if 'processing_results' in st.session_state:
                results = st.session_state['processing_results']
                
                st.success(f"🎉 Feather 파일이 성공적으로 저장되었습니다!")
                st.write(f"📁 저장 경로: {results['file_path']}")
                st.write(f"📊 파일 크기: {results['file_size']:.2f} MB")
                st.write(f"📈 데이터 행 수: {results['resampled_rows']:,} (원본 대비 {results['reduction_ratio']:.1f}%)")
                
                if os.path.exists(results['file_path']):
                    with open(results['file_path'], 'rb') as f:
                        st.download_button(
                            label="📥 Feather 파일 다운로드",
                            data=f,
                            file_name=results['filename'],
                            mime="application/octet-stream",
                            help="처리된 데이터를 Feather 형식으로 다운로드",
                            type="primary",
                            key="download_persistent_state"
                        )
                else:
                    st.error("⚠️ 파일을 찾을 수 없습니다. 다시 처리해주세요.")
                
                with st.expander("📋 처리된 데이터 샘플 (처음 5행)", expanded=False):
                    if 'resampled_df' in st.session_state:
                        st.dataframe(st.session_state['resampled_df'].head())
                    else:
                        st.error("데이터를 불러올 수 없습니다.")
                
                st.info("💡 '데이터 시각화' 탭에서 데이터를 시각화할 수 있습니다!")
        
        else:
            st.info("📂 데이터 처리를 시작하려면 아래 버튼을 클릭하세요.")

            with st.expander("🔍 현재 세션 상태 (디버깅)", expanded=False):
                st.write("세션 상태 키들:", list(st.session_state.keys()))
                
                if 'files_uploaded' in st.session_state:
                    st.write("✅ 파일 업로드 완료됨")
                else:
                    st.write("❌ 파일 업로드 정보 없음")
                    
                if 'folder_path' in st.session_state:
                    folder_path = st.session_state['folder_path']
                    st.write("저장된 폴더 경로:", folder_path)
                    if os.path.exists(folder_path):
                        files = os.listdir(folder_path)
                        csv_files = [f for f in files if f.endswith('.csv')]
                        st.write("폴더 내 CSV 파일들:", csv_files)
                        st.write(f"CSV 파일 수: {len(csv_files)}")
                    else:
                        st.write("❌ 폴더가 존재하지 않음")
                else:
                    st.write("❌ folder_path가 세션에 없음")
                    if 'files_uploaded' in st.session_state:
                        folder_path = os.path.join(os.getcwd(), "data", "temp_uploads")
                        if os.path.exists(folder_path):
                            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
                            if csv_files:
                                st.session_state['folder_path'] = folder_path
                                st.session_state['csv_files'] = csv_files
                                st.write("✅ folder_path 복구 완료!")
                                st.rerun()

            can_process = False
            folder_path = None
            csv_files = []
            time_columns = {}
            
            if ('folder_path' in st.session_state and 
                'time_columns' in st.session_state and 
                os.path.exists(st.session_state['folder_path'])):
                
                folder_path = st.session_state['folder_path']
                time_columns = st.session_state['time_columns']
                csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
                if csv_files:
                    can_process = True
            
            elif not can_process:
                default_folder = os.path.join(os.getcwd(), "data", "temp_uploads")
                if os.path.exists(default_folder):
                    csv_files = [f for f in os.listdir(default_folder) if f.endswith('.csv')]
                    if csv_files:
                        folder_path = default_folder
                        st.session_state['folder_path'] = folder_path
                        st.session_state['csv_files'] = csv_files
                        
                        if 'time_columns' not in st.session_state:
                            time_columns = {
                                'year': 'year',
                                'month': 'month', 
                                'day': 'day',
                                'hour': 'hour',
                                'minute': 'minute',
                                'second': 'second'
                            }
                            st.session_state['time_columns'] = time_columns
                        else:
                            time_columns = st.session_state['time_columns']
                        
                        can_process = True
            
            if can_process:
                st.success(f"✅ 처리 가능: {len(csv_files)}개 CSV 파일 발견")
            else:
                st.error("❌ 처리할 CSV 파일이 없습니다. 파일을 다시 업로드해주세요.")

            if st.button("🔄 처리 시작", type="primary", disabled=not can_process):
                try:
                    if not os.path.exists(folder_path):
                        st.error(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
                        st.stop()
                    
                    if not csv_files:
                        st.error("❌ 지정된 폴더에 CSV 파일이 없습니다.")
                        st.stop()
                    
                    st.info(f"📂 총 {len(csv_files)} 개의 CSV 파일을 발견했습니다.")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_data = []

                    for i, file in enumerate(csv_files):
                        status_text.text(f"처리 중: {file} ({i+1}/{len(csv_files)})")
                        progress_bar.progress((i + 1) / len(csv_files))
                        
                        try:
                            df = pd.read_csv(os.path.join(folder_path, file), low_memory=False)
                            
                            valid_time_columns = {k: v for k, v in time_columns.items() if v is not None}
                            
                            if valid_time_columns:
                                missing_columns = []
                                for key, col_name in valid_time_columns.items():
                                    if col_name not in df.columns:
                                        missing_columns.append(col_name)

                                if missing_columns:
                                    st.warning(f"파일 {file}에서 다음 시간 컬럼이 없습니다: {missing_columns}")
                                    continue
                                else:
                                    if 'year' in valid_time_columns:
                                        if df[valid_time_columns['year']].max() < 100:
                                            df[valid_time_columns['year']] = df[valid_time_columns['year']] + 2000
                                    
                                    time_col_list = [valid_time_columns[col] for col in ['year', 'month', 'day', 'hour', 'minute', 'second'] 
                                                    if col in valid_time_columns]
                                    
                                    df['timestamp'] = pd.to_datetime(df[time_col_list])
                                    df = df.set_index('timestamp')
                                    df.drop(columns=list(valid_time_columns.values()), inplace=True)

                            all_data.append(df)
                            
                        except Exception as e:
                            st.warning(f"⚠️ 파일 처리 중 오류 발생: {file}\n{str(e)}")    
                    
                    if not all_data:
                        st.error("❌ 처리할 수 있는 데이터가 없습니다.")
                        st.stop()
                    
                    status_text.text("파일 병합 중...")
                    
                    merged_df = pd.concat(all_data)
                    
                    status_text.text("데이터 정렬 및 중복 제거 중...")
                    merged_df = merged_df.sort_index()
                    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
                    
                    if remove_spikes:
                        status_text.text("스파이크 노이즈(999) 제거 중...")
                        merged_df = replace_999_with_neighbors_mean(merged_df)
                    
                    status_text.text(f"{sampling_rate} 주기로 리샘플링 중...")
                    if sampling_method == "median":
                        resampled_df = merged_df.resample(sampling_rate).median()
                    elif sampling_method == "mean":
                        resampled_df = merged_df.resample(sampling_rate).mean()
                    elif sampling_method == "min":
                        resampled_df = merged_df.resample(sampling_rate).min()
                    else:
                        resampled_df = merged_df.resample(sampling_rate).max()
                    
                    status_text.text("Feather 파일 저장 중...")
                    
                    save_dir = os.path.dirname(os.path.join(folder_path, output_filename))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    
                    resampled_df_save = resampled_df.reset_index()
                    save_path = os.path.join(folder_path, output_filename)
                    resampled_df_save.to_feather(save_path)
                    
                    file_size = os.path.getsize(save_path) / (1024 * 1024)
                    
                    st.session_state['resampled_df'] = resampled_df
                    st.session_state['processing_complete'] = True
                    st.session_state['processing_results'] = {
                        'merged_rows': len(merged_df),
                        'merged_cols': len(merged_df.columns),
                        'time_range': f"{merged_df.index.min()} ~ {merged_df.index.max()}",
                        'resampled_rows': len(resampled_df),
                        'reduction_ratio': len(resampled_df)/len(merged_df)*100,
                        'file_path': save_path,
                        'filename': output_filename,
                        'file_size': file_size
                    }
                    
                    status_text.text("✅ 처리 완료!")
                    st.success(f"🎉 Feather 파일이 성공적으로 저장되었습니다!")
                    st.write(f"📁 저장 경로: {save_path}")
                    st.write(f"📊 파일 크기: {file_size:.2f} MB")
                    st.write(f"📈 데이터 행 수: {len(resampled_df):,} (원본 대비 {len(resampled_df)/len(merged_df)*100:.1f}%)")
                    
                    if os.path.exists(save_path):
                        with open(save_path, 'rb') as f:
                            st.download_button(
                                label="📥 Feather 파일 다운로드",
                                data=f,
                                file_name=output_filename,
                                mime="application/octet-stream",
                                help="처리된 데이터를 Feather 형식으로 다운로드",
                                type="primary",
                                key="download_after_processing"
                            )
                    
                    with st.expander("📋 처리된 데이터 샘플 (처음 5행)", expanded=False):
                        st.dataframe(resampled_df.head())
                    
                    st.info("💡 '데이터 시각화' 탭에서 데이터를 시각화할 수 있습니다!")
                    
                except Exception as e:
                    st.error(f"❌ 전체 처리 중 오류 발생: {str(e)}")
                    keys_to_clear = ['processing_complete', 'resampled_df', 'processing_results']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]

    # 탭 2: 데이터 시각화
    with tab2:
        if 'resampled_df' in st.session_state and st.session_state['resampled_df'] is not None:
            st.subheader("📊 처리된 데이터 시각화")
            
            df = st.session_state['resampled_df']
            
            # 다변량 시계열 데이터 동시 관찰
            st.title("🚀 신호 관찰 및 상호 관계 보기")
            
            st.success(f"✅ 데이터 로딩 완료! Shape: {df.shape}")

            selected_cols = st.multiselect("Plot할 컬럼을 선택하세요", df.columns.tolist(), key="csv_viz_columns")

            if selected_cols:
                st.subheader("📉 다운샘플 비율 설정 (1/N)")
                downsample_rate = st.slider("다운샘플 비율", min_value=1, max_value=100, value=10, key="csv_downsample")

                crosshair = st.checkbox("▶️ 십자선 Hover 활성화", value=True, key="csv_crosshair")

                fig = go.Figure()

                for col in selected_cols:
                    y = df[col].iloc[::downsample_rate]
                    x = df.index[::downsample_rate]
                    fig.add_trace(go.Scattergl(
                        x=x,
                        y=y,
                        mode='lines',
                        name=col,
                        showlegend=True,
                        hoverinfo='x',
                        hovertemplate=''
                    ))

                fig.update_layout(
                    title="📊 Plotly 그래프 (다운샘플링 적용)",
                    dragmode="zoom",
                    xaxis=dict(
                        rangeslider=dict(visible=False)
                    ),
                    height=600
                )

                if crosshair:
                    fig.update_layout(
                        hovermode="x",
                        xaxis=dict(
                            showspikes=True,
                            spikemode='across',
                            spikesnap='cursor',
                            spikecolor="red",
                            spikethickness=1
                        ),
                        yaxis=dict(
                            showspikes=True,
                            spikemode='across',
                            spikesnap='cursor',
                            spikecolor="blue",
                            spikethickness=1
                        )
                    )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("시각화할 컬럼을 선택해주세요.")
        else:
            st.info("시각화할 데이터가 없습니다. 먼저 '데이터 처리' 탭에서 CSV 파일을 처리해주세요.")

# =====================================
# 메인 애플리케이션
# =====================================
def main():
    st.title("🚀 데이터 변환 통합 도구")
    
    # 사이드바 렌더링
    render_sidebar()
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📊 Excel → Feather 변환기", "📈 CSV → Feather 변환기", "📋 사용법 안내"])
    
    with tab1:
        tab_excel_converter()
    
    with tab2:
        tab_csv_converter()
    
    with tab3:
        render_usage_guide()

def render_usage_guide():
    """사용법 안내 탭"""
    st.header("📋 프로그램 기능 안내")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 주요 기능")
        st.markdown("""
        **Tab 1: Excel → Feather 변환기**
        - 대용량 Excel 파일 처리 및 Feather 변환
        - 날짜 컬럼 자동 식별 및 보호
        - 문자-숫자 자동 매핑 변환
        - 모든 시트 일괄 처리 기능
        - ZIP 압축 다운로드 지원
        
        **Tab 2: CSV → Feather 변환기**  
        - 다중 CSV 파일 병합 처리
        - 시간 인덱스 자동 조정
        - 스파이크 노이즈(999) 제거
        - 리샘플링 및 데이터 정제
        - 실시간 데이터 시각화
        """)
    
    with col2:
        st.subheader("⚙️ 고급 기능")
        st.markdown("""
        **데이터 전처리**
        - NaN/Inf 값 자동 처리 및 보간
        - 문자값 빈도 분석 및 매핑
        - 날짜 컬럼 자동 보호 메커니즘
        - 컬럼 선택적 제거 기능
        
        **시각화 도구**
        - Plotly 기반 고성능 WebGL 렌더링
        - 다운샘플링을 통한 대용량 데이터 처리
        - 십자선 Hover 및 인터랙티브 줌/팬
        - 다변량 시계열 동시 관찰
        - 시간축 기반 시각화 지원
        """)
    
    st.markdown("---")
    st.subheader("📝 사용 순서")
    
    st.markdown("""
    **Excel → Feather 변환 시:**
    1. Excel 파일 업로드 및 읽기 설정
    2. 시트 선택 (자동 감지)
    3. 문자-숫자 매핑 설정 (필요시)
    4. 불필요한 컬럼 제거 (선택사항)
    5. Feather 파일로 저장 또는 다운로드
    6. 모든 시트 일괄 처리 (다중 시트인 경우)
    
    **CSV → Feather 변환 시:**
    1. 다중 CSV 파일 업로드
    2. 시간 컬럼 설정 (year, month, day 등)
    3. 리샘플링 옵션 설정
    4. 데이터 처리 실행
    5. 시각화 탭에서 결과 확인
    6. Feather 파일 다운로드
    """)
    
    st.markdown("---")
    st.info("💡 **사용 팁**: 각 탭은 독립적으로 작동하므로, 필요에 따라 원하는 변환 도구를 선택하여 사용하세요. 대용량 데이터 처리 시 메모리 사용량을 고려하여 적절한 다운샘플링을 적용하는 것을 권장합니다.")
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <small>이 도구는 다양한 형식의 데이터를 효율적인 Feather 형식으로 변환하고 분석하기 위해 설계되었습니다.</small>
    </div>
    """, unsafe_allow_html=True)

# =====================================
# 앱 실행
# =====================================
if __name__ == "__main__":
    main()