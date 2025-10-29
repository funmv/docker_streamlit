"""
통합 데이터 분석 시스템 - 메인 앱 (v2)
YAML 설정 기반 Excel/CSV 데이터 로더 및 시각화
- 타임스탬프 제외 옵션 추가
- 샘플링 기능 추가
"""
import streamlit as st
# UI 컴포넌트 임포트
import importlib
import sys

# utils 모듈 강제 리로드
if 'utils.visualization' in sys.modules:
    importlib.reload(sys.modules['utils.visualization'])
if 'utils.ui_components' in sys.modules:
    importlib.reload(sys.modules['utils.ui_components'])

# from utils import render_config_tab, render_loading_tab, render_visualization_tab

# 한글 폰트 설정
try:
    from matplotlib import font_manager, rc
    import matplotlib.pyplot as plt
    # Windows 환경
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        # Linux 환경
        from matplotlib import font_manager, rc
        import matplotlib.pyplot as plt
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)  
        plt.rcParams['axes.unicode_minus'] = False
    except:
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

# 페이지 설정
st.set_page_config(
    page_title="통합 데이터 분석 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# UI 컴포넌트 임포트
from utils import render_config_tab, render_loading_tab, render_visualization_tab


# Session State 초기화
def init_session_state():
    """세션 상태 초기화"""
    if 'config' not in st.session_state:
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
                'exclude_from_output': False  # 새로운 옵션
            },
            'sampling': {  # 새로운 섹션
                'enabled': False,
                'method': 'every_n',  # 'every_n', 'mean', 'median', 'first', 'last'
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
    
    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None
    
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None


def main():
    """메인 애플리케이션"""
    # 세션 상태 초기화
    init_session_state()
    
    # 타이틀
    st.title("🔧 통합 데이터 분석 시스템")
    st.markdown("Excel/CSV 파일을 YAML 설정으로 자동 처리하고 가시화합니다.")
    
    # 탭 생성
    tabs = st.tabs(["⚙️ YAML 설정", "📂 데이터 로딩", "📊 데이터 가시화"])
    
    with tabs[0]:
        render_config_tab()
    
    with tabs[1]:
        render_loading_tab()
    
    with tabs[2]:
        render_visualization_tab()


if __name__ == "__main__":
    main()