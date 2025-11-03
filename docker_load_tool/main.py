"""
통합 데이터 분석 시스템 - Streamlit Frontend
"""
import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.data_service import DataService
from backend.file_service import FileService
from utils.yaml_utils import get_default_config, dict_to_yaml_string, yaml_string_to_dict, load_yaml_file
from frontend.config_ui import render_config_tab
from frontend.loading_ui import render_loading_tab
from frontend.viz_ui import render_visualization_tab

# 한글 폰트 설정
try:
    from matplotlib import font_manager, rc
    import matplotlib.pyplot as plt
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
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


def init_session_state():
    """세션 상태 초기화"""
    if 'config' not in st.session_state:
        st.session_state.config = get_default_config()
    
    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None
    
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    
    # 서비스 인스턴스
    if 'data_service' not in st.session_state:
        st.session_state.data_service = DataService()
    
    if 'file_service' not in st.session_state:
        st.session_state.file_service = FileService()


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