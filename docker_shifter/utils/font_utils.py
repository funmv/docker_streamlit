"""
한글 폰트 설정 유틸리티
"""
import matplotlib.pyplot as plt


def setup_korean_font():
    """한글 폰트 설정 함수"""
    try:
        from matplotlib import font_manager, rc
        # Windows 환경
        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
    except:
        try:
            # Linux 환경
            from matplotlib import font_manager, rc
            font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font_name)
            plt.rcParams['axes.unicode_minus'] = False
        except:
            # 폰트 로드 실패 시 기본 폰트 사용
            plt.rcParams['axes.unicode_minus'] = False
