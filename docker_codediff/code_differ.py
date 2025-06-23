import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import difflib
import re
from typing import List, Tuple, Dict

# 한글 폰트 설정
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

# matplotlib 경고 제거를 위한 설정
plt.rcParams['figure.max_open_warning'] = 50

# 페이지 설정
st.set_page_config(page_title="두 코드의 차이점 분석", layout="wide")

# 세션 상태 초기화
if 'code1' not in st.session_state:
    st.session_state.code1 = ""
if 'code2' not in st.session_state:
    st.session_state.code2 = ""
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'selected_changes' not in st.session_state:
    st.session_state.selected_changes = {}

def get_line_changes(code1: str, code2: str) -> List[Dict]:
    """두 코드 간의 차이점을 분석하여 변경사항을 반환"""
    lines1 = code1.splitlines()
    lines2 = code2.splitlines()
    
    # SequenceMatcher를 사용하여 더 정확한 차이점 분석
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    changes = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue  # 동일한 부분은 건너뜀
        elif tag in ['delete', 'insert', 'replace']:
            change = {
                'type': tag,
                'line_num': i1 + 1,  # 1-based line number
                'old_lines': lines1[i1:i2] if tag in ['delete', 'replace'] else [],
                'new_lines': lines2[j1:j2] if tag in ['insert', 'replace'] else []
            }
            changes.append(change)
    
    return changes

def highlight_code_differences(code1: str, code2: str) -> str:
    """코드의 차이점을 HTML로 하이라이트 (통합된 단일 뷰)"""
    lines1 = code1.splitlines()
    lines2 = code2.splitlines()
    
    # 라인별 차이 분석
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    
    highlighted_lines = []
    line_num = 1
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # 동일한 부분
            for i in range(i1, i2):
                content = lines1[i].replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                if not content.strip():
                    content = "&nbsp;"
                highlighted_lines.append(f'<div class="code-line"><span class="line-number">{line_num}</span><span class="change-indicator">&nbsp;</span><span class="line-content">{content}</span></div>')
                line_num += 1
        elif tag == 'delete':
            # 삭제된 부분 (코드1에만 있음)
            for i in range(i1, i2):
                content = lines1[i].replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                if not content.strip():
                    content = "&nbsp;"
                highlighted_lines.append(f'<div class="code-line deleted"><span class="line-number">{line_num}</span><span class="change-indicator">-</span><span class="line-content">{content}</span></div>')
                line_num += 1
        elif tag == 'insert':
            # 추가된 부분 (코드2에만 있음)
            for j in range(j1, j2):
                content = lines2[j].replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                if not content.strip():
                    content = "&nbsp;"
                highlighted_lines.append(f'<div class="code-line added"><span class="line-number">{line_num}</span><span class="change-indicator">+</span><span class="line-content">{content}</span></div>')
                line_num += 1
        elif tag == 'replace':
            # 변경된 부분 - 먼저 삭제된 라인들 표시
            for i in range(i1, i2):
                content = lines1[i].replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                if not content.strip():
                    content = "&nbsp;"
                highlighted_lines.append(f'<div class="code-line deleted"><span class="line-number">{line_num}</span><span class="change-indicator">-</span><span class="line-content">{content}</span></div>')
                line_num += 1
            # 그 다음 추가된 라인들 표시
            for j in range(j1, j2):
                content = lines2[j].replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                if not content.strip():
                    content = "&nbsp;"
                highlighted_lines.append(f'<div class="code-line added"><span class="line-number">{line_num}</span><span class="change-indicator">+</span><span class="line-content">{content}</span></div>')
                line_num += 1
    
    return '\n'.join(highlighted_lines)

def apply_selected_changes(original_code: str, new_code: str, selected_changes: Dict[int, bool]) -> Tuple[str, List[int]]:
    """선택된 변경사항만 적용하여 새로운 코드 생성하고 변경된 라인 번호 반환"""
    if not selected_changes:
        return original_code, []
    
    changes = get_line_changes(original_code, new_code)
    lines = original_code.splitlines()
    changed_line_numbers = []
    
    # 변경사항을 역순으로 적용 (라인 번호 변경 방지)
    line_offset = 0
    for i, change in enumerate(reversed(changes)):
        change_id = len(changes) - 1 - i
        if selected_changes.get(change_id, False):
            line_num = change['line_num'] - 1 + line_offset  # 0-based index + offset
            
            # 기존 라인 삭제
            for _ in range(len(change['old_lines'])):
                if line_num < len(lines):
                    lines.pop(line_num)
                    line_offset -= 1
            
            # 새 라인 삽입
            for j, new_line in enumerate(change['new_lines']):
                lines.insert(line_num + j, new_line)
                changed_line_numbers.append(line_num + j + 1)  # 1-based for display
                line_offset += 1
    
    return '\n'.join(lines), sorted(set(changed_line_numbers))

def highlight_final_code(code: str, changed_lines: List[int]) -> str:
    """최종 코드를 하이라이트하여 변경된 부분 표시"""
    lines = code.splitlines()
    highlighted_lines = []
    
    for i, line in enumerate(lines):
        line_num = i + 1
        content = line.replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
        if not content.strip():
            content = "&nbsp;"
            
        # 변경된 라인인지 확인
        if line_num in changed_lines:
            highlighted_lines.append(f'<div class="final-code-line applied-change"><span class="final-line-number">{line_num}</span><span class="final-change-indicator">*</span><span class="final-line-content">{content}</span></div>')
        else:
            highlighted_lines.append(f'<div class="final-code-line"><span class="final-line-number">{line_num}</span><span class="final-change-indicator">&nbsp;</span><span class="final-line-content">{content}</span></div>')
    
    return '\n'.join(highlighted_lines)

# CSS 스타일 정의 - 정리된 버전 (사용되지 않는 클래스 제거)
st.markdown("""
<style>
.code-container {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.375rem;
    padding: 1rem;
    margin: 1rem 0;
    font-family: 'Courier New', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    line-height: 1.15;
    white-space: nowrap;
    max-height: 1200px;
    overflow: auto;
}

.code-line {
    margin: 0;
    padding: 6px 8px;
    border-radius: 3px;
    display: block;
    min-height: 1.4em;
    white-space: nowrap;
    overflow: visible;
}

.line-number {
    display: inline-block;
    width: 50px;
    text-align: right;
    margin-right: 10px;
    color: #6c757d;
    font-weight: 500;
    font-size: 0.9rem;
    vertical-align: top;
    /* 라인넘버 선택 방지 */
    user-select: none;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    /* 추가적인 스타일링으로 구분감 향상 */
    opacity: 0.7;
    background-color: rgba(108, 117, 125, 0.1);
    border-radius: 3px;
    padding: 2px 4px;
}

.change-indicator {
    display: inline-block;
    width: 20px;
    text-align: center;
    margin-right: 10px;
    color: #495057;
    font-weight: bold;
    font-size: 1rem;
    vertical-align: top;
    /* 변경사항 표시 선택 방지 */
    user-select: none;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    /* 시각적 구분을 위한 스타일링 */
    opacity: 0.8;
    background-color: rgba(73, 80, 87, 0.1);
    border-radius: 3px;
    padding: 2px 4px;
}

.line-content {
    display: inline;
    vertical-align: top;
    white-space: pre;
    /* 코드 내용만 선택 가능하도록 설정 */
    user-select: text;
    -webkit-user-select: text;
    -moz-user-select: text;
    -ms-user-select: text;
}

.code-line.added {
    background-color: #d4edda;
    color: #155724;
    border-left: 4px solid #28a745;
}

.code-line.deleted {
    background-color: #f8d7da;
    color: #721c24;
    border-left: 4px solid #dc3545;
}

.code-line.applied-change {
    background-color: #fff3cd !important;
    border-left: 4px solid #ffc107;
}

.final-code-line {
    margin: 0;
    padding: 6px 8px;
    border-radius: 3px;
    display: block;
    min-height: 1.4em;
    white-space: nowrap;
    overflow: visible;
}

.final-line-number {
    display: inline-block;
    width: 50px;
    text-align: right;
    margin-right: 10px;
    color: #6c757d;
    font-weight: 500;
    font-size: 0.9rem;
    vertical-align: top;
    /* 최종 코드 라인넘버도 선택 방지 */
    user-select: none;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    opacity: 0.7;
    background-color: rgba(108, 117, 125, 0.1);
    border-radius: 3px;
    padding: 2px 4px;
}

.final-change-indicator {
    display: inline-block;
    width: 20px;
    text-align: center;
    margin-right: 10px;
    color: #495057;
    font-weight: bold;
    font-size: 1rem;
    vertical-align: top;
    /* 최종 코드 변경사항 표시도 선택 방지 */
    user-select: none;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    opacity: 0.8;
    background-color: rgba(73, 80, 87, 0.1);
    border-radius: 3px;
    padding: 2px 4px;
}

.final-line-content {
    display: inline;
    vertical-align: top;
    white-space: pre;
    /* 최종 코드 내용만 선택 가능 */
    user-select: text;
    -webkit-user-select: text;
    -moz-user-select: text;
    -ms-user-select: text;
}

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

/* 코드 복사 시 더 나은 사용자 경험을 위한 hover 효과 */
.code-line:hover .line-number {
    background-color: rgba(108, 117, 125, 0.2);
}

.code-line:hover .change-indicator {
    background-color: rgba(73, 80, 87, 0.2);
}

.final-code-line:hover .final-line-number {
    background-color: rgba(108, 117, 125, 0.2);
}

.final-code-line:hover .final-change-indicator {
    background-color: rgba(73, 80, 87, 0.2);
}
</style>
""", unsafe_allow_html=True)

# 메인 애플리케이션
st.title("🔍 두 코드 비교 분석 도구")

# 탭 생성
tab1, tab2 = st.tabs(["📝 코드 입력", "🔄 코드 비교 및 적용"])

# 첫 번째 탭: 코드 입력
with tab1:
    st.header("📝 코드 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔹 원본 코드 (코드1)")
        code1_input = st.text_area(
            "원본 코드를 입력하세요:",
            value=st.session_state.code1,
            height=400,
            key="code1_input",
            help="분석할 원본 코드를 여기에 붙여넣으세요."
        )
        
        if st.button("코드1 저장", key="save_code1"):
            st.session_state.code1 = code1_input
            st.success("✅ 코드1이 저장되었습니다!")
    
    with col2:
        st.subheader("🔹 비교 코드 (코드2)")
        code2_input = st.text_area(
            "비교할 코드를 입력하세요:",
            value=st.session_state.code2,
            height=400,
            key="code2_input",
            help="원본 코드와 비교할 코드를 여기에 붙여넣으세요."
        )
        
        if st.button("코드2 저장", key="save_code2"):
            st.session_state.code2 = code2_input
            st.success("✅ 코드2가 저장되었습니다!")
    
    # 코드 비교 버튼
    st.markdown("---")
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        if st.button("🔍 코드 비교 실행", key="compare_codes", type="primary"):
            # 현재 입력된 코드를 직접 사용 (저장 버튼을 누르지 않아도 됨)
            current_code1 = code1_input.strip()
            current_code2 = code2_input.strip()
            
            if current_code1 and current_code2:
                # 세션 상태도 업데이트
                st.session_state.code1 = current_code1
                st.session_state.code2 = current_code2
                
                st.session_state.comparison_results = get_line_changes(
                    current_code1, 
                    current_code2
                )
                st.success("✅ 코드 비교가 완료되었습니다! '코드 비교 및 적용' 탭에서 결과를 확인하세요.")
            else:
                st.error("❌ 두 코드를 모두 입력해주세요!")

# 두 번째 탭: 코드 비교 및 적용
with tab2:
    st.header("🔄 코드 비교 및 적용")
    
    if st.session_state.comparison_results is not None:
        changes = st.session_state.comparison_results
        
        if not changes:
            st.info("🎉 두 코드가 동일합니다! 변경사항이 없습니다.")
        else:
            st.subheader(f"📊 발견된 변경사항: {len(changes)}개")
            
            # 코드 비교 시각화
            st.subheader("🔍 코드 비교 시각화")
            highlighted_code = highlight_code_differences(
                st.session_state.code1, 
                st.session_state.code2
            )
            
            st.markdown("**통합 코드 비교 (코드1 → 코드2 변경사항)**")
            st.markdown("- 🔴 **빨간색**: 삭제된 라인 (코드1에서 제거)")
            st.markdown("- 🟢 **초록색**: 추가된 라인 (코드2에서 추가)")
            st.markdown("- ⚪ **흰색**: 변경되지 않은 라인")
            st.markdown("- 💡 **팁**: 라인 번호와 +/- 기호는 선택되지 않으므로 순수한 코드만 복사할 수 있습니다!")
            
            # 컬럼 구조 설명 추가
            st.markdown("**📋 화면 구성**: [라인번호] [+/-표시] [코드내용] - 코드내용만 선택/복사 가능")
            
            st.markdown(f'<div class="code-container">{highlighted_code}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
    
    else:
        st.info("🔍 먼저 '코드 입력' 탭에서 두 코드를 입력하고 비교를 실행해주세요.")

# 사이드바 - 공통 설정
with st.sidebar:
    
    # 통계 정보
    if st.session_state.comparison_results:
        st.markdown("### 📊 분석 통계")
        st.metric("발견된 변경사항", len(st.session_state.comparison_results))
        selected_count = sum(1 for v in st.session_state.selected_changes.values() if v)
        st.metric("선택된 변경사항", selected_count)
    
    # 하단 정보
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
            🛠️ <strong>버전:</strong> V.1.6 (06-23-2025)<br>
            <hr>
        </div>
        """, 
        unsafe_allow_html=True
    )