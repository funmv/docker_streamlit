import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from io import BytesIO
from difflib import SequenceMatcher
from collections import defaultdict
import openpyxl

# 한글 폰트 설정 (matplotlib 사용시)
try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc
    # Windows 환경
    try:
        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # Linux 환경
        try:
            font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font_name)  
            plt.rcParams['axes.unicode_minus'] = False
        except:
            # 폰트 로드 실패 시 기본 폰트 사용
            plt.rcParams['axes.unicode_minus'] = False
    
    # matplotlib 경고 제거를 위한 설정
    plt.rcParams['figure.max_open_warning'] = 50
except ImportError:
    pass

def is_tag_name(text):
    """태그명인지 판단하는 함수 (숫자와 영어대문자 혼합)"""
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    text = str(text).strip()
    if len(text) < 2:
        return False
    
    # 태그명 패턴: 영어 대문자와 숫자가 포함된 조합
    # 예: T5_CTIMR1, 3TY5904, 3STDL5477 등
    pattern = r'^[A-Z0-9_]+$'
    has_pattern = bool(re.match(pattern, text))
    has_digit = any(c.isdigit() for c in text)
    has_upper = any(c.isupper() for c in text)
    
    return has_pattern and has_digit and has_upper

def is_description(text):
    """태그 설명인지 판단하는 함수 (한글, 영문, 숫자 혼합)"""
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    text = str(text).strip()
    if len(text) < 1:
        return False
    
    # 태그명 패턴이면 제외
    if is_tag_name(text):
        return False
    
    # 설명 패턴: 한글, 영문, 숫자, 공백, 특수문자 포함
    # 예: '대기온도', '1st metal temp', 'GT5 MW', 'RH Bore temp' 등
    return True

def find_tag_description_rows(df):
    """엑셀에서 태그설명 행과 태그명 행을 찾는 함수"""
    tag_rows = []
    description_rows = []
    
    # 상위 100개 행만 체크 (전체 행 수와 100 중 작은 값)
    max_rows_to_check = min(100, df.shape[0])

    # 각 행에서 태그명과 설명의 개수를 계산
    for row_idx in range(max_rows_to_check):
        tag_count = 0
        desc_count = 0
        
        for col_idx in range(df.shape[1]):
            cell_value = df.iloc[row_idx, col_idx]
            
            if is_tag_name(cell_value):
                tag_count += 1
            elif is_description(cell_value):
                desc_count += 1
        
        # 태그가 3개 이상 있는 행을 태그명 행으로 간주
        if tag_count >= 3:
            tag_rows.append((row_idx, tag_count))
        
        # 설명이 3개 이상 있는 행을 설명 행으로 간주
        if desc_count >= 3:
            description_rows.append((row_idx, desc_count))
    
    # 가장 많은 태그/설명을 가진 행 선택
    main_tag_row = max(tag_rows, key=lambda x: x[1])[0] if tag_rows else None
    main_desc_row = max(description_rows, key=lambda x: x[1])[0] if description_rows else None
    
    return main_tag_row, main_desc_row, tag_rows, description_rows

def extract_tags_from_excel(file_content, file_name):
    """엑셀 파일에서 태그명과 설명을 자동으로 추출"""
    try:
        # 첫 번째 시트만 읽기
        df = pd.read_excel(BytesIO(file_content), sheet_name=0, header=None)
        
        st.write(f"📋 {file_name} 파일 분석 중...")
        st.write(f"- 파일 크기: {df.shape[0]}행 x {df.shape[1]}열")
        
        # 태그설명 행과 태그명 행 찾기
        main_tag_row, main_desc_row, tag_rows, description_rows = find_tag_description_rows(df)
        
        if main_tag_row is None:
            st.warning(f"⚠️ {file_name}에서 태그명 행을 찾을 수 없습니다.")
            return {}
        
        if main_desc_row is None:
            st.warning(f"⚠️ {file_name}에서 태그설명 행을 찾을 수 없습니다.")
            # 태그명만 있는 경우 태그명을 설명으로 사용
            main_desc_row = main_tag_row
        
        st.write(f"- 태그명 행: {main_tag_row + 1} (후보: {len(tag_rows)}개)")
        st.write(f"- 설명 행: {main_desc_row + 1} (후보: {len(description_rows)}개)")
        
        # 태그명과 설명 매핑
        tag_description_map = {}
        
        # 각 열에서 태그명과 설명 추출
        for col_idx in range(df.shape[1]):
            tag_name = df.iloc[main_tag_row, col_idx]
            description = df.iloc[main_desc_row, col_idx]
            
            # 태그명이 유효한 경우만 처리
            if is_tag_name(tag_name):
                # 설명이 유효하지 않으면 태그명을 설명으로 사용
                if not is_description(description):
                    description = tag_name
                else:
                    description = str(description).strip()
                
                tag_description_map[description] = str(tag_name).strip()
        
        if not tag_description_map:
            st.warning(f"⚠️ {file_name}에서 유효한 태그-설명 쌍을 찾을 수 없습니다.")
            return {}
        
        st.success(f"✅ {file_name}에서 {len(tag_description_map)}개의 태그-설명 쌍을 추출했습니다.")
        
        # 추출된 태그-설명 샘플 표시
        with st.expander(f"📝 {file_name} 추출 결과 미리보기"):
            sample_items = list(tag_description_map.items())[:5]
            for desc, tag in sample_items:
                st.write(f"- **{desc}** → `{tag}`")
            if len(tag_description_map) > 5:
                st.write(f"... 외 {len(tag_description_map) - 5}개")
        
        return tag_description_map
        
    except Exception as e:
        st.error(f"❌ {file_name} 파일 처리 중 오류 발생: {str(e)}")
        return {}

def calculate_similarity(tag1, tag2):
    """두 태그명 간의 유사도를 계산 (0~1 사이 값)"""
    return SequenceMatcher(None, tag1, tag2).ratio()

def find_similar_tags(tags_dict, similarity_threshold=0.7):
    """유사한 태그들을 그룹화"""
    st.session_state.similar_pairs = []

    all_tags = []
    for file_name, tag_data in tags_dict.items():
        for description, tag_name in tag_data.items():
            all_tags.append({
                'file': file_name,
                'description': description,
                'tag': tag_name
            })
    
    st.write(f"🔍 총 {len(all_tags)}개의 태그를 분석 중...")
    
    # 유사 태그 그룹 찾기
    similar_groups = []
    processed_indices = set()
    
    for i, tag_info in enumerate(all_tags):
        if i in processed_indices:
            continue
            
        current_group = [tag_info]
        processed_indices.add(i)
        
        for j, other_tag_info in enumerate(all_tags):
            if j in processed_indices or j <= i:
                continue
                
            # 태그명 유사도만 계산
            similarity = calculate_similarity(tag_info['tag'], other_tag_info['tag'])
            
            if similarity >= similarity_threshold:
                if 'similar_pairs' not in st.session_state:
                    st.session_state.similar_pairs = []
                st.session_state.similar_pairs.append(
                    f"✅ 유사 태그 발견: '{tag_info['tag']}' vs '{other_tag_info['tag']}' (유사도: {similarity:.3f})"
                )
                current_group.append(other_tag_info)
                processed_indices.add(j)            
        
        similar_groups.append(current_group)
    
    return similar_groups

def create_comparison_table(similar_groups, file_names):
    """비교 테이블 생성"""
    # 겹치는 태그 그룹 (2개 이상 파일에서 발견)
    overlapping_groups = [group for group in similar_groups if len(set(item['file'] for item in group)) > 1]
    
    # 독립적인 태그 그룹 (1개 파일에서만 발견)
    unique_groups = [group for group in similar_groups if len(set(item['file'] for item in group)) == 1]
    
    st.write(f"🔍 분석 결과:")
    st.write(f"- 전체 그룹 수: {len(similar_groups)}")
    st.write(f"- 공통 태그 그룹: {len(overlapping_groups)}")
    st.write(f"- 독립 태그 그룹: {len(unique_groups)}")
    
    # 테이블 데이터 준비
    table_data = []

    # 겹치는 태그들 - 각 태그를 개별 행으로 추가
    for group_idx, group in enumerate(overlapping_groups):
        for item in group:
            row = {
                '구분': '공통 태그', 
                '태그명': item['tag'], 
                '설명': item['description'],
                '파일': item['file'], 
                '그룹_인덱스': group_idx
            }
            
            # 각 파일별 정보 추가
            for file_name in file_names:
                if file_name == item['file']:
                    row[f'{file_name}_태그명'] = item['tag']
                    row[f'{file_name}_설명'] = item['description']
                    row[f'{file_name}_존재'] = '✓'
                else:
                    # 같은 그룹 내에서 다른 파일에 있는 태그 찾기
                    other_items = [g_item for g_item in group if g_item['file'] == file_name]
                    if other_items:
                        row[f'{file_name}_태그명'] = other_items[0]['tag']
                        row[f'{file_name}_설명'] = other_items[0]['description']
                        row[f'{file_name}_존재'] = '✓'
                    else:
                        row[f'{file_name}_태그명'] = '-'
                        row[f'{file_name}_설명'] = '-'
                        row[f'{file_name}_존재'] = '✗'
            
            table_data.append(row)    
    
    # 독립적인 태그들 추가
    for group in unique_groups:
        for item in group:
            row = {
                '구분': f'독립 태그 ({item["file"]})', 
                '태그명': item['tag'], 
                '설명': item['description'],
                '파일': item['file'], 
                '그룹_인덱스': -1
            }
            
            for file_name in file_names:
                if file_name == item['file']:
                    row[f'{file_name}_태그명'] = item['tag']
                    row[f'{file_name}_설명'] = item['description']
                    row[f'{file_name}_존재'] = '✓'
                else:
                    row[f'{file_name}_태그명'] = '-'
                    row[f'{file_name}_설명'] = '-'
                    row[f'{file_name}_존재'] = '✗'
            
            table_data.append(row)
    
    return pd.DataFrame(table_data)

def handle_file_upload(uploaded_files):
    """업로드된 파일들을 처리하는 함수"""
    tags_dict = {}
    file_names = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name.split('.')[0]  # 확장자 제거
        file_names.append(file_name)
        
        status_text.text(f"처리 중: {file_name} ({i+1}/{len(uploaded_files)})")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        # 엑셀 파일에서 태그 추출
        file_content = uploaded_file.read()
        tag_data = extract_tags_from_excel(file_content, file_name)
        
        if tag_data:
            tags_dict[file_name] = tag_data
    
    status_text.text("파일 처리 완료!")
    
    # 세션 상태에 저장
    st.session_state.tags_dict = tags_dict
    st.session_state.file_names = file_names
    st.session_state.files_processed = True

def main():
    st.set_page_config(page_title="엑셀 태그 관리 시스템", layout="wide")
    
    st.title("🏷️ 엑셀 태그 관리 시스템")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    similarity_threshold = st.sidebar.slider(
        "유사도 임계값", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="이 값 이상의 유사도를 가진 태그들을 유사 태그로 분류합니다."
    )
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📁 파일 업로드", "📊 분석 결과", "📋 비교 테이블"])
    
    with tab1:
        st.header("📁 엑셀 파일 업로드")
        st.markdown("**엑셀 파일을 업로드하여 태그명과 설명을 추출합니다.**")
        
        uploaded_files = st.file_uploader(
            "엑셀 파일들을 선택하세요 (여러 파일 선택 가능)",
            type=['xlsx', 'xls'],
            accept_multiple_files=True,
            help="각 엑셀 파일의 첫 번째 시트에서 태그명과 설명을 자동으로 추출합니다."
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)}개의 파일이 선택되었습니다.")
            
            # 파일 목록 표시
            with st.expander("📋 선택된 파일 목록"):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {file.name}")
            
            if st.button("🚀 파일 분석 시작", type="primary"):
                handle_file_upload(uploaded_files)
        
        # 파일 형식 가이드
        with st.expander("📖 파일 형식 가이드"):
            st.markdown("""
            **태그명 인식 조건:**
            - 영어 대문자와 숫자가 혼합된 형태
            - 예시: `T5_CTIMR1`, `3TY5904`, `ABC123`, `XYZ_456`
            
            **설명 추출 방법:**
            - 태그명과 설명이 별도 행에 있는 경우 자동 매핑
            - 설명이 없는 경우 태그명을 설명으로 사용
            
            **지원 파일 형식:**
            - Excel 파일 (.xlsx, .xls)
            - 각 파일의 첫 번째 시트만 분석
            """)
    
    with tab2:
        st.header("📊 분석 결과")
        
        if hasattr(st.session_state, 'files_processed') and st.session_state.files_processed:
            tags_dict = st.session_state.tags_dict
            file_names = st.session_state.file_names
            
            if tags_dict:
                # 파일별 태그 정보
                st.subheader("📋 파일별 태그 정보")
                cols = st.columns(min(len(file_names), 4))
                
                for i, (file_name, data) in enumerate(tags_dict.items()):
                    with cols[i % len(cols)]:
                        st.metric(f"📁 {file_name}", f"{len(data)}개 태그")
                
                # 유사도 분석
                st.subheader("🔍 태그 유사도 분석")
                
                with st.spinner("태그 유사도를 분석하는 중..."):
                    similar_groups = find_similar_tags(tags_dict, similarity_threshold)
                
                overlapping_count = len([group for group in similar_groups if len(set(item['file'] for item in group)) > 1])
                unique_count = len([group for group in similar_groups if len(set(item['file'] for item in group)) == 1])
                
                # 분석 결과 요약
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🤝 공통 태그 그룹", overlapping_count)
                with col2:
                    st.metric("🏠 독립 태그 그룹", unique_count)
                with col3:
                    st.metric("📊 전체 태그 그룹", len(similar_groups))
                
                # 유사 태그 발견 결과
                if hasattr(st.session_state, 'similar_pairs') and st.session_state.similar_pairs:
                    with st.expander(f"🔍 유사 태그 발견 결과 ({len(st.session_state.similar_pairs)}개)", expanded=False):
                        for pair in st.session_state.similar_pairs:
                            st.write(pair)
                
                # 세션 상태에 분석 결과 저장
                st.session_state.similar_groups = similar_groups
                
            else:
                st.warning("⚠️ 분석할 데이터가 없습니다. 파일을 다시 업로드해주세요.")
        else:
            st.info("📁 먼저 '파일 업로드' 탭에서 엑셀 파일을 업로드하고 분석을 시작해주세요.")
    
    with tab3:
        st.header("📋 태그 비교 테이블")
        
        if (hasattr(st.session_state, 'files_processed') and st.session_state.files_processed and
            hasattr(st.session_state, 'similar_groups')):
            
            tags_dict = st.session_state.tags_dict
            file_names = st.session_state.file_names
            similar_groups = st.session_state.similar_groups
            
            comparison_df = create_comparison_table(similar_groups, file_names)
            
            if not comparison_df.empty:
                # 파일별 색상 매핑
                default_colors = ['#fff2e6', '#e6ffe6', '#ffe6f3', '#f3e6ff', '#e6fff3', '#fff3e6']
                file_colors = {}
                
                for i, file_name in enumerate(file_names):
                    if i < len(default_colors):
                        file_colors[file_name] = default_colors[i]
                    else:
                        file_colors[file_name] = '#f0f0f0'
                
                # 스타일 적용 함수
                def style_table(row):
                    original_row = comparison_df.iloc[row.name]
                    
                    if original_row['구분'] == '공통 태그':
                        group_idx = original_row['그룹_인덱스']
                        common_tag_colors = ['#cce7ff', '#d4edda', '#fff3cd', '#f8d7da']
                        color = common_tag_colors[group_idx % len(common_tag_colors)]
                        return [f'background-color: {color}'] * len(row)
                    else:
                        file_name = original_row['파일']
                        color = file_colors.get(file_name, '#f0f0f0')
                        return [f'background-color: {color}'] * len(row)

                # 색상 범례 표시
                st.subheader("🎨 색상 범례")
                
                legend_cols = st.columns(min(len(file_names) + 3, 6))
                
                # 공통 태그 색상 범례
                common_colors = ['#cce7ff', '#d4edda', '#fff3cd', '#f8d7da']
                for i in range(min(4, len(legend_cols))):
                    with legend_cols[i]:
                        st.markdown(f"**공통 태그 그룹 {i+1}**")
                        st.markdown(f'<div style="background-color: {common_colors[i]}; padding: 5px; border-radius: 3px; text-align: center;">그룹 {i+1}</div>', unsafe_allow_html=True)
                
                # 독립 태그 색상 범례
                for i, file_name in enumerate(file_names):
                    col_index = i + 4
                    if col_index < len(legend_cols):
                        with legend_cols[col_index]:
                            color = file_colors.get(file_name, '#f0f0f0')
                            st.markdown(f"**{file_name}**")
                            st.markdown(f'<div style="background-color: {color}; padding: 5px; border-radius: 3px; text-align: center;">{file_name}</div>', unsafe_allow_html=True)

                # 테이블 표시
                display_columns = [col for col in comparison_df.columns if col != '그룹_인덱스']
                display_df = comparison_df[display_columns]
                styled_df = display_df.style.apply(style_table, axis=1)

                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=600
                )
                
                # 테이블 다운로드
                csv_data = comparison_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 비교 테이블 CSV 다운로드",
                    data=csv_data,
                    file_name="excel_tag_comparison_table.csv",
                    mime="text/csv"
                )
                
                # 상세 분석 결과
                with st.expander("🔍 상세 분석 결과"):
                    st.subheader("🤝 공통 태그 분석")
                    common_tags = comparison_df[comparison_df['구분'] == '공통 태그']
                    if not common_tags.empty:
                        unique_common_tags = common_tags.drop_duplicates(subset=['태그명'])
                        for _, row in unique_common_tags.iterrows():
                            st.write(f"**{row['태그명']}** - {row['설명']}")
                            same_tag_rows = common_tags[common_tags['태그명'] == row['태그명']]
                            for _, tag_row in same_tag_rows.iterrows():
                                file_name = tag_row['파일']
                                tag_key = f'{file_name}_태그명'
                                desc_key = f'{file_name}_설명'
                                st.write(f"  - {file_name}: {tag_row[tag_key]} ({tag_row[desc_key]})")
                    else:
                        st.write("공통 태그가 없습니다.")
                    
                    st.subheader("🏠 독립 태그 분석")
                    unique_tags = comparison_df[comparison_df['구분'] != '공통 태그']
                    if not unique_tags.empty:
                        for file_name in file_names:
                            file_unique_tags = unique_tags[unique_tags['파일'] == file_name]
                            if not file_unique_tags.empty:
                                st.write(f"**{file_name} 전용 태그:**")
                                for _, row in file_unique_tags.iterrows():
                                    st.write(f"  - {row['태그명']}: {row['설명']}")
                    else:
                        st.write("독립 태그가 없습니다.")
            
            else:
                st.warning("⚠️ 비교할 데이터가 없습니다.")
        else:
            st.info("📊 먼저 '분석 결과' 탭에서 분석을 완료해주세요.")

if __name__ == "__main__":
    main()