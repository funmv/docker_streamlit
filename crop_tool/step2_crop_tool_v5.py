import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import json
import tempfile
from datetime import datetime

# 한글 폰트 설정
try:
    from matplotlib import font_manager, rc
    # Windows 환경
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
except:
    try:
        # Linux 환경
        from matplotlib import font_manager, rc
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
    except:
        pass

# 페이지 설정
st.set_page_config(page_title="시계열 데이터 분석 및 분할", layout="wide")

# 세션 스테이트 초기화
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = {}
if 'cropped_data' not in st.session_state:
    st.session_state.cropped_data = []
if 'selected_files' not in st.session_state:
    st.session_state.selected_files = []
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'split_condition' not in st.session_state:
    st.session_state.split_condition = None
if 'feature_mapping' not in st.session_state:
    st.session_state.feature_mapping = {}


def read_hdf5_file(uploaded_file):
    """HDF5 파일 읽기 (header_metadata 포함)"""
    try:
        # UploadedFile을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # HDF5 읽기
            df = pd.read_hdf(tmp_path, key='data')

            # 메타데이터 복원 (header_metadata 포함)
            try:
                import tables
                import json

                with tables.open_file(tmp_path, 'r') as h5file:
                    group = h5file.get_node('/data')
                    if hasattr(group._v_attrs, 'pandas_attrs'):
                        attrs_json = group._v_attrs.pandas_attrs
                        pandas_attrs = json.loads(attrs_json)

                        # header_metadata 파싱
                        if 'header_metadata' in pandas_attrs:
                            header_meta = pandas_attrs['header_metadata']
                            # 문자열이면 다시 JSON 파싱
                            if isinstance(header_meta, str):
                                header_meta = json.loads(header_meta)
                            df.attrs['header_metadata'] = header_meta

                        # 기타 attrs도 저장
                        for key, value in pandas_attrs.items():
                            if key != 'header_metadata':
                                df.attrs[key] = value
            except:
                pass

            return df
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return None


def save_to_hdf5(df, file_path):
    """HDF5 파일로 저장"""
    try:
        # DataFrame 저장
        df.to_hdf(file_path, key='data', mode='w', format='fixed')

        # 메타데이터 저장 (attrs가 있으면)
        if hasattr(df, 'attrs') and df.attrs:
            try:
                import tables
                import json

                with tables.open_file(file_path, 'r+') as h5file:
                    group = h5file.get_node('/data')
                    attrs_json = json.dumps(df.attrs, default=str)
                    group._v_attrs.pandas_attrs = attrs_json
            except:
                pass

        return True
    except Exception as e:
        st.error(f"파일 저장 오류: {e}")
        return False


def get_searchable_feature_options(df):
    """
    DataFrame에서 검색 가능한 특징 옵션 생성
    - Description과 tag_name을 모두 포함하여 둘 다 검색 가능
    - 반환: (옵션 리스트, 표시명→실제컬럼명 매핑 딕셔너리)
    """
    feature_options = []
    feature_mapping = {}  # display_name -> actual_column_name

    columns = df.columns.tolist()

    # header_metadata에서 tag_name 가져오기
    header_meta = df.attrs.get('header_metadata', {})
    desc_list = header_meta.get('description', [])
    tag_list = header_meta.get('tag_name', header_meta.get('tag', []))

    for idx, col in enumerate(columns):
        # tag 정보 찾기
        tag = None
        if tag_list and idx < len(tag_list):
            tag = tag_list[idx]
            # 'Unnamed:' 로 시작하면 유효하지 않은 tag로 취급
            if tag and str(tag).startswith('Unnamed:'):
                tag = None

        # 표시명 생성: "Description [tag]" 형태
        if tag and tag != col:
            display_name = f"{col} [{tag}]"
        else:
            display_name = col

        feature_options.append(display_name)
        feature_mapping[display_name] = col

    return feature_options, feature_mapping


def plot_timeseries(data_dict, selected_files, selected_features):
    """시계열 데이터 플로팅"""
    if not selected_files or not selected_features:
        st.warning("파일과 특징을 선택해주세요.")
        return

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    fig = go.Figure()

    # y축 도메인 계산
    n_features = len(selected_features)
    spacing = 0.05
    height_per_plot = (1.0 - spacing * (n_features - 1)) / n_features

    # 각 특징별로 trace 추가
    for feat_idx, feature in enumerate(selected_features):
        # y축 도메인 계산 (위에서부터)
        y_start = 1.0 - (feat_idx + 1) * height_per_plot - feat_idx * spacing
        y_end = 1.0 - feat_idx * height_per_plot - feat_idx * spacing

        # 부동소수점 오차 방지를 위한 클리핑 (0~1 범위)
        y_start = max(0.0, min(1.0, y_start))
        y_end = max(0.0, min(1.0, y_end))

        # 파일별로 trace 추가
        for file_idx, file_name in enumerate(selected_files):
            df = data_dict[file_name]

            if feature in df.columns:
                color = colors[file_idx % len(colors)]
                yaxis_name = f'y{feat_idx + 1}' if feat_idx > 0 else 'y'

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[feature],
                        name=f"{file_name} - {feature}",
                        line=dict(color=color),
                        legendgroup=file_name,
                        showlegend=(feat_idx == 0),
                        yaxis=yaxis_name
                    )
                )

        # y축 설정
        yaxis_dict = {
            'domain': [y_start, y_end],
            'anchor': 'x',
            'title': feature
        }

        if feat_idx == 0:
            fig.update_layout(yaxis=yaxis_dict)
        else:
            fig.update_layout(**{f'yaxis{feat_idx + 1}': yaxis_dict})

    # 레이아웃 설정
    fig.update_layout(
        height=300 * n_features,
        xaxis=dict(title='Index'),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


def split_by_condition(df, column, operator, threshold, offset_left=0, offset_right=0, min_length=10):
    """조건에 따라 데이터 분할 (오프셋 지원)"""
    if column not in df.columns:
        st.error(f"'{column}' 컬럼을 찾을 수 없습니다.")
        return []

    # 조건에 맞는 행 찾기
    if operator == '>':
        mask = df[column] > threshold
    elif operator == '>=':
        mask = df[column] >= threshold
    elif operator == '<':
        mask = df[column] < threshold
    elif operator == '<=':
        mask = df[column] <= threshold
    elif operator == '==':
        mask = df[column] == threshold
    else:
        st.error(f"알 수 없는 연산자: {operator}")
        return []

    # 연속된 True 구간 찾기
    diff = np.diff(np.concatenate(([False], mask.values, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    segments = []

    # 각 구간에 offset 적용하여 추출
    for start, end in zip(starts, ends):
        # offset 적용 (범위를 벗어나지 않도록 클리핑)
        seg_start = max(0, start - offset_left)
        seg_end = min(len(df), end + offset_right)

        segment_length = seg_end - seg_start

        # 최소 길이 필터링
        if segment_length < min_length:
            continue

        segment_data = df.iloc[seg_start:seg_end].copy()
        segments.append({
            'start': seg_start,
            'end': seg_end,
            'data': segment_data,
            'original_start': int(start),
            'original_end': int(end),
            'adjusted_start': int(seg_start),
            'adjusted_end': int(seg_end)
        })

    return segments


def _apply_loaded_features():
    """
    file_uploader(on_change) 콜백:
    업로드된 JSON 파일에서 특징 리스트를 읽어 multiselect 위젯의 세션 값을 갱신.
    콜백은 위젯 재렌더링 직전에 실행되므로 st.rerun() 없이도 즉시 반영됨.
    """
    loaded_file = st.session_state.get('load_features_uploader')
    if loaded_file is None:
        # 사용자가 업로드 파일을 제거한 경우 — 별도 처리 없이 종료
        return

    feature_options = st.session_state.get('_current_feature_options', [])

    try:
        loaded_features = json.loads(loaded_file.read().decode('utf-8'))
        if not isinstance(loaded_features, list):
            st.session_state['_features_load_message'] = (
                "❌ JSON 파일은 특징 이름의 리스트여야 합니다."
            )
            return

        valid_features = [f for f in loaded_features if f in feature_options]
        invalid_count = len(loaded_features) - len(valid_features)

        # multiselect 위젯의 세션 값을 직접 갱신 (위젯 렌더 전이므로 안전)
        st.session_state['tab1_feature_select'] = valid_features

        if invalid_count > 0:
            st.session_state['_features_load_message'] = (
                f"⚠️ {len(valid_features)}개 특징을 불러왔습니다. "
                f"({invalid_count}개는 현재 파일에 없어 제외됨)"
            )
        else:
            st.session_state['_features_load_message'] = (
                f"✅ {len(valid_features)}개 특징을 불러왔습니다."
            )
    except Exception as e:
        st.session_state['_features_load_message'] = f"❌ 불러오기 실패: {e}"


def save_segments(segments, base_filename, output_dir):
    """추출된 구간을 HDF5 파일로 저장"""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, segment in enumerate(segments):
        output_filename = f"{base_filename}_segment_{idx+1}_{timestamp}.h5"
        output_path = os.path.join(output_dir, output_filename)

        if save_to_hdf5(segment['data'], output_path):
            saved_files.append(output_path)

    return saved_files


# ==================== UI 시작 ====================
st.title("🔍 시계열 데이터 분석 및 분할 도구")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["📊 데이터 로드 및 분할", "📈 분할된 데이터 보기", "🔄 일괄 처리"])

# ==================== 탭 1: 데이터 로드 및 분할 ====================
with tab1:
    st.header("데이터 로드 및 조건 기반 분할")

    # 사이드바: 파일 업로드
    with st.sidebar:
        st.subheader("1️⃣ HDF5 파일 업로드")

        uploaded_files = st.file_uploader(
            "Drag and drop file here\nLimit 200MB per file • HDF5",
            type=['h5', 'hdf5'],
            accept_multiple_files=True,
            help="여러 개의 HDF5 파일을 선택하거나 드래그 앤 드롭하세요"
        )

        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)}개 파일 업로드됨")

            if st.button("📂 파일 읽기", type="primary"):
                st.session_state.loaded_data = {}  # 기존 데이터 초기화
                with st.spinner("파일 읽는 중..."):
                    for uploaded_file in uploaded_files:
                        try:
                            df = read_hdf5_file(uploaded_file)
                            if df is not None:
                                file_name = uploaded_file.name
                                st.session_state.loaded_data[file_name] = df
                                st.success(f"✅ {file_name} 로드 완료 ({df.shape[0]} rows × {df.shape[1]} cols)")
                        except Exception as e:
                            st.error(f"❌ {uploaded_file.name} 로드 실패: {e}")

    # 메인 영역
    if st.session_state.loaded_data:
        st.success(f"✅ 총 {len(st.session_state.loaded_data)}개 파일 로드됨")

        # 파일 선택
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.selected_files = st.multiselect(
                "분석할 파일 선택",
                options=list(st.session_state.loaded_data.keys()),
                default=list(st.session_state.loaded_data.keys()),
                key='tab1_file_select'
            )

        if st.session_state.selected_files:
            # 첫 번째 파일의 컬럼을 기준으로
            first_df = st.session_state.loaded_data[st.session_state.selected_files[0]]

            # 검색 가능한 옵션 생성 (Description + tag)
            feature_options, feature_mapping = get_searchable_feature_options(first_df)

            # 세션에 매핑 저장 (다른 곳에서 사용)
            st.session_state.feature_mapping = feature_mapping

            # 콜백에서 사용할 수 있도록 현재 옵션을 세션에 저장
            st.session_state['_current_feature_options'] = feature_options

            with col2:
                # 위젯 초기값을 세션 상태로 관리 (default 인자와의 충돌 경고 방지)
                if 'tab1_feature_select' not in st.session_state:
                    st.session_state['tab1_feature_select'] = feature_options[
                        :min(3, len(feature_options))
                    ]
                else:
                    # 파일이 바뀌어 더 이상 존재하지 않는 항목은 자동 제거
                    st.session_state['tab1_feature_select'] = [
                        f for f in st.session_state['tab1_feature_select']
                        if f in feature_options
                    ]

                selected_display = st.multiselect(
                    "시각화할 특징 선택 (Description 또는 Tag로 검색)",
                    options=feature_options,
                    key='tab1_feature_select'
                )
                # 표시명 → 실제 컬럼명 변환
                st.session_state.selected_features = [
                    feature_mapping.get(d, d) for d in selected_display
                ]

                # ===== 저장 / 불러오기 한 줄에 배치 =====
                save_col, load_col = st.columns([1, 1])
                with save_col:
                    if selected_display:
                        features_json = json.dumps(
                            selected_display, ensure_ascii=False, indent=2
                        )
                        st.download_button(
                            label="💾 현재 선택 특징 저장 (JSON)",
                            data=features_json.encode('utf-8'),
                            file_name=f"selected_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            key='save_features_btn',
                            use_container_width=True
                        )
                with load_col:
                    # ===== 저장된 특징 불러오기 (on_change 콜백으로 위젯 값 갱신) =====
                    with st.expander("📂 저장된 특징 불러오기", expanded=False):
                        st.file_uploader(
                            "JSON 파일 선택",
                            type=['json'],
                            key='load_features_uploader',
                            label_visibility='collapsed',
                            on_change=_apply_loaded_features
                        )
                        # 콜백에서 남긴 메시지 표시 (한 번만)
                        msg = st.session_state.pop('_features_load_message', None)
                        if msg:
                            if msg.startswith('❌'):
                                st.error(msg)
                            elif msg.startswith('⚠️'):
                                st.warning(msg)
                            else:
                                st.success(msg)

            # 원본 데이터 시각화
            if st.session_state.selected_features:
                st.subheader("📊 원본 데이터 시각화")
                plot_timeseries(
                    st.session_state.loaded_data,
                    st.session_state.selected_files,
                    st.session_state.selected_features
                )

            # 분할 조건 설정
            st.subheader("✂️ 데이터 분할 조건 설정")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                target_file = st.selectbox(
                    "대상 파일",
                    options=st.session_state.selected_files
                )

            target_df = st.session_state.loaded_data[target_file]
            numeric_cols = target_df.select_dtypes(include=[np.number]).columns.tolist()

            with col2:
                split_column = st.selectbox(
                    "분할 기준 컬럼",
                    options=numeric_cols
                )

            with col3:
                operator = st.selectbox(
                    "연산자",
                    options=['>', '>=', '<', '<=', '==']
                )

            with col4:
                threshold = st.number_input(
                    "임계값",
                    value=float(target_df[split_column].median()) if split_column else 0.0
                )

            # 오프셋 및 최소 길이 설정
            col1, col2, col3 = st.columns(3)

            with col1:
                offset_left = st.number_input(
                    "좌측 오프셋 (샘플)",
                    value=0,
                    step=10,
                    help="조건 만족 시작 지점보다 앞쪽으로 포함할 샘플 수"
                )

            with col2:
                offset_right = st.number_input(
                    "우측 오프셋 (샘플)",
                    value=0,
                    step=10,
                    help="조건 만족 종료 지점보다 뒤쪽으로 포함할 샘플 수"
                )

            with col3:
                min_length = st.slider(
                    "최소 구간 길이 (샘플)",
                    min_value=1,
                    max_value=3000,
                    value=500,
                    step=10,
                    help="이 값보다 짧은 구간은 제외됩니다"
                )

            if st.button("🔪 데이터 분할 실행", type="primary"):
                with st.spinner("분할 중..."):
                    segments = split_by_condition(
                        target_df,
                        split_column,
                        operator,
                        threshold,
                        offset_left,
                        offset_right,
                        min_length
                    )

                    if segments:
                        st.session_state.cropped_data = segments
                        st.session_state.split_condition = {
                            'file': target_file,
                            'column': split_column,
                            'operator': operator,
                            'threshold': threshold,
                            'offset_left': offset_left,
                            'offset_right': offset_right,
                            'min_length': min_length
                        }
                        st.success(f"✅ {len(segments)}개 구간 추출 완료!")

                        # 구간 정보 표시
                        for idx, seg in enumerate(segments):
                            st.info(
                                f"구간 {idx+1}: 원본 [{seg['original_start']}:{seg['original_end']}] → "
                                f"조정 [{seg['adjusted_start']}:{seg['adjusted_end']}] "
                                f"(길이: {len(seg['data'])})"
                            )
                    else:
                        st.warning("⚠️ 조건을 만족하는 구간이 없습니다.")

            # 분할 결과 표시
            if st.session_state.cropped_data:
                st.subheader("📋 분할된 구간 정보")

                summary_data = []
                for idx, segment in enumerate(st.session_state.cropped_data):
                    summary_data.append({
                        '구간 번호': idx + 1,
                        '원본 시작': segment.get('original_start', segment['start']),
                        '원본 종료': segment.get('original_end', segment['end']),
                        '조정 시작': segment.get('adjusted_start', segment['start']),
                        '조정 종료': segment.get('adjusted_end', segment['end']),
                        '길이': len(segment['data']),
                        '행 수': segment['data'].shape[0],
                        '열 수': segment['data'].shape[1]
                    })

                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

                # 저장 옵션
                st.subheader("💾 분할 데이터 저장")

                col1, col2 = st.columns(2)
                with col1:
                    base_name = target_file.replace('.h5', '').replace('.hdf5', '')
                    output_filename = st.text_input(
                        "출력 파일명 (기본값)",
                        value=base_name
                    )

                with col2:
                    output_dir = st.text_input(
                        "출력 디렉토리",
                        value="./output"
                    )

                if st.button("💾 파일로 저장"):
                    with st.spinner("저장 중..."):
                        saved_files = save_segments(
                            st.session_state.cropped_data,
                            output_filename,
                            output_dir
                        )
                        st.success(f"✅ {len(saved_files)}개 파일 저장 완료!")
                        for file in saved_files:
                            st.text(f"📄 {file}")

# ==================== 탭 2: 분할된 데이터 보기 ====================
with tab2:
    st.header("분할된 구간 데이터 시각화")

    if not st.session_state.cropped_data:
        st.info("ℹ️ 먼저 '데이터 로드 및 분할' 탭에서 데이터를 분할해주세요.")
    else:
        # 분할 조건 표시
        if st.session_state.split_condition:
            cond = st.session_state.split_condition
            st.info(
                f"📊 분할 조건: {cond['file']} 파일의 {cond['column']} "
                f"{cond['operator']} {cond['threshold']}, "
                f"오프셋: [좌:{cond.get('offset_left', 0)}, 우:{cond.get('offset_right', 0)}], "
                f"최소 길이: {cond['min_length']}"
            )

        st.subheader("📋 구간 선택 및 특징 선택")

        col1, col2 = st.columns(2)

        with col1:
            # 구간 선택 (다중 선택)
            available_segments = list(range(len(st.session_state.cropped_data)))
            selected_segments = st.multiselect(
                "시각화할 구간 선택",
                options=available_segments,
                default=available_segments[:min(3, len(available_segments))],
                format_func=lambda x: f"구간 {x+1}",
                key='tab2_segment_select'
            )

        with col2:
            # 특징 선택
            if st.session_state.cropped_data:
                first_segment = st.session_state.cropped_data[0]
                segment_df = first_segment['data']

                # 검색 가능한 옵션 생성 (세션에 저장된 매핑 사용 또는 새로 생성)
                if hasattr(st.session_state, 'feature_mapping') and st.session_state.feature_mapping:
                    # 기존 매핑에서 현재 컬럼에 해당하는 것만 필터링
                    available_features = segment_df.columns.tolist()
                    reverse_mapping = {v: k for k, v in st.session_state.feature_mapping.items()}
                    feature_options_tab2 = [
                        reverse_mapping.get(col, col) for col in available_features
                    ]
                    feature_mapping_tab2 = {
                        reverse_mapping.get(col, col): col for col in available_features
                    }
                else:
                    # 매핑이 없으면 새로 생성
                    feature_options_tab2, feature_mapping_tab2 = get_searchable_feature_options(segment_df)

                selected_display_tab2 = st.multiselect(
                    "플롯할 특징 선택 (Description 또는 Tag로 검색)",
                    options=feature_options_tab2,
                    default=feature_options_tab2[:min(3, len(feature_options_tab2))],
                    key='tab2_feature_select'
                )
                # 표시명 → 실제 컬럼명 변환
                selected_plot_features = [
                    feature_mapping_tab2.get(d, d) for d in selected_display_tab2
                ]

        if selected_segments and selected_plot_features:
            # 각 구간별로 플롯
            for seg_idx in selected_segments:
                segment = st.session_state.cropped_data[seg_idx]
                df = segment['data']

                st.subheader(f"구간 {seg_idx + 1}")
                st.caption(
                    f"원본 인덱스: [{segment.get('original_start', segment['start'])}:{segment.get('original_end', segment['end'])}], "
                    f"조정 인덱스: [{segment.get('adjusted_start', segment['start'])}:{segment.get('adjusted_end', segment['end'])}], "
                    f"길이: {len(df)}"
                )

                # Figure 생성 (단일 x축, 여러 y축 방식)
                fig = go.Figure()

                # y축 도메인 계산
                n_features = len(selected_plot_features)
                spacing = 0.05
                height_per_plot = (1.0 - spacing * (n_features - 1)) / n_features

                # 각 특징별로 trace 추가
                for feat_idx, feature in enumerate(selected_plot_features):
                    if feature in df.columns:
                        # y축 도메인 계산 (위에서부터)
                        y_start = 1.0 - (feat_idx + 1) * height_per_plot - feat_idx * spacing
                        y_end = 1.0 - feat_idx * height_per_plot - feat_idx * spacing

                        # 부동소수점 오차 방지를 위한 클리핑 (0~1 범위)
                        y_start = max(0.0, min(1.0, y_start))
                        y_end = max(0.0, min(1.0, y_end))

                        yaxis_name = f'y{feat_idx + 1}' if feat_idx > 0 else 'y'

                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df[feature],
                                name=feature,
                                line=dict(width=2),
                                yaxis=yaxis_name
                            )
                        )

                        # y축 설정
                        yaxis_dict = {
                            'domain': [y_start, y_end],
                            'anchor': 'x',
                            'title': feature
                        }

                        # DIO 신호인지 확인 (0과 1만 있으면)
                        unique_vals = df[feature].dropna().unique()
                        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                            yaxis_dict['autorange'] = 'reversed'

                        if feat_idx == 0:
                            fig.update_layout(yaxis=yaxis_dict)
                        else:
                            fig.update_layout(**{f'yaxis{feat_idx + 1}': yaxis_dict})

                fig.update_layout(
                    height=250 * n_features,
                    showlegend=True,
                    hovermode='x unified',
                    xaxis=dict(
                        title="Index",
                        showspikes=True,
                        spikemode='across',
                        spikethickness=1,
                        spikedash='dot',
                        spikecolor='#999999'
                    )
                )

                config = {
                    'displayModeBar': True,
                    'displaylogo': False
                }

                st.plotly_chart(fig, use_container_width=True, config=config)
                st.divider()
        else:
            st.warning("구간과 특징을 선택해주세요.")



# ==================== 탭 3: 일괄 처리 ====================
with tab3:
    st.header("여러 파일 일괄 처리")
    st.info("💡 탭1에서 설정한 분할 조건을 여러 파일에 동시에 적용할 수 있습니다.")

    # 분할 조건 확인
    if st.session_state.split_condition is None:
        st.warning("⚠️ 먼저 탭1에서 분할 조건을 설정하고 '데이터 분할 실행' 버튼을 클릭해주세요.")
    else:
        # 저장된 조건 표시
        st.subheader("📋 저장된 분할 조건")
        cond = st.session_state.split_condition

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("파일", cond['file'])
            st.metric("기준 컬럼", cond['column'])
        with col2:
            st.metric("연산자", cond['operator'])
            st.metric("임계값", f"{cond['threshold']:.4f}")
        with col3:
            st.metric("좌측 오프셋", cond.get('offset_left', 0))
            st.metric("우측 오프셋", cond.get('offset_right', 0))
        with col4:
            st.metric("최소 구간 길이", f"{cond['min_length']} 샘플")

        st.divider()

        # 파일 선택
        st.subheader("1️⃣ 처리할 파일 선택")

        col1, col2 = st.columns([3, 1])

        with col1:
            # 로컬 파일 선택
            batch_files = st.file_uploader(
                "HDF5 파일 선택 (여러 개 가능)",
                type=['h5', 'hdf5'],
                accept_multiple_files=True,
                key='batch_uploader',
                help="일괄 처리할 HDF5 파일들을 선택하세요"
            )

        with col2:
            if batch_files:
                st.info(f"📁 {len(batch_files)}개 파일 선택됨")

        if batch_files:
            # 파일명 설정
            st.subheader("2️⃣ 출력 설정")

            col1, col2 = st.columns(2)
            with col1:
                add_prefix = st.checkbox("파일명에 접두사 추가", value=True)
                if add_prefix:
                    prefix = st.text_input("접두사", value="cropped_", key='batch_prefix')
                else:
                    prefix = ""

            with col2:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = st.text_input(
                    "ZIP 파일명",
                    value=f"batch_cropped_{timestamp}.zip",
                    key='batch_zip_name'
                )

            st.divider()

            # 일괄 처리 실행
            st.subheader("3️⃣ 일괄 처리 실행")

            col1, col2 = st.columns([1, 1])

            with col1:
                # 미리보기 옵션
                preview_mode = st.checkbox("미리보기 모드 (파일 생성 안 함)", value=False)

            with col2:
                process_button = st.button("🚀 일괄 처리 시작", type="primary", key='batch_start')

            if process_button:
                import zipfile
                from io import BytesIO

                progress_bar = st.progress(0)
                status_text = st.empty()

                results = []
                total_segments = 0

                # 메모리에 ZIP 파일 생성
                zip_buffer = BytesIO()

                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, uploaded_file in enumerate(batch_files):
                        status_text.text(f"처리 중: {uploaded_file.name} ({idx+1}/{len(batch_files)})")

                        try:
                            # 파일 읽기
                            df = read_hdf5_file(uploaded_file)

                            if df is None:
                                results.append({
                                    'file': uploaded_file.name,
                                    'status': '❌ 실패',
                                    'message': '파일 읽기 실패',
                                    'segments': 0
                                })
                                continue

                            # 조건에 맞는 컬럼이 있는지 확인
                            if cond['column'] not in df.columns:
                                results.append({
                                    'file': uploaded_file.name,
                                    'status': '❌ 실패',
                                    'message': f"컬럼 '{cond['column']}'을 찾을 수 없음",
                                    'segments': 0
                                })
                                continue

                            # 분할 실행
                            segments = split_by_condition(
                                df,
                                cond['column'],
                                cond['operator'],
                                cond['threshold'],
                                cond.get('offset_left', 0),
                                cond.get('offset_right', 0),
                                cond['min_length']
                            )

                            if segments:
                                if not preview_mode:
                                    # ZIP에 추가
                                    base_name = uploaded_file.name.replace('.h5', '').replace('.hdf5', '')
                                    saved_count = 0

                                    for seg_idx, segment in enumerate(segments):
                                        # 임시 파일에 HDF5 저장
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                                            tmp_path = tmp.name

                                        try:
                                            if save_to_hdf5(segment['data'], tmp_path):
                                                # ZIP에 추가
                                                output_filename = f"{prefix}{base_name}_seg{seg_idx+1}_{timestamp}.h5"
                                                zip_file.write(tmp_path, output_filename)
                                                saved_count += 1
                                        finally:
                                            # 임시 파일 삭제
                                            if os.path.exists(tmp_path):
                                                os.unlink(tmp_path)

                                    results.append({
                                        'file': uploaded_file.name,
                                        'status': '✅ 성공',
                                        'message': f'{saved_count}개 구간 저장됨',
                                        'segments': saved_count
                                    })
                                else:
                                    results.append({
                                        'file': uploaded_file.name,
                                        'status': '👁️ 미리보기',
                                        'message': f'{len(segments)}개 구간 발견',
                                        'segments': len(segments)
                                    })

                                total_segments += len(segments)
                            else:
                                results.append({
                                    'file': uploaded_file.name,
                                    'status': '⚠️ 조건 불만족',
                                    'message': '조건을 만족하는 구간이 없음',
                                    'segments': 0
                                })

                        except Exception as e:
                            results.append({
                                'file': uploaded_file.name,
                                'status': '❌ 오류',
                                'message': str(e),
                                'segments': 0
                            })

                        progress_bar.progress((idx + 1) / len(batch_files))

                status_text.text("✅ 일괄 처리 완료!")

                # 결과 표시
                st.subheader("📊 처리 결과")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                # 다운로드 버튼
                if not preview_mode and total_segments > 0:
                    zip_buffer.seek(0)

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="📥 ZIP 파일 다운로드",
                            data=zip_buffer.getvalue(),
                            file_name=zip_filename,
                            mime="application/zip",
                            type="primary",
                            use_container_width=True
                        )

                    st.success(f"🎉 총 {total_segments}개 구간이 추출되었습니다! ZIP 파일을 다운로드하세요.")
                elif preview_mode:
                    st.info(f"👁️ 미리보기 완료: 총 {total_segments}개 구간 발견")
                else:
                    st.warning("⚠️ 추출된 구간이 없어 ZIP 파일이 생성되지 않았습니다.")




