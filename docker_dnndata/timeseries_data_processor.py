import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime
import json

# =================================================================================
# 한글 폰트 설정
# =================================================================================
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

# matplotlib 경고 제거를 위한 설정
plt.rcParams['figure.max_open_warning'] = 50

# =================================================================================
# 유틸리티 함수들
# =================================================================================
def initialize_session_state():
    """세션 상태 초기화"""
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = []
    if 'normalization_params' not in st.session_state:
        st.session_state.normalization_params = None
    if 'normalized_dataset' not in st.session_state:
        st.session_state.normalized_dataset = None
    if 'shifted_dataset' not in st.session_state:
        st.session_state.shifted_dataset = None
    if 'reshaped_dataset' not in st.session_state:
        st.session_state.reshaped_dataset = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None

def load_npy_dataset(uploaded_file):
    """NPY 파일에서 데이터셋 로드"""
    try:
        # NPY 파일 로드
        dataset = np.load(uploaded_file, allow_pickle=True).item()
        
        # 데이터셋 구조 확인
        required_keys = ['train_inputs', 'train_outputs', 'val_inputs', 'val_outputs', 'metadata']
        missing_keys = [key for key in required_keys if key not in dataset]
        
        if missing_keys:
            st.error(f"❌ 데이터셋에 필요한 키가 없습니다: {missing_keys}")
            return None, None
        
        # 특징 이름 추출
        metadata = dataset['metadata']
        feature_names = []
        
        if 'feature_info' in metadata:
            time_feature_names = metadata['feature_info'].get('time_feature_names', [])
            data_feature_names = metadata['feature_info'].get('data_feature_names', [])
            feature_names = time_feature_names + data_feature_names
        
        # 특징 이름이 없으면 기본 이름 사용
        if not feature_names and len(dataset['train_inputs']) > 0:
            total_features = dataset['train_inputs'].shape[2]
            feature_names = [f"Feature_{i+1}" for i in range(total_features)]
        
        return dataset, feature_names
        
    except Exception as e:
        st.error(f"❌ 파일 로드 중 오류 발생: {str(e)}")
        return None, None

def display_dataset_info(dataset, feature_names):
    """데이터셋 정보 표시"""
    st.subheader("📊 데이터셋 정보")
    
    # 기본 정보
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎓 훈련 샘플", f"{len(dataset['train_inputs']):,}")
    with col2:
        st.metric("🔬 검증 샘플", f"{len(dataset['val_inputs']):,}")
    with col3:
        st.metric("📈 특징 수", len(feature_names))
    with col4:
        st.metric("📊 총 샘플", f"{len(dataset['train_inputs']) + len(dataset['val_inputs']):,}")
    
    # 형태 정보
    with st.expander("📋 데이터 형태 정보"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**훈련 데이터:**")
            st.write(f"- 입력 형태: {dataset['train_inputs'].shape}")
            st.write(f"- 출력 형태: {dataset['train_outputs'].shape}")
        with col2:
            st.write("**검증 데이터:**")
            st.write(f"- 입력 형태: {dataset['val_inputs'].shape}")
            st.write(f"- 출력 형태: {dataset['val_outputs'].shape}")
    
    # 특징 이름
    with st.expander("🏷️ 특징 이름 목록"):
        feature_df = pd.DataFrame({
            '인덱스': range(len(feature_names)),
            '특징명': feature_names
        })
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    # 메타데이터 정보
    if 'metadata' in dataset:
        with st.expander("📋 메타데이터 정보"):
            metadata = dataset['metadata']
            st.json(metadata, expanded=False)

def calculate_normalization_params(train_inputs):
    """정규화 파라미터 계산 (Min-Max 정규화)"""
    num_features = train_inputs.shape[2]
    params = {}
    
    for i in range(num_features):
        feature_data = train_inputs[:, :, i].flatten()
        params[i] = {
            'min': float(np.min(feature_data)),
            'max': float(np.max(feature_data)),
            'range': float(np.max(feature_data) - np.min(feature_data))
        }
    
    return params

def apply_normalization(data, normalization_params):
    """Min-Max 정규화 적용"""
    normalized_data = data.copy()
    
    for i, params in normalization_params.items():
        if params['range'] > 0:  # 분모가 0이 되는 것을 방지
            normalized_data[:, :, i] = (data[:, :, i] - params['min']) / params['range']
        else:
            normalized_data[:, :, i] = 0  # 상수인 경우 0으로 설정
    
    return normalized_data

def apply_feature_shift(data, feature_idx, shift_amount):
    """특정 특징에 시프트 적용"""
    shifted_data = data.copy()
    
    if shift_amount > 0:  # 오른쪽으로 시프트 (지연)
        shifted_data[:, shift_amount:, feature_idx] = data[:, :-shift_amount, feature_idx]
        shifted_data[:, :shift_amount, feature_idx] = data[:, 0:1, feature_idx]  # 첫 번째 값으로 패딩
    elif shift_amount < 0:  # 왼쪽으로 시프트 (앞당김)
        shift_amount = abs(shift_amount)
        shifted_data[:, :-shift_amount, feature_idx] = data[:, shift_amount:, feature_idx]
        shifted_data[:, -shift_amount:, feature_idx] = data[:, -1:, feature_idx]  # 마지막 값으로 패딩
    
    return shifted_data

def select_features_from_dataset(dataset, input_features, output_features):
    """선택된 특징으로 데이터셋 재구성"""
    selected_dataset = {}
    
    # 입력 특징 선택
    selected_dataset['train_inputs'] = dataset['train_inputs'][:, :, input_features]
    selected_dataset['val_inputs'] = dataset['val_inputs'][:, :, input_features]
    
    # 출력 특징 선택
    selected_dataset['train_outputs'] = dataset['train_outputs'][:, :, output_features]
    selected_dataset['val_outputs'] = dataset['val_outputs'][:, :, output_features]
    
    # 메타데이터 업데이트
    if 'metadata' in dataset:
        selected_dataset['metadata'] = dataset['metadata'].copy()
        selected_dataset['metadata']['selected_input_features'] = input_features
        selected_dataset['metadata']['selected_output_features'] = output_features
    
    return selected_dataset

# =================================================================================
# 탭별 메인 함수들
# =================================================================================
def tab_data_input():
    """탭1: 데이터 입력"""
    st.header("📁 데이터 입력")
    st.markdown("NPY 형식의 다변량 시계열 데이터셋을 업로드하세요.")
    
    uploaded_file = st.file_uploader(
        "NPY 파일 선택",
        type=['npy'],
        help="딥러닝 학습용으로 전처리된 NPY 파일을 업로드하세요."
    )
    
    if uploaded_file is not None:
        with st.spinner("📊 데이터 로딩 중..."):
            dataset, feature_names = load_npy_dataset(uploaded_file)
            
            if dataset is not None:
                # 세션 상태에 저장
                st.session_state.dataset = dataset
                st.session_state.feature_names = feature_names
                st.session_state.metadata = dataset.get('metadata', {})
                
                st.success("✅ 데이터셋이 성공적으로 로드되었습니다!")
                
                # 데이터셋 정보 표시
                display_dataset_info(dataset, feature_names)
                
                # 샘플 데이터 미리보기
                with st.expander("👀 샘플 데이터 미리보기"):
                    if len(dataset['train_inputs']) > 0:
                        # 샘플 선택 방법 선택
                        col_method, col_select = st.columns([1, 2])
                        
                        with col_method:
                            selection_method = st.radio(
                                "샘플 선택 방법",
                                ["목록에서 선택", "직접 입력"],
                                key="sample_selection_method"
                            )
                        
                        with col_select:
                            if selection_method == "목록에서 선택":
                                sample_idx = st.selectbox(
                                    "미리볼 샘플 선택",
                                    range(min(10, len(dataset['train_inputs']))),
                                    key="sample_preview_select"
                                )
                            else:
                                max_sample = len(dataset['train_inputs']) - 1
                                sample_idx = st.number_input(
                                    f"샘플 번호 입력 (0~{max_sample})",
                                    min_value=0,
                                    max_value=max_sample,
                                    value=0,
                                    key="sample_preview_input"
                                )
                        
                        # 시각화할 특징 선택
                        max_features = min(5, len(feature_names))
                        selected_features = st.multiselect(
                            f"시각화할 특징 선택 (최대 {max_features}개)",
                            range(len(feature_names)),
                            default=list(range(max_features)),
                            format_func=lambda x: f"{x}: {feature_names[x]}",
                            key="preview_features"
                        )
                        
                        if selected_features:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**입력 시퀀스**")
                                fig_input = go.Figure()
                                input_sample = dataset['train_inputs'][sample_idx]
                                
                                for feature_idx in selected_features[:max_features]:
                                    fig_input.add_trace(go.Scatter(
                                        y=input_sample[:, feature_idx],
                                        mode='lines+markers',
                                        name=feature_names[feature_idx],
                                        line=dict(width=2)
                                    ))
                                
                                fig_input.update_layout(
                                    title="입력 시퀀스",
                                    xaxis_title="Time Steps",
                                    yaxis_title="Values",
                                    height=300
                                )
                                st.plotly_chart(fig_input, use_container_width=True)
                            
                            with col2:
                                st.markdown("**출력 시퀀스**")
                                fig_output = go.Figure()
                                output_sample = dataset['train_outputs'][sample_idx]
                                
                                for feature_idx in selected_features[:max_features]:
                                    fig_output.add_trace(go.Scatter(
                                        y=output_sample[:, feature_idx],
                                        mode='lines+markers',
                                        name=feature_names[feature_idx],
                                        line=dict(width=2)
                                    ))
                                
                                fig_output.update_layout(
                                    title="출력 시퀀스",
                                    xaxis_title="Time Steps",
                                    yaxis_title="Values",
                                    height=300
                                )
                                st.plotly_chart(fig_output, use_container_width=True)

def tab_normalization():
    """탭2: 정규화"""
    st.header("📏 데이터 정규화")
    
    if st.session_state.dataset is None:
        st.warning("⚠️ 먼저 '데이터 입력' 탭에서 데이터를 로드해주세요.")
        return
    
    dataset = st.session_state.dataset
    feature_names = st.session_state.feature_names
    
    st.markdown("Min-Max 정규화를 통해 모든 특징을 [0, 1] 범위로 정규화합니다.")
    
    # 정규화 파라미터 계산 또는 로드
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**정규화 파라미터 설정 방법:**")
        param_source = st.radio(
            "파라미터 소스 선택",
            ["데이터에서 자동 계산", "파일에서 로드"],
            key="param_source_method"
        )
    
    with col2:
        if param_source == "파일에서 로드":
            uploaded_param_file = st.file_uploader(
                "파라미터 파일 업로드",
                type=['json'],
                help="이전에 저장한 정규화 파라미터 JSON 파일을 업로드하세요.",
                key="param_file_upload"
            )
    
    # 파라미터 계산 또는 로드 실행
    if param_source == "데이터에서 자동 계산":
        if st.button("🔢 정규화 파라미터 계산", key="calc_norm_params"):
            with st.spinner("정규화 파라미터 계산 중..."):
                norm_params = calculate_normalization_params(dataset['train_inputs'])
                st.session_state.normalization_params = norm_params
                st.success("✅ 정규화 파라미터가 계산되었습니다!")
    
    elif param_source == "파일에서 로드":
        if uploaded_param_file is not None:
            if st.button("📁 파라미터 파일 로드", key="load_norm_params"):
                try:
                    with st.spinner("파라미터 파일 로딩 중..."):
                        param_data = json.load(uploaded_param_file)
                        
                        # 파라미터 형식 검증
                        if 'normalization_params' in param_data:
                            loaded_params = param_data['normalization_params']
                            
                            # 키를 정수로 변환 (JSON에서는 문자열로 저장됨)
                            norm_params = {}
                            for key, value in loaded_params.items():
                                norm_params[int(key)] = value
                            
                            st.session_state.normalization_params = norm_params
                            st.success("✅ 파라미터 파일이 로드되었습니다!")
                            
                            # 로드된 파라미터 정보 표시
                            st.info(f"📊 로드된 특징 수: {len(norm_params)}개")
                            
                        else:
                            st.error("❌ 올바른 파라미터 파일 형식이 아닙니다.")
                
                except Exception as e:
                    st.error(f"❌ 파라미터 파일 로드 중 오류: {str(e)}")
    
    # 정규화 파라미터 표시 및 수정
    if st.session_state.normalization_params is not None:
        st.subheader("📊 정규화 파라미터")
        
        norm_params = st.session_state.normalization_params.copy()
        
        # 파라미터 표시 및 수정 인터페이스
        with st.expander("✏️ 정규화 파라미터 확인 및 수정"):
            st.markdown("각 특징의 Min/Max 값을 확인하고 필요시 수정할 수 있습니다.")
            
            param_data = []
            for i, params in norm_params.items():
                param_data.append({
                    '특징 인덱스': i,
                    '특징명': feature_names[i] if i < len(feature_names) else f'Feature_{i}',
                    '최솟값': params['min'],
                    '최댓값': params['max'],
                    '범위': params['range']
                })
            
            param_df = pd.DataFrame(param_data)
            st.dataframe(param_df, use_container_width=True, hide_index=True)
            
            # 수정 인터페이스
            st.markdown("**파라미터 수정:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                modify_feature = st.selectbox(
                    "수정할 특징 선택",
                    range(len(feature_names)),
                    format_func=lambda x: f"{x}: {feature_names[x]}",
                    key="modify_feature_select"
                )
            
            with col2:
                current_min = norm_params[modify_feature]['min']
                new_min = st.number_input(
                    "새로운 최솟값",
                    value=current_min,
                    key="new_min_value"
                )
            
            with col3:
                current_max = norm_params[modify_feature]['max']
                new_max = st.number_input(
                    "새로운 최댓값",
                    value=current_max,
                    key="new_max_value"
                )
            
            if st.button("📝 파라미터 업데이트", key="update_params"):
                if new_max > new_min:
                    norm_params[modify_feature]['min'] = new_min
                    norm_params[modify_feature]['max'] = new_max
                    norm_params[modify_feature]['range'] = new_max - new_min
                    st.session_state.normalization_params = norm_params
                    st.success(f"✅ {feature_names[modify_feature]} 특징의 파라미터가 업데이트되었습니다!")
                    st.rerun()
                else:
                    st.error("❌ 최댓값은 최솟값보다 커야 합니다.")
        
        # 정규화 파라미터 저장 옵션
        st.markdown("---")
        st.subheader("💾 정규화 파라미터 저장")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            param_filename = st.text_input(
                "저장할 파일명",
                value="normalization_params",
                key="param_save_filename"
            )
        
        with col2:
            if st.button("💾 파라미터 저장", key="save_norm_params"):
                try:
                    # 파라미터를 JSON 형식으로 준비
                    save_data = {
                        'normalization_params': st.session_state.normalization_params,
                        'feature_names': feature_names,
                        'creation_time': datetime.now().isoformat(),
                        'total_features': len(feature_names),
                        'data_info': {
                            'train_samples': len(dataset['train_inputs']),
                            'input_shape': list(dataset['train_inputs'].shape),
                            'description': 'Min-Max normalization parameters'
                        }
                    }
                    
                    # JSON 문자열로 변환
                    json_str = json.dumps(save_data, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="📁 파라미터 파일 다운로드",
                        data=json_str,
                        file_name=f"{param_filename}.json",
                        mime="application/json",
                        help="정규화 파라미터를 JSON 파일로 다운로드합니다."
                    )
                    
                    st.success("✅ 파라미터 파일이 준비되었습니다!")
                    
                except Exception as e:
                    st.error(f"❌ 파라미터 저장 중 오류: {str(e)}")
        
        # 저장된 파라미터 미리보기
        with st.expander("👀 저장될 파라미터 미리보기"):
            preview_data = {
                'normalization_params': st.session_state.normalization_params,
                'feature_names': feature_names[:5] + (['...'] if len(feature_names) > 5 else []),  # 처음 5개만 표시
                'total_features': len(feature_names)
            }
            st.json(preview_data)
        
        # 정규화 적용
        if st.button("🎯 정규화 적용", key="apply_normalization"):
            with st.spinner("정규화 적용 중..."):
                normalized_dataset = {}
                
                # 모든 데이터에 정규화 적용
                normalized_dataset['train_inputs'] = apply_normalization(
                    dataset['train_inputs'], norm_params
                )
                normalized_dataset['train_outputs'] = apply_normalization(
                    dataset['train_outputs'], norm_params
                )
                normalized_dataset['val_inputs'] = apply_normalization(
                    dataset['val_inputs'], norm_params
                )
                normalized_dataset['val_outputs'] = apply_normalization(
                    dataset['val_outputs'], norm_params
                )
                
                # 메타데이터 복사
                if 'metadata' in dataset:
                    normalized_dataset['metadata'] = dataset['metadata'].copy()
                    normalized_dataset['metadata']['normalization_applied'] = True
                    normalized_dataset['metadata']['normalization_params'] = norm_params
                
                st.session_state.normalized_dataset = normalized_dataset
                st.success("✅ 정규화가 완료되었습니다!")
        
        # 정규화 결과 확인
        if st.session_state.normalized_dataset is not None:
            st.subheader("📈 정규화 결과 확인")
            
            normalized_dataset = st.session_state.normalized_dataset
            
            # 정규화 전후 비교를 위한 선택 옵션들
            col1, col2, col3 = st.columns(3)
            
            with col1:
                data_type = st.selectbox(
                    "데이터 타입 선택",
                    ["train_inputs", "train_outputs", "val_inputs", "val_outputs"],
                    key="norm_compare_data_type"
                )
            
            with col2:
                feature_to_compare = st.selectbox(
                    "비교할 특징 선택",
                    range(len(feature_names)),
                    format_func=lambda x: f"{x}: {feature_names[x]}",
                    key="norm_compare_feature"
                )
            
            with col3:
                sample_count = st.number_input(
                    "분석할 샘플 수",
                    min_value=10,
                    max_value=min(100000, len(normalized_dataset[data_type])),
                    value=min(100, len(normalized_dataset[data_type])),
                    key="norm_sample_count"
                )
            
            # 데이터 타입 한글 표시 매핑
            data_type_korean = {
                "train_inputs": "훈련용 입력",
                "train_outputs": "훈련용 라벨",
                "val_inputs": "검증용 입력", 
                "val_outputs": "검증용 라벨"
            }
            
            st.markdown(f"**선택된 데이터**: {data_type_korean[data_type]} - {feature_names[feature_to_compare]}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**정규화 전**")
                original_data = dataset[data_type][:sample_count, :, feature_to_compare].flatten()
                
                fig_before = go.Figure()
                fig_before.add_trace(go.Histogram(
                    x=original_data,
                    nbinsx=50,
                    name="정규화 전",
                    marker_color='lightblue'
                ))
                fig_before.update_layout(
                    title=f"정규화 전 분포<br>{data_type_korean[data_type]}",
                    xaxis_title="값",
                    yaxis_title="빈도",
                    height=350
                )
                st.plotly_chart(fig_before, use_container_width=True)
                
                # 통계 정보
                st.markdown("**통계 정보:**")
                stats_before = pd.DataFrame({
                    '통계량': ['최솟값', '최댓값', '평균', '표준편차', '중앙값'],
                    '값': [
                        f"{np.min(original_data):.6f}",
                        f"{np.max(original_data):.6f}",
                        f"{np.mean(original_data):.6f}",
                        f"{np.std(original_data):.6f}",
                        f"{np.median(original_data):.6f}"
                    ]
                })
                st.dataframe(stats_before, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**정규화 후**")
                normalized_data = normalized_dataset[data_type][:sample_count, :, feature_to_compare].flatten()
                
                fig_after = go.Figure()
                fig_after.add_trace(go.Histogram(
                    x=normalized_data,
                    nbinsx=50,
                    name="정규화 후",
                    marker_color='lightgreen'
                ))
                fig_after.update_layout(
                    title=f"정규화 후 분포<br>{data_type_korean[data_type]}",
                    xaxis_title="값 (0~1 범위)",
                    yaxis_title="빈도",
                    height=350
                )
                st.plotly_chart(fig_after, use_container_width=True)
                
                # 통계 정보
                st.markdown("**통계 정보:**")
                stats_after = pd.DataFrame({
                    '통계량': ['최솟값', '최댓값', '평균', '표준편차', '중앙값'],
                    '값': [
                        f"{np.min(normalized_data):.6f}",
                        f"{np.max(normalized_data):.6f}",
                        f"{np.mean(normalized_data):.6f}",
                        f"{np.std(normalized_data):.6f}",
                        f"{np.median(normalized_data):.6f}"
                    ]
                })
                st.dataframe(stats_after, use_container_width=True, hide_index=True)
            
            # 정규화 품질 검증
            st.markdown("---")
            st.subheader("🔍 정규화 품질 검증")
            
            # 모든 데이터 타입에 대한 정규화 범위 확인
            validation_results = []
            for dt in ["train_inputs", "train_outputs", "val_inputs", "val_outputs"]:
                data = normalized_dataset[dt]
                min_val = np.min(data)
                max_val = np.max(data)
                
                # 범위 벗어남 체크 (0~1 범위)
                out_of_range = (min_val < -0.001) or (max_val > 1.001)  # 작은 오차 허용
                
                validation_results.append({
                    '데이터 타입': data_type_korean[dt],
                    '최솟값': f"{min_val:.6f}",
                    '최댓값': f"{max_val:.6f}",
                    '정규화 상태': '✅ 정상' if not out_of_range else '❌ 범위 벗어남'
                })
            
            validation_df = pd.DataFrame(validation_results)
            st.dataframe(validation_df, use_container_width=True, hide_index=True)
            
            # 정규화 파라미터 확인
            with st.expander("📊 적용된 정규화 파라미터"):
                if st.session_state.normalization_params:
                    param_list = []
                    for i, params in st.session_state.normalization_params.items():
                        param_list.append({
                            '특징 인덱스': i,
                            '특징명': feature_names[i] if i < len(feature_names) else f'Feature_{i}',
                            '최솟값': f"{params['min']:.6f}",
                            '최댓값': f"{params['max']:.6f}",
                            '범위': f"{params['range']:.6f}"
                        })
                    
                    param_df = pd.DataFrame(param_list)
                    st.dataframe(param_df, use_container_width=True, hide_index=True)

def tab_feature_shift():
    """탭3: 특징 시프트"""
    st.header("↔️ 특징 시프트")
    
    # 사용할 데이터셋 결정
    if st.session_state.normalized_dataset is not None:
        current_dataset = st.session_state.normalized_dataset
        dataset_name = "정규화된 데이터셋"
    elif st.session_state.dataset is not None:
        current_dataset = st.session_state.dataset
        dataset_name = "원본 데이터셋"
    else:
        st.warning("⚠️ 먼저 '데이터 입력' 탭에서 데이터를 로드해주세요.")
        return
    
    feature_names = st.session_state.feature_names
    st.markdown(f"현재 사용 중인 데이터: **{dataset_name}**")
    st.markdown("특정 특징을 시간 축에서 앞당기거나 지연시킬 수 있습니다.")
    
    # 시프트 설정
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data_type = st.selectbox(
            "데이터 타입 선택",
            ["train_inputs", "train_outputs", "val_inputs", "val_outputs"],
            key="shift_data_type"
        )
    
    with col2:
        shift_feature = st.selectbox(
            "시프트할 특징 선택",
            range(len(feature_names)),
            format_func=lambda x: f"{x}: {feature_names[x]}",
            key="shift_feature_select"
        )
    
    with col3:
        shift_amount = st.number_input(
            "시프트 양 (틱)",
            min_value=-50,
            max_value=50,
            value=0,
            help="양수: 오른쪽 시프트(지연), 음수: 왼쪽 시프트(앞당김)",
            key="shift_amount"
        )
    
    with col4:
        st.markdown("**시프트 방향:**")
        if shift_amount > 0:
            st.info("🔜 오른쪽 시프트 (지연)")
        elif shift_amount < 0:
            st.info("🔙 왼쪽 시프트 (앞당김)")
        else:
            st.info("➡️ 시프트 없음")
    
    # 시프트 미리보기
    if shift_amount != 0:
        with st.expander("👀 시프트 미리보기"):
            sample_data = current_dataset[data_type][0, :, shift_feature]  # 첫 번째 샘플
            shifted_sample = apply_feature_shift(
                current_dataset[data_type][:1], shift_feature, shift_amount
            )[0, :, shift_feature]
            
            fig_preview = go.Figure()
            fig_preview.add_trace(go.Scatter(
                y=sample_data,
                mode='lines+markers',
                name='원본',
                line=dict(color='blue', width=2)
            ))
            fig_preview.add_trace(go.Scatter(
                y=shifted_sample,
                mode='lines+markers',
                name=f'시프트 ({shift_amount}틱)',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_preview.update_layout(
                title=f"{feature_names[shift_feature]} 특징 시프트 미리보기",
                xaxis_title="Time Steps",
                yaxis_title="값",
                height=400
            )
            st.plotly_chart(fig_preview, use_container_width=True)
    
    # 시프트 적용
    if st.button("🎯 시프트 적용", key="apply_shift"):
        if shift_amount == 0:
            st.warning("⚠️ 시프트 양이 0이므로 변경사항이 없습니다.")
        else:
            with st.spinner("시프트 적용 중..."):
                # 현재 데이터셋 복사
                if st.session_state.shifted_dataset is None:
                    shifted_dataset = {}
                    for key in current_dataset.keys():
                        if isinstance(current_dataset[key], np.ndarray):
                            shifted_dataset[key] = current_dataset[key].copy()
                        else:
                            shifted_dataset[key] = current_dataset[key]
                else:
                    shifted_dataset = st.session_state.shifted_dataset
                
                # 선택된 데이터에 시프트 적용
                shifted_dataset[data_type] = apply_feature_shift(
                    shifted_dataset[data_type], shift_feature, shift_amount
                )
                
                # 메타데이터 업데이트
                if 'metadata' in shifted_dataset:
                    if 'shift_history' not in shifted_dataset['metadata']:
                        shifted_dataset['metadata']['shift_history'] = []
                    
                    shifted_dataset['metadata']['shift_history'].append({
                        'data_type': data_type,
                        'feature_index': shift_feature,
                        'feature_name': feature_names[shift_feature],
                        'shift_amount': shift_amount,
                        'timestamp': datetime.now().isoformat()
                    })
                
                st.session_state.shifted_dataset = shifted_dataset
                st.success(f"✅ {feature_names[shift_feature]} 특징에 {shift_amount}틱 시프트가 적용되었습니다!")
    
    # 시프트 히스토리 표시
    if (st.session_state.shifted_dataset is not None and 
        'metadata' in st.session_state.shifted_dataset and
        'shift_history' in st.session_state.shifted_dataset['metadata']):
        
        st.subheader("📋 시프트 히스토리")
        
        shift_history = st.session_state.shifted_dataset['metadata']['shift_history']
        if shift_history:
            history_df = pd.DataFrame(shift_history)
            history_df = history_df[['data_type', 'feature_name', 'shift_amount', 'timestamp']]
            history_df.columns = ['데이터 타입', '특징명', '시프트 양', '적용 시간']
            
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
            if st.button("🗑️ 시프트 히스토리 초기화", key="clear_shift_history"):
                st.session_state.shifted_dataset = None
                st.success("✅ 시프트 히스토리가 초기화되었습니다!")
                st.rerun()
        else:
            st.info("📝 아직 적용된 시프트가 없습니다.")

def tab_sequence_reshape():
    """탭4: 시퀀스 길이 조정"""
    st.header("📐 시퀀스 길이 조정")
    
    # 사용할 데이터셋 결정
    if st.session_state.shifted_dataset is not None:
        current_dataset = st.session_state.shifted_dataset
        dataset_name = "시프트 적용된 데이터셋"
    elif st.session_state.normalized_dataset is not None:
        current_dataset = st.session_state.normalized_dataset
        dataset_name = "정규화된 데이터셋"
    elif st.session_state.dataset is not None:
        current_dataset = st.session_state.dataset
        dataset_name = "원본 데이터셋"
    else:
        st.warning("⚠️ 먼저 '데이터 입력' 탭에서 데이터를 로드해주세요.")
        return
    
    feature_names = st.session_state.feature_names
    st.markdown(f"현재 사용 중인 데이터: **{dataset_name}**")
    
    # 현재 데이터 형태 정보 표시
    st.subheader("📊 현재 데이터 형태")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**입력 데이터:**")
        current_input_shape = current_dataset['train_inputs'].shape
        st.write(f"- 훈련 입력: {current_input_shape}")
        st.write(f"- 검증 입력: {current_dataset['val_inputs'].shape}")
        st.write(f"- 현재 Lookback 길이: **{current_input_shape[1]}**")
    
    with col2:
        st.markdown("**출력 데이터:**")
        current_output_shape = current_dataset['train_outputs'].shape
        st.write(f"- 훈련 출력: {current_output_shape}")
        st.write(f"- 검증 출력: {current_dataset['val_outputs'].shape}")
        st.write(f"- 현재 Horizon 길이: **{current_output_shape[1]}**")
    
    st.markdown("---")
    
    # 새로운 시퀀스 길이 설정
    st.subheader("⚙️ 새로운 시퀀스 길이 설정")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_lookback = st.number_input(
            "새로운 Lookback 길이",
            min_value=1,
            max_value=current_input_shape[1],
            value=min(current_input_shape[1], 50),
            help="입력 시퀀스의 길이 (과거를 얼마나 볼 것인가)",
            key="new_lookback_length"
        )
    
    with col2:
        new_horizon = st.number_input(
            "새로운 Horizon 길이", 
            min_value=1,
            max_value=current_output_shape[1],
            value=min(current_output_shape[1], 10),
            help="출력 시퀀스의 길이 (미래를 얼마나 예측할 것인가)",
            key="new_horizon_length"
        )
    
    with col3:
        st.markdown("**데이터 추출 방법:**")
        st.info("📋 Lookback: 뒷부분에서 추출\n📋 Horizon: 앞부분에서 추출")
        st.markdown("*추출 방법이 고정되어 시퀀스 연속성을 보장합니다.*")
    
    # 변경 사항 미리보기
    if new_lookback != current_input_shape[1] or new_horizon != current_output_shape[1]:
        st.markdown("### 🔄 변경 사항 미리보기")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**변경 전 → 변경 후**")
            change_df = pd.DataFrame({
                '구분': ['Lookback 길이', 'Horizon 길이', '총 시퀀스 길이'],
                '변경 전': [
                    current_input_shape[1],
                    current_output_shape[1], 
                    current_input_shape[1] + current_output_shape[1]
                ],
                '변경 후': [
                    new_lookback,
                    new_horizon,
                    new_lookback + new_horizon
                ]
            })
            st.dataframe(change_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**예상 형태 변화**")
            shape_df = pd.DataFrame({
                '데이터 타입': ['train_inputs', 'train_outputs', 'val_inputs', 'val_outputs'],
                '현재 형태': [
                    str(current_dataset['train_inputs'].shape),
                    str(current_dataset['train_outputs'].shape),
                    str(current_dataset['val_inputs'].shape),
                    str(current_dataset['val_outputs'].shape)
                ],
                '변경 후 형태': [
                    f"({current_dataset['train_inputs'].shape[0]}, {new_lookback}, {current_dataset['train_inputs'].shape[2]})",
                    f"({current_dataset['train_outputs'].shape[0]}, {new_horizon}, {current_dataset['train_outputs'].shape[2]})",
                    f"({current_dataset['val_inputs'].shape[0]}, {new_lookback}, {current_dataset['val_inputs'].shape[2]})",
                    f"({current_dataset['val_outputs'].shape[0]}, {new_horizon}, {current_dataset['val_outputs'].shape[2]})"
                ]
            })
            st.dataframe(shape_df, use_container_width=True, hide_index=True)
        
        # 샘플 미리보기
        with st.expander("👀 변경 결과 미리보기"):
            sample_idx = 0  # 첫 번째 샘플로 미리보기
            
            # 현재 데이터 추출
            current_input = current_dataset['train_inputs'][sample_idx]
            current_output = current_dataset['train_outputs'][sample_idx]
            
            # 새로운 길이로 변환 (미리보기)
            # Lookback: 뒷부분에서 추출, Horizon: 앞부분에서 추출
            new_input_preview = current_input[-new_lookback:]  # 뒷부분에서
            new_output_preview = current_output[:new_horizon]  # 앞부분에서
            
            # 첫 번째 특징만 시각화
            feature_idx = 0
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature_{feature_idx}'
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_input = go.Figure()
                fig_input.add_trace(go.Scatter(
                    y=current_input[:, feature_idx],
                    mode='lines+markers',
                    name=f'현재 입력 (길이: {len(current_input)})',
                    line=dict(color='blue', width=2)
                ))
                fig_input.add_trace(go.Scatter(
                    y=new_input_preview[:, feature_idx],
                    mode='lines+markers',
                    name=f'변경 후 입력 (길이: {len(new_input_preview)})',
                    line=dict(color='red', width=2, dash='dash')
                ))
                fig_input.update_layout(
                    title=f"입력 시퀀스 변화 미리보기<br>{feature_name}",
                    xaxis_title="Time Steps",
                    yaxis_title="값",
                    height=300
                )
                st.plotly_chart(fig_input, use_container_width=True)
            
            with col2:
                fig_output = go.Figure()
                fig_output.add_trace(go.Scatter(
                    y=current_output[:, feature_idx],
                    mode='lines+markers',
                    name=f'현재 출력 (길이: {len(current_output)})',
                    line=dict(color='green', width=2)
                ))
                fig_output.add_trace(go.Scatter(
                    y=new_output_preview[:, feature_idx],
                    mode='lines+markers',
                    name=f'변경 후 출력 (길이: {len(new_output_preview)})',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                fig_output.update_layout(
                    title=f"출력 시퀀스 변화 미리보기<br>{feature_name}",
                    xaxis_title="Time Steps",
                    yaxis_title="값",
                    height=300
                )
                st.plotly_chart(fig_output, use_container_width=True)
    
    # 시퀀스 길이 변경 적용
    if st.button("🎯 시퀀스 길이 변경 적용", key="apply_sequence_reshape"):
        if new_lookback == current_input_shape[1] and new_horizon == current_output_shape[1]:
            st.warning("⚠️ 변경사항이 없습니다.")
        else:
            with st.spinner("시퀀스 길이 변경 적용 중..."):
                try:
                    reshaped_dataset = {}
                    
                    # 각 데이터 타입별로 크기 조정
                    # Lookback (입력): 뒷부분에서 추출
                    for data_type in ['train_inputs', 'val_inputs']:
                        current_data = current_dataset[data_type]
                        reshaped_dataset[data_type] = current_data[:, -new_lookback:, :]
                    
                    # Horizon (출력): 앞부분에서 추출
                    for data_type in ['train_outputs', 'val_outputs']:
                        current_data = current_dataset[data_type]
                        reshaped_dataset[data_type] = current_data[:, :new_horizon, :]
                    
                    # 메타데이터 복사 및 업데이트
                    if 'metadata' in current_dataset:
                        reshaped_dataset['metadata'] = current_dataset['metadata'].copy()
                        reshaped_dataset['metadata']['sequence_reshaped'] = True
                        reshaped_dataset['metadata']['reshape_info'] = {
                            'original_lookback': current_input_shape[1],
                            'original_horizon': current_output_shape[1],
                            'new_lookback': new_lookback,
                            'new_horizon': new_horizon,
                            'extraction_method': 'lookback_from_end_horizon_from_start',
                            'reshape_timestamp': datetime.now().isoformat()
                        }
                    
                    # 기타 정보 복사
                    for key in current_dataset.keys():
                        if key not in reshaped_dataset and key not in ['train_inputs', 'train_outputs', 'val_inputs', 'val_outputs']:
                            reshaped_dataset[key] = current_dataset[key]
                    
                    # 세션 상태에 저장
                    st.session_state.reshaped_dataset = reshaped_dataset
                    
                    st.success("✅ 시퀀스 길이 변경이 완료되었습니다!")
                    
                except Exception as e:
                    st.error(f"❌ 시퀀스 길이 변경 중 오류: {str(e)}")
    
    # 변경 결과 표시
    if hasattr(st.session_state, 'reshaped_dataset') and st.session_state.reshaped_dataset is not None:
        st.markdown("---")
        st.subheader("✅ 시퀀스 길이 변경 완료")
        
        reshaped_dataset = st.session_state.reshaped_dataset
        
        # 변경 후 형태 정보
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎓 훈련 샘플", f"{len(reshaped_dataset['train_inputs']):,}")
        with col2:
            st.metric("🔬 검증 샘플", f"{len(reshaped_dataset['val_inputs']):,}")
        with col3:
            st.metric("📈 새 Lookback", reshaped_dataset['train_inputs'].shape[1])
        with col4:
            st.metric("📊 새 Horizon", reshaped_dataset['train_outputs'].shape[1])
        
        # 상세 정보
        with st.expander("📋 변경된 데이터 상세 정보"):
            info_df = pd.DataFrame({
                '데이터 타입': ['train_inputs', 'train_outputs', 'val_inputs', 'val_outputs'],
                '변경 후 형태': [
                    str(reshaped_dataset['train_inputs'].shape),
                    str(reshaped_dataset['train_outputs'].shape),
                    str(reshaped_dataset['val_inputs'].shape),
                    str(reshaped_dataset['val_outputs'].shape)
                ]
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)
            
            if 'reshape_info' in reshaped_dataset.get('metadata', {}):
                reshape_info = reshaped_dataset['metadata']['reshape_info']
                st.json(reshape_info)

def tab_feature_selection():
    """탭5: 입출력 특징 선정"""
    st.header("🎯 입출력 특징 선정")
    
    # 사용할 데이터셋 결정 (우선순위: reshaped > shifted > normalized > original)
    if hasattr(st.session_state, 'reshaped_dataset') and st.session_state.reshaped_dataset is not None:
        current_dataset = st.session_state.reshaped_dataset
        dataset_name = "시퀀스 길이 조정된 데이터셋"
    elif st.session_state.shifted_dataset is not None:
        current_dataset = st.session_state.shifted_dataset
        dataset_name = "시프트 적용된 데이터셋"
    elif st.session_state.normalized_dataset is not None:
        current_dataset = st.session_state.normalized_dataset
        dataset_name = "정규화된 데이터셋"
    elif st.session_state.dataset is not None:
        current_dataset = st.session_state.dataset
        dataset_name = "원본 데이터셋"
    else:
        st.warning("⚠️ 먼저 '데이터 입력' 탭에서 데이터를 로드해주세요.")
        return
    
    feature_names = st.session_state.feature_names
    st.markdown(f"현재 사용 중인 데이터: **{dataset_name}**")
    st.markdown("모델 학습에 사용할 입력 특징과 출력 특징을 선택하세요.")
    
    # 특징 선택 인터페이스
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 입력 특징 선택")
        input_features = st.multiselect(
            "모델 입력으로 사용할 특징들을 선택하세요",
            range(len(feature_names)),
            default=list(range(min(5, len(feature_names)))),  # 기본값: 처음 5개 특징
            format_func=lambda x: f"{x}: {feature_names[x]}",
            key="input_features_select"
        )
        
        if input_features:
            st.write(f"**선택된 입력 특징 수:** {len(input_features)}")
            input_feature_names = [feature_names[i] for i in input_features]
            for i, name in enumerate(input_feature_names):
                st.write(f"  {i+1}. {name}")
    
    with col2:
        st.subheader("📤 출력 특징 선택")
        output_features = st.multiselect(
            "모델 출력으로 사용할 특징들을 선택하세요",
            range(len(feature_names)),
            default=list(range(min(3, len(feature_names)))),  # 기본값: 처음 3개 특징
            format_func=lambda x: f"{x}: {feature_names[x]}",
            key="output_features_select"
        )
        
        if output_features:
            st.write(f"**선택된 출력 특징 수:** {len(output_features)}")
            output_feature_names = [feature_names[i] for i in output_features]
            for i, name in enumerate(output_feature_names):
                st.write(f"  {i+1}. {name}")
    
    # 특징 선택 유효성 검사
    if not input_features:
        st.error("❌ 최소 1개 이상의 입력 특징을 선택해야 합니다.")
        return
    
    if not output_features:
        st.error("❌ 최소 1개 이상의 출력 특징을 선택해야 합니다.")
        return
    
    # 선택된 특징으로 데이터셋 재구성
    if st.button("🎯 선택된 특징으로 데이터셋 재구성", key="reconstruct_dataset"):
        with st.spinner("데이터셋 재구성 중..."):
            try:
                selected_dataset = select_features_from_dataset(
                    current_dataset, input_features, output_features
                )
                
                # 선택된 특징 이름들 저장
                selected_input_names = [feature_names[i] for i in input_features]
                selected_output_names = [feature_names[i] for i in output_features]
                
                selected_dataset['selected_input_names'] = selected_input_names
                selected_dataset['selected_output_names'] = selected_output_names
                
                st.session_state.selected_features = selected_dataset
                st.success("✅ 선택된 특징으로 데이터셋이 재구성되었습니다!")
                
            except Exception as e:
                st.error(f"❌ 데이터셋 재구성 중 오류 발생: {str(e)}")
    
    # 재구성된 데이터셋 정보 표시
    if st.session_state.selected_features is not None:
        st.markdown("---")
        st.subheader("✅ 재구성된 데이터셋 정보")
        
        selected_dataset = st.session_state.selected_features
        
        # 기본 정보
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎓 훈련 샘플", f"{len(selected_dataset['train_inputs']):,}")
        with col2:
            st.metric("🔬 검증 샘플", f"{len(selected_dataset['val_inputs']):,}")
        with col3:
            st.metric("📥 입력 특징", len(input_features))
        with col4:
            st.metric("📤 출력 특징", len(output_features))
        
        # 형태 정보
        with st.expander("📋 재구성된 데이터 형태"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**훈련 데이터:**")
                st.write(f"- 입력 형태: {selected_dataset['train_inputs'].shape}")
                st.write(f"- 출력 형태: {selected_dataset['train_outputs'].shape}")
            with col2:
                st.write("**검증 데이터:**")
                st.write(f"- 입력 형태: {selected_dataset['val_inputs'].shape}")
                st.write(f"- 출력 형태: {selected_dataset['val_outputs'].shape}")
        
        # 선택된 특징 요약
        with st.expander("🏷️ 선택된 특징 요약"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**입력 특징:**")
                input_df = pd.DataFrame({
                    '인덱스': input_features,
                    '특징명': selected_dataset['selected_input_names']
                })
                st.dataframe(input_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**출력 특징:**")
                output_df = pd.DataFrame({
                    '인덱스': output_features,
                    '특징명': selected_dataset['selected_output_names']
                })
                st.dataframe(output_df, use_container_width=True, hide_index=True)
        
        # 샘플 데이터 미리보기
        with st.expander("👀 재구성된 데이터 미리보기"):
            sample_idx = st.selectbox(
                "미리볼 샘플 선택",
                range(min(5, len(selected_dataset['train_inputs']))),
                key="final_sample_preview"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**입력 시퀀스 (선택된 특징)**")
                fig_input = go.Figure()
                input_sample = selected_dataset['train_inputs'][sample_idx]
                
                for i, feature_name in enumerate(selected_dataset['selected_input_names']):
                    fig_input.add_trace(go.Scatter(
                        y=input_sample[:, i],
                        mode='lines+markers',
                        name=feature_name,
                        line=dict(width=2)
                    ))
                
                fig_input.update_layout(
                    title="입력 시퀀스 (선택된 특징)",
                    xaxis_title="Time Steps",
                    yaxis_title="값",
                    height=300
                )
                st.plotly_chart(fig_input, use_container_width=True)
            
            with col2:
                st.markdown("**출력 시퀀스 (선택된 특징)**")
                fig_output = go.Figure()
                output_sample = selected_dataset['train_outputs'][sample_idx]
                
                for i, feature_name in enumerate(selected_dataset['selected_output_names']):
                    fig_output.add_trace(go.Scatter(
                        y=output_sample[:, i],
                        mode='lines+markers',
                        name=feature_name,
                        line=dict(width=2)
                    ))
                
                fig_output.update_layout(
                    title="출력 시퀀스 (선택된 특징)",
                    xaxis_title="Time Steps",
                    yaxis_title="값",
                    height=300
                )
                st.plotly_chart(fig_output, use_container_width=True)
        
        # 데이터셋 저장 및 다운로드
        st.subheader("💾 최종 데이터셋 저장")
        
        dataset_filename = st.text_input(
            "저장할 파일명",
            value="processed_dataset",
            key="final_dataset_filename"
        )
        
        if st.button("📦 최종 데이터셋 생성", key="generate_final_dataset"):
            try:
                with st.spinner("최종 데이터셋 생성 중..."):
                    # 메타데이터 업데이트
                    final_metadata = selected_dataset.get('metadata', {}).copy()
                    final_metadata.update({
                        'processing_complete': True,
                        'final_input_features': input_features,
                        'final_output_features': output_features,
                        'final_input_feature_names': selected_dataset['selected_input_names'],
                        'final_output_feature_names': selected_dataset['selected_output_names'],
                        'processing_timestamp': datetime.now().isoformat(),
                        'final_shapes': {
                            'train_inputs': selected_dataset['train_inputs'].shape,
                            'train_outputs': selected_dataset['train_outputs'].shape,
                            'val_inputs': selected_dataset['val_inputs'].shape,
                            'val_outputs': selected_dataset['val_outputs'].shape
                        }
                    })
                    
                    # 최종 데이터셋 구성
                    final_dataset = {
                        'train_inputs': selected_dataset['train_inputs'],
                        'train_outputs': selected_dataset['train_outputs'],
                        'val_inputs': selected_dataset['val_inputs'],
                        'val_outputs': selected_dataset['val_outputs'],
                        'metadata': final_metadata,
                        'input_feature_names': selected_dataset['selected_input_names'],
                        'output_feature_names': selected_dataset['selected_output_names']
                    }
                    
                    # NPY 형식으로 저장 준비
                    import io
                    buffer = io.BytesIO()
                    np.save(buffer, final_dataset)
                    dataset_data = buffer.getvalue()
                
                st.download_button(
                    label="💾 최종 데이터셋 다운로드",
                    data=dataset_data,
                    file_name=f"{dataset_filename}.npy",
                    mime="application/octet-stream",
                    help="처리된 최종 데이터셋을 다운로드합니다."
                )
                
                st.success("✅ 최종 데이터셋이 생성되었습니다! 다운로드 버튼을 클릭하세요.")
                
                # 사용 예시 코드
                with st.expander("🐍 Python 사용 예시 코드"):
                    st.code(f"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# 최종 데이터셋 로드
dataset = np.load('{dataset_filename}.npy', allow_pickle=True).item()

# 데이터 접근
train_inputs = dataset['train_inputs']    # Shape: {selected_dataset['train_inputs'].shape}
train_outputs = dataset['train_outputs']  # Shape: {selected_dataset['train_outputs'].shape}
val_inputs = dataset['val_inputs']        # Shape: {selected_dataset['val_inputs'].shape}
val_outputs = dataset['val_outputs']      # Shape: {selected_dataset['val_outputs'].shape}

# 특징 이름 확인
input_features = dataset['input_feature_names']   # {selected_dataset['selected_input_names']}
output_features = dataset['output_feature_names'] # {selected_dataset['selected_output_names']}

# 메타데이터 확인
metadata = dataset['metadata']
print("처리 완료:", metadata['processing_complete'])
print("선택된 입력 특징:", metadata['final_input_feature_names'])
print("선택된 출력 특징:", metadata['final_output_feature_names'])

# PyTorch 데이터로더 생성 예시
train_dataset = TensorDataset(
    torch.FloatTensor(train_inputs),
    torch.FloatTensor(train_outputs)
)
val_dataset = TensorDataset(
    torch.FloatTensor(val_inputs),
    torch.FloatTensor(val_outputs)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"훈련 데이터: {{len(train_dataset):,}}개 샘플")
print(f"검증 데이터: {{len(val_dataset):,}}개 샘플")
print(f"입력 차원: {{train_inputs.shape[1:]}}")
print(f"출력 차원: {{train_outputs.shape[1:]}}")
                    """, language="python")
                
            except Exception as e:
                st.error(f"❌ 최종 데이터셋 생성 중 오류: {str(e)}")

# =================================================================================
# 메인 애플리케이션
# =================================================================================
def main():
    # 페이지 설정 및 초기화
    st.set_page_config(
        page_title="다변량 시계열 데이터 처리 앱", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    setup_korean_font()
    initialize_session_state()
    
    # 메인 타이틀
    st.title("🔬 다변량 시계열 데이터 처리 도구")
    st.markdown("다변량 시계열 데이터의 전처리부터 딥러닝 모델 학습용 데이터 준비까지")
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 데이터 입력", 
        "📏 정규화", 
        "↔️ 특징 시프트", 
        "📐 시퀀스 길이 조정",
        "🎯 입출력 특징 선정"
    ])
    
    # 각 탭 실행
    with tab1:
        tab_data_input()
    
    with tab2:
        tab_normalization()
    
    with tab3:
        tab_feature_shift()
    
    with tab4:
        tab_sequence_reshape()
    
    with tab5:
        tab_feature_selection()
    
    # 사이드바 정보
    with st.sidebar:
        st.header("📊 현재 상태")
        
        # 데이터 로드 상태
        if st.session_state.dataset is not None:
            st.success("✅ 데이터 로드 완료")
        else:
            st.error("❌ 데이터 미로드")
        
        # 정규화 상태
        if st.session_state.normalized_dataset is not None:
            st.success("✅ 정규화 완료")
        else:
            st.warning("⚠️ 정규화 미완료")
        
        # 시프트 상태
        if st.session_state.shifted_dataset is not None:
            st.success("✅ 시프트 적용됨")
        else:
            st.info("ℹ️ 시프트 미적용")
        
        # 시퀀스 길이 조정 상태
        if hasattr(st.session_state, 'reshaped_dataset') and st.session_state.reshaped_dataset is not None:
            st.success("✅ 시퀀스 길이 조정 완료")
        else:
            st.info("ℹ️ 시퀀스 길이 조정 미적용")
        
        # 특징 선택 상태
        if st.session_state.selected_features is not None:
            st.success("✅ 특징 선택 완료")
        else:
            st.warning("⚠️ 특징 선택 미완료")
        
        st.markdown("---")
        
        # 진행 상황 표시
        progress_steps = [
            ("데이터 로드", st.session_state.dataset is not None),
            ("정규화", st.session_state.normalized_dataset is not None),
            ("시프트 적용", st.session_state.shifted_dataset is not None),
            ("시퀀스 길이 조정", hasattr(st.session_state, 'reshaped_dataset') and st.session_state.reshaped_dataset is not None),
            ("특징 선택", st.session_state.selected_features is not None)
        ]
        
        completed_steps = sum(1 for _, completed in progress_steps if completed)
        progress = completed_steps / len(progress_steps)
        
        st.subheader("📈 진행률")
        st.progress(progress)
        st.write(f"{completed_steps}/{len(progress_steps)} 단계 완료 ({progress:.0%})")
        
        # 각 단계별 상태
        for step_name, completed in progress_steps:
            if completed:
                st.write(f"✅ {step_name}")
            else:
                st.write(f"⭕ {step_name}")
        
        # 사용 가이드
        st.markdown("---")
        with st.expander("📖 사용 가이드"):
            st.markdown("""
            **1단계: 데이터 입력**
            - NPY 형식의 전처리된 데이터셋 업로드
            - 데이터 구조 및 특징 확인
            
            **2단계: 정규화** 
            - Min-Max 정규화 파라미터 계산
            - 필요시 파라미터 수동 조정
            - 모든 데이터에 정규화 적용
            
            **3단계: 특징 시프트**
            - 특정 특징을 시간축에서 이동
            - 양수: 지연, 음수: 앞당김
            - 여러 특징에 순차적 적용 가능
            
            **4단계: 시퀀스 길이 조정**
            - Lookback과 Horizon 길이 사용자 정의
            - 데이터 추출 방법 선택 가능
            - 시퀀스 변화 미리보기 제공
            
            **5단계: 입출력 특징 선정**
            - 모델 입력/출력 특징 선택
            - 최종 데이터셋 생성 및 다운로드
            - PyTorch 사용 예시 코드 제공
            """)
        
        # 풋노트
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
                🛠️ <strong>버전:</strong> V.2.0 (06-12-2025)<br>
                <hr>
            </div>
            """, 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()

