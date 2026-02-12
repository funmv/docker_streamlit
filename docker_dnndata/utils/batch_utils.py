"""
배치 처리 유틸리티
"""
import streamlit as st
import random
from typing import List, Dict, Tuple
from .file_utils import load_data_file
from .data_utils import apply_time_delay


def process_batch_files(files: List, selected_features: List[str], delays: Dict[str, int]) -> List[Dict]:
    """배치로 여러 파일에 지연 처리를 적용하는 함수"""
    processed_files = []

    for i, file in enumerate(files):
        try:
            # 파일 로드
            df = load_data_file(file)
            if df is None:
                continue

            # 선택된 특징들이 파일에 존재하는지 확인
            missing_features = [feat for feat in selected_features if feat not in df.columns]
            if missing_features:
                st.warning(f"⚠️ {file.name}에서 누락된 특징: {missing_features}")
                continue

            # 지연 처리 적용
            processed_df = df.copy()
            for feature in selected_features:
                delay = delays.get(feature, 0)
                if delay != 0:
                    shifted_series = apply_time_delay(df, feature, delay)
                    processed_df[feature] = shifted_series

            # 처리된 데이터 정보 저장
            processed_files.append({
                'original_name': file.name,
                'processed_name': f"{file.name.split('.')[0]}_batch_shifted.feather",
                'dataframe': processed_df,
                'shape': processed_df.shape,
                'applied_delays': {feat: delays[feat] for feat in selected_features if delays.get(feat, 0) != 0}
            })

        except Exception as e:
            st.error(f"❌ {file.name} 처리 중 오류: {str(e)}")
            continue

    return processed_files


def split_files_train_val(files: List, train_ratio: float = 0.8) -> Tuple[List, List]:
    """파일들을 훈련용과 검증용으로 분할"""
    total_files = len(files)
    train_size = int(total_files * train_ratio)

    # 파일들을 섞어서 분할
    shuffled_files = files.copy()
    random.shuffle(shuffled_files)

    train_files = shuffled_files[:train_size]
    val_files = shuffled_files[train_size:]

    return train_files, val_files
