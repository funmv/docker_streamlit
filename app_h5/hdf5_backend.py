"""
HDF5 데이터 분석 백엔드
데이터 로드, 처리, 계산 로직
"""
import pandas as pd
import numpy as np
import h5py
import json
import logging
from typing import List, Dict, Tuple, Optional
import io
import tempfile
import os


class HDF5Backend:
    """HDF5 파일 처리 및 데이터 계산"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_hdf5(self, file_bytes: bytes, filename: str) -> pd.DataFrame:
        """
        HDF5 파일(bytes)을 로드하여 DataFrame 반환
        
        Args:
            file_bytes: HDF5 파일 bytes
            filename: 파일 이름
        
        Returns:
            DataFrame
        """
        self.logger.info(f"=== HDF5 파일 로드: {filename} ===")
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            # DataFrame 로드
            df = pd.read_hdf(tmp_path, key='data')
            self.logger.info(f"✅ DataFrame 로드 완료: shape={df.shape}")
            
            # 메타데이터 로드 시도
            try:
                import tables
                
                with tables.open_file(tmp_path, 'r') as h5file:
                    group = h5file.get_node('/data')
                    
                    if hasattr(group._v_attrs, 'pandas_attrs'):
                        attrs_json = group._v_attrs.pandas_attrs
                        df.attrs = json.loads(attrs_json)
                        self.logger.info(f"✅ 메타데이터 로드 완료")
            
            except Exception as e:
                self.logger.debug(f"메타데이터 로드 실패 (무시): {e}")
            
            return df
        
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def calculate_expression(
        self,
        df: pd.DataFrame,
        expression: str,
        feature_map: Dict[str, str],
        shifts: Dict[str, int] = None,
        thresholds: Dict[str, float] = None,
        min_range: Optional[float] = None, # New
        max_range: Optional[float] = None  # New
    ) -> Tuple[pd.Series, bool, str]:
        """
        수식을 계산하여 결과 반환
        
        Args:
            df: 원본 DataFrame
            expression: 계산 수식 (예: "A + B / C")
            feature_map: 변수 -> 특징명 매핑 (예: {"A": "feature1", "B": "feature2"})
            shifts: 변수별 shift 값 (예: {"B": 32, "C": -10})
            thresholds: 변수별 임계값 (예: {"A": 10.0, "B": 5.0})
            min_range: 계산 결과 최소값 (이 값 미만은 0)
            max_range: 계산 결과 최대값 (이 값 초과는 0)
        
        Returns:
            (결과 Series, 성공 여부, 오류 메시지)
        """
        self.logger.info(f"=== 수식 계산 시작 ===")
        self.logger.info(f"수식: {expression}")
        self.logger.info(f"변수 매핑: {feature_map}")
        self.logger.info(f"Shift: {shifts}")
        self.logger.info(f"임계값: {thresholds}")
        self.logger.info(f"결과 범위 (Min): {min_range}")
        self.logger.info(f"결과 범위 (Max): {max_range}")
        
        if not expression or not feature_map:
            return None, False, "수식 또는 변수가 없습니다."
        
        try:
            # 변수를 실제 데이터로 치환할 namespace 생성
            namespace = {}
            
            for var_name, feature_name in feature_map.items():
                if feature_name not in df.columns:
                    return None, False, f"특징 '{feature_name}'을 찾을 수 없습니다."
                
                # 원본 데이터
                data = df[feature_name].values.copy()
                
                # Shift 적용 (계산용)
                if shifts and var_name in shifts:
                    shift_val = shifts[var_name]
                    if shift_val != 0:
                        self.logger.info(f" 	{var_name} Shift: {shift_val}")
                        if shift_val > 0:
                            # 우측 shift: 앞에 NaN 추가
                            data = np.concatenate([np.full(shift_val, np.nan), data[:-shift_val]])
                        else:
                            # 좌측 shift: 뒤에 NaN 추가
                            data = np.concatenate([data[-shift_val:], np.full(-shift_val, np.nan)])
                
                # 임계값 적용
                if thresholds and var_name in thresholds:
                    threshold = thresholds[var_name]
                    if not np.isnan(threshold):
                        self.logger.info(f" 	{var_name} 임계값: {threshold} (미만은 0 처리)")
                        # 임계값보다 작은 값은 0으로 처리
                        data = np.where(data < threshold, 0, data)
                
                namespace[var_name] = data
            
            # numpy 함수 추가
            namespace['np'] = np
            
            # 수식 계산
            with np.errstate(divide='ignore', invalid='ignore'):
                result = eval(expression, {"__builtins__": {}}, namespace)
            
            # 0으로 나누기 등으로 발생한 Inf, -Inf 값을 NaN으로 대체
            inf_mask = np.isinf(result)
            if np.any(inf_mask):
                num_infs = np.sum(inf_mask)
                self.logger.warning(
                    f"수식 계산 중 {num_infs}개의 Inf/ -Inf 값이 발견되었습니다. "
                    f"0에 가까운 값으로 나눈 것으로 보입니다. 이 값들을 NaN으로 대체합니다."
                )
                # Inf와 -Inf를 NaN으로 교체
                result[inf_mask] = np.nan
            
            # Series로 변환
            result_series = pd.Series(result, index=df.index)

            # --- [ ⭐ 수정된 부분 시작 (Clipping) ⭐ ] ---
            if min_range is not None and max_range is not None:
                if min_range > max_range:
                    return None, False, "Clipping 최소값이 최대값보다 클 수 없습니다."
                
                # 범위를 벗어나는 값들의 마스크 생성 (NaN은 이 비교에서 False가 됨)
                outside_range_mask = (result_series < min_range) | (result_series > max_range)
                
                num_clipped = np.sum(outside_range_mask)
                if num_clipped > 0:
                    self.logger.info(
                        f"Clipping: {num_clipped}개의 값이 범위 "
                        f"[{min_range}, {max_range}]를 벗어나 0으로 처리됩니다."
                    )
                    # 범위를 벗어난 값을 0으로 설정
                    result_series.loc[outside_range_mask] = 0
            # --- [ ⭐ 수정된 부분 종료 (Clipping) ⭐ ] ---

            self.logger.info(f"✅ 수식 계산 완료")
            self.logger.info(f" 	결과 통계: mean={np.nanmean(result_series):.2f}, std={np.nanstd(result_series):.2f}")
            
            return result_series, True, ""
        
        except Exception as e:
            error_msg = f"수식 계산 오류: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return None, False, error_msg
    
    def get_statistics(self, data: pd.Series, start_idx: int = None, end_idx: int = None) -> Dict[str, float]:
        """
        데이터의 통계 계산
        
        Args:
            data: 데이터 Series
            start_idx: 시작 인덱스 (None이면 전체)
            end_idx: 종료 인덱스 (None이면 전체)
        
        Returns:
            통계 딕셔너리
        """
        if start_idx is not None and end_idx is not None:
            data = data.iloc[start_idx:end_idx+1]
        
        return {
            '평균': np.nanmean(data),
            '표준편차': np.nanstd(data),
            '최소값': np.nanmin(data),
            '최대값': np.nanmax(data),
            '중앙값': np.nanmedian(data),
            '데이터 개수': len(data) - np.isnan(data).sum()
        }
    
    def merge_multiple_files(
        self,
        dataframes: List[pd.DataFrame],
        expression: str,
        feature_map: Dict[str, str],
        start_idx: int,
        end_idx: int,
        shifts: Dict[str, int] = None,
        thresholds: Dict[str, float] = None,
        min_range: Optional[float] = None, # New
        max_range: Optional[float] = None  # New
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        여러 파일의 계산 결과를 평균
        
        Args:
            dataframes: DataFrame 리스트
            expression: 수식
            feature_map: 변수 매핑
            start_idx: 시작 인덱스
            end_idx: 종료 인덱스
            shifts: Shift 값
            thresholds: 임계값
            min_range: 계산 결과 최소값
            max_range: 계산 결과 최대값
        
        Returns:
            (평균 Series, 통계)
        """
        self.logger.info(f"=== 다중 파일 병합 시작: {len(dataframes)}개 파일 ===")
        
        results = []
        
        for i, df in enumerate(dataframes):
            result, success, error = self.calculate_expression(
                df, expression, feature_map, shifts, thresholds,
                min_range, max_range # New params
            )
            
            if success:
                # 선택 구간만 추출
                result_segment = result.iloc[start_idx:end_idx+1]
                results.append(result_segment.values)
                self.logger.info(f" 	파일 {i+1}: 계산 완료")
            else:
                self.logger.warning(f" 	파일 {i+1}: 계산 실패 - {error}")
        
        if not results:
            return None, {}
        
        # 평균 계산
        avg_result = np.nanmean(results, axis=0)
        avg_series = pd.Series(avg_result)
        
        # 통계 계산
        stats = self.get_statistics(avg_series)
        
        self.logger.info(f"✅ 병합 완료: {len(results)}개 파일 평균")
        
        return avg_series, stats