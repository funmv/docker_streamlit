"""
HDF5 데이터 분석 도구 - Backend
데이터 처리, 분석, 계산 로직
"""
import pandas as pd
import numpy as np
import io
import tempfile
import os
import json
import logging
from typing import List, Dict, Tuple, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HDF5Analyzer:
    """HDF5 파일 분석 클래스"""
    
    @staticmethod
    def load_hdf5(file_bytes: bytes) -> pd.DataFrame:
        """HDF5 파일(bytes)을 로드하여 DataFrame 반환"""
        logger.info("=== HDF5 파일 로드 시작 ===")
        
        with io.BytesIO(file_bytes) as buffer:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                tmp.write(buffer.read())
                tmp_path = tmp.name
            
            try:
                # DataFrame 로드
                df = pd.read_hdf(tmp_path, key='data')
                logger.info(f"✅ DataFrame 로드 성공: shape={df.shape}")
                
                # 메타데이터 로드 시도
                try:
                    import tables
                    with tables.open_file(tmp_path, 'r') as h5file:
                        group = h5file.get_node('/data')
                        if hasattr(group._v_attrs, 'pandas_attrs'):
                            attrs_json = group._v_attrs.pandas_attrs
                            df.attrs = json.loads(attrs_json)
                except Exception as e:
                    pass  # 메타데이터 로드 실패는 무시
                
                return df
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> List[str]:
        """숫자형 컬럼만 추출"""
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return cols
    
    @staticmethod
    def get_time_column(df: pd.DataFrame) -> Optional[str]:
        """시간 관련 컬럼 찾기"""
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['time', 'timestamp', 'date', 'datetime']):
                logger.info(f"✅ 시간 컬럼 발견: {col}")
                return col
        
        # 시간 컬럼이 없으면 인덱스 확인
        if pd.api.types.is_datetime64_any_dtype(df.index):
            logger.info("✅ 인덱스가 datetime 타입")
            return None
        
        return None


class ExpressionCalculator:
    """인덱스 기반 수식 계산기 - 특징을 A, B, C로 참조"""
    
    @staticmethod
    def calculate_custom(
        df: pd.DataFrame,
        selected_features: List[str],
        expression: str
    ) -> pd.Series:
        """
        사용자 정의 수식 계산
        
        Args:
            df: DataFrame
            selected_features: 선택된 특징명 리스트 (순서대로 A, B, C, ...)
            expression: A, B, C를 사용한 수식 (예: "(A + B) / 2")
        
        Returns:
            계산 결과 Series
        """
        logger.info("=== 사용자 정의 수식 계산 시작 ===")
        logger.info(f"수식: {expression}")
        
        # 네임스페이스 구성
        namespace = {
            'np': np,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'log10': np.log10,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'pi': np.pi,
            'e': np.e
        }
        
        # 선택된 특징을 A, B, C로 매핑
        for idx, col in enumerate(selected_features):
            var_name = chr(65 + idx)  # A, B, C, ...
            namespace[var_name] = df[col]
        
        try:
            # 수식 평가
            result = eval(expression, {"__builtins__": {}}, namespace)
            
            # Series로 변환
            if isinstance(result, pd.Series):
                logger.info(f"✅ 계산 성공")
                return result
            elif isinstance(result, (int, float, np.ndarray)):
                logger.info(f"✅ 계산 성공 (스칼라 → Series 변환)")
                return pd.Series(result, index=df.index)
            else:
                logger.error(f"❌ 유효하지 않은 결과 타입: {type(result)}")
                raise ValueError("수식 결과가 유효하지 않습니다.")
        
        except Exception as e:
            logger.error(f"❌ 수식 계산 오류: {e}")
            available_vars = [k for k in namespace.keys() if not k.startswith('_') and len(k) == 1]
            error_msg = f"수식 오류: {str(e)}\n"
            error_msg += f"수식: {expression}\n"
            error_msg += f"사용 가능한 변수: {', '.join(sorted(available_vars))}"
            raise ValueError(error_msg)


class RangeSelector:
    """구간 선택 관련 유틸리티 - Index 기반"""
    
    @staticmethod
    def create_range_mask(
        df: pd.DataFrame,
        selected_range: Tuple[int, int]
    ) -> pd.Series:
        """
        선택된 인덱스 구간에 대한 마스크 생성
        
        Args:
            df: DataFrame
            selected_range: (start_idx, end_idx)
        
        Returns:
            Boolean mask Series
        """
        start_idx, end_idx = selected_range
        
        # 인덱스 유효성 검사
        start_idx = max(0, int(start_idx))
        end_idx = min(len(df) - 1, int(end_idx))
        
        # 마스크 생성
        mask = pd.Series([False] * len(df), index=df.index)
        mask.iloc[start_idx:end_idx+1] = True
        
        return mask


class StatisticsCalculator:
    """통계 계산 유틸리티"""
    
    @staticmethod
    def calculate_statistics(data: pd.Series, label: str) -> Dict:
        """데이터 통계 계산"""
        stats = {
            '항목': label,
            '평균': f"{data.mean():.4f}",
            '표준편차': f"{data.std():.4f}",
            '최소값': f"{data.min():.4f}",
            '최대값': f"{data.max():.4f}",
            '중앙값': f"{data.median():.4f}",
            '데이터 수': str(len(data))  # 문자열로 변환하여 Arrow 호환성 확보
        }
        
        return stats