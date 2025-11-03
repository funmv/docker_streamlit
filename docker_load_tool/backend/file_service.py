"""
파일 저장 서비스
HDF5 형식만 지원
"""
import pandas as pd
import tempfile
import logging
import json
from typing import Dict


class FileService:
    """파일 저장 서비스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def save_to_hdf5(self, df: pd.DataFrame, compression='gzip') -> bytes:
        """
        DataFrame을 HDF5로 저장하여 bytes 반환
        
        Args:
            df: 저장할 DataFrame (attrs 포함)
            compression: 압축 방식
        
        Returns:
            HDF5 파일 bytes
        """
        self.logger.info(f"=== HDF5 저장 시작 ===")
        self.logger.info(f"DataFrame shape: {df.shape}")
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
            tmp_path = tmp.name
        
        try:
            # DataFrame을 HDF5로 저장
            df.to_hdf(
                tmp_path, 
                key='data', 
                mode='w',
                format='fixed',
                complevel=9 if compression == 'gzip' else 0,
                complib=compression if compression != 'gzip' else 'zlib'
            )
            
            self.logger.info(f"✅ DataFrame HDF5 저장 완료")
            
            # 메타데이터를 그룹 속성으로 저장
            if hasattr(df, 'attrs'):
                try:
                    import tables
                    
                    with tables.open_file(tmp_path, 'r+') as h5file:
                        group = h5file.get_node('/data')
                        attrs_json = json.dumps(df.attrs, default=str)
                        group._v_attrs.pandas_attrs = attrs_json
                        
                        self.logger.info(f"✅ HDF5 메타데이터 저장 완료")
                
                except Exception as e:
                    self.logger.error(f"❌ HDF5 메타데이터 저장 실패: {e}")
            
            # 파일 읽기
            with open(tmp_path, 'rb') as f:
                file_bytes = f.read()
            
            self.logger.info(f"✅ HDF5 저장 완료: {len(file_bytes)} bytes")
            
            return file_bytes
        
        finally:
            # 임시 파일 삭제
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def load_from_hdf5(self, file_path: str) -> pd.DataFrame:
        """
        HDF5에서 DataFrame과 메타데이터 로드
        
        Args:
            file_path: HDF5 파일 경로
        
        Returns:
            DataFrame (attrs에 메타데이터 포함)
        """
        self.logger.info(f"=== HDF5 로드 시작: {file_path} ===")
        
        # DataFrame 로드
        df = pd.read_hdf(file_path, key='data')
        self.logger.info(f"DataFrame 로드 완료: shape={df.shape}")
        
        # 메타데이터 로드
        try:
            import tables
            
            with tables.open_file(file_path, 'r') as h5file:
                group = h5file.get_node('/data')
                
                if hasattr(group._v_attrs, 'pandas_attrs'):
                    attrs_json = group._v_attrs.pandas_attrs
                    df.attrs = json.loads(attrs_json)
                    
                    self.logger.info(f"✅ HDF5 메타데이터 로드 완료")
                else:
                    self.logger.warning("⚠️ HDF5 파일에 pandas_attrs가 없습니다.")
        
        except Exception as e:
            self.logger.error(f"❌ HDF5 메타데이터 로드 실패: {e}")
        
        return df
    
    def extract_date_range(self, df: pd.DataFrame) -> str:
        """
        DataFrame에서 timestamp 컬럼의 날짜 범위 추출
        
        Returns:
            날짜 범위 문자열 (예: "20250101_20250131" 또는 "20250115")
        """
        ts_col = None
        for col in df.columns:
            if 'timestamp' in str(col).lower() or 'datetime' in str(col).lower() or 'date' in str(col).lower():
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    ts_col = col
                    break
        
        if ts_col is None:
            return ""
        
        try:
            valid_dates = df[ts_col].dropna()
            
            if len(valid_dates) == 0:
                return ""
            
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            
            min_str = pd.to_datetime(min_date).strftime('%Y%m%d')
            max_str = pd.to_datetime(max_date).strftime('%Y%m%d')
            
            if min_str == max_str:
                return min_str
            else:
                return f"{min_str}_{max_str}"
        except:
            return ""