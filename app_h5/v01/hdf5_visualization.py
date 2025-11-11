"""
HDF5 데이터 분석 도구 - Visualization (Index 기반)
Plotly 기반 시각화 모듈
"""
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PlotlyVisualizer:
    """Plotly 기반 시각화 클래스 - Index 기반 X축"""
    
    @staticmethod
    def create_interactive_plot(
        df: pd.DataFrame, 
        x_col: Optional[str], 
        y_col: str, 
        title: str, 
        selected_range: Optional[Tuple[int, int]] = None
    ):
        """
        Plotly 기반 인터랙티브 플롯 생성 (Index 기반)
        
        Args:
            df: DataFrame
            x_col: timestamp 컬럼명 (hover 표시용, None이면 표시 안함)
            y_col: Y축 컬럼명
            title: 그래프 제목
            selected_range: 선택된 인덱스 범위 (start_idx, end_idx)
        """
        fig = go.Figure()
        
        # X축은 ALWAYS integer index
        x_values = np.arange(len(df))
        
        # Timestamp 데이터 준비 (hover용)
        timestamp_str = None
        if x_col and x_col in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    timestamp_str = df[x_col].dt.strftime('%Y-%m-%d %H:%M:%S').values
                else:
                    timestamp_str = pd.to_datetime(df[x_col]).dt.strftime('%Y-%m-%d %H:%M:%S').values
            except:
                timestamp_str = df[x_col].astype(str).values
        
        # Hover 템플릿 구성
        if timestamp_str is not None:
            hover_template = (
                '<b>Index</b>: %{x}<br>'
                '<b>Time</b>: %{customdata}<br>'
                '<b>Value</b>: %{y:.4f}<extra></extra>'
            )
            customdata = timestamp_str
        else:
            hover_template = (
                '<b>Index</b>: %{x}<br>'
                '<b>Value</b>: %{y:.4f}<extra></extra>'
            )
            customdata = None
        
        # 메인 데이터 플롯
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df[y_col].values,
            mode='lines',
            name=y_col,
            line=dict(color='blue', width=2),
            customdata=customdata,
            hovertemplate=hover_template
        ))
        
        # 선택된 구간 강조
        if selected_range is not None:
            start_idx, end_idx = selected_range
            
            # 인덱스 유효성 검사
            start_idx = max(0, int(start_idx))
            end_idx = min(len(df) - 1, int(end_idx))
            
            if start_idx <= end_idx:
                # 선택 구간의 데이터
                selected_x = x_values[start_idx:end_idx+1]
                selected_y = df[y_col].values[start_idx:end_idx+1]
                
                if timestamp_str is not None:
                    selected_customdata = timestamp_str[start_idx:end_idx+1]
                else:
                    selected_customdata = None
                
                # 선택된 구간을 빨간색으로 표시
                fig.add_trace(go.Scatter(
                    x=selected_x,
                    y=selected_y,
                    mode='lines',
                    name='선택 구간',
                    line=dict(color='red', width=3),
                    customdata=selected_customdata,
                    hovertemplate=hover_template
                ))
                
                # 구간 시작/종료 수직선
                y_min = df[y_col].min()
                y_max = df[y_col].max()
                y_margin = (y_max - y_min) * 0.05
                y_range = [y_min - y_margin, y_max + y_margin]
                
                # 시작 수직선
                fig.add_trace(go.Scatter(
                    x=[start_idx, start_idx],
                    y=y_range,
                    mode='lines',
                    name='구간 시작',
                    line=dict(color='green', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # 종료 수직선
                fig.add_trace(go.Scatter(
                    x=[end_idx, end_idx],
                    y=y_range,
                    mode='lines',
                    name='구간 종료',
                    line=dict(color='orange', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # X축 레이블
        if x_col:
            x_label = f"Index (Time: {x_col})"
        else:
            x_label = "Index"
        
        # 레이아웃 설정
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_col,
            hovermode='closest',
            height=500,
            showlegend=True,
            dragmode='select',  # box select 모드
            selectdirection='h',  # 수평 방향만 선택
        )
        
        # X축 타입 명시
        fig.update_xaxes(type='linear')
        
        # 선택 안내
        fig.add_annotation(
            text="💡 마우스로 구간을 드래그하여 선택하세요",
            xref="paper", yref="paper",
            x=0.5, y=1.08,
            showarrow=False,
            font=dict(size=12, color="gray"),
            bgcolor="lightyellow",
            bordercolor="orange",
            borderwidth=1
        )
        
        return fig
    
    @staticmethod
    def create_combined_result_plot(
        df: pd.DataFrame,
        x_col: Optional[str],
        feature_names: List[str],
        result: pd.Series,
        expression: str,
        selected_range: Optional[Tuple[int, int]] = None,
        feature_shifts: Optional[dict] = None
    ):
        """
        연산 결과와 사용된 특징들을 함께 표시하는 플롯
        모든 서브플롯에 마우스 포인터에 따라 십자선 표시
        특징별 Shift 적용 가능
        
        Args:
            df: DataFrame
            x_col: timestamp 컬럼명
            feature_names: 사용된 특징명 리스트
            result: 계산 결과 Series
            expression: 수식
            selected_range: 선택된 인덱스 범위
            feature_shifts: 특징별 shift 값 딕셔너리 {feature_name: shift_value}
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        if feature_shifts is None:
            feature_shifts = {}
        
        logger.info(f"=== Combined 플롯 생성: {len(feature_names)}개 특징 ===")
        
        # 서브플롯 개수 = 결과 1개 + 특징 N개
        n_plots = len(feature_names) + 1
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=n_plots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=[f"계산 결과: {expression}"] + [f"{chr(65+i)} = {name}" for i, name in enumerate(feature_names)]
        )
        
        # X축 데이터
        x_values = np.arange(len(df))
        
        # Timestamp 준비
        timestamp_str = None
        if x_col and x_col in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    timestamp_str = df[x_col].dt.strftime('%Y-%m-%d %H:%M:%S').values
                else:
                    timestamp_str = pd.to_datetime(df[x_col]).dt.strftime('%Y-%m-%d %H:%M:%S').values
            except:
                timestamp_str = df[x_col].astype(str).values
        
        # Hover 템플릿
        if timestamp_str is not None:
            hover_template = '<b>Index</b>: %{x}<br><b>Time</b>: %{customdata}<br><b>Value</b>: %{y:.4f}<extra></extra>'
            customdata = timestamp_str
        else:
            hover_template = '<b>Index</b>: %{x}<br><b>Value</b>: %{y:.4f}<extra></extra>'
            customdata = None
        
        # 1. 계산 결과 플롯 (첫 번째)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=result.values,
                mode='lines',
                name='결과',
                line=dict(color='red', width=2),
                customdata=customdata,
                hovertemplate=hover_template
            ),
            row=1, col=1
        )
        
        # 선택 구간 표시 (결과 플롯에만)
        if selected_range is not None:
            start_idx, end_idx = selected_range
            start_idx = max(0, int(start_idx))
            end_idx = min(len(df) - 1, int(end_idx))
            
            if start_idx <= end_idx:
                selected_x = x_values[start_idx:end_idx+1]
                selected_y = result.values[start_idx:end_idx+1]
                
                fig.add_trace(
                    go.Scatter(
                        x=selected_x,
                        y=selected_y,
                        mode='lines',
                        name='선택 구간',
                        line=dict(color='orange', width=3),
                        customdata=customdata[start_idx:end_idx+1] if customdata is not None else None,
                        hovertemplate=hover_template,
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # 2. 특징 플롯들
        colors = ['blue', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, feature_name in enumerate(feature_names):
            color = colors[i % len(colors)]
            
            # Shift 적용
            shift = feature_shifts.get(feature_name, 0)
            
            if shift == 0:
                # Shift 없음
                x_plot = x_values
                y_plot = df[feature_name].values
                customdata_plot = customdata
            else:
                # Shift 적용
                if shift > 0:
                    # 우측 이동: 앞부분 NaN, 뒷부분 데이터
                    x_plot = x_values
                    y_plot = np.concatenate([np.full(shift, np.nan), df[feature_name].values[:-shift]])
                    if customdata is not None:
                        customdata_plot = np.concatenate([[''] * shift, customdata[:-shift]])
                    else:
                        customdata_plot = None
                else:
                    # 좌측 이동: 앞부분 데이터, 뒷부분 NaN
                    shift_abs = abs(shift)
                    x_plot = x_values
                    y_plot = np.concatenate([df[feature_name].values[shift_abs:], np.full(shift_abs, np.nan)])
                    if customdata is not None:
                        customdata_plot = np.concatenate([customdata[shift_abs:], [''] * shift_abs])
                    else:
                        customdata_plot = None
            
            # 플롯 제목에 shift 정보 표시
            if shift != 0:
                subplot_title = f"{chr(65+i)} = {feature_name} (Shift: {shift:+d})"
            else:
                subplot_title = f"{chr(65+i)} = {feature_name}"
            
            # 서브플롯 제목 업데이트
            fig.layout.annotations[i+1].update(text=subplot_title)
            
            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode='lines',
                    name=f"{chr(65+i)}",
                    line=dict(color=color, width=1.5),
                    customdata=customdata_plot,
                    hovertemplate=hover_template
                ),
                row=i+2, col=1
            )
        
        # X축 레이블
        x_label = f"Index (Time: {x_col})" if x_col else "Index"
        
        # 레이아웃 설정
        fig.update_layout(
            height=250 * n_plots,  # 각 플롯당 250px
            showlegend=False,
            hovermode='x',  # x축 기준 hover
            title_text="연산 결과 및 사용 특징"
        )
        
        # X축은 마지막 플롯에만 레이블 표시
        fig.update_xaxes(title_text=x_label, row=n_plots, col=1)
        fig.update_xaxes(
            type='linear',
            showspikes=True,  # 수직선 표시
            spikemode='across',  # 전체 플롯 관통
            spikesnap='cursor',  # 커서 위치
            spikecolor='gray',  # 회색
            spikethickness=1,  # 선 두께
            spikedash='dot'  # 점선
        )
        
        # Y축 레이블
        fig.update_yaxes(title_text="결과", row=1, col=1)
        for i, feature_name in enumerate(feature_names):
            fig.update_yaxes(title_text=f"{chr(65+i)}", row=i+2, col=1)
        
        return fig