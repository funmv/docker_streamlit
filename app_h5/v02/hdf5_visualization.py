"""
HDF5 데이터 시각화 모듈
Plotly 기반 인터랙티브 그래프 생성
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple


class HDF5Visualizer:
    """데이터 시각화 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_combined_plot(
        self,
        df: pd.DataFrame,
        result_series: pd.Series,
        expression: str,
        feature_map: Dict[str, str],
        shifts: Dict[str, int] = None,
        thresholds: Dict[str, float] = None, # 이 인자는 여전히 받지만, 사용되지 않음
        selected_range: Tuple[int, int] = None,
        min_range: Optional[float] = None,
        max_range: Optional[float] = None
    ) -> go.Figure:
        """
        계산 결과 + 모든 사용 특징을 단일 x축, 다중 y축 플롯으로 생성
        
        Args:
            df: 원본 DataFrame
            result_series: 계산 결과
            expression: 수식
            feature_map: 변수 매핑
            shifts: Shift 값
            thresholds: 임계값 (더 이상 사용되지 않음)
            selected_range: 선택된 구간 (start_idx, end_idx)
            min_range: 계산 결과 최소값 (제목 표시용)
            max_range: 계산 결과 최대값 (제목 표시용)
        
        Returns:
            Plotly Figure
        """
        self.logger.info(f"=== Combined (Unified Hover) 플롯 생성 ===")
        
        fig = go.Figure()
        
        # 1. 플롯할 데이터 목록 생성
        plots_to_make = []
        
        # 1a. 계산 결과
        result_title = f"계산 결과: {expression}"
        if min_range is not None and max_range is not None:
            result_title += f" (Clipped: [{min_range}, {max_range}] -> 0)"

        plots_to_make.append({
            'data': result_series,
            'name': f"계산 결과: {expression}", # 범례 이름
            'title': result_title,          # Y축 제목
            'is_dio': False
        })
        
        # 1b. 사용된 특징들
        for var_name, feature_name in feature_map.items():
            if feature_name in df.columns:
                data = df[feature_name]
                
                # DIO 신호인지 확인 (0과 1만 있으면)
                unique_vals = data.dropna().unique()
                is_dio = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
                
                # 플롯 제목 (Shift만 포함, 임계값 제거)
                plot_title = f"{var_name} = {feature_name}"
                if shifts and var_name in shifts and shifts[var_name] != 0:
                    plot_title += f" (Shift: {shifts[var_name]:+d})"
                # --- [ ⭐ 수정된 부분 (임계값 표시 제거) ⭐ ] ---
                # if thresholds and var_name in thresholds and not np.isnan(thresholds[var_name]):
                #     plot_title += f" (임계값: {thresholds[var_name]:.1f})"
                # --- [ ⭐ 수정된 부분 종료 ⭐ ] ---
                
                plots_to_make.append({
                    'data': data,
                    'name': feature_name, # 범례(legend)에 표시될 이름
                    'title': plot_title,   # Y축에 표시될 이름
                    'is_dio': is_dio
                })

        # 2. y축 도메인 계산
        n_features = len(plots_to_make)
        if n_features == 0:
            self.logger.warning("플롯할 데이터가 없습니다.")
            return go.Figure() # 플롯할 게 없음
            
        spacing = 0.05
        if n_features > 5:
            spacing = 0.02 
            
        height_per_plot = (1.0 - spacing * (n_features - 1)) / n_features

        # X축 데이터 (인덱스)
        x_values = df.index
        
        # 3. 각 특징별로 trace 추가
        layout_updates = {}
        
        for feat_idx, plot_info in enumerate(plots_to_make):
            y_start = 1.0 - (feat_idx + 1) * height_per_plot - feat_idx * spacing
            y_end = 1.0 - feat_idx * height_per_plot - feat_idx * spacing
            
            y_start = max(0.0, min(1.0, y_start))
            y_end = max(0.0, min(1.0, y_end))
            
            yaxis_name = f'y{feat_idx + 1}' if feat_idx > 0 else 'y'
            
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=plot_info['data'],
                    name=plot_info['name'],
                    line=dict(width=2),
                    yaxis=yaxis_name
                )
            )
            
            # y축 설정
            yaxis_dict = {
                'domain': [y_start, y_end],
                'anchor': 'x',
                'title': plot_info['title'],
                'zeroline': False,
                'showline': True,
            }
            
            # DIO 신호인 경우 y축 뒤집기
            if plot_info['is_dio']:
                yaxis_dict['autorange'] = 'reversed'
            
            if feat_idx == 0:
                layout_updates['yaxis'] = yaxis_dict
            else:
                layout_updates[f'yaxis{feat_idx + 1}'] = yaxis_dict
        
        # 4. (Optional) Hover 템플릿
        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in str(col).lower() or 'datetime' in str(col).lower():
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    timestamp_col = col
                    break
        
        if timestamp_col:
            timestamps = df[timestamp_col].dt.strftime('%Y-%m-%d %H:%M:%S').values
            customdata = np.column_stack([x_values, timestamps])
            hover_template = (
                "<b>Index:</b> %{customdata[0]}<br>"
                "<b>Time:</b> %{customdata[1]}<br>"
                "<b>Value:</b> %{y:.2f}<br>"
                "<extra></extra>"
            )
        else:
            customdata = np.column_stack([x_values])
            hover_template = (
                "<b>Index:</b> %{customdata[0]}<br>"
                "<b>Value:</b> %{y:.2f}<br>"
                "<extra></extra>"
            )
            
        fig.update_traces(customdata=customdata, hovertemplate=hover_template)

        # 5. 전체 레이아웃 업데이트
        fig.update_layout(
            height=max(500, 180 * n_features),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            
            hovermode='x unified',
            xaxis=dict(
                title="Index",
                showspikes=True,
                spikemode='across',
                spikethickness=1,
                spikedash='dot',
                spikecolor='#999999'
            ),
            # --- [ ⭐ 핵심 수정 사항 (dragmode) ⭐ ] ---
            dragmode='zoom', # 초기 툴바 모드를 'zoom'으로 설정
            # --- [ ⭐ 수정된 부분 종료 ⭐ ] ---
            
            **layout_updates
        )
        
        # 6. 선택 구간 표시 (vrect)
        if selected_range:
            fig.add_vrect(
                x0=selected_range[0], x1=selected_range[1],
                fillcolor="LightSalmon", opacity=0.3,
                layer="below", line_width=0,
            )

        return fig