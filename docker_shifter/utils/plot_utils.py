"""
플롯 생성 유틸리티
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict
from .data_utils import apply_time_delay, get_data_segment
from .file_utils import load_data_file


def create_multivariate_plot(df: pd.DataFrame, selected_cols: List[str],
                            delays: Dict[str, int], downsample_rate: int = 1,
                            crosshair: bool = True, num_segments: int = 3,
                            selected_segment: int = 0) -> go.Figure:
    """기본 다변량 시계열 플롯을 생성하는 함수"""
    # 데이터 구간 선택
    df_segment = get_data_segment(df, num_segments, selected_segment)

    fig = go.Figure()

    for col in selected_cols:
        delay = delays.get(col, 0)

        # 1단계: 선택된 구간에서 시간 지연 적용
        y_data = apply_time_delay(df_segment, col, delay)

        # 2단계: 지연 적용된 데이터에 다운샘플링 적용
        y = y_data.iloc[::downsample_rate]
        x = df_segment.index[::downsample_rate]

        # 지연값이 있는 경우 레이블에 표시
        label = f"{col} (delay: {delay})" if delay != 0 else col

        fig.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode='lines',
            name=label,
            showlegend=True,
            hoverinfo='x',
            hovertemplate=''
        ))

    # 구간 정보를 제목에 추가
    segment_info = f"구간 {selected_segment + 1}/{num_segments}"
    fig.update_layout(
        title=f"📊 다변량 시계열 신호 분석 ({segment_info})",
        dragmode="zoom",
        xaxis=dict(
            rangeslider=dict(visible=False),
            title="시간 인덱스"
        ),
        yaxis=dict(
            title="신호 값"
        ),
        height=600
    )

    if crosshair:
        fig.update_layout(
            hovermode="x",
            xaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="red",
                spikethickness=1,
                title="시간 인덱스"
            ),
            yaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="blue",
                spikethickness=1,
                title="신호 값"
            )
        )

    return fig


def create_combined_plot(df: pd.DataFrame, delay_cols: List[str],
                        delays: Dict[str, int], reference_cols: List[str] = None,
                        downsample_rate: int = 1, crosshair: bool = True,
                        num_segments: int = 3, selected_segment: int = 0) -> go.Figure:
    """지연 적용된 컬럼과 기준 컬럼을 함께 표시하는 플롯을 생성하는 함수"""
    # 데이터 구간 선택
    df_segment = get_data_segment(df, num_segments, selected_segment)

    fig = go.Figure()

    # 지연 적용된 컬럼들 추가
    for col in delay_cols:
        delay = delays.get(col, 0)

        # 1단계: 선택된 구간에서 시간 지연 적용
        y_data = apply_time_delay(df_segment, col, delay)

        # 2단계: 지연 적용된 데이터에 다운샘플링 적용
        y = y_data.iloc[::downsample_rate]
        x = df_segment.index[::downsample_rate]

        # 지연값이 있는 경우 레이블에 표시
        label = f"{col} (delay: {delay:+d})" if delay != 0 else f"{col} (original)"

        fig.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode='lines',
            name=label,
            showlegend=True,
            hoverinfo='x',
            hovertemplate='',
            line=dict(width=2)  # 지연 적용된 신호는 두꺼운 선
        ))

    # 기준 컬럼들 추가 (지연 적용 안됨)
    if reference_cols:
        for col in reference_cols:
            # 1단계: 선택된 구간의 원본 데이터 (지연 적용 안함)
            y_data = df_segment[col]

            # 2단계: 다운샘플링 적용
            y = y_data.iloc[::downsample_rate]
            x = df_segment.index[::downsample_rate]

            fig.add_trace(go.Scattergl(
                x=x,
                y=y,
                mode='lines',
                name=f"{col} (reference)",
                showlegend=True,
                hoverinfo='x',
                hovertemplate='',
                line=dict(width=1, dash='dot')  # 기준 신호는 점선으로 구분
            ))

    # 구간 정보를 제목에 추가
    segment_info = f"구간 {selected_segment + 1}/{num_segments}"
    fig.update_layout(
        title=f"📊 시간 지연 적용 신호 vs 기준 신호 비교 ({segment_info})",
        dragmode="zoom",
        xaxis=dict(
            rangeslider=dict(visible=False),
            title="시간 인덱스"
        ),
        yaxis=dict(
            title="신호 값"
        ),
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    if crosshair:
        fig.update_layout(
            hovermode="x",
            xaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="red",
                spikethickness=1,
                title="시간 인덱스"
            ),
            yaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="blue",
                spikethickness=1,
                title="신호 값"
            )
        )

    return fig


def create_multi_file_plot(selected_files: List, selected_features: List[str],
                          downsample_rate: int = 1, crosshair: bool = True,
                          num_segments: int = 3, selected_segment: int = 0) -> go.Figure:
    """선택된 파일들의 특징들을 플롯하는 함수 (탭1,2 방식과 동일)"""
    fig = go.Figure()

    # 파일별로 처리
    for file in selected_files:
        try:
            df = load_data_file(file)
            if df is None:
                continue

            # 데이터 구간 선택
            df_segment = get_data_segment(df, num_segments, selected_segment)

            # 선택된 특징들 처리
            for feature in selected_features:
                if feature in df.columns:
                    # 1단계: 선택된 구간의 원본 데이터
                    y_data = df_segment[feature]

                    # 2단계: 다운샘플링 적용
                    y = y_data.iloc[::downsample_rate]
                    x = df_segment.index[::downsample_rate]

                    # 파일명과 특징명을 포함한 레이블
                    file_name = file.name.split('.')[0]  # 확장자 제거
                    label = f"{file_name}_{feature}"

                    fig.add_trace(go.Scattergl(
                        x=x,
                        y=y,
                        mode='lines',
                        name=label,
                        showlegend=True,
                        hoverinfo='x',
                        hovertemplate=''
                    ))

        except Exception as e:
            st.warning(f"⚠️ {file.name} 플롯 생성 중 오류: {str(e)}")
            continue

    # 구간 정보를 제목에 추가
    segment_info = f"구간 {selected_segment + 1}/{num_segments}"
    fig.update_layout(
        title=f"📊 다중 파일 특징 비교 ({segment_info})",
        dragmode="zoom",
        xaxis=dict(
            rangeslider=dict(visible=False),
            title="시간 인덱스"
        ),
        yaxis=dict(
            title="신호 값"
        ),
        height=600
    )

    if crosshair:
        fig.update_layout(
            hovermode="x",
            xaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="red",
                spikethickness=1,
                title="시간 인덱스"
            ),
            yaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor="blue",
                spikethickness=1,
                title="신호 값"
            )
        )

    return fig
