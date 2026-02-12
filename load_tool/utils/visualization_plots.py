"""
시각화 플롯 모듈
Streamlit 시각화 함수들
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render_timeseries_plot(df, numeric_cols, datetime_cols):
    """시계열 그래프"""
    st.subheader("📈 시계열 그래프")

    col1, col2 = st.columns([3, 1])

    with col1:
        # X축 옵션 생성
        x_options = ["Index (순서)"]  # 인덱스 옵션 추가

        if datetime_cols:
            x_options.extend(datetime_cols)  # datetime 컬럼 추가

        # 다른 모든 컬럼도 선택 가능하게 (datetime 제외)
        other_cols = [col for col in df.columns.tolist() if col not in datetime_cols]
        x_options.extend(other_cols)

        x_col = st.selectbox("X축", options=x_options, key='ts_x')

        # Y축 선택 (다중 선택)
        y_cols = st.multiselect("Y축 변수 (다중 선택 가능)", options=numeric_cols, key='ts_y')

    with col2:
        plot_type = st.selectbox("그래프 타입", ['선 그래프', '점 그래프', '선+점'])
        show_legend = st.checkbox("범례 표시", value=True)

    if y_cols:
        # 메타데이터에서 단위와 태그 정보 가져오기
        meta = {}
        if st.session_state.metadata:
            if isinstance(st.session_state.loaded_data, dict):
                selected_sheet = list(st.session_state.loaded_data.keys())[0]
                meta = st.session_state.metadata.get(selected_sheet, {}).get('header_metadata', {})
            else:
                meta = st.session_state.metadata.get('header_metadata', {})

        # 선택된 변수들의 정보 표시
        if meta and ('unit' in meta or 'tag_name' in meta):
            with st.expander("📋 선택된 변수 정보"):
                info_data = []
                for col in y_cols:
                    try:
                        col_idx = df.columns.tolist().index(col)
                        unit = 'N/A'
                        tag = 'N/A'

                        if 'unit' in meta and col_idx < len(meta['unit']):
                            unit_val = meta['unit'][col_idx]
                            unit = str(unit_val) if pd.notna(unit_val) else 'N/A'

                        if 'tag_name' in meta and col_idx < len(meta['tag_name']):
                            tag_val = meta['tag_name'][col_idx]
                            tag = str(tag_val) if pd.notna(tag_val) else 'N/A'

                        info_data.append({
                            '변수명': col,
                            '단위': unit,
                            '태그명': tag
                        })
                    except:
                        continue

                if info_data:
                    info_df = pd.DataFrame(info_data)
                    st.dataframe(info_df, use_container_width=True)

        # 그래프 생성
        fig = go.Figure()

        mode = 'lines' if plot_type == '선 그래프' else 'markers' if plot_type == '점 그래프' else 'lines+markers'

        # X축 데이터 결정
        if x_col == "Index (순서)":
            x_data = df.index
            x_title = "Index"
        else:
            x_data = df[x_col]
            x_title = x_col

        for y_col in y_cols:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=df[y_col],
                mode=mode,
                name=y_col,
                connectgaps=False,
                line=dict(width=2) if 'lines' in mode else None,
                marker=dict(size=6) if 'markers' in mode else None
            ))

        fig.update_layout(
            title='시계열 데이터',
            xaxis_title=x_title,
            yaxis_title='값',
            hovermode='x unified',
            showlegend=show_legend,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)


def render_scatter_plot(df, numeric_cols):
    """산점도"""
    st.subheader("🔵 산점도")

    col1, col2, col3 = st.columns(3)

    with col1:
        x_col = st.selectbox("X축", options=numeric_cols, key='scatter_x')
    with col2:
        y_col = st.selectbox("Y축", options=numeric_cols, key='scatter_y')
    with col3:
        color_col = st.selectbox("색상 (선택)", options=[None] + numeric_cols, key='scatter_color')

    if x_col and y_col:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f'{x_col} vs {y_col}',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)


def render_histogram(df, numeric_cols):
    """히스토그램"""
    st.subheader("📊 히스토그램")

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_col = st.selectbox("변수 선택", options=numeric_cols, key='hist_col')

    with col2:
        n_bins = st.slider("Bins 수", 10, 100, 30)

    if selected_col:
        fig = px.histogram(
            df,
            x=selected_col,
            nbins=n_bins,
            title=f'{selected_col} 분포',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # 기본 통계
        with st.expander("📈 기본 통계"):
            stats = df[selected_col].describe()
            st.dataframe(stats, use_container_width=True)


def render_boxplot(df, numeric_cols):
    """박스플롯"""
    st.subheader("📦 박스플롯")

    selected_cols = st.multiselect("변수 선택 (다중 선택)", options=numeric_cols, key='box_cols')

    if selected_cols:
        fig = go.Figure()

        for col in selected_cols:
            fig.add_trace(go.Box(
                y=df[col],
                name=col,
                boxmean='sd'
            ))

        fig.update_layout(
            title='박스플롯',
            yaxis_title='값',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(df, numeric_cols):
    """상관관계 히트맵"""
    st.subheader("🔥 상관관계 히트맵")

    selected_cols = st.multiselect(
        "변수 선택 (다중 선택, 최소 2개)",
        options=numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))],
        key='corr_cols'
    )

    if len(selected_cols) >= 2:
        corr_matrix = df[selected_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='상관관계 히트맵',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
