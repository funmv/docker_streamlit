"""
데이터 시각화 UI 탭
시계열 플롯만 지원
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import tempfile


def render_visualization_tab():
    """데이터 시각화 탭 렌더링"""
    st.header("📊 데이터 가시화")
    
    # HDF5 파일 업로드 옵션
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_h5 = st.file_uploader(
            "HDF5 파일 불러오기 (선택사항)",
            type=['h5', 'hdf5'],
            key='viz_h5'
        )
    
    with col2:
        if uploaded_h5:
            if st.button("📥 HDF5 로드", use_container_width=True):
                try:
                    import os
                    
                    # 임시 파일로 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                        tmp.write(uploaded_h5.read())
                        tmp_path = tmp.name
                    
                    try:
                        file_service = st.session_state.file_service
                        df = file_service.load_from_hdf5(tmp_path)
                        
                        st.session_state.metadata = {
                            'source_name': df.attrs.get('source_name', 'hdf5_file'),
                            'header_metadata': df.attrs.get('header_metadata', {}),
                            'shape': df.shape,
                            'columns': df.columns.tolist(),
                            'dtypes': {str(k): str(v) for k, v in df.dtypes.items()}
                        }
                        
                        header_meta = df.attrs.get('header_metadata', {})
                        
                        msg_parts = ["✅ HDF5 로드 완료!"]
                        if header_meta:
                            msg_parts.append(f"메타데이터: {len(header_meta)}개 필드")
                        
                        st.success(" | ".join(msg_parts))
                        
                        st.session_state.loaded_data = df
                        st.rerun()
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        
                except Exception as e:
                    st.error(f"❌ 로드 실패: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # 로드된 데이터 확인
    if st.session_state.loaded_data is None:
        st.info("ℹ️ 먼저 '데이터 로딩' 탭에서 데이터를 로드하거나, HDF5 파일을 업로드해주세요.")
        return
    
    data = st.session_state.loaded_data
    
    # 다중 시트인 경우 선택
    if isinstance(data, dict):
        selected_sheet = st.selectbox("시트 선택", options=list(data.keys()), key='viz_sheet_select')
        df = data[selected_sheet]
    else:
        df = data
    
    st.divider()
    
    # 메타데이터 정보
    if st.session_state.metadata:
        with st.expander("ℹ️ 헤더 메타데이터 확인"):
            if isinstance(st.session_state.metadata, dict):
                if isinstance(data, dict):
                    meta = st.session_state.metadata.get(selected_sheet, {})
                else:
                    meta = st.session_state.metadata
                
                header_meta = meta.get('header_metadata', {})
                if header_meta:
                    st.markdown("**사용 가능한 헤더 정보:**")
                    
                    if 'description' in header_meta:
                        with st.container():
                            st.markdown("##### 📝 설명(Description)")
                            desc_list = header_meta['description'][:10]
                            st.write(", ".join([str(d) for d in desc_list if pd.notna(d)]))
                    
                    if 'unit' in header_meta:
                        with st.container():
                            st.markdown("##### 📏 단위(Unit)")
                            unit_list = header_meta['unit'][:10]
                            st.write(", ".join([str(u) for u in unit_list if pd.notna(u)]))
                    
                    if 'tag_name' in header_meta:
                        with st.container():
                            st.markdown("##### 🏷️ 태그명(Tag)")
                            tag_list = header_meta['tag_name'][:10]
                            st.write(", ".join([str(t) for t in tag_list if pd.notna(t)]))
                    
                    if st.checkbox("전체 메타데이터 보기", key='show_full_meta'):
                        st.json(header_meta)
                else:
                    st.info("메타데이터가 없습니다.")
            else:
                st.info("메타데이터가 없습니다.")
    
    st.divider()
    
    # 숫자형 컬럼 찾기
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Boolean 컬럼을 숫자형으로 변환
    df_plot = df.copy()
    converted_cols = []
    
    if bool_cols:
        for col in bool_cols:
            df_plot[col] = df_plot[col].astype(float)
            if col not in numeric_cols:
                numeric_cols.append(col)
                converted_cols.append(col)
    
    if not numeric_cols:
        st.warning("⚠️ 숫자형 또는 Boolean 컬럼이 없습니다.")
        return
    
    # 시계열 그래프
    render_timeseries_plot(df_plot, numeric_cols, datetime_cols)


def render_timeseries_plot(df, numeric_cols, datetime_cols):
    """시계열 그래프"""
    st.subheader("📈 시계열 그래프")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # X축 옵션
        x_options = ["Index (순서)"]
        
        if datetime_cols:
            x_options.extend(datetime_cols)
        
        other_cols = [col for col in df.columns.tolist() if col not in datetime_cols]
        x_options.extend(other_cols)
        
        x_col = st.selectbox("X축", options=x_options, key='ts_x')
        y_cols = st.multiselect("Y축 변수 (다중 선택 가능)", options=numeric_cols, key='ts_y')
    
    with col2:
        plot_type = st.selectbox("그래프 타입", ['선 그래프', '점 그래프', '선+점'])
        show_legend = st.checkbox("범례 표시", value=True)
    
    if y_cols:
        # 메타데이터에서 단위와 태그 정보
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