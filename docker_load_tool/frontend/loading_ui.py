"""
데이터 로딩 UI 탭
"""
import streamlit as st
import pandas as pd
import io
import tempfile
import os


def render_loading_tab():
    """데이터 로딩 탭 렌더링"""
    st.header("📂 데이터 로딩")
    
    file_type = st.session_state.config['file_info']['file_type']
    
    # 파일 업로더
    if file_type == 'csv':
        uploaded_files = st.file_uploader(
            "CSV 파일 업로드 (여러 개 선택 가능)",
            type=['csv'],
            accept_multiple_files=True,
            key='csv_upload'
        )
    else:
        uploaded_files = st.file_uploader(
            "Excel 파일 업로드",
            type=['xlsx', 'xls'],
            key='excel_upload'
        )
        if uploaded_files:
            uploaded_files = [uploaded_files]
    
    if uploaded_files:
        if len(uploaded_files) > 1:
            st.info(f"📁 {len(uploaded_files)}개 파일 선택됨")
            for i, f in enumerate(uploaded_files, 1):
                st.caption(f"{i}. {f.name}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 데이터 로드 시작", type="primary", use_container_width=True):
                progress_text = st.empty()
                progress_bar = st.empty()
                
                with st.spinner("데이터를 로딩하는 중..."):
                    try:
                        def update_progress(sheet_name, current, total):
                            progress_text.text(f"처리 중: {current}/{total} - {sheet_name}")
                            progress_bar.progress(current / total)
                        
                        # 데이터 서비스 사용
                        data_service = st.session_state.data_service
                        
                        all_data = {}
                        
                        for idx, uploaded_file in enumerate(uploaded_files, 1):
                            if len(uploaded_files) > 1:
                                progress_text.text(f"파일 {idx}/{len(uploaded_files)} 처리 중: {uploaded_file.name}")
                                progress_bar.progress(idx / len(uploaded_files))
                            
                            # 데이터 로드
                            data = data_service.load_data(
                                uploaded_file, 
                                st.session_state.config, 
                                file_type,
                                progress_callback=update_progress
                            )
                            
                            file_base_name = uploaded_file.name.rsplit('.', 1)[0]
                            
                            if isinstance(data, pd.DataFrame):
                                df_clean = data_service.prepare_for_display(data)
                                all_data[file_base_name] = df_clean
                            else:
                                for sheet_name, sheet_df in data.items():
                                    combined_name = f"{file_base_name}_{sheet_name}"
                                    df_clean = data_service.prepare_for_display(sheet_df)
                                    all_data[combined_name] = df_clean
                        
                        progress_text.empty()
                        progress_bar.empty()
                        
                        # session_state에 저장
                        if len(all_data) == 1:
                            final_df = list(all_data.values())[0]
                            st.session_state.loaded_data = final_df
                            
                            st.session_state.metadata = {
                                'source_name': final_df.attrs.get('source_name', 'unknown'),
                                'header_metadata': final_df.attrs.get('header_metadata', {}),
                                'shape': final_df.shape,
                                'columns': final_df.columns.tolist(),
                                'dtypes': {str(k): str(v) for k, v in final_df.dtypes.items()}
                            }
                        else:
                            st.session_state.loaded_data = all_data
                            
                            st.session_state.metadata = {
                                sheet: {
                                    'source_name': df.attrs.get('source_name', sheet),
                                    'header_metadata': df.attrs.get('header_metadata', {}),
                                    'shape': df.shape,
                                    'columns': df.columns.tolist(),
                                    'dtypes': {str(k): str(v) for k, v in df.dtypes.items()}
                                }
                                for sheet, df in all_data.items()
                            }
                        
                        st.success("✅ 데이터 로딩 완료!")
                        
                        # 메타데이터 정보
                        if st.session_state.metadata:
                            header_meta = st.session_state.metadata.get('header_metadata', {})
                            if header_meta:
                                metadata_info = []
                                if 'description' in header_meta:
                                    metadata_info.append(f"✅ Description: {len(header_meta['description'])}개")
                                if 'unit' in header_meta:
                                    metadata_info.append(f"✅ Unit: {len(header_meta['unit'])}개")
                                if 'tag_name' in header_meta:
                                    metadata_info.append(f"✅ Tag_name: {len(header_meta['tag_name'])}개")
                                
                                if metadata_info:
                                    st.info("📋 **메타데이터 발견:**\n" + "\n".join(metadata_info))
                        
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ 데이터 로딩 실패: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with col2:
            if st.button("🗑️ 초기화", use_container_width=True):
                keys_to_remove = ['loaded_data', 'metadata']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # 로드된 데이터 표시
    if st.session_state.loaded_data is not None:
        st.divider()
        st.subheader("📊 로드된 데이터")
        
        data = st.session_state.loaded_data
        
        # 단일 DataFrame
        if isinstance(data, pd.DataFrame):
            st.markdown(f"**Shape:** {data.shape[0]:,} rows × {data.shape[1]:,} columns")
            
            with st.expander("🔍 데이터 미리보기", expanded=True):
                n_rows = st.slider("표시할 행 수", 5, 100, 10, key='single_preview_rows')
                
                # PyArrow 호환을 위한 안전한 표시
                preview_df = data.head(n_rows).copy()
                for col in preview_df.columns:
                    if preview_df[col].dtype == 'object':
                        preview_df[col] = preview_df[col].astype(str).replace('nan', '').replace('None', '')
                
                # st.dataframe(preview_df, use_container_width=True)  #width='stretch
                st.dataframe(preview_df, width='stretch')  #
            
            if st.session_state.metadata:
                with st.expander("ℹ️ 메타데이터"):
                    meta = st.session_state.metadata
                    if 'header_metadata' in meta and meta['header_metadata']:
                        st.json(meta['header_metadata'])
            
            with st.expander("📈 기본 통계"):
                stats_df = data.describe(include='all').reset_index()
                
                # PyArrow 호환을 위한 안전한 표시
                for col in stats_df.columns:
                    if stats_df[col].dtype == 'object':
                        stats_df[col] = stats_df[col].astype(str).replace('nan', '').replace('None', '')
                
                # st.dataframe(stats_df, use_container_width=True)
                st.dataframe(stats_df, width='stretch')
            
            # HDF5 저장
            st.divider()
            st.subheader("💾 데이터 저장")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**저장 형식:** HDF5")
            
            with col2:
                file_name = st.text_input("파일명 (확장자 제외)", value="output_data", key='single_file_name')
                include_date = st.checkbox("파일명에 날짜 추가", value=False, key='single_include_date')
            
            with col3:
                st.write("")
                st.write("")
                if st.button("💾 HDF5 저장", type="primary", use_container_width=True, key='single_save_btn'):
                    try:
                        file_service = st.session_state.file_service
                        
                        # 날짜 범위 추출
                        date_str = file_service.extract_date_range(data) if include_date else ''
                        
                        # HDF5로 저장
                        file_bytes = file_service.save_to_hdf5(data, compression='gzip')
                        
                        # 파일명 생성
                        if date_str:
                            download_name = f"{file_name}_{date_str}.h5"
                        else:
                            download_name = f"{file_name}.h5"
                        
                        st.download_button(
                            label=f"⬇️ {download_name} 다운로드",
                            data=file_bytes,
                            file_name=download_name,
                            mime='application/x-hdf5',
                            use_container_width=True
                        )
                        
                        st.success("✅ HDF5 파일 준비 완료!")
                    
                    except Exception as e:
                        st.error(f"❌ 저장 실패: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # 다중 DataFrame
        else:
            st.markdown(f"**총 {len(data)}개 시트 로드됨**")
            
            selected_sheet = st.selectbox("시트 선택", options=list(data.keys()), key='loading_sheet_select')
            df = data[selected_sheet]
            
            st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]:,} columns")
            
            with st.expander("🔍 데이터 미리보기", expanded=True):
                n_rows = st.slider("표시할 행 수", 5, 100, 10, key='multi_sheet_preview_rows')
                
                # PyArrow 호환을 위한 안전한 표시
                preview_df = df.head(n_rows).copy()
                for col in preview_df.columns:
                    if preview_df[col].dtype == 'object':
                        preview_df[col] = preview_df[col].astype(str).replace('nan', '').replace('None', '')
                
                st.dataframe(preview_df, width='stretch')
            
            if st.session_state.metadata and selected_sheet in st.session_state.metadata:
                with st.expander("ℹ️ 메타데이터"):
                    meta = st.session_state.metadata[selected_sheet]
                    if 'header_metadata' in meta and meta['header_metadata']:
                        st.json(meta['header_metadata'])
            
            with st.expander("📈 기본 통계"):
                stats_df = df.describe(include='all')
                
                # PyArrow 호환을 위한 안전한 표시
                for col in stats_df.columns:
                    if stats_df[col].dtype == 'object':
                        stats_df[col] = stats_df[col].astype(str).replace('nan', '').replace('None', '')
                
                st.dataframe(stats_df, width='stretch')
            
            # 저장 옵션
            st.divider()
            st.subheader("💾 데이터 저장")
            
            all_sheet_names = list(data.keys())
            
            col_select1, col_select2 = st.columns([3, 1])
            
            with col_select1:
                save_mode = st.radio(
                    "저장 모드",
                    ["개별 저장", "선택한 시트 병합", "모든 시트 병합"],
                    horizontal=True,
                    key='multi_save_mode'
                )
            
            with col_select2:
                if save_mode != "개별 저장":
                    sort_by_time = st.checkbox("시간순 정렬", value=True, key='sort_by_time')
                else:
                    sort_by_time = False
            
            selected_sheets = []
            if save_mode == "선택한 시트 병합":
                selected_sheets = st.multiselect(
                    "병합할 시트 선택",
                    options=all_sheet_names,
                    default=all_sheet_names[:min(3, len(all_sheet_names))],
                    key='selected_sheets_to_merge'
                )
                if not selected_sheets:
                    st.warning("⚠️ 병합할 시트를 최소 1개 이상 선택해주세요.")
            elif save_mode == "모든 시트 병합":
                selected_sheets = all_sheet_names
                st.info(f"📊 총 {len(selected_sheets)}개 시트를 병합합니다.")
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**저장 형식:** HDF5")
                if save_mode == "개별 저장":
                    save_all = st.checkbox("모든 시트 저장", value=False, key='multi_save_all')
                else:
                    save_all = False
            
            with col2:
                file_name = st.text_input("파일명 (확장자 제외)", value="output_data", key='multi_file_name')
                include_date = st.checkbox("파일명에 날짜 추가", value=True, key='multi_include_date')
            
            with col3:
                st.write("")
                st.write("")
                if st.button("💾 HDF5 저장", type="primary", use_container_width=True, key='multi_save_btn'):
                    try:
                        file_service = st.session_state.file_service
                        
                        # 병합 모드
                        if save_mode in ["선택한 시트 병합", "모든 시트 병합"]:
                            if not selected_sheets:
                                st.error("❌ 병합할 시트를 선택해주세요.")
                            else:
                                dfs_to_merge = [data[sheet] for sheet in selected_sheets]
                                
                                # timestamp 컬럼 찾기
                                has_timestamp = False
                                ts_col = None
                                for df_temp in dfs_to_merge:
                                    for col in df_temp.columns:
                                        if 'timestamp' in str(col).lower() or 'datetime' in str(col).lower():
                                            if pd.api.types.is_datetime64_any_dtype(df_temp[col]):
                                                has_timestamp = True
                                                ts_col = col
                                                break
                                    if has_timestamp:
                                        break
                                
                                merged_df = pd.concat(dfs_to_merge, ignore_index=True)
                                
                                if sort_by_time and has_timestamp and ts_col:
                                    merged_df = merged_df.sort_values(by=ts_col).reset_index(drop=True)
                                    st.success(f"✅ {len(selected_sheets)}개 시트를 시간순으로 병합했습니다.")
                                else:
                                    st.success(f"✅ {len(selected_sheets)}개 시트를 입력 순서대로 병합했습니다.")
                                
                                date_str = file_service.extract_date_range(merged_df) if include_date else ''
                                
                                file_bytes = file_service.save_to_hdf5(merged_df, compression='gzip')
                                
                                if date_str:
                                    download_name = f"{file_name}_{date_str}_merged.h5"
                                else:
                                    download_name = f"{file_name}_merged.h5"
                                
                                st.download_button(
                                    label=f"⬇️ {download_name} 다운로드",
                                    data=file_bytes,
                                    file_name=download_name,
                                    mime='application/x-hdf5',
                                    use_container_width=True
                                )
                        
                        # 개별 저장 모드
                        elif save_mode == "개별 저장":
                            if save_all:
                                st.info(f"총 {len(data)}개의 HDF5 파일을 생성합니다...")
                                
                                import zipfile
                                zip_buffer = io.BytesIO()
                                
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    for sheet_name, sheet_df in data.items():
                                        import re
                                        safe_name = re.sub(r'[<>:"/\\|?*]', '_', sheet_name)
                                        
                                        date_str = file_service.extract_date_range(sheet_df) if include_date else ''
                                        
                                        file_bytes = file_service.save_to_hdf5(sheet_df, compression='gzip')
                                        
                                        if date_str:
                                            file_name_in_zip = f"{file_name}_{date_str}_{safe_name}.h5"
                                        else:
                                            file_name_in_zip = f"{file_name}_{safe_name}.h5"
                                        
                                        zip_file.writestr(file_name_in_zip, file_bytes)
                                
                                zip_buffer.seek(0)
                                
                                if include_date:
                                    first_sheet = list(data.values())[0]
                                    date_str = file_service.extract_date_range(first_sheet)
                                    if date_str:
                                        zip_name = f"{file_name}_{date_str}_all_sheets.zip"
                                    else:
                                        zip_name = f"{file_name}_all_sheets.zip"
                                else:
                                    zip_name = f"{file_name}_all_sheets.zip"
                                
                                st.download_button(
                                    label=f"⬇️ {zip_name} (전체 다운로드)",
                                    data=zip_buffer,
                                    file_name=zip_name,
                                    mime='application/zip',
                                    use_container_width=True
                                )
                                st.success(f"✅ {len(data)}개의 HDF5 파일이 ZIP으로 압축되었습니다!")
                            else:
                                # 선택된 시트만
                                date_str = file_service.extract_date_range(df) if include_date else ''
                                
                                file_bytes = file_service.save_to_hdf5(df, compression='gzip')
                                
                                if date_str:
                                    download_name = f"{file_name}_{date_str}_{selected_sheet}.h5"
                                else:
                                    download_name = f"{file_name}_{selected_sheet}.h5"
                                
                                st.download_button(
                                    label=f"⬇️ {download_name} 다운로드",
                                    data=file_bytes,
                                    file_name=download_name,
                                    mime='application/x-hdf5',
                                    use_container_width=True
                                )
                                st.success("✅ HDF5 파일 준비 완료!")
                    
                    except Exception as e:
                        st.error(f"❌ 저장 실패: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())