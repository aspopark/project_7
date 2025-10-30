# app.py
import streamlit as st
import json
import time # 지연 시간 부여 등 필요 시 활용
# import sys # 디버깅용 (현재 사용되지 않아 제거)
# import re # 정규표현식 사용 (extract_problem_info_from_llm_message 제거로 불필요)
from typing import Dict, Any, List

# 프로젝트 내 서비스 임포트
from azure_storage_manager import AzureStorageManager
from azure_search_manager import AzureSearchManager
from ai_service import AIService
from data_parser import CDRParser # CDRParser 임포트

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit 페이지 설정
st.set_page_config(layout="wide", page_title="CDR 파일 유효성 검증 시스템") 

# --- 서비스 초기화 ---
# @st.cache_resource를 사용하여 서비스 인스턴스를 캐싱합니다.
# 이 캐시는 스크립트 재실행 시에도 객체를 유지합니다.
@st.cache_resource
def get_storage_manager_cached():
    return AzureStorageManager()

@st.cache_resource
def get_search_manager_cached():
    return AzureSearchManager()

@st.cache_resource
def get_ai_service_cached():
    return AIService()

@st.cache_resource
def get_cdr_parser_cached():
    return CDRParser() # CDRParser도 캐싱하여 재실행 시 인스턴스를 유지합니다.

storage_manager = get_storage_manager_cached()
search_manager = get_search_manager_cached()
ai_service = get_ai_service_cached()
cdr_parser = get_cdr_parser_cached() # CDRParser 인스턴스를 app.py에서 가져옵니다.

# --- 도우미 함수: extract_problem_info_from_llm_message 제거됨 ---
# LLM이 직접 JSON 형태로 structured issue를 반환하므로, 더 이상 필요 없습니다.
# def extract_problem_info_from_llm_message(issue_description: str) -> tuple[int | None, str | None]:
#     ... (제거) ...

# --- Streamlit UI 시작 ---
st.title("📞 CDR 파일 유효성 검증 시스템")
st.markdown("---")

# --- 1. CDR 포맷 정의 관리 섹션 ---
st.header("1. CDR 포맷 정의 관리")
st.markdown("CDR 파일 검증을 위해 Fixed Length 또는 CSV 형식의 포맷 정의 파일을 업로드하고 저장하세요. "
            "이 정의들은 AI가 CDR 파일의 형식을 식별하고 유효성을 검증하는 데 사용됩니다.")

uploaded_format_file_widget = st.file_uploader("CDR 포맷 정의 파일 업로드 (JSON 형식, 예: fixed_length.json, csv_format.json)", 
                                        type=["json"], 
                                        key="format_uploader_section")

# format_file도 session_state로 관리하여 Streamlit 재실행 시 정보 유지
if uploaded_format_file_widget is not None:
    if 'last_uploaded_format_filename' not in st.session_state or \
       st.session_state['last_uploaded_format_filename'] != uploaded_format_file_widget.name or \
       'format_content_raw' not in st.session_state:
        st.session_state['last_uploaded_format_filename'] = uploaded_format_file_widget.name
        st.session_state['format_content_raw'] = uploaded_format_file_widget.getvalue().decode("utf-8")
        st.session_state['format_file_uploaded'] = True

if st.session_state.get('format_file_uploaded', False):
    format_file_name_for_ui = st.session_state['last_uploaded_format_filename']
    format_content_for_ui = st.session_state['format_content_raw']
    
    st.subheader(f"업로드된 '{format_file_name_for_ui}' 파일 내용 미리보기:")
    st.code(format_content_for_ui, language="json")

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        if st.button(f"'{format_file_name_for_ui}' Azure Storage 및 AI Search에 저장", use_container_width=True):
            with st.spinner("포맷 정의 저장 및 색인 중..."):
                # ai_service.upload_and_index_format 단일 호출로 통합
                # ai_service 내부에서 storage_manager와 search_manager를 적절히 호출합니다.
                success = ai_service.upload_and_index_format(format_file_name_for_ui, format_content_for_ui)

                if success:
                    st.success(f"✔️ '{format_file_name_for_ui}' 포맷 정의가 Azure Blob Storage 및 AI Search에 성공적으로 저장/색인되었습니다.")
                    st.session_state["format_list_refreshed"] = True 
                else:
                    st.error("❌ 포맷 정의 저장 및 색인에 실패했습니다. 로그를 확인해주세요.")
    with col2:
        # 이 버튼을 누르면 Streamlit 재실행 (rerun)되며, 그때 아래 사이드바 목록이 새로고침됩니다.
        if st.button("저장된 포맷 정의 목록 새로고침", use_container_width=True):
            st.rerun() # 강제로 앱을 다시 실행하여 목록을 최신화합니다.

# --- 저장된 포맷 정의 목록 항상 표시 (사이드바) ---
# 이 블록을 if 조건문 밖으로 이동하여 항상 실행되도록 합니다.
available_formats = storage_manager.list_definitions() # storage_manager.list_definitions 호출
if available_formats:
    st.sidebar.subheader("✨ 현재 저장된 CDR 포맷 정의")
    for fmt in available_formats:
        st.sidebar.markdown(f"- `{fmt}`")
else:
    st.sidebar.info("저장된 포맷 정의가 아직 없습니다. 파일을 업로드해 주세요.")


st.markdown("---")

# --- 2. CDR 파일 업로드 및 유효성 검증 섹션 ---
st.header("2. CDR 파일 업로드 및 유효성 검증")
st.markdown("검증할 CDR 파일을 업로드하면 AI가 자동으로 포맷을 식별하고 유효성 검사를 수행합니다.")

current_uploaded_cdr_file_data = st.file_uploader("검증할 CDR 파일 업로드 (예: .txt, .csv)", 
                                                 type=["txt", "csv"], 
                                                 key="cdr_uploader_section")

# 파일 업로더에 파일이 새로 들어왔거나, 이전에 없던 파일인 경우 session_state에 저장
if current_uploaded_cdr_file_data is not None:
    if 'last_uploaded_cdr_filename' not in st.session_state or \
       st.session_state['last_uploaded_cdr_filename'] != current_uploaded_cdr_file_data.name or \
       'cdr_content_raw' not in st.session_state:
        st.session_state['last_uploaded_cdr_filename'] = current_uploaded_cdr_file_data.name
        st.session_state['cdr_content_raw'] = current_uploaded_cdr_file_data.getvalue().decode("utf-8")
        st.session_state['cdr_file_uploaded'] = True
        # 파일이 새로 업로드되면 이전 검증 결과 및 식별 정보 초기화
        st.session_state['validation_issues'] = None
        st.session_state['identified_format_name'] = None
        st.session_state['format_definition_content'] = None
        st.session_state['identified_format_type'] = None
        st.session_state['identified_format_source'] = None
        st.session_state['identified_format_confidence'] = None
        st.session_state['identified_format_reason'] = None


# session_state에서 cdr_content_raw를 가져오거나, 파일이 없으면 None
cdr_file_name = st.session_state.get('last_uploaded_cdr_filename')
cdr_content = st.session_state.get('cdr_content_raw')
identified_format_name = st.session_state.get('identified_format_name')
identified_format_type = st.session_state.get('identified_format_type')
format_definition_content = st.session_state.get('format_definition_content')
validation_issues = st.session_state.get('validation_issues')
identified_format_source = st.session_state.get('identified_format_source')
identified_format_confidence = st.session_state.get('identified_format_confidence')
identified_format_reason = st.session_state.get('identified_format_reason')

# 만약 cdr_content가 session_state에 있고 아직 식별 또는 검증되지 않았다면 UI 표시
if cdr_content and identified_format_name is None and validation_issues is None:
    st.info(f"업로드된 파일: **`{cdr_file_name}`**. AI가 파일 형식을 식별할 준비 중입니다.")
    st.subheader("CDR 파일 내용 미리보기 (첫 1000자)")
    st.text_area("파일 내용", cdr_content[:1000], height=150, help="용량 문제로 파일의 첫 부분만 보여줍니다.", key="cdr_preview_initial")
    if st.button("AI 기반 CDR 파일 형식 식별 및 유효성 검증 시작", type="primary", use_container_width=True):
        # 버튼 클릭 시 검증 로직 시작
        st.session_state['validation_issues'] = None # 이전 결과 초기화
        st.session_state['identified_format_name'] = None
        st.session_state['format_definition_content'] = None
        st.session_state['identified_format_type'] = None
        st.session_state['identified_format_source'] = None
        st.session_state['identified_format_confidence'] = None
        st.session_state['identified_format_reason'] = None

        with st.spinner("AI가 CDR 파일 형식을 식별 중입니다..."):
            identified_format_data: Dict[str, Any] = ai_service.identify_cdr_format(cdr_content) 
            
            # session_state에 식별된 정보 저장
            st.session_state['identified_format_name'] = identified_format_data.get("format_name")
            st.session_state['identified_format_type'] = identified_format_data.get("format_type")
            st.session_state['format_definition_content'] = identified_format_data.get("format_definition_content")
            st.session_state['identified_format_source'] = identified_format_data.get("source")
            st.session_state['identified_format_confidence'] = identified_format_data.get("confidence")
            st.session_state['identified_format_reason'] = identified_format_data.get("reason")
        
        # Streamlit 앱을 다시 실행하여 UI 업데이트
        st.rerun()

# 포맷 식별이 완료되면 해당 정보를 표시
if identified_format_name and identified_format_type != "unknown":
    st.info(f"업로드된 파일: **`{cdr_file_name}`**. AI가 파일 형식을 식별했습니다.")
    st.subheader("CDR 파일 내용 미리보기 (첫 1000자)")
    st.text_area("파일 내용", cdr_content[:1000], height=150, help="용량 문제로 파일의 첫 부분만 보여줍니다.", key="cdr_preview_identified")

    st.success(f"AI가 식별한 CDR 파일 형식: **`{identified_format_name}`** (유형: {identified_format_type}, 출처: {identified_format_source}, 신뢰도: {identified_format_confidence:.2f})")
    # 추론된 포맷인 경우 특별 메시지
    if identified_format_name.startswith("inferred_"):
        st.info("💡 이 포맷은 AI가 CDR 파일 내용을 분석하여 추론한 것입니다. 정확도를 높이려면 이 포맷 정의를 저장하고 Search에 색인하세요.")

    st.markdown("**적용된 포맷 정의 내용:**")
    st.json(format_definition_content, expanded=False)
    st.info(f"식별 이유: {identified_format_reason}")
    
    # 포맷 식별 후, 유효성 검증을 진행할 준비가 된 상태
    if validation_issues is None: # 아직 유효성 검증이 실행되지 않았으면
        if st.button("AI 기반 CDR 파일 유효성 검증 실행", type="primary", use_container_width=True):
            with st.spinner(f"'{identified_format_name}' 형식으로 CDR 파일 유효성 검증 중입니다. 잠시만 기다려 주세요..."):
                # validate_cdr_file에 cdr_parser_instance 전달
                validation_issues_result = ai_service.validate_cdr_file(cdr_content, identified_format_name, format_definition_content, cdr_parser)
                st.session_state['validation_issues'] = validation_issues_result
            st.rerun() # 검증 결과 업데이트를 위해 재실행
elif cdr_content and identified_format_name == "unknown": # CDR 파일이 업로드되었으나 AI가 식별에 실패한 경우
    st.error("AI가 CDR 파일 형식을 식별하지 못했습니다. 포맷 정의가 Azure AI Search에 정확히 등록되었는지 확인하거나, 파일 내용을 변경하여 다시 시도해 주세요.")


# 검증 결과가 session_state에 있고 비어있지 않으면 표시
if validation_issues is not None:
    if validation_issues: # 문제점이 있을 경우
        st.error("❌ CDR 파일에서 **문제가 발견되었습니다!**")
        st.subheader("🔍 발견된 문제점:")
        
        # Streamlit selectbox에 표시할 문제점 요약 리스트
        issue_options_display = []
        for i, issue in enumerate(validation_issues):
            issue_line = issue.get('line', 'N/A')
            issue_type = issue.get('issue_type', 'UNKNOWN')
            issue_field = issue.get('field') # field가 None일 수 있음
            
            summary = f"라인 {issue_line}: [{issue_type}] "
            if issue_field:
                summary += f"필드 '{issue_field}' - "
            summary += issue.get('description', '설명 없음')
            issue_options_display.append(summary)
        
        selected_issue_index = st.selectbox("아래 목록에서 상세 내용을 볼 문제를 선택하세요:", 
                                            options=range(len(issue_options_display)), 
                                            format_func=lambda x: issue_options_display[x],
                                            key="issue_selector")

        if selected_issue_index is not None:
            selected_issue = validation_issues[selected_issue_index]
            
            line_number_to_show_for_content = selected_issue.get('line')
            # extract_problem_info_from_llm_message 함수는 이제 제거되었으므로 직접 사용하지 않습니다.
            # LLM이 JSON으로 반환했으므로, selected_issue['field']에 이미 필드명이 있습니다.
            problem_field_name_from_json = selected_issue.get('field') 
            
            st.subheader(f"⚠️ 문제 CDR 레코드 상세 정보 (원본 파일 라인 {line_number_to_show_for_content})")
            
            cdr_lines_original = cdr_content.replace('\r', '').split('\n') 
            
            if 0 < line_number_to_show_for_content <= len(cdr_lines_original):
                problem_line_content = cdr_lines_original[line_number_to_show_for_content - 1] 
                st.code(problem_line_content, language="text")

                # CDRParser를 사용하여 문제 라인을 다시 파싱하여 필드별로 상세 표시
                if identified_format_name and format_definition_content: # 식별된 포맷 정보가 있는 경우
                    st.markdown("---")
                    st.markdown("#### 필드별 상세 매핑 및 비교:")
                    
                    # format_definition_content는 이미 Dict 형태
                    format_def_obj_for_parser: Dict[str, Any] = format_definition_content 
                    parsed_line_data = cdr_parser.parse_cdr_line(problem_line_content, identified_format_name, format_def_obj_for_parser)
                    
                    if "error" in parsed_line_data:
                        st.warning(f"데이터 파싱 중 오류 발생: {parsed_line_data['error']}")
                    else:
                        fields_def = format_def_obj_for_parser.get('fields', [])
                        
                        st.json(parsed_line_data, expanded=False) # 파싱된 전체 데이터도 보여줄 수 있습니다.

                        st.markdown("##### 포맷 정의 vs. 파싱된 데이터:")
                        
                        table_data_rows = [] # 2D 리스트로 변경
                        # 헤더 추가
                        if format_def_obj_for_parser.get('format_type') == 'fixed_length':
                            table_data_rows.append(["필드명", "정의된 길이", "정의된 타입", "파싱된 값", "파싱된 값 길이", "오류 여부"])
                            for field_def in fields_def:
                                field_name = field_def.get('name')
                                defined_length = field_def.get('length')
                                defined_type = field_def.get('type', 'string')
                                parsed_value = parsed_line_data.get(field_name)
                                
                                parsed_value_str = str(parsed_value) if parsed_value is not None else ""
                                parsed_value_length = len(parsed_value_str)
                                
                                error_status = "❌" if (field_name == problem_field_name_from_json and selected_issue.get('issue_type') != 'LENGTH_MISMATCH') or "PARSE_ERROR" in parsed_value_str else "✅"
                                if selected_issue.get('issue_type') == 'LENGTH_MISMATCH' and defined_length != parsed_value_length and parsed_value_str:
                                    error_status = "❌" # 길이 불일치는 별도로 체크하여 표시
                                
                                table_data_rows.append([
                                    field_name, 
                                    str(defined_length), 
                                    defined_type, 
                                    parsed_value_str, 
                                    str(parsed_value_length), 
                                    error_status
                                ])
                        elif format_def_obj_for_parser.get('format_type') == 'csv':
                            table_data_rows.append(["필드명", "정의된 타입", "파싱된 값", "오류 여부"])
                            for field_def in fields_def:
                                field_name = field_def.get('name')
                                defined_type = field_def.get('type', 'string')
                                parsed_value = parsed_line_data.get(field_name)
                                parsed_value_str = str(parsed_value) if parsed_value is not None else ""
                                error_status = "❌" if field_name == problem_field_name_from_json or "PARSE_ERROR" in parsed_value_str else "✅"
                                table_data_rows.append([
                                    field_name, 
                                    defined_type, 
                                    parsed_value_str, 
                                    error_status
                                ])
                        else:
                            st.warning("알 수 없는 포맷 타입이므로 필드 상세 정보를 표시할 수 없습니다.")
                        
                        if table_data_rows:
                            # Streamlit table은 리스트 오브 리스트를 기대합니다.
                            st.table(table_data_rows) 

                else:
                    st.warning("포맷 정의 정보를 불러올 수 없습니다.")
            else:
                st.error("원본 파일에서 해당 라인 번호를 찾을 수 없습니다.")

    else: # 문제점이 없을 경우
        st.success("✅ CDR 파일이 포맷 정의에 따라 완벽하게 유효합니다! 🎉")
        
        # 추가적으로, 파일의 요약 통계나 일부 레코드 예시 등을 보여줄 수 있습니다.
        # 예: st.subheader("파일 요약 정보:")
        #     st.write(f"총 라인 수: {len(cdr_content.splitlines())}")
        #     st.write(f"검증 대상 데이터 라인 수: {len(cdr_content.splitlines()) - (1 if cdr_file_name.endswith('.csv') else 0)}")
