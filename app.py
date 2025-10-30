# app.py
import streamlit as st
import json
import time # 지연 시간 부여 등 필요 시 활용
from azure_storage_manager import AzureStorageManager
from azure_search_manager import AzureSearchManager
from ai_service import AIService
from data_parser import CDRParser
import logging
import sys # 디버깅용
import re # 정규표현식 사용

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit 페이지 설정
st.set_page_config(layout="wide", page_title="CDR 파일 유효성 검증 시스템") 

# --- 서비스 초기화 ---
@st.cache_resource
def get_storage_manager():
    return AzureStorageManager()

@st.cache_resource
def get_search_manager():
    return AzureSearchManager()

@st.cache_resource
def get_ai_service():
    return AIService()

@st.cache_resource
def get_cdr_parser():
    return CDRParser()

storage_manager = get_storage_manager()
search_manager = get_search_manager()
ai_service = get_ai_service()
cdr_parser = get_cdr_parser()

# --- 도우미 함수: 문제 정보 추출 ---
def extract_problem_info_from_llm_message(issue_description: str) -> tuple[int | None, str | None]:
    """
    LLM이 반환한 문제 설명에서 라인 번호와 필드 이름을 추출합니다.
    예시: "라인 34: duration 필드: 길이 불일치"
    """
    problem_line_num = None
    problem_field_name = None

    # 라인 번호 추출 (LLM 메시지 시작 부분에서 '라인 N:' 패턴 찾기)
    line_match = re.search(r'^라인 (\d+):', issue_description) 
    if line_match:
        problem_line_num = int(line_match.group(1))

    # 필드 이름 추출
    field_match = re.search(r'([a-zA-Z0-9_]+) 필드:', issue_description)
    if field_match:
        problem_field_name = field_match.group(1).strip()
    
    return problem_line_num, problem_field_name

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
                if storage_manager.upload_format_definition(format_file_name_for_ui, format_content_for_ui):
                    st.success(f"✔️ '{format_file_name_for_ui}'이(가) Azure Blob Storage에 저장되었습니다.")
                    if search_manager.upload_format_document(format_file_name_for_ui, format_content_for_ui):
                        st.success(f"✔️ '{format_file_name_for_ui}' 포맷 정의가 Azure AI Search에 성공적으로 색인되었습니다.")
                        st.session_state["format_list_refreshed"] = True 
                    else:
                        st.error("❌ Azure AI Search 색인에 실패했습니다. 로그를 확인해주세요.")
                else:
                    st.error("❌ Azure Blob Storage에 저장 실패했습니다. 로그를 확인해주세요.")
    with col2:
        # 이 버튼을 누르면 Streamlit 재실행 (rerun)되며, 그때 아래 사이드바 목록이 새로고침됩니다.
        if st.button("저장된 포맷 정의 목록 새로고침", use_container_width=True):
            st.rerun() # 강제로 앱을 다시 실행하여 목록을 최신화합니다.

# --- 저장된 포맷 정의 목록 항상 표시 (사이드바) ---
# 이 블록을 if 조건문 밖으로 이동하여 항상 실행되도록 합니다.
available_formats = storage_manager.list_definitions()
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
        # 파일이 새로 업로드되면 이전 검증 결과 및 식별 정보 초기화
        st.session_state['validation_issues'] = None
        st.session_state['identified_format_name'] = None
        st.session_state['format_definition_content'] = None

# session_state에서 cdr_content_raw를 가져오거나, 파일이 없으면 None
cdr_file_name = st.session_state.get('last_uploaded_cdr_filename')
cdr_content = st.session_state.get('cdr_content_raw')
identified_format_name = st.session_state.get('identified_format_name')
format_definition_content = st.session_state.get('format_definition_content')
validation_issues = st.session_state.get('validation_issues')


if cdr_content: # cdr_content가 session_state에 있으면 처리
    st.info(f"업로드된 파일: **`{cdr_file_name}`**. AI가 파일 형식을 식별할 준비 중입니다.")
    st.subheader("CDR 파일 내용 미리보기 (첫 1000자)")
    st.text_area("파일 내용", cdr_content[:1000], height=150, help="용량 문제로 파일의 첫 부분만 보여줍니다.")

    # 검증 시작 버튼을 누를 때만 검증 로직 실행
    if st.button("AI 기반 CDR 파일 형식 식별 및 유효성 검증 시작", type="primary", use_container_width=True):
        st.session_state['validation_issues'] = None # 이전 결과 초기화
        st.session_state['identified_format_name'] = None
        st.session_state['format_definition_content'] = None
        
        with st.spinner("AI가 CDR 파일 형식을 식별 중입니다..."):
            identified_format_name, format_definition_content = ai_service.identify_cdr_format(cdr_content) 
            st.session_state['identified_format_name'] = identified_format_name
            st.session_state['format_definition_content'] = format_definition_content
        
        if identified_format_name and format_definition_content:
            st.success(f"AI가 식별한 CDR 파일 형식: **`{identified_format_name}`**")
            # 추론된 포맷인 경우 특별 메시지
            if identified_format_name and identified_format_name.startswith("inferred_"):
                st.info("💡 이 포맷은 AI가 CDR 파일 내용을 분석하여 추론한 것입니다. 정확도를 높이려면 이 포맷 정의를 저장하고 Search에 색인하세요.")

            st.markdown("**적용된 포맷 정의 내용:**")
            st.json(json.loads(format_definition_content))

            with st.spinner(f"'{identified_format_name}' 형식으로 CDR 파일 유효성 검증 중입니다. 잠시만 기다려 주세요..."):
                validation_issues = ai_service.validate_cdr_file(cdr_content, identified_format_name, format_definition_content)
                st.session_state['validation_issues'] = validation_issues 
        else:
            st.error("AI가 CDR 파일 형식을 식별하지 못했습니다. 포맷 정의가 Azure AI Search에 정확히 등록되었는지 확인하거나, 파일 내용을 변경하여 다시 시도해 주세요.")
            st.session_state['validation_issues'] = None

# 검증 결과가 session_state에 있고 비어있지 않으면 표시
if validation_issues is not None and identified_format_name and format_definition_content: # 포맷 식별 성공해야 상세 표시
    if validation_issues:
        st.error("❌ CDR 파일에서 **문제가 발견되었습니다!**")
        st.subheader("🔍 발견된 문제점:")
        
        issue_options_display = []
        for issue in validation_issues:
            issue_options_display.append(f"라인 {issue['line']}: {issue['issue']}")
        
        selected_issue_index = st.selectbox("아래 목록에서 상세 내용을 볼 문제를 선택하세요:", 
                                            options=range(len(issue_options_display)), 
                                            format_func=lambda x: issue_options_display[x],
                                            key="issue_selector")

        if selected_issue_index is not None:
            selected_issue = validation_issues[selected_issue_index]
            
            line_number_to_show_for_content = selected_issue['line'] 
            _, problem_field_name = extract_problem_info_from_llm_message(selected_issue['issue']) 
            
            st.subheader(f"⚠️ 문제 CDR 레코드 상세 정보 (원본 파일 라인 {line_number_to_show_for_content})")
            
            cdr_lines_original = cdr_content.replace('\r', '').split('\n') 
            
            if 0 < line_number_to_show_for_content <= len(cdr_lines_original):
                problem_line_content = cdr_lines_original[line_number_to_show_for_content - 1] 
                st.code(problem_line_content, language="text")

                parsed_line_data = cdr_parser.parse_cdr_line(problem_line_content, identified_format_name, format_definition_content)
                
                if "error" in parsed_line_data:
                    st.warning(f"데이터 파싱 중 오류 발생: {parsed_line_data['error']}")
                elif format_definition_content: 
                    st.markdown("---")
                    st.markdown("#### 필드별 상세 매핑 및 비교:")
                    
                    format_def_obj = json.loads(format_definition_content)
                    format_type = format_def_obj.get('format_type')
                    
                    if format_type == "fixed_length":
                        fields_def = format_def_obj.get('fields', [])
                        table_data = []
                        for field in fields_def:
                            field_name = field.get('name')
                            start = field.get('start')
                            length = field.get('length')
                            field_type = field.get('type')
                            
                            actual_raw_value = ""
                            if start is not None and length is not None and start < len(problem_line_content):
                                actual_raw_value = problem_line_content[start : min(start + length, len(problem_line_content))]
                            
                            display_value = f"`{actual_raw_value}`"
                            if field_name and problem_field_name and field_name.lower() == problem_field_name.lower():
                                display_value = f"**<span style='color:red;'>{display_value}</span>**"

                            table_data.append({
                                "필드명": field_name,
                                "정의_시작": start,
                                "정의_길이": length,
                                "정의_타입": field_type,
                                "실제_값": display_value,
                                "문제": "🚨" if (field_name and problem_field_name and field_name.lower() == problem_field_name.lower()) else ""
                            })
                        st.markdown("##### 고정 길이 포맷 상세 매핑:")
                        st.markdown(f"`record_length`: `{format_def_obj.get('record_length')}` (정의된 총 레코드 길이)")
                        st.markdown(f"`실제 라인 길이`: `{len(problem_line_content)}`")
                        st.markdown("오류 필드는 **빨간색**으로 강조됩니다.")
                        st.markdown(
                            f"| 필드명 | 정의_시작 | 정의_길이 | 정의_타입 | 실제_값 (라인 내 추출) | 문제 |\n"
                            f"|:---|:--------:|:---------:|:--------:|:--------------------|:----:|\n"
                            + "\n".join([
                                f"| {d['필드명']} | {d['정의_시작']} | {d['정의_길이']} | {d['정의_타입']} | {d['실제_값']} | {d['문제']} |"
                                for d in table_data
                            ])
                            , unsafe_allow_html=True
                        )

                    elif format_type == "csv":
                        schema_def = format_def_obj.get('schema', {})
                        header_fields = schema_def.get('header', [])
                        types = schema_def.get('types', {})
                        delimiter = schema_def.get('delimiter', ',')

                        raw_values = problem_line_content.split(delimiter)
                        
                        table_data = []
                        for i, field_name in enumerate(header_fields):
                            field_type = types.get(field_name, 'string')
                            actual_value = raw_values[i].strip() if i < len(raw_values) else ''

                            display_value = f"`{actual_value}`"
                            if field_name and problem_field_name and field_name.lower() == problem_field_name.lower():
                                display_value = f"**<span style='color:red;'>{display_value}</span>**"

                            table_data.append({
                                "필드명": field_name,
                                "정의_타입": field_type,
                                "실제_값": display_value,
                                "문제": "🚨" if (field_name and problem_field_name and field_name.lower() == problem_field_name.lower()) else ""
                            })
                        st.markdown("##### CSV 포맷 상세 매핑:")
                        st.markdown(f"**구분자 정의:** `{delimiter}`")
                        st.markdown(f"**정의된 필드 개수 (헤더):** `{len(header_fields)}`")
                        st.markdown(f"**실제 필드 개수 (이 라인):** `{len(raw_values)}`")
                        st.markdown("오류 필드는 **빨간색**으로 강조됩니다.")
                        st.markdown(
                            f"| 필드명 | 정의_타입 | 실제_값 | 문제 |\n"
                            f"|:---|:--------:|:----------|:----:|\n"
                            + "\n".join([
                                f"| {d['필드명']} | {d['정의_타입']} | {d['실제_값']} | {d['문제']} |"
                                for d in table_data
                            ])
                            , unsafe_allow_html=True
                        )
                    else:
                        st.warning(f"지원되지 않는 포맷 타입 '{format_type}'입니다.")
                else:
                    st.error("포맷 정의 내용을 찾을 수 없어 상세 매핑을 표시할 수 없습니다.")
            elif line_number_to_show_for_content == -1: 
                st.markdown(f"**🔴 일반적인 문제 내용:** <span style='color:red; font-weight:bold;'>{selected_issue['issue']}</span>", unsafe_allow_html=True)
                st.info("이 문제는 LLM이 특정 라인에 직접 연결되지 않는 파일 전체 또는 일반적인 규칙 위반으로 판단한 경우입니다. 상세 매핑은 지원되지 않습니다.")
            else: 
                st.warning(f"LLM에서 보고한 라인 번호({line_number_to_show_for_content})가 CDR 파일의 유효 범위를 벗어나거나, 파일 내용에 해당 라인이 없습니다.")
            
            st.markdown(f"**🔴 LLM 원본 문제 내용:** <span style='color:red; font-weight:bold;'>{selected_issue['issue']}</span>", unsafe_allow_html=True)
    else:
        st.success("🎉 CDR 파일에서 문제가 발견되지 않았습니다. 파일이 유효합니다!")
        st.session_state['validation_issues'] = [] # 문제 없음을 명시적으로 저장

elif cdr_content is None and current_uploaded_cdr_file_data is None: 
    st.info("위에서 CDR 파일을 업로드해주세요.")
