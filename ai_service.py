# ai_service.py
import json
import logging
import os
import re
from typing import List, Dict, Any, Optional
import numpy as np # << 수정: numpy 임포트 추가 (필요시 pip install numpy)

# Azure SDK 및 Streamlit 관련 라이브러리 임포트
from openai import AzureOpenAI

# 프로젝트 내 커스텀 모듈 임포트
import config
from azure_storage_manager import AzureStorageManager
from azure_search_manager import AzureSearchManager
from data_parser import CDRParser # CDRParser 클래스를 임포트합니다.

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIService:
    def __init__(self):
        self.storage_manager = AzureStorageManager()
        self.search_manager = AzureSearchManager()
        # self.cdr_parser는 app.py에서 관리하고 전달받도록 변경 (cache_resource 때문)

        # Azure OpenAI 클라이언트 초기화
        try:
            self.openai_client = AzureOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.AZURE_OPENAI_API_VERSION
            )
            logging.info("AzureOpenAI client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize AzureOpenAI client: {e}", exc_info=True)
            self.openai_client = None # 클라이언트 초기화 실패 시 None으로 설정

        # 프롬프트 템플릿 로드
        self._load_prompt_templates()

    def _load_prompt_templates(self):
        """프로젝트의 prompts 폴더에서 프롬프트 템플릿을 로드합니다."""
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")

        def load_template(filename):
            filepath = os.path.join(prompts_dir, filename)
            if not os.path.exists(filepath):
                logging.error(f"Prompt template file not found: {filepath}")
                raise FileNotFoundError(f"Prompt template file not found: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()

        try:
            self.identify_format_system_prompt_template = load_template("identify_format_system.txt")
            self.identify_format_general_prompt_template = load_template("identify_format_general.txt")
            self.infer_format_definition_prompt_template = load_template("infer_format_definition.txt")
            self.validate_cdr_system_prompt_template = load_template("validate_cdr_system.txt")
            # self.map_issue_fields_template = load_template("map_issue_fields.txt") # 필요 시 추가

            logging.info("All prompt templates loaded successfully.")
        except FileNotFoundError:
            logging.error("Failed to load prompt templates. Please ensure files exist in 'prompts' directory.", exc_info=True)
            # 앱 구동이 불가하게 만들 수 있으므로 raise 대신 적절한 에러 처리 필요
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading prompt templates: {e}", exc_info=True)

    def _call_openai_chat_completion(self, system_message: str, user_message: str, max_tokens: int = 1500) -> Optional[str]:
        """
        Azure OpenAI Chat Completion API를 호출하는 내부 도우미 함수.
        """
        if not self.openai_client:
            logging.error("OpenAI client is not initialized. Cannot make API call.")
            return None
        try:
            response = self.openai_client.chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens,
                temperature=0.1, # 일관된 결과를 위해 낮은 온도로 설정
            )
            # LLM 응답이 messages 리스트에 담겨 있으므로, 마지막 메시지의 content를 반환
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}", exc_info=True) # exc_info=True 추가하여 상세 스택 트레이스 로깅
            return None

    def upload_and_index_format(self, format_name: str, format_content_str: str, format_type: str = None):
        """
        CDR 포맷 정의를 Blob Storage에 업로드하고, Azure AI Search에 색인합니다.
        주인님의 새로운 스키마를 지원하도록 field_names 추출 로직을 수정합니다.
        """
        try:
            # 1. Blob Storage에 업로드 (주인님의 storage_manager.upload_format_definition 호출)
            storage_success = self.storage_manager.upload_format_definition(
                format_name, # file_name (format_name을 직접 전달)
                format_content_str # content
            )
            if not storage_success:
                logging.error(f"Failed to upload format '{format_name}' to Blob Storage.")
                return False

            # 2. Azure AI Search에 색인
            format_data = json.loads(format_content_str)
            if not format_type:
                format_type = format_data.get("format_type", "unknown")

            # << 수정: 인덱싱 시 사용될 필드명 추출 로직 (schema.header 사용) >>
            field_names = []
            if format_data.get("schema") and format_data["schema"].get("header"):
                field_names = format_data["schema"]["header"] # schema.header 리스트에서 필드명 추출
            
            field_names_str = " ".join(field_names)

            # SearchManager의 upload_format_document 호출
            search_success = self.search_manager.upload_format_document(
                format_name, # file_name_from_upload
                format_content_str # content_from_upload (내부에서 필드명 처리)
            )

            if not search_success:
                logging.error(f"Failed to index format '{format_name}' in Azure AI Search.")
                return False

            logging.info(f"Successfully processed and indexed CDR format '{format_name}'.")
            return True

        except json.JSONDecodeError:
            logging.error(f"Invalid JSON format for '{format_name}'.")
            return False
        except Exception as e:
            logging.error(f"An error occurred during upload_and_index_format for '{format_name}': {e}", exc_info=True)
            return False

    def _analyze_cdr_sample_structure(self, cdr_sample_lines: List[str]) -> Dict[str, Any]:
        """
        CDR 샘플 라인을 Python 기반으로 분석하여 구조적 힌트를 반환합니다.
        다양한 구분자(콤마, 세미콜론, 탭, 파이프)를 확인하고, 가장 유력한 구분자와
        라인 길이 일관성 여부를 파악하여 포맷 유형을 추론합니다.
        """
        if not cdr_sample_lines:
            return {
                "inferred_query_format_type": "unknown",
                "meta_query_fields": [],
                "all_lines_have_same_length": False,
                "detected_delimiter": None,
                "avg_line_length": 0
            }

        # << 수정: 주석 및 빈 라인 제외 (분석 대상에서 제외) >>
        relevant_sample_lines = [
            line.strip() for line in cdr_sample_lines 
            if line.strip() and not line.strip().startswith('#')
        ]
        
        # << 추가: 주석에서 포맷 힌트 추출 >>
        comment_hints = []
        for line in cdr_sample_lines:
            if line.strip().startswith('#'):
                # 주석 라인에서 괄호 안의 내용이나 키워드 추출 시도
                match_paren = re.search(r'\((.*?)\)', line) # (구분자: 세미콜론 ';') 같은 패턴
                if match_paren:
                    comment_hints.append(match_paren.group(1).replace(':', '')) # '구분자 세미콜론'
                
                # 파일명 부분이나 "청구 이벤트", "사용량 데이터" 같은 키워드 추출
                match_keywords = re.findall(r'[a-zA-Z0-9_]+', line)
                # 특정 단어 필터링 (예: 'cdr', 'csv', 'file', '통신', '요금', '구분자', '세미콜론' 등 너무 일반적인 단어)
                filtered_keywords = [
                    kw for kw in match_keywords 
                    if kw.lower() not in ['cdr', 'csv', 'file', '통신', '요금', '구분자', '세미콜론', '등'] and len(kw) > 2
                ]
                comment_hints.extend(filtered_keywords)


        if not relevant_sample_lines:
            return {
                "inferred_query_format_type": "unknown",
                "meta_query_fields": comment_hints, # << 수정: 주석에서 얻은 힌트는 그대로 전달
                "all_lines_have_same_length": False,
                "detected_delimiter": None,
                "avg_line_length": 0
            }


        # 1. 라인 길이 분석 (Fixed Length 가능성)
        line_lengths = [len(line) for line in relevant_sample_lines]
        all_lines_have_same_length = False
        if len(line_lengths) > 0 and len(set(line_lengths)) == 1:
            all_lines_have_same_length = True
        elif len(line_lengths) > 1 and len(set(line_lengths)) <= 2: # 헤더/푸터 등으로 2가지 길이만 존재할 수 있음
             # 가장 많은 길이와 다른 길이가 하나만 존재하고 그 수가 적다면 일관성 있는 것으로 간주
             lengths_counts = {length: line_lengths.count(length) for length in set(line_lengths)}
             # 예를 들어 {100: 9, 90: 1} 이면 일관성이 있다고 볼 수 있음
             if max(lengths_counts.values()) >= len(relevant_sample_lines) - 1 : # 최소 한 개 라인 빼고 모두 동일
                all_lines_have_same_length = True
                
        avg_line_length = np.mean(line_lengths) if line_lengths else 0


        # 2. 구분자 개수 분석 (CSV/ delimited 가능성)
        common_delimiters = [',', ';', '\t', '|'] # 일반적인 구분자
        delimiter_consistency = {} # {구분자: (일관성 여부 True/False, 평균 필드 개수)}
        best_delimiter = None
        max_consistent_fields = 0
        
        for delimiter in common_delimiters:
            # << 수정: 구분자로 split 했을 때 필드 개수가 1보다 커야 유의미한 구분자로 판단 >>
            if not relevant_sample_lines or not relevant_sample_lines[0].count(delimiter) > 0:
                continue # 해당 구분자가 라인에 없거나 분리할 필드가 없으면 다음 구분자로

            field_counts = [line.count(delimiter) for line in relevant_sample_lines]
            
            has_consistent_delimiter = False
            if len(field_counts) > 0 and len(set(field_counts)) == 1: # 모든 라인의 구분자 개수가 동일
                has_consistent_delimiter = True
            elif len(field_counts) > 1 and len(set(field_counts)) <= 2: # 2가지 구분자 개수만 존재 (헤더 등)
                counts_of_counts = {count: field_counts.count(count) for count in set(field_counts)}
                if max(counts_of_counts.values()) >= len(relevant_sample_lines) - 1: # 최소 한 개 라인 빼고 모두 동일
                    has_consistent_delimiter = True
            
            avg_field_count = np.mean(field_counts) if field_counts else 0
            delimiter_consistency[delimiter] = (has_consistent_delimiter, avg_field_count)

            # 가장 많은 필드를 생성하는 일관성 있는 구분자를 찾음
            # << 수정: avg_field_count > 0 조건을 추가하여 의미 없는 빈 필드 제거 >>
            if has_consistent_delimiter and avg_field_count > max_consistent_fields and avg_field_count > 0: 
                max_consistent_fields = avg_field_count
                best_delimiter = delimiter

        
        inferred_query_format_type = "unknown"
        # << 수정: meta_query_fields 초기화 방식을 변경 >>
        meta_query_fields = [] # Python 분석이 추출하는 필드명 리스트
        detected_delimiter = None

        if best_delimiter and max_consistent_fields > 0: # 일관성 있는 구분자가 있고, 필드가 하나 이상
            inferred_query_format_type = "csv"
            detected_delimiter = best_delimiter
            
            # << 수정: 첫 번째 데이터 라인에서 잠재적 필드명 추출 (데이터 값 필터링) >>
            if relevant_sample_lines:
                first_data_line = relevant_sample_lines[0]
                potential_fields = [f.strip() for f in first_data_line.split(detected_delimiter) if f.strip()]
                # 데이터 값보다는 좀 더 일반적인 필드 패턴을 Search 쿼리 키워드로 사용
                # 숫자이거나 숫자.숫자 형태가 아닌 것만 필터링 (데이터 값 제외)
                meta_query_fields = [
                    f for f in potential_fields 
                    if re.fullmatch(r'[a-zA-Z_]+[0-9]*', f) and not f.isdigit() and not f.replace('.', '', 1).isdigit()
                ][:7] 
                logging.debug(f"Inferred potential CSV header fields (filtered): {meta_query_fields}")
            
        elif all_lines_have_same_length: # 구분자 일관성이 없는데 라인 길이는 일관성 있음
            inferred_query_format_type = "fixed_length"
        
        # Fixed Length이면서 동시에 일관성 있는 구분자도 있는 경우 (둘 다 충족)
        # 일반적으로 CSV가 더 상세한 구조 정보이므로 CSV 우선 (규칙 3)
        if inferred_query_format_type == "csv" and all_lines_have_same_length:
             logging.debug("CDR sample is both CSV and Fixed Length candidate, prioritizing CSV.")

        # << 수정: 최종 Search 쿼리에 사용될 힌트는 주석 힌트와 (필터링된) 메타 필드를 조합 >>
        final_search_hints = []
        final_search_hints.extend(comment_hints)
        final_search_hints.extend(meta_query_fields)
        
        # 중복 제거 및 적절히 조합
        final_search_hints_str = " ".join(sorted(list(set(final_search_hints))))

        logging.info(f"CDR sample analysis: type={inferred_query_format_type}, search_hints='{final_search_hints_str}', all_lines_same_len={all_lines_have_same_length}, detected_delimiter='{detected_delimiter}'")
        return {
            "inferred_query_format_type": inferred_query_format_type,
            "meta_query_fields": final_search_hints_str, # Search 쿼리에 사용될 통합 힌트
            "all_lines_have_same_length": all_lines_have_same_length,
            "detected_delimiter": detected_delimiter,
            "avg_line_length": avg_line_length
        }

    # << 수정: cdr_file_name 인자 제거 (파일명을 가지고 추론하지 않기 위함) >>
    def identify_cdr_format(self, cdr_file_content: str) -> Dict[str, Any]:
        """
        CDR 파일의 포맷을 식별하거나 추론합니다.
        """
        identified_format_data = {
            "format_name": "unknown",
            "format_type": "unknown",
            "format_definition_content": {},
            "source": "failure", # "search", "llm_inferred"
            "confidence": 0.0,
            "reason": ""
        }

        cdr_sample_lines = cdr_file_content.splitlines()[:config.MAX_LINES_FOR_LLM_VALIDATION] # config에서 최대 라인 수 제한
        if not cdr_sample_lines:
            identified_format_data["reason"] = "Empty CDR file content."
            return identified_format_data

        # 1. Python 기반 CDR 샘플 분석
        analysis_result = self._analyze_cdr_sample_structure(cdr_sample_lines)
        inferred_type = analysis_result["inferred_query_format_type"]
        meta_fields_str = analysis_result["meta_query_fields"] # << 수정: 이제 meta_query_fields는 이미 결합된 문자열
        avg_line_length = analysis_result["avg_line_length"]
        detected_delimiter = analysis_result["detected_delimiter"] # << 탐지된 구분자

        # 2. Azure AI Search를 통한 포맷 정의 검색 시도
        search_query_parts = []
        
        # << 수정: 파일명 키워드를 Search 쿼리에 추가하는 로직 제거 >>
        
        if meta_fields_str:
            search_query_parts.append(meta_fields_str)
        
        if detected_delimiter: # << 수정: 탐지된 구분자를 쿼리에 추가하여 Search 정확도 높임
            # Search Index의 format_content 필드에 "delimiter:;" 와 같은 패턴이 저장되어 있으면 좋음.
            # 지금은 일반 텍스트 쿼리에 포함하여 relevance를 높임.
            search_query_parts.append(f'"{detected_delimiter}"') 
            
        # "CDR Format"은 일반적인 쿼리이므로 항상 추가 (문맥 유지)
        if "CDR Format" not in search_query_parts: # 중복 방지
             search_query_parts.append("CDR Format")

        search_query = " ".join(search_query_parts).strip()
        if not search_query:
            search_query = "*" # 빈 쿼리 방지
        
        logging.info(f"Initial Search Query: '{search_query}' (Inferred Type: {inferred_type})")

        retrieved_formats = self.search_manager.search_format(
            query=search_query,
            format_type=inferred_type if inferred_type != "unknown" else None, # 'unknown'인 경우 필터 적용 안함
            top=3 # 상위 3개까지 가져와서 LLM에게 더 많은 컨텍스트 제공
        )

        logging.info(f"AI Search for format type '{inferred_type}' with query '{search_query}' returned {len(retrieved_formats)} results.")

        # 3. LLM을 사용하여 Search 결과 검증 및 Fallback 로직 수행
        # LLM에게 전달할 user_message_content를 prepare
        llm_user_message_content = f"CDR Sample:\n```\n{cdr_file_content}\n```\n\nPython Analysis Hint: Inferred Type is '{inferred_type}'. Average Line Length: {avg_line_length}. "

        if retrieved_formats:
            # Search 결과가 있다면, LLM에게 검증 요청
            search_results_str = json.dumps([{"format_name": f["format_name"], "format_type": f["format_type"], "format_content_snippet": json.dumps(f["format_content"])[:200] + "..."} for f in retrieved_formats], indent=2)
            
            system_prompt = self.identify_format_system_prompt_template.format(
                inferred_type_hint=inferred_type,
                meta_fields_hint=meta_fields_str,
                search_results=search_results_str
            )
            llm_response = self._call_openai_chat_completion(system_prompt, llm_user_message_content)

            if llm_response:
                try:
                    # << 수정: LLM 응답에서 마크다운 코드 블록 표식 제거 >>
                    cleaned_llm_response = llm_response.strip()
                    if cleaned_llm_response.startswith("```json"):
                        cleaned_llm_response = cleaned_llm_response[len("```json"):].strip()
                    if cleaned_llm_response.endswith("```"):
                        cleaned_llm_response = cleaned_llm_response[:-len("```")].strip()

                    response_json = json.loads(cleaned_llm_response) # << 전처리된 문자열로 파싱 시도
                    identified_format_name = response_json.get("identified_format_name")
                    confidence_score = response_json.get("confidence", 0.0) # LLM이 제시한 신뢰도
                    reason = response_json.get("reason", "Identified by LLM with search context.")

                    if identified_format_name:
                        # 식별된 포맷 정의 전체를 가져오기 (Search 결과에서 찾아오거나 Blob에서 다운로드)
                        final_format = next((f for f in retrieved_formats if f["format_name"] == identified_format_name), None)
                        if final_format:
                            identified_format_data.update({
                                "format_name": final_format["format_name"],
                                "format_type": final_format["format_type"],
                                "format_definition_content": final_format["format_content"],
                                "source": "search_validated_by_llm",
                                "confidence": confidence_score,
                                "reason": reason
                            })
                            logging.info(f"AI가 최종적으로 식별한 포맷: {identified_format_data['format_name']} (LLM 응답: {identified_format_name}, 파이썬 검증 결과 신뢰도: {confidence_score})")
                            return identified_format_data
                        else:
                            logging.warning(f"LLM이 식별한 포맷 '{identified_format_name}'이 Search 결과에 없음. Fallback 시작.")
                    else:
                        logging.warning(f"LLM 응답에 identified_format_name이 없음. 응답: {llm_response}. Fallback 시작.")
                except json.JSONDecodeError as e:
                    logging.error(f"LLM 응답 JSON 파싱 오류: {e}, 응답: {llm_response}. Fallback 시작.", exc_info=True)
                except Exception as e:
                    logging.error(f"LLM 응답 처리 중 오류 발생: {e}. Fallback 시작.", exc_info=True)
            else:
                logging.info(f"LLM으로부터 응답 없음. Fallback 시작.")
        else:
            logging.info("Azure AI Search에서 관련 포맷 정의를 찾지 못함. LLM 일반 식별/추론 Fallback 시작.")

        # Fallback 1: LLM에게 일반 포맷 유형 식별 요청
        logging.info("Fallback 1: LLM에게 일반 포맷 유형 식별 요청.")
        general_system_prompt = self.identify_format_general_prompt_template.format(
            inferred_type_hint=inferred_type,
            meta_fields_hint=meta_fields_str,
            sample_content_snippet='\n'.join(cdr_sample_lines)
        )
        llm_general_response = self._call_openai_chat_completion(general_system_prompt, llm_user_message_content)

        if llm_general_response:
            try:
                # << 수정: LLM 응답에서 마크다운 코드 블록 표식 제거 >>
                cleaned_llm_response = llm_general_response.strip()
                if cleaned_llm_response.startswith("```json"):
                    cleaned_llm_response = cleaned_llm_response[len("```json"):].strip()
                if cleaned_llm_response.endswith("```"):
                    cleaned_llm_response = cleaned_llm_response[:-len("```")].strip()
                
                response_json = json.loads(cleaned_llm_response) # << 전처리된 문자열로 파싱 시도
                llm_inferred_type = response_json.get("inferred_type")
                llm_reason = response_json.get("reason", "LLM inferred general type.")
                if llm_inferred_type:
                    logging.info(f"LLM이 일반 유형을 식별함: {llm_inferred_type}. 해당 유형으로 Search 재시도.")
                    re_retrieved_formats = self.search_manager.search_format(
                        query=meta_fields_str, # 필드명 힌트만으로 검색
                        format_type=llm_inferred_type,
                        top=1
                    )
                    if re_retrieved_formats:
                        final_format = re_retrieved_formats[0]
                        identified_format_data.update({
                            "format_name": final_format["format_name"],
                            "format_type": final_format["format_type"],
                            "format_definition_content": final_format["format_content"],
                            "source": "llm_general_type_and_search",
                            "confidence": 0.7, # Search 결과가 있으므로 어느 정도 신뢰
                            "reason": llm_reason
                        })
                        logging.info(f"LLM이 식별한 일반 유형으로 Search 성공: {final_format['format_name']}")
                        return identified_format_data
                    else:
                        logging.info("LLM이 식별한 일반 유형으로 Search 재시도 실패. 추론 단계로 이동.")
                else:
                    logging.warning(f"LLM 일반 유형 응답에 'inferred_type' 없음: {llm_general_response}. 추론 단계로 이동.")
            except json.JSONDecodeError as e:
                logging.error(f"LLM 일반 유형 응답 JSON 파싱 오류: {e}, 응답: {llm_general_response}", exc_info=True)
            except Exception as e:
                logging.error(f"LLM 일반 유형 처리 중 오류 발생: {e}", exc_info=True)
        else:
            logging.info("LLM으로부터 일반 유형 식별 응답 없음. 추론 단계로 이동.")

        # Fallback 2: LLM에게 포맷 정의 자체를 추론 요청
        logging.info("Fallback 2: LLM에게 CDR 샘플 기반 포맷 정의 추론 요청.")
        infer_system_prompt = self.infer_format_definition_prompt_template.format(
            sample_content_snippet='\n'.join(cdr_sample_lines),
            inferred_type_hint=inferred_type,
            meta_fields_hint=meta_fields_str
        )
        llm_infer_response = self._call_openai_chat_completion(infer_system_prompt, llm_user_message_content)

        if llm_infer_response:
            try:
                # << 수정: LLM 응답에서 마크다운 코드 블록 표식 제거 >>
                cleaned_llm_response = llm_infer_response.strip()
                if cleaned_llm_response.startswith("```json"):
                    cleaned_llm_response = cleaned_llm_response[len("```json"):].strip()
                if cleaned_llm_response.endswith("```"):
                    cleaned_llm_response = cleaned_llm_response[:-len("```")].strip()

                inferred_json_definition = json.loads(cleaned_llm_response) # << 전처리된 문자열로 파싱 시도
                inferred_format_name = inferred_json_definition.get("format_name", "llm_inferred_format")
                inferred_format_type = inferred_json_definition.get("format_type", "unknown")
                
                # 추론된 포맷 정의도 Search 인덱스에 추가할 수 있지만, 여기서는 바로 사용
                identified_format_data.update({
                    "format_name": inferred_format_name,
                    "format_type": inferred_format_type,
                    "format_definition_content": inferred_json_definition,
                    "source": "llm_inferred",
                    "confidence": 0.6, # 추론된 것이므로 신뢰도는 약간 낮게 설정
                    "reason": "LLM inferred format definition from sample."
                })
                logging.info(f"LLM이 포맷 정의를 추론함: {inferred_format_name}")
                return identified_format_data
            except json.JSONDecodeError as e:
                logging.error(f"LLM 추론 응답 JSON 파싱 오류: {e}, 응답: {llm_infer_response}", exc_info=True)
            except Exception as e:
                logging.error(f"LLM 추론 처리 중 오류 발생: {e}", exc_info=True)
        else:
            logging.error("LLM으로부터 추론 응답 없음. CDR 포맷 식별 및 추론 최종 실패.")
        
        logging.error("CDR 포맷 식별 및 추론 최종 실패.")
        return identified_format_data

    # cdr_parser_instance 인자를 추가하여 app.py에서 전달받도록 변경
    def validate_cdr_file(self, cdr_content: str, identified_format_name: str, format_definition_content: Dict[str, Any], cdr_parser_instance: CDRParser) -> List[Dict[str, Any]]:
        """
        CDR 파일 내용을 식별된 포맷 정의에 따라 유효성을 검증하고, 문제점을 반환합니다.
        """
        issues = []
        if not self.openai_client:
            issues.append({"line": -1, "description": "OpenAI client is not initialized, cannot perform validation."})
            return issues

        try:
            # 1. 포맷 정의에서 'format_type', 'record_length', 'delimiter' 정보 추출
            format_type = format_definition_content.get("format_type", "unknown")
            record_length = format_definition_content.get("record_length", 0) # Fixed Length일 경우 필요
            
            # << 수정: delimiter는 format_definition.schema.delimiter에서 가져옴 >>
            delimiter = format_definition_content.get('schema', {}).get('delimiter')


            # 프롬프트에 전달할 record_length_defined 값 구성 (KeyError: 'record_length_defined' 해결)
            record_length_defined_for_prompt = f"레코드 정의 길이는 {record_length}입니다." if format_type == "fixed_length" and record_length > 0 else "고정 길이 레코드가 아니므로 정의된 레코드 길이가 없습니다."
            logging.debug(f"record_length_defined_for_prompt: {record_length_defined_for_prompt}") # 디버깅용 로그

            # 프롬프트에 전달할 delimiter_defined 값 구성 (KeyError: 'delimiter_defined' 해결)
            delimiter_defined_for_prompt = ""
            if format_type == "csv" and delimiter:
                delimiter_defined_for_prompt = f"데이터 필드 구분자는 '{delimiter}'입니다."
            elif format_type == "csv": # CSV인데 delimiter가 명시 안된 경우
                delimiter_defined_for_prompt = "CSV 포맷이지만 구분자가 명시되지 않았습니다 (기본 콤마(,) 사용)."
            else: # Fixed Length 또는 unknown 포맷
                delimiter_defined_for_prompt = "CSV 포맷이 아니므로 데이터 필드 구분자는 없습니다."
            logging.debug(f"delimiter_defined_for_prompt: {delimiter_defined_for_prompt}")


            # 2. CDR 파일을 라인별로 처리하며 Python 기반 1차 파싱 및 기본 검증
            cdr_lines = cdr_content.strip().splitlines()
            processed_lines_for_llm = []

            for i, line in enumerate(cdr_lines):
                line_number = i + 1
                stripped_line = line.strip()
                if not stripped_line: # 비어있는 라인은 스킵
                    # LLM에게도 전달하지 않아 규칙 1에 부합
                    continue
                if stripped_line.startswith('#'): # 주석 라인 스킵 (CDR 파일 특성 고려)
                    # LLM에게도 전달하지 않아 규칙 1에 부합
                    continue
                # 필드명만 나열된 헤더 라인도 스킵
                # 간단한 휴리스틱으로 판단. CSV 포맷이면서, 첫 라인이고, 알파벳/콤마로만 구성된 경우
                # 이 로직은 주석 처리하여 LLM의 규칙 1에 의존하도록 변경
                # is_potential_header_line = False
                # if format_type == "csv" and i == 0 and re.fullmatch(r'[a-zA-Z0-9_, ]+', stripped_line):
                #     is_potential_header_line = True
                # if is_potential_header_line:
                #    continue

                actual_length = len(stripped_line)
                
                # Python 기반으로 파싱 시도 (LLM에게 전달할 파싱 결과 생성)
                parsed_data = {}
                try:
                    # CDRParser 인스턴스를 통해 parse_cdr_line 메서드를 호출
                    # cdr_parser_instance 인자를 사용하도록 변경
                    parsed_data = cdr_parser_instance.parse_cdr_line(stripped_line, identified_format_name, format_definition_content)
                except Exception as e:
                    logging.warning(f"Line {line_number} parsing failed by Python: {e}", exc_info=True)
                    # 파싱 실패 시에도 원본 라인을 LLM에 전달하여 판단하게 함

                # LLM 프롬프트에 들어갈 각 라인 정보 구성 (주석/빈 라인은 위에서 이미 스킵됨)
                # 헤더 라인은 파싱될 수 있으나, LLM이 규칙에 따라 무시하게 됨
                processed_lines_for_llm.append(
                    f"Line {line_number} (Actual Length: {actual_length}):\n"
                    f"Original: `{stripped_line}`\n"
                    f"Parsed Data (Python): `{json.dumps(parsed_data, ensure_ascii=False, default=str) if parsed_data else 'Parsing Failed'}`\n" # << datetime 객체 직렬화를 위해 default=str 추가
                )

            # LLM에 너무 많은 라인을 한 번에 보내지 않도록 제한
            llm_input_lines = processed_lines_for_llm[:config.MAX_LINES_FOR_LLM_VALIDATION]
            remaining_lines_count = len(processed_lines_for_llm) - len(llm_input_lines)
            
            user_message_content = f"검증할 CDR 샘플 데이터 (첫 {len(llm_input_lines)}개 라인):\n```\n" + "\n".join(llm_input_lines) + "\n```"
            if remaining_lines_count > 0:
                user_message_content += f"\n... (총 {remaining_lines_count}개 라인 생략됨)"


            # 3. Azure OpenAI (LLM) 기반 최종 유효성 검증
            system_prompt_content = self.validate_cdr_system_prompt_template.format(
                format_type=format_type,
                format_definition=json.dumps(format_definition_content, ensure_ascii=False, indent=2, default=str), # << datetime 객체 직렬화를 위해 default=str 추가
                record_length_defined=record_length_defined_for_prompt,
                delimiter_defined=delimiter_defined_for_prompt
            )
            logging.info("Calling LLM for CDR file validation.")
            llm_validation_response = self._call_openai_chat_completion(system_prompt_content, user_message_content)

            if llm_validation_response:
                try:
                    # << 수정: LLM 응답에서 마크다운 코드 블록 표식 제거 >>
                    cleaned_llm_response = llm_validation_response.strip()
                    if cleaned_llm_response.startswith("```json"):
                        cleaned_llm_response = cleaned_llm_response[len("```json"):].strip()
                    if cleaned_llm_response.endswith("```"):
                        cleaned_llm_response = cleaned_llm_response[:-len("```")].strip()
                    
                    parsed_issues = json.loads(cleaned_llm_response) # << 전처리된 문자열로 파싱 시도
                    if isinstance(parsed_issues, list):
                        issues.extend(parsed_issues)
                        logging.info(f"LLM identified {len(parsed_issues)} issues.")
                    else:
                        logging.error(f"LLM 응답이 예상한 JSON 배열 형식이 아님: {llm_validation_response}")
                        issues.append({"line": -1, "description": f"LLM 응답 형식 오류. LLM 응답: {llm_validation_response[:200]}..."})
                except json.JSONDecodeError as e:
                    logging.error(f"LLM 유효성 검증 응답 JSON 파싱 오류: {e}, 응답: {llm_validation_response}", exc_info=True)
                    issues.append({"line": -1, "description": "LLM 응답 파싱 오류. 자세한 로그 확인 필요."})
            else:
                issues.append({"line": -1, "description": "LLM으로부터 유효성 검증 응답을 받지 못했습니다."})

        except Exception as e:
            logging.error(f"CDR 파일 유효성 검증 중 예상치 못한 오류 발생: {e}", exc_info=True)
            issues.append({"line": -1, "description": f"서버 내부 오류: {e}"})
        
        return issues