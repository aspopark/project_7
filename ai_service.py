# ai_service.py
import json
import logging
import os
import re
from typing import List, Dict, Any, Optional

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
        # self.cdr_parser = CDRParser() # << CDRParser는 app.py에서 관리하고 전달받도록 변경 (cache_resource 때문)

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
            # search_manager의 upload_format_document 호출에 맞춰 document 구성
            # ai_service에서 upload_format_document의 file_name_from_upload와 content_from_upload를 인자로 기대함
            # 내부적으로 이 함수가 JSON 파싱을 수행하므로, format_content_str을 그대로 전달
            search_success = self.search_manager.upload_format_document(
                format_name, # file_name_from_upload
                format_content_str # content_from_upload
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
        """
        if not cdr_sample_lines:
            return {"inferred_query_format_type": "unknown", "meta_query_fields": [], "all_lines_have_same_length": False}

        # 라인 길이 분석
        line_lengths = [len(line.strip()) for line in cdr_sample_lines if line.strip()]
        unique_lengths = set(line_lengths)

        is_fixed_length_candidate = False
        all_lines_have_same_length = False

        if len(line_lengths) > 0 and len(unique_lengths) == 1:
            is_fixed_length_candidate = True
            all_lines_have_same_length = True # 모든 라인 길이가 완벽하게 동일
        elif len(unique_lengths) == 2:
            # 두 가지 길이 값이 있고, 한 라인만 길이가 다른 경우 (헤더/푸터 후보)
            lengths_counts = {length: line_lengths.count(length) for length in unique_lengths}
            if min(lengths_counts.values()) == 1: # 한 길이가 1번만 나타나면 후보
                is_fixed_length_candidate = True


        # 콤마(,) 개수 분석 (CSV 가능성)
        comma_counts = [line.count(',') for line in cdr_sample_lines if line.strip()]
        unique_comma_counts = set(comma_counts)
        has_consistent_commas = False

        if len(comma_counts) > 0 and len(unique_comma_counts) == 1:
            has_consistent_commas = True
        elif len(unique_comma_counts) == 2:
            # 두 가지 콤마 개수 값이 있고, 한 라인만 콤마 개수가 다른 경우 (헤더/푸터 후보)
            commas_counts_dict = {count: comma_counts.count(count) for count in unique_comma_counts}
            if min(commas_counts_dict.values()) == 1: # 한 콤마 개수가 1번만 나타나면 후보
                has_consistent_commas = True


        inferred_query_format_type = "unknown"
        meta_query_fields = []

        if is_fixed_length_candidate and not has_consistent_commas:
            # 라인 길이는 일정한데 콤마 개수는 불규칙 -> Fixed Length
            inferred_query_format_type = "fixed_length"
        elif has_consistent_commas and not is_fixed_length_candidate:
            # 콤마 개수는 일정한데 라인 길이는 불규칙 (CSV 가능성 높음, Fixed Length는 아님) -> CSV
            inferred_query_format_type = "csv"
            # CSV로 추정될 경우, 첫 번째 라인에서 잠재적 필드명 추출
            if cdr_sample_lines and cdr_sample_lines[0].strip():
                first_line = cdr_sample_lines[0].strip()
                potential_fields = [f.strip() for f in first_line.split(',') if f.strip()]
                # 영문자, 숫자, 언더스코어 조합의 필드명으로만 간주 (정확도를 위해 상위 5개 정도만 사용)
                meta_query_fields = [f for f in potential_fields if re.fullmatch(r'[a-zA-Z0-9_]+', f)][:5]
                logging.debug(f"Inferred potential CSV header fields: {meta_query_fields}")
        elif is_fixed_length_candidate and has_consistent_commas:
            # 둘 다 일관성 있는 경우 -> CSV일 가능성이 더 높음 (Fixed Length이면서 콤마 포함도 가능하므로)
            inferred_query_format_type = "csv"
            if cdr_sample_lines and cdr_sample_lines[0].strip():
                first_line = cdr_sample_lines[0].strip()
                potential_fields = [f.strip() for f in first_line.split(',') if f.strip()]
                meta_query_fields = [f for f in potential_fields if re.fullmatch(r'[a-zA-Z0-9_]+', f)][:5]

        logging.info(f"CDR sample analysis: type={inferred_query_format_type}, fields={meta_query_fields}, all_lines_same_len={all_lines_have_same_length}")
        return {
            "inferred_query_format_type": inferred_query_format_type,
            "meta_query_fields": meta_query_fields,
            "all_lines_have_same_length": all_lines_have_same_length,
            "avg_line_length": sum(line_lengths) / len(line_lengths) if line_lengths else 0
        }

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
        meta_fields_str = " ".join(analysis_result["meta_query_fields"])
        avg_line_length = analysis_result["avg_line_length"]

        # 2. Azure AI Search를 통한 포맷 정의 검색 시도
        search_query = f"{meta_fields_str} CDR Format" if meta_fields_str else "CDR Format"
        
        retrieved_formats = self.search_manager.search_format(
            query=search_query,
            format_type=inferred_type if inferred_type != "unknown" else None,
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
                    # LLM 응답이 JSON 형태라고 가정하고 파싱
                    response_json = json.loads(llm_response)
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
                response_json = json.loads(llm_general_response)
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
                inferred_json_definition = json.loads(llm_infer_response)
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

    # cdr_parser 인자를 추가하여 app.py에서 전달받도록 변경
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
            delimiter = format_definition_content.get("delimiter") # CSV일 경우 필요 (None 가능)

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
                # >> LLM 규칙 1번에 의해 이 판단은 LLM에게 위임하는 것이 좋습니다.
                #    다만, CDRParser.parse_cdr_line으로 넘겨서 파싱이 시도되는 것은 의미가 있습니다.

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
                    # LLM 응답이 JSON 배열 형태라고 가정하고 파싱
                    parsed_issues = json.loads(llm_validation_response)
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