# ai_service.py
from openai import AzureOpenAI
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME
from azure_search_manager import AzureSearchManager
import logging
import json
import os 
import re 
import numpy as np 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIService:
    def __init__(self):
        try:
            self.client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            )
            self.search_manager = AzureSearchManager()
            self.identify_format_system_prompt_template = self._load_prompt_template("identify_format_system.txt")
            self.identify_format_general_prompt_template = self._load_prompt_template("identify_format_general.txt")
            self.validate_cdr_system_prompt_template = self._load_prompt_template("validate_cdr_system.txt")
            self.infer_format_definition_prompt_template = self._load_prompt_template("infer_format_definition.txt") 

        except Exception as e:
            logging.error(f"Azure OpenAI 클라이언트 초기화 오류: {e}")
            raise

    def _load_prompt_template(self, filename: str) -> str:
        """'prompts/' 디렉토리에서 프롬프트 템플릿 파일을 로드합니다."""
        script_dir = os.path.dirname(__file__) 
        filepath = os.path.join(script_dir, "prompts", filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"프롬프트 파일 '{filepath}'을 찾을 수 없습니다. 경로를 확인해주세요.")
            raise
        except Exception as e:
            logging.error(f"프롬pt 파일 '{filepath}' 로드 중 오류 발생: {e}")
            raise

    def _call_llm(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 1500) -> str:
        """Azure OpenAI LLM을 호출하여 응답을 받습니다."""
        try:
            logged_messages = []
            for msg in messages:
                logged_msg = msg.copy()
                if isinstance(logged_msg.get('content'), str) and len(logged_msg['content']) > 500:
                    logged_msg['content'] = logged_msg['content'][:500] + "\n... (콘텐츠가 길어 일부만 로깅됨)"
                logged_messages.append(logged_msg)
            logging.info(f"LLM에 전송하는 메시지: {logged_messages}")
            
            response = self.client.chat.completions.create( 
                model=AZURE_OPENAI_DEPLOYMENT_NAME, 
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw_llm_response = response.choices[0].message.content
            logging.info(f"LLM으로부터 받은 원본 응답: '{raw_llm_response}'")
            return raw_llm_response
        except Exception as e:
            logging.error(f"LLM 호출 오류: {e}")
            return f"LLM 호출 중 오류 발생: {e}"

    def _python_based_format_validation(self, cdr_sample_content: str, format_name: str, format_content: str) -> dict:
        """
        파이썬 코드로 CDR 샘플이 주어진 포맷 정의에 얼마나 일치하는지 검증합니다.
        LLM에게 추가 정보로 제공되어 판단에 활용될 수 있습니다.
        """
        result = {
            "is_fixed_length_like": False,
            "is_csv_like": False,
            "match_confidence": 0.0, # 0.0 ~ 1.0
            "reason": ""
        }
        
        try:
            format_def_obj = {} 
            if format_content and format_name != "general_check": 
                try:
                    format_def_obj = json.loads(format_content)
                    if not isinstance(format_def_obj, dict): 
                        raise ValueError(f"JSON root는 dict이어야 하지만, {type(format_def_obj)}입니다.")
                except json.JSONDecodeError as jde:
                    result["reason"] = f"파이썬 기반 검증: 포맷 정의 JSON 파싱 실패. 오류: {jde}. 원본: {format_content[:100]}..."
                    logging.error(result["reason"])
                    return result
                except ValueError as ve: 
                    result["reason"] = f"파이썬 기반 검증: 포맷 정의 JSON 객체 타입 불일치. 오류: {ve}. 원본: {format_content[:100]}..."
                    logging.error(result["reason"])
                    return result
                except Exception as e:
                    result["reason"] = f"파이썬 기반 검증: 포맷 정의 JSON 처리 중 알 수 없는 오류: {e}. 원본: {format_content[:100]}..."
                    logging.error(result["reason"])
                    return result

            format_type = format_def_obj.get('format_type')
            
            raw_lines = cdr_sample_content.replace('\r', '').strip().split('\n')
            relevant_sample_lines = [
                line.strip() for line in raw_lines 
                if line.strip() and not line.strip().startswith('#')
            ]

            if not relevant_sample_lines:
                result["reason"] = "분석할 유효한 데이터 라인이 없습니다."
                return result

            first_few_lines = relevant_sample_lines[:min(len(relevant_sample_lines), 10)]

            if format_type == "fixed_length" and "record_length" in format_def_obj and "fields" in format_def_obj:
                expected_record_length = format_def_obj["record_length"]
                matched_lines = 0
                for line in first_few_lines:
                    if len(line) == expected_record_length:
                        matched_lines += 1
                
                result["is_fixed_length_like"] = True
                result["match_confidence"] = matched_lines / len(first_few_lines)
                result["reason"] = f"Fixed-Length 형식. 정의된 레코드 길이 {expected_record_length}에 {matched_lines}/{len(first_few_lines)} 라인이 일치합니다."
            
            elif format_type == "csv" and "schema" in format_def_obj and "delimiter" in format_def_obj["schema"] and "header" in format_def_obj["schema"]:
                delimiter = format_def_obj["schema"]["delimiter"]
                expected_field_count = len(format_def_obj["schema"]["header"])
                matched_lines = 0
                
                lines_to_check = first_few_lines
                if len(lines_to_check) > 0 and (lines_to_check[0].count(delimiter) + 1 == expected_field_count) and \
                   (len(lines_to_check) > 1 and (lines_to_check[1].count(delimiter) + 1 != expected_field_count)): 
                    lines_to_check = lines_to_check[1:] 
                    result["reason"] += " (헤더 라인 제외됨)"

                for line in lines_to_check:
                    if line.count(delimiter) + 1 == expected_field_count:
                        matched_lines += 1
                
                result["is_csv_like"] = True
                result["match_confidence"] = matched_lines / len(lines_to_check) if lines_to_check else 0.0
                result["reason"] = f"CSV 형식. 정의된 구분자 '{delimiter}' 기준으로 {expected_field_count}개 필드에 {matched_lines}/{len(lines_to_check)} 라인이 일치합니다."
            elif format_name == "general_check": 
                comma_counts_per_line = [line.count(',') for line in first_few_lines]
                avg_comma_count = np.mean(comma_counts_per_line) if comma_counts_per_line else 0
                
                line_lengths = [len(line) for line in first_few_lines if len(line) > 0]
                all_lengths_nearly_same = False
                if len(line_lengths) >= 3 and line_lengths[0] > 0:
                    first_len = line_lengths[0]
                    all_lengths_nearly_same = all(abs(l - first_len) / first_len <= 0.02 for l in line_lengths)

                if avg_comma_count >= 0.8 and (sum(1 for c in comma_counts_per_line if c > 0) >= len(first_few_lines) * 0.7) and not all_lengths_nearly_same:
                    result["is_csv_like"] = True
                    result["match_confidence"] = 0.9 
                    result["reason"] = "파이썬 일반 유추: CSV 형식으로 강력히 추정됨."
                elif avg_comma_count < 0.2 and all_lengths_nearly_same and len(first_few_lines) >= 3:
                    result["is_fixed_length_like"] = True
                    result["match_confidence"] = 0.9 
                    result["reason"] = "파이썬 일반 유추: Fixed-Length 형식으로 강력히 추정됨."
                else:
                    result["reason"] = "파이썬 일반 유추: 불확실."
            else: 
                result["reason"] = f"파이썬 기반 검증: 포맷 정의에 필요한 필드(format_type, record_length/schema)가 누락되었습니다. 정의된 타입: {format_type}"

        except Exception as e: 
            result["reason"] = f"파이썬 기반 검증 중 알 수 없는 오류 발생: {e}. 원본 format_content: {format_content[:100]}..."
            logging.error(result["reason"])
            
        return result

    def infer_format_definition(self, cdr_sample_content: str) -> tuple[str | None, str | None]:
        """
        LLM에게 CDR 샘플을 분석하여 JSON 포맷 정의를 추론하도록 요청합니다.
        반환값: (추론된 포맷 파일명, 추론된 포맷 정의 JSON 문자열)
        """
        logging.info("AI Search에서 포맷을 찾지 못하여 LLM에게 포맷 정의 추론을 요청합니다.")
        llm_sample_content = cdr_sample_content[:min(len(cdr_sample_content), 4000)]
        
        messages = [
            {"role": "system", "content": self.infer_format_definition_prompt_template},
            {"role": "user", "content": f"다음 CDR 파일의 내용을 분석하여 포맷 정의를 생성해주세요:\n\n```\n{llm_sample_content}\n```"}
        ]
        
        raw_llm_response = self._call_llm(messages, temperature=0.3)
        
        if raw_llm_response:
            json_match = re.search(r'```json\s*(\{.*\})\s*```', raw_llm_response, re.DOTALL)
            json_str = json_match.group(1) if json_match else raw_llm_response.strip()

            if json_str.lower() == "unknown":
                logging.info("LLM이 포맷 정의 추론에 실패하거나 'unknown'을 반환했습니다.")
                return None, None

            try:
                inferred_format_obj = json.loads(json_str)
                
                if not isinstance(inferred_format_obj, dict):
                    raise ValueError(f"LLM이 반환한 JSON root는 객체여야 하지만, {type(inferred_format_obj)}입니다.")
                
                inferred_format_name = inferred_format_obj.get("format_name")
                inferred_format_type = inferred_format_obj.get("format_type")

                if inferred_format_name and inferred_format_type and \
                   ((inferred_format_type == "fixed_length" and "record_length" in inferred_format_obj and "fields" in inferred_format_obj) or \
                    (inferred_format_type == "csv" and "schema" in inferred_format_obj and "delimiter" in inferred_format_obj["schema"])):
                    logging.info(f"LLM이 포맷 정의를 성공적으로 추론했습니다: {inferred_format_name}")
                    return inferred_format_name, json_str 
                else:
                    logging.warning(f"LLM이 유효한 포맷 정의를 추론했으나 필수 필드가 누락되었거나 형식이 맞지 않습니다. 추론 결과: {json_str[:100]}...")
                    return None, None
            except json.JSONDecodeError as jde:
                logging.error(f"LLM이 추론한 포맷 정의가 유효한 JSON이 아닙니다. 오류: {jde}. 원본:\n{json_str}")
                return None, None
            except ValueError as ve:
                logging.error(f"LLM이 추론한 포맷 정의 JSON 객체 타입 불일치. 오류: {ve}. 원본:\n{json_str}")
                return None, None
            except Exception as e:
                logging.error(f"LLM 추론 결과 처리 중 알 수 없는 오류: {e}. 원본:\n{json_str}")
                return None, None
        
        logging.info("LLM으로부터 응답이 없거나 파싱할 내용이 없습니다.")
        return None, None


    def identify_cdr_format(self, cdr_sample_content: str) -> tuple[str | None, str | None]:
        search_sample_content = cdr_sample_content[:2000]
        llm_sample_content = cdr_sample_content[:min(len(cdr_sample_content), 4000)]

        retrieved_format_name, retrieved_format_content = self.search_manager.search_format(search_sample_content)

        if not retrieved_format_name or not retrieved_format_content:
            logging.warning("Azure AI Search에서 적합한 포맷을 찾지 못했습니다. LLM에게 일반적인 식별 또는 추론을 시도합니다.")
            
            python_general_check_result = self._python_based_format_validation(
                llm_sample_content, "general_check", json.dumps({}) 
            )

            context_general_prompt = self.identify_format_general_prompt_template.format(
                python_validation_result=json.dumps(python_general_check_result, indent=2)
            )
            messages_general = [
                {"role": "system", "content": context_general_prompt},
                {"role": "user", "content": f"다음 CDR 파일의 형식을 식별해주세요:\n\n```\n{llm_sample_content}\n```"}
            ]
            identified_format_from_general_llm = self._call_llm(messages_general, temperature=0.0).strip().lower()
            
            if identified_format_from_general_llm.endswith(".json"):
                doc_id_for_llm_identified = identified_format_from_general_llm.replace(".json", "").replace("_", "-")
                try:
                    document_from_search = self.search_manager.search_client.get_document(key=doc_id_for_llm_identified, select=["format_content"])
                    logging.info(f"LLM 일반 식별: 포맷 '{identified_format_from_general_llm}'에 대한 정의를 AI Search에서 로드했습니다.")
                    return identified_format_from_general_llm, document_from_search['format_content']
                except Exception as e:
                    logging.error(f"LLM 일반 식별: 포맷 '{identified_format_from_general_llm}'의 정의를 AI Search에서 로드 실패: {e}. 이제 LLM 추론을 시도합니다.")
            
            inferred_name, inferred_content = self.infer_format_definition(llm_sample_content)
            if inferred_name and inferred_content:
                return inferred_name, inferred_content
            
            logging.warning("LLM이 일반적인 식별 및 포맷 정의 추론에 모두 실패했습니다.")
            return None, None

        try:
            format_obj_check = json.loads(retrieved_format_content)
            if not isinstance(format_obj_check, dict):
                 raise ValueError(f"retrieved_format_content root는 dict이어야 하지만, {type(format_obj_check)}입니다.")

            python_specific_check = self._python_based_format_validation(
                llm_sample_content, retrieved_format_name, retrieved_format_content
            )
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"AI Search에서 가져온 '{retrieved_format_name}'의 format_content가 유효한 JSON 객체가 아닙니다. 오류: {e}. 이제 LLM 추론을 시도합니다.")
            inferred_name, inferred_content = self.infer_format_definition(llm_sample_content)
            if inferred_name and inferred_content:
                logging.info(f"Search 정의 손상으로 LLM이 새로운 포맷 정의를 추론했습니다: {inferred_name}")
                return inferred_name, inferred_content
            return None, None 

        system_specific_prompt = self.identify_format_system_prompt_template.format(
            format_name=retrieved_format_name, 
            format_content=retrieved_format_content,
            python_validation_result=json.dumps(python_specific_check, indent=2) 
        )
        messages_specific = [
            {"role": "system", "content": system_specific_prompt},
            {"role": "user", "content": f"다음 CDR 파일의 형식을 식별하고 위 포맷 정의와의 일치 여부를 판단해주세요:\n\n```\n{llm_sample_content}\n```"} 
        ]
        
        final_identified_format_file = self._call_llm(messages_specific, temperature=0.2).strip() 

        logging.info(f"AI Search 제안 포맷: {retrieved_format_name}, LLM 최종 식별 응답: '{final_identified_format_file}'") 

        is_python_confident_match = False
        if python_specific_check["match_confidence"] >= 0.7:
            if retrieved_format_name.lower().startswith("fixed_length") and python_specific_check["is_fixed_length_like"]:
                is_python_confident_match = True
            elif retrieved_format_name.lower().startswith("csv") and python_specific_check["is_csv_like"]:
                is_python_confident_match = True

        if final_identified_format_file == retrieved_format_name or is_python_confident_match:
            logging.info(f"AI가 최종적으로 식별한 포맷: {retrieved_format_name} (LLM 응답: {final_identified_format_file}, 파이썬 검증 결과 신뢰도: {python_specific_check['match_confidence']})") 
            return retrieved_format_name, retrieved_format_content
        else:
            if final_identified_format_file.startswith("inferred_") and final_identified_format_file.endswith(".json"):
                logging.info(f"LLM이 Search 결과와 다르지만 자체적으로 추론한 포맷({final_identified_format_file})을 선택합니다. 추론 내용을 다시 요청합니다.")
                inferred_name_rerun, inferred_content_rerun = self.infer_format_definition(llm_sample_content) 
                if inferred_name_rerun and inferred_content_rerun and inferred_name_rerun == final_identified_format_file:
                    logging.info(f"LLM의 추론 포맷({inferred_name_rerun})의 내용을 성공적으로 재획득했습니다.")
                    return inferred_name_rerun, inferred_content_rerun
                else:
                    logging.warning(f"LLM의 최종 식별 응답({final_identified_format_file})은 추론된 포맷을 나타내지만, 실제 추론 내용 재획득에 실패했습니다. 'unknown'으로 처리합니다.")
                    return None, None

            logging.warning(f"AI Search 결과({retrieved_format_name})와 AI 최종 식별 결과('{final_identified_format_file}')가 다릅니다. 'unknown'으로 처리합니다. 파이썬 검증 결과 신뢰도: {python_specific_check['match_confidence']}. 최종적으로 LLM 추론을 시도합니다.")
            inferred_name, inferred_content = self.infer_format_definition(llm_sample_content)
            if inferred_name and inferred_content:
                logging.info(f"LLM 판단 불일치 후 최종적으로 추론된 포맷 정의: {inferred_name}")
                return inferred_name, inferred_content
            return None, None 

    def validate_cdr_file(self, cdr_content: str, format_definition_name: str, format_definition_content: str) -> list[dict]:
        format_def_obj = json.loads(format_definition_content) # 포맷 정의 JSON 파싱
        
        # Fixed Length 포맷의 경우 record_length_defined 값을 프롬프트에 전달하기 위해 추출
        record_length_defined = format_def_obj.get('record_length')
        
        # CSV 포맷의 경우 delimiter_defined 및 header_field_count 값을 프롬프트에 전달하기 위해 추출
        delimiter_defined = format_def_obj.get('schema', {}).get('delimiter')
        header_field_count = len(format_def_obj.get('schema', {}).get('header', []))

        # system_prompt_content를 포맷 정의 타입에 따라 다르게 구성
        if format_def_obj.get('format_type') == "fixed_length" and record_length_defined is not None:
            system_prompt_content = self.validate_cdr_system_prompt_template.format(
                format_name=format_definition_name, 
                format_content=format_definition_content,
                record_length_defined=record_length_defined
            )
        elif format_def_obj.get('format_type') == "csv" and delimiter_defined is not None and header_field_count > 0:
            system_prompt_content = self.validate_cdr_system_prompt_template.format(
                format_name=format_definition_name, 
                format_content=format_definition_content,
                delimiter_defined=delimiter_defined,
                header_field_count=header_field_count
            )
        else: # 기본값 또는 오류 처리 (예: LLM에게 추가 정보 없음)
            system_prompt_content = self.validate_cdr_system_prompt_template.format(
                format_name=format_definition_name, 
                format_content=format_definition_content,
                record_length_defined="unknown", # 더미 값, 실제 프롬프트에서 사용 안 되도록 조정 필요
                delimiter_defined="unknown",
                header_field_count="unknown"
            )


        # CDR 라인에 실제 길이 및 유형 정보를 포함하여 LLM에 전달
        cdr_lines_with_info = []
        cdr_original_lines = cdr_content.replace('\r', '').split('\n')
        
        # 헤더 라인을 LLM이 검증 대상에서 제외하도록 돕기 위해 미리 파싱
        header_line_text = ""
        actual_data_start_line = -1
        
        # CSV의 경우 첫번째 라인이 헤더일 수 있음
        if format_def_obj.get('format_type') == "csv" and len(cdr_original_lines) > 0 and \
            cdr_original_lines[0].count(delimiter_defined) + 1 == header_field_count: # CSV이고 필드 수가 일치하면 헤더로 간주
            header_line_text = cdr_original_lines[0].strip()
            actual_data_start_line = 1 # 0번 인덱스는 헤더이므로 실제 데이터는 1번부터
            
        for i, line in enumerate(cdr_original_lines):
            line_stripped = line.strip()
            line_type_hint = "데이터 레코드"
            
            # 주석 라인
            if line_stripped.startswith('#'):
                line_type_hint = "주석 라인 (검증 대상 제외)"
            # CSV 헤더 라인
            elif format_def_obj.get('format_type') == "csv" and i == actual_data_start_line-1 and line_stripped == header_line_text:
                line_type_hint = "헤더 라인 (검증 대상 제외)"
            # 빈 라인
            elif not line_stripped:
                line_type_hint = "빈 라인 (검증 대상 제외)"
            
            # 길이 정보 추가 (Fixed Length, CSV 공통)
            length_info = f"실제 길이: {len(line)}자"
            
            # CSV인 경우 필드 개수 정보도 추가
            field_count_info = ""
            if format_def_obj.get('format_type') == "csv" and delimiter_defined:
                field_count_info = f", 실제 필드 개수: {line.count(delimiter_defined) + 1}개"
            
            cdr_lines_with_info.append(f"라인 {i+1} ({line_type_hint}, {length_info}{field_count_info}): {line}")
        
        user_message_content = f"다음 CDR 파일을 원본 라인 번호를 유지하며 검증해 주세요:\n\n```\n" + "\n".join(cdr_lines_with_info) + "\n```"

        messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_message_content}
        ]

        raw_validation_result = self._call_llm(messages)
        logging.info(f"AI 유효성 검증 원본 결과:\n{raw_validation_result}")

        issues = []
        issue_blocks = raw_validation_result.strip().split('\n---')

        for block in issue_blocks:
            block_lines = block.strip().split('\n')
            if not block_lines:
                continue

            first_line = block_lines[0].strip()
            
            if "No issues found." in first_line:
                return []
            
            match = re.search(r'^라인 (\d+): (.+)', first_line)
            if match:
                original_line_num = int(match.group(1))
                issue_desc = match.group(2).strip()
                # 추가적인 상세 설명이 있다면 issue_desc에 이어서 붙임
                # LLM은 이제 한 줄에 1개의 문제점만 보고하도록 지시했으므로, 이 부분은 더 단순하게 처리될 수 있음
                issues.append({"line": original_line_num, "issue": issue_desc})
            elif first_line: 
                logging.warning(f"LLM이 파싱 불가능한 형식으로 오류 보고: '{first_line}' (라인 번호 패턴 없음)")
                issues.append({"line": -1, "issue": first_line}) 
        
        unique_issues = []
        seen_issues = set()
        for issue in issues:
            issue_tuple = (issue['line'], issue['issue'])
            if issue_tuple not in seen_issues:
                unique_issues.append(issue)
                seen_issues.add(issue_tuple)
        
        unique_issues.sort(key=lambda x: x['line']) 
        return unique_issues