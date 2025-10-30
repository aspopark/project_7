# data_parser.py
import json
import logging
from datetime import datetime
import re
from typing import Dict, Any, Union, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _convert_value_to_type(value: str, target_type: str) -> Union[str, int, float, bool, datetime, None]:
    """
    문자열 값을 지정된 타입으로 변환합니다. 변환 실패 시 ValueError를 발생시킵니다.
    """
    if value is None or value.strip() == "":
        return None # 빈 값은 None으로 처리

    value = value.strip()

    if target_type == "string":
        return value
    elif target_type == "integer":
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"'{value}'는 유효한 정수 형식이 아닙니다.")
    elif target_type == "float":
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"'{value}'는 유효한 실수 형식이 아닙니다.")
    elif target_type == "boolean":
        if value.lower() in ['true', '1', 't', 'y', 'yes']:
            return True
        elif value.lower() in ['false', '0', 'n']:
            return False
        else:
            raise ValueError(f"'{value}'는 유효한 부울 형식이 아닙니다.")
    elif target_type == "datetime":
        try:
            # ISO 8601 with/without timezone
            # YYYY-MM-DD HH:MM:SS
            # YYYYMMDDHHMMSS
            if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$", value):
                 # datetime.fromisoformat은 Z를 +00:00으로 처리하지 못하는 경우가 있어 명시적 대체
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            elif re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?$", value):
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            elif re.match(r"^\d{14}$", value) and value.isdigit():
                return datetime.strptime(value, "%Y%m%d%H%M%S")
            else:
                raise ValueError(f"'{value}'는 유효한 날짜/시간 형식이 아닙니다.")
        except ValueError:
            raise ValueError(f"'{value}'는 유효한 날짜/시간 형식이 아닙니다.")
    else:
        logging.warning(f"Unknown target type '{target_type}'. Returning raw string.")
        return value


class CDRParser:
    def parse_fixed_length(self, cdr_line: str, format_definition: dict) -> dict: 
        """
        고정 길이(fixed_length) CDR 라인을 포맷 정의에 따라 파싱합니다.
        format_definition은 'fields' 키를 가진 dict 형태입니다.
        """
        fields_def = format_definition.get('fields', [])
        record_length_defined = format_definition.get('record_length') # 전체 레코드 길이도 파싱 정의에 있을 수 있음
        
        parsed_data = {}
        try:
            # 전체 라인 길이가 정의된 길이와 다르면 경고는 발생시키지만, 필드 파싱은 정의된 길이에 따라 진행
            if record_length_defined is not None and len(cdr_line) != record_length_defined:
                 logging.warning(f"CDRParser: 파싱하려는 라인({len(cdr_line)})과 정의된 레코드 길이({record_length_defined}) 불일치. 원본 라인: '{cdr_line}'")

            current_pos = 0 # 각 필드 정의에 'start'가 명시되어 있지 않은 경우를 위한 기본값

            for field_def in fields_def:
                name = field_def['name']
                start = field_def.get('start', current_pos) # 'start'가 없으면 이전 필드 바로 다음
                length = field_def['length']
                field_type = field_def.get('type', 'string')

                if name is None or length is None:
                    logging.warning(f"Malformed field definition (name or length missing): {field_def}. Skipping.")
                    continue
                
                raw_value = ""
                if start < len(cdr_line): # 필드 시작 위치가 라인 안에 있는지 확인
                    # 정의된 length만큼 추출, 하지만 라인 끝을 넘어가지 않게
                    raw_value = cdr_line[start : min(start + length, len(cdr_line))]
                
                try:
                    parsed_data[name] = _convert_value_to_type(raw_value, field_type)
                except ValueError as ve:
                    parsed_data[name] = f"PARSE_ERROR: '{raw_value}' (expected {field_type}, error: {ve})"
                    logging.debug(f"Type conversion error for field '{name}' in fixed_length ('{raw_value}' as {field_type}): {ve}")
                except Exception as e:
                    parsed_data[name] = f"UNKNOWN_PARSE_ERROR: '{raw_value}'"
                    logging.debug(f"Unknown error during parsing fixed_length field '{name}': {e}", exc_info=True)
                
                # 다음 필드의 current_pos 기본값 업데이트 (만약 다음 필드에 start가 명시되어 있지 않을 경우 대비)
                current_pos = start + length

            return parsed_data
        except Exception as e:
            logging.error(f"고정 길이 파싱 중 오류: {e}. CDR 라인: '{cdr_line}'", exc_info=True)
            return {"error": f"고정 길이 파싱 오류: {e}"}

    def parse_csv(self, cdr_line: str, format_definition: dict) -> dict:
        """
        CSV CDR 라인을 포맷 정의에 따라 파싱합니다.
        주인님의 포맷 정의 스키마 (schema 객체 내 delimiter, header, types)를 지원합니다.
        """
        # << 수정: format_definition에서 schema 객체와 그 내부 필드들을 가져옵니다. >>
        schema = format_definition.get('schema', {})
        delimiter = schema.get('delimiter', ',') # schema.delimiter 사용, 기본값은 콤마
        header_fields = schema.get('header', []) # schema.header 사용
        field_types = schema.get('types', {}) # schema.types 사용 (필드명: 타입 맵)
        
        values = cdr_line.split(delimiter)
        
        # LLM에게 필드 개수 불일치를 보고하게 했으므로 여기서는 경고만 로깅
        if len(values) != len(header_fields):
            logging.warning(f"CDRParser: 파싱하려는 라인({len(values)} 필드)과 정의된 헤더 필드 개수({len(header_fields)} 필드) 불일치. 원본 라인: '{cdr_line}'")

        parsed_data = {}
        for i, field_name in enumerate(header_fields): # << 수정: header_fields를 기반으로 파싱
            field_type = field_types.get(field_name, 'string') # << 수정: types 맵에서 타입 가져옴
            
            field_value_raw = values[i] if i < len(values) else "" 
            
            try:
                parsed_data[field_name] = _convert_value_to_type(field_value_raw, field_type)
            except ValueError as ve: 
                parsed_data[field_name] = f"PARSE_ERROR: '{field_value_raw}' (expected {field_type}, error: {ve})"
                logging.debug(f"Type conversion error for field '{field_name}' in csv ('{field_value_raw}' as {field_type}): {ve}")
            except Exception as e:
                parsed_data[name] = f"UNKNOWN_PARSE_ERROR: '{field_value_raw}'" # << 수정: name 대신 field_name 사용
                logging.debug(f"Unknown error during parsing csv field '{field_name}': {e}", exc_info=True)

        return parsed_data

    def parse_cdr_line(self, cdr_line: str, format_definition_name: str, format_definition_content: Union[str, Dict[str, Any]]) -> dict:
        """
        주어진 포맷 정의에 따라 단일 CDR 라인을 파싱합니다.
        AI가 식별한 포맷 정의 이름을 기반으로 Fixed Length 또는 CSV 파서를 선택합니다.
        ai_service.py에서는 format_definition_content를 이미 JSON 객체로 가지고 있으므로,
        여기서는 format_definition_content가 문자열일 경우만 json.loads를 수행하고,
        이미 Dict인 경우 그대로 사용하도록 처리합니다.
        """
        format_def_obj: Dict[str, Any] = {}
        if isinstance(format_definition_content, str):
            try:
                format_def_obj = json.loads(format_definition_content)
            except json.JSONDecodeError:
                logging.error(f"포맷 정의 JSON 파싱 오류: {format_definition_content}. (format_definition_name: {format_definition_name})", exc_info=True)
                return {"error": "잘못된 포맷 정의입니다 (JSON 형식 아님)."}
        elif isinstance(format_definition_content, dict):
            format_def_obj = format_definition_content
        else:
            logging.error(f"예상치 못한 포맷 정의 콘텐츠 타입: {type(format_definition_content)}. (format_definition_name: {format_definition_name})", exc_info=True)
            return {"error": "예상치 못한 포맷 정의 콘텐츠 타입입니다."}


        format_type = format_def_obj.get('format_type')
        if format_type == "fixed_length":
            return self.parse_fixed_length(cdr_line, format_def_obj)
        elif format_type == "csv":
            return self.parse_csv(cdr_line, format_def_obj)
        else:
            logging.warning(f"지원되지 않는 포맷 정의 타입입니다: '{format_type}'. 'fixed_length' 또는 'csv'여야 합니다. (format_definition_name: {format_definition_name})")
            return {"error": f"지원되지 않는 포맷 정의 타입입니다: '{format_type}'"}