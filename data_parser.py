# data_parser.py
import json
import logging
from datetime import datetime
import re # 정규표현식 추가

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CDRParser:
    def _cast_type(self, value: str, field_type: str):
        """값을 지정된 타입으로 변환 시도."""
        if field_type == "int":
            try:
                return int(value) if value else None
            except ValueError:
                raise ValueError(f"'{value}'는 유효한 정수 형식이 아닙니다.")
        elif field_type == "float":
            try:
                return float(value) if value else None
            except ValueError:
                raise ValueError(f"'{value}'는 유효한 실수 형식이 아닙니다.")
        elif field_type == "bool":
            if value.lower() in ['true', '1', 'y']:
                return True
            elif value.lower() in ['false', '0', 'n']:
                return False
            else:
                raise ValueError(f"'{value}'는 유효한 부울 형식이 아닙니다.")
        elif field_type == "datetime":
            try:
                # ISO 8601 형식과 일반적인 YYYY-MM-DD HH:MM:SS, YYYYMMDDHHMMSS 형식 모두 시도
                if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", value):
                    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                elif re.match(r"^\d{14}$", value) and value.isdigit():
                    return datetime.strptime(value, "%Y%m%d%H%M%S")
                elif re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$", value): # ISO 8601 with timezone
                    # fromisoformat은 Python 3.7+에서 Z를 +00:00으로 자동으로 처리.
                    # 하지만 이전 버전 호환성이나 다른 포맷을 위해 수동 처리 로직 유지.
                    return datetime.fromisoformat(value.replace('Z', '+00:00') if 'Z' in value else value) 
                else:
                    raise ValueError(f"'{value}'는 유효한 날짜/시간 형식이 아닙니다.")

            except ValueError:
                raise ValueError(f"'{value}'는 유효한 날짜/시간 형식이 아닙니다.")
        return value 

    def parse_fixed_length(self, cdr_line: str, format_definition: dict) -> dict: 
        """
        고정 길이(fixed_length) CDR 라인을 포맷 정의에 따라 파싱합니다.
        format_definition은 이제 'fields' 키를 가진 dict 형태입니다.
        """
        fields_def = format_definition.get('fields', [])
        record_length_defined = format_definition.get('record_length')
        
        parsed_data = {}
        try:
            # 전체 라인 길이가 정의된 길이와 다르면 경고는 발생시키지만, 필드 파싱은 정의된 길이에 따라 진행
            if record_length_defined is not None and len(cdr_line) != record_length_defined:
                 logging.warning(f"CDRParser: 파싱하려는 라인({len(cdr_line)})과 정의된 레코드 길이({record_length_defined}) 불일치. 원본 라인: '{cdr_line}'")

            # 파싱 전, 현재 라인 길이가 정의된 총 필드 길이를 초과하는지 여부를 저장
            line_exceeds_defined_length = len(cdr_line) > record_length_defined if record_length_defined is not None else False

            current_pos = 0
            for field_def in fields_def: 
                name = field_def['name']
                start = field_def.get('start', current_pos) 
                length = field_def['length']
                field_type = field_def.get('type', 'string')

                field_value_raw = ""
                # 필드 값 추출 시, 정의된 length 만큼만 정확히 추출.
                # 라인 길이가 정의된 시작 + 길이보다 짧아도 에러가 아닌, 가능한 만큼만 추출
                # start가 라인 길이보다 길거나, length가 0이면 빈 문자열.
                if start < len(cdr_line):
                    # 정의된 length까지 추출. 만약 라인이 짧으면 라인의 끝까지.
                    field_value_raw = cdr_line[start : min(start + length, len(cdr_line))]
                
                # 타입 변환 시도
                try:
                    parsed_data[name] = self._cast_type(field_value_raw.strip(), field_type)
                except ValueError as ve: # _cast_type에서 발생하는 ValueError를 처리
                    parsed_data[name] = f"PARSE_ERROR: '{field_value_raw.strip()}' is not a valid {field_type}. Detail: {ve}"
                except Exception as e: # 기타 예상치 못한 타입 변환 오류
                    parsed_data[name] = f"PARSE_ERROR: Unexpected error parsing '{field_value_raw.strip()}' as {field_type}. Detail: {e}"


                current_pos = start + length 
            return parsed_data
        except Exception as e:
            logging.error(f"고정 길이 파싱 중 오류: {e}. CDR 라인: '{cdr_line}'")
            return {"error": f"고정 길이 파싱 오류: {e}"}

    def parse_csv(self, cdr_line: str, format_definition: dict) -> dict:
        """
        CSV CDR 라인을 포맷 정의에 따라 파싱합니다.
        format_definition은 이제 'schema' 키를 가진 dict 형태입니다.
        """
        schema_def = format_definition.get('schema', {})
        delimiter = schema_def.get('delimiter', ',')
        header = schema_def.get('header', [])
        types = schema_def.get('types', {})
        
        values = cdr_line.split(delimiter)
        
        if len(values) != len(header):
            logging.warning(f"CDRParser: 파싱하려는 라인({len(values)} 필드)과 정의된 헤더({len(header)} 필드) 불일치: '{cdr_line}'")

        parsed_data = {}
        for i, h_name in enumerate(header):
            field_value_raw = values[i].strip() if i < len(values) else "" 
            field_type = types.get(h_name, 'string') 
            try:
                parsed_data[h_name] = self._cast_type(field_value_raw, field_type)
            except ValueError as ve: # _cast_type에서 발생하는 ValueError를 처리
                parsed_data[h_name] = f"PARSE_ERROR: '{field_value_raw}' is not a valid {field_type}. Detail: {ve}"
            except Exception as e: # 기타 예상치 못한 타입 변환 오류
                parsed_data[h_name] = f"PARSE_ERROR: Unexpected error parsing '{field_value_raw}' as {field_type}. Detail: {e}"

        return parsed_data

    def parse_cdr_line(self, cdr_line: str, format_definition_name: str, format_definition_content: str) -> dict:
        """
        주어진 포맷 정의에 따라 단일 CDR 라인을 파싱합니다.
        AI가 식별한 포맷 정의 이름을 기반으로 Fixed Length 또는 CSV 파서를 선택합니다.
        """
        try:
            format_def_obj = json.loads(format_definition_content)
        except json.JSONDecodeError:
            logging.error(f"포맷 정의 JSON 파싱 오류: {format_definition_content}")
            return {"error": "잘못된 포맷 정의입니다 (JSON 형식 아님)."}

        format_type = format_def_obj.get('format_type')
        if format_type == "fixed_length":
            return self.parse_fixed_length(cdr_line, format_def_obj)
        elif format_type == "csv":
            return self.parse_csv(cdr_line, format_def_obj)
        else:
            return {"error": f"지원되지 않는 포맷 정의 타입입니다: '{format_type}'. 'fixed_length' 또는 'csv'여야 합니다."}
