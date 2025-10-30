# azure_search_manager.py
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, 
    SearchField, 
    SearchFieldDataType, 
    SimpleField, 
    ScoringProfile, 
    TextWeights,
)
from azure.core.credentials import AzureKeyCredential
from config import AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME
import logging
import json
# import numpy as np # 이제 _analyze_cdr_sample_structure 함수에서만 사용되므로 여기에선 불필요.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AzureSearchManager:
    def __init__(self):
        self.credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
        self.index_client = SearchIndexClient(AZURE_SEARCH_ENDPOINT, self.credential)
        self._create_or_update_index()
        self.search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, self.credential)


    def _create_or_update_index(self):
        """CDR 포맷 정의를 위한 Azure AI Search 인덱스를 생성하거나 업데이트합니다."""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True, facetable=True),
            SearchField(name="format_name", type=SearchFieldDataType.String, searchable=True, filterable=True, sortable=True),
            SearchField(name="format_content", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="format_type", type=SearchFieldDataType.String, searchable=True, filterable=True) ,
            # ai_service에서 'field_names'를 인덱스에 추가하고 있으므로, 여기에 추가하는 것이 좋습니다.
            SearchField(name="field_names", type=SearchFieldDataType.String, searchable=True, filterable=True)
        ]
        
        text_weights = TextWeights(weights={"format_name": 1.5, "format_content": 3.0, "format_type": 5.0, "field_names": 2.0}) 
        scoring_profile = ScoringProfile(
            name="format_scoring", 
            text_weights=text_weights,
        )

        index = SearchIndex(
            name=AZURE_SEARCH_INDEX_NAME, 
            fields=fields, 
            scoring_profiles=[scoring_profile], 
            default_scoring_profile="format_scoring",
        )

        try:
            self.index_client.create_or_update_index(index)
            logging.info(f"Azure AI Search 인덱스 '{AZURE_SEARCH_INDEX_NAME}' 생성 또는 업데이트 완료.")
        except Exception as e:
            logging.error(f"Azure AI Search 인덱스 생성/업데이트 오류: {e}", exc_info=True)
            raise

    def upload_format_document(self, file_name_from_upload: str, content_from_upload: str) -> bool:
        """
        포맷 정의를 Azure AI Search 인덱스에 문서로 업로드(색인)합니다.
        주인님의 새로운 스키마를 지원하도록 field_names 추출 로직을 수정합니다.
        """
        inferred_format_type = "unknown"
        actual_format_name = file_name_from_upload 
        
        try:
            format_obj = json.loads(content_from_upload)
            inferred_format_type = format_obj.get("format_type", "unknown")
            if format_obj.get("format_name"):
                actual_format_name = format_obj["format_name"]

            # << 수정: 인덱싱 시 사용될 필드명 추출 로직 (schema.header 사용) >>
            field_names = []
            if format_obj.get("schema") and format_obj["schema"].get("header"):
                field_names = format_obj["schema"]["header"] # schema.header 리스트에서 필드명 추출
            
            field_names_str = " ".join(field_names)

        except json.JSONDecodeError:
            logging.warning(f"'{file_name_from_upload}'의 content가 유효한 JSON이 아닙니다. format_type을 'unknown'으로 설정합니다. 원본: {content_from_upload[:100]}...")
            field_names_str = "" # 파싱 실패 시 빈 문자열
        except Exception as e:
            logging.error(f"'{file_name_from_upload}'의 JSON 파싱 중 오류 발생: {e}. 원본: {content_from_upload[:100]}...", exc_info=True)
            field_names_str = "" # 파싱 실패 시 빈 문자열
        
        doc_id = actual_format_name.lower().replace(".json", "").replace("_", "-") 

        document = {
            "id": doc_id, 
            "format_name": actual_format_name,
            "format_content": content_from_upload, 
            "format_type": inferred_format_type,
            "field_names": field_names_str # 'field_names' 필드를 문서에 추가
        }
        try:
            result = self.search_client.upload_documents([document])
            # result는 Success 또는 Failure 여부를 포함한 리스트이므로 더 자세한 로그 가능
            for res in result:
                if res.succeeded:
                    logging.info(f"CDR 포맷 '{actual_format_name}' (ID: {doc_id})이(가) Azure AI Search에 성공적으로 색인되었습니다.")
                else:
                    logging.error(f"CDR 포맷 '{actual_format_name}' (ID: {doc_id}) 색인 실패: {res.error_message}")
            return True
        except Exception as e:
            logging.error(f"CDR 포맷 '{actual_format_name}' (ID: {doc_id}) 색인 오류: {e}", exc_info=True)
            return False

    def search_format(self, query: str = "*", format_type: str = None, top: int = 1) -> list[dict]:
        """
        쿼리를 기반으로 Azure AI Search에서 가장 적합한 포맷 정의를 검색합니다.
        ai_service.py에서 이미 CDR 샘플을 분석하여 format_type을 추론하므로,
        여기서는 그 format_type을 필터로 활용합니다.
        """
        try:
            filter_clause = None
            if format_type and format_type != "unknown": # "unknown"일 때는 필터링하지 않음
                filter_clause = f"format_type eq '{format_type}'"
                logging.debug(f"Azure Search applying filter: {filter_clause}")
            
            # 주인님의 원래 search_format 내부에 있던 CDR 샘플 분석 및 format_type 유추 로직은
            # ai_service.py의 _analyze_cdr_sample_structure 함수에서 수행하므로,
            # 이 메서드에서는 해당 로직을 모두 제거합니다.
            # (raw_lines = query.replace('\r', '').strip().split('\n') ... 이하 모든 관련 로직 제거)
            # numpy 임포트도 더 이상 필요 없습니다.

            results = self.search_client.search(
                search_text=query,
                filter=filter_clause, # format_type을 필터로 사용
                query_type="full", # OR, AND, NOT 등을 지원하는 full Lucene 쿼리 사용
                select=["id", "format_name", "format_type", "format_content"],
                top=top
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result["id"],
                    "format_name": result["format_name"],
                    "format_type": result["format_type"],
                    "format_content": json.loads(result["format_content"]) 
                })
            logging.info(f"Azure Search for query '{query}' (filter: {filter_clause}) returned {len(formatted_results)} results.")
            return formatted_results

        except Exception as e:
            logging.error(f"Azure Search에서 포맷 정의 검색 중 오류: {e}. 쿼리: '{query}', 필터: '{format_type}'", exc_info=True)
            return []