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
import numpy as np # 통계 계산을 위해 추가

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
            SearchField(name="format_type", type=SearchFieldDataType.String, searchable=True, filterable=True) 
        ]
        
        text_weights = TextWeights(weights={"format_name": 1.5, "format_content": 3.0, "format_type": 5.0}) 
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
            logging.error(f"Azure AI Search 인덱스 생성/업데이트 오류: {e}")
            raise

    def upload_format_document(self, file_name_from_upload: str, content_from_upload: str) -> bool:
        """포맷 정의를 Azure AI Search 인덱스에 문서로 업로드(색인)합니다."""
        inferred_format_type = "unknown"
        actual_format_name = file_name_from_upload 
        
        try:
            format_obj = json.loads(content_from_upload)
            inferred_format_type = format_obj.get("format_type", "unknown")
            if format_obj.get("format_name"):
                actual_format_name = format_obj["format_name"]

        except json.JSONDecodeError:
            logging.warning(f"'{file_name_from_upload}'의 content가 유효한 JSON이 아닙니다. format_type을 'unknown'으로 설정합니다. 원본: {content_from_upload[:100]}...")
        except Exception as e:
            logging.error(f"'{file_name_from_upload}'의 JSON 파싱 중 오류 발생: {e}. 원본: {content_from_upload[:100]}...")
        
        doc_id = actual_format_name.lower().replace(".json", "").replace("_", "-") 

        document = {
            "id": doc_id, 
            "format_name": actual_format_name,
            "format_content": content_from_upload, 
            "format_type": inferred_format_type 
        }
        try:
            result = self.search_client.upload_documents([document])
            logging.info(f"CDR 포맷 '{actual_format_name}'이(가) Azure AI Search에 색인되었습니다. 결과: {result}")
            return True
        except Exception as e:
            logging.error(f"CDR 포맷 '{actual_format_name}' 색인 오류: {e}")
            return False

    def search_format(self, query: str) -> tuple[str | None, str | None]:
        """쿼리를 기반으로 Azure AI Search에서 가장 적합한 포맷 정의를 검색합니다."""
        try:
            inferred_query_format_type = "unknown"
            
            raw_lines = query.replace('\r', '').strip().split('\n')
            relevant_sample_lines = [
                line.strip() for line in raw_lines 
                if line.strip() and not line.strip().startswith('#')
            ]
            
            meta_query_parts = []
            
            if len(relevant_sample_lines) >= 3: # 최소 3줄 이상이 있어야 패턴 분석의 신뢰성 높음
                first_few_lines = relevant_sample_lines[:min(len(relevant_sample_lines), 10)]

                comma_counts_per_line = [line.count(',') for line in first_few_lines]
                avg_comma_count = np.mean(comma_counts_per_line) if comma_counts_per_line else 0

                non_zero_comma_counts = [c for c in comma_counts_per_line if c > 0]
                comma_counts_consistent = False
                if len(non_zero_comma_counts) >= 2:
                    if len(set(non_zero_comma_counts)) == 1: # 모든 콤마 개수가 동일하다면
                        comma_counts_consistent = True
                    elif len(non_zero_comma_counts) > 2 and np.std(non_zero_comma_counts) / np.mean(non_zero_comma_counts) < 0.1: # 표준편차가 평균의 10% 이내
                         comma_counts_consistent = True

                line_lengths = [len(line) for line in first_few_lines if len(line) > 0]
                
                all_lengths_nearly_same = False
                if len(line_lengths) >= 3 and line_lengths[0] > 0:
                    first_len = line_lengths[0]
                    all_lengths_nearly_same = all(abs(l - first_len) / first_len <= 0.01 for l in line_lengths) 
                
                # ----------------- 유추 로직 대폭 강화 (구분자 우선) -----------------
                # 1. CSV로 판단하는 경우 (콤마가 강력한 지표, 다른 조건보다 우선)
                #    - 평균 콤마 개수가 1개 이상이고, 콤마 개수가 라인별로 일관되게 나타나야 함.
                #    - 또는 콤마가 많고 (0.5개 이상) 라인 길이가 Fixed-Length처럼 엄격하게 동일하지 않을 경우
                if (avg_comma_count >= 1.0 and comma_counts_consistent) or \
                   (avg_comma_count >= 0.5 and not all_lengths_nearly_same):
                    inferred_query_format_type = "csv"
                    meta_query_parts.append("CSV format delimiter comma fields")
                    if len(relevant_sample_lines) >= 2 and relevant_sample_lines[0].count(',') > 0 and relevant_sample_lines[1].count(',') > 0:
                        potential_fields = [f'"{field.strip()}"' for field in relevant_sample_lines[0].split(',')[:5]] # 상위 5개 필드만, 따옴표로 감싸서 정확도 높임
                        meta_query_parts.extend(potential_fields)
                # 2. Fixed-Length로 판단하는 경우 (콤마가 거의 없고 + 길이 일관성)
                elif avg_comma_count < 0.2 and all_lengths_nearly_same and len(first_few_lines) >= 3:
                    inferred_query_format_type = "fixed_length"
                    meta_query_parts.append("fixed_length format record_length")
                # ----------------- 유추 로직 대폭 강화 끝 -----------------
            
            # CDR 샘플 내용과 유추된 메타-질의어를 결합
            # meta_query_parts를 먼저 두어 Search가 "타입"에 더 높은 가중치를 주도록 유도
            combined_search_text = " ".join(meta_query_parts + [query]) if meta_query_parts else query
            
            search_filter_str = None
            if inferred_query_format_type != "unknown":
                search_filter_str = f"format_type eq '{inferred_query_format_type}'"

            logging.info(f"CDR 샘플 기반 유추된 포맷 타입: '{inferred_query_format_type}'")
            logging.info(f"AI Search 쿼리 텍스트 (강화된 메타-질의어 포함): '{combined_search_text}', 필터: '{search_filter_str}'") 

            results = self.search_client.search(
                search_text=combined_search_text, 
                top=1, 
                select=["format_name", "format_content", "format_type"],
                filter=search_filter_str 
            )
            for result in results:
                logging.info(f"AI Search에서 검색된 최적의 포맷: '{result['format_name']}' (format_type: {result.get('format_type')})") 
                return result['format_name'], result['format_content']
            logging.info(f"AI Search에서 적합한 포맷을 찾지 못했습니다.") 
            return None, None
        except Exception as e:
            logging.error(f"Azure AI Search 쿼리 오류: {e}")
            return None, None