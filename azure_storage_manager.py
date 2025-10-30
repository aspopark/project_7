# azure_storage_manager.py
from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME, AZURE_STORAGE_CDR_CONTAINER_NAME
import logging
from azure.core.exceptions import ResourceExistsError # << 수정: ResourceExistsError 임포트 추가

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AzureStorageManager:
    def __init__(self):
        try:
            # Blob Service 클라이언트 초기화
            self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
            # 포맷 정의용 컨테이너 클라이언트
            self.definitions_container_client = self.blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
            # CDR 파일용 컨테이너 클라이언트
            self.cdrs_container_client = self.blob_service_client.get_container_client(AZURE_STORAGE_CDR_CONTAINER_NAME)
            # 컨테이너가 존재하지 않으면 생성
            self._create_containers_if_not_exist()
        except Exception as e:
            logging.error(f"Azure Storage 연결 오류: {e}", exc_info=True) # << 수정: exc_info=True 추가
            raise # 초기화 실패 시 애플리케이션 시작을 중단합니다.

    def _create_containers_if_not_exist(self):
        """두 컨테이너(포맷 정의, CDR 파일)가 존재하지 않으면 생성합니다."""
        for container_name, client in [
            (AZURE_STORAGE_CONTAINER_NAME, self.definitions_container_client),
            (AZURE_STORAGE_CDR_CONTAINER_NAME, self.cdrs_container_client)
        ]:
            try:
                client.create_container()
                logging.info(f"컨테이너 '{container_name}'가 생성되었습니다.")
            except ResourceExistsError: # << 수정: ResourceExistsError 예외를 직접 잡음
                logging.info(f"컨테이너 '{container_name}'가 이미 존재합니다. 스킵합니다.")
            except Exception as e:
                logging.error(f"컨테이너 '{container_name}' 생성 오류: {e}", exc_info=True) # << 수정: exc_info=True 추가

    def upload_file(self, file_content: str, blob_name: str, container_client) -> bool:
        """지정된 컨테이너에 파일을 업로드합니다."""
        try:
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(file_content.encode('utf-8'), overwrite=True) # 파일 내용을 UTF-8로 인코딩하여 업로드
            logging.info(f"'{blob_name}' 파일을 Azure Blob Storage에 성공적으로 업로드했습니다.")
            return True
        except Exception as e:
            logging.error(f"'{blob_name}' 파일 업로드 오류: {e}", exc_info=True) # << 수정: exc_info=True 추가
            return False

    def download_file(self, blob_name: str, container_client) -> str | None:
        """지정된 컨테이너에서 파일을 다운로드하고 문자열로 반환합니다."""
        try:
            blob_client = container_client.get_blob_client(blob_name)
            download_stream = blob_client.download_blob()
            logging.info(f"'{blob_name}' 파일을 Azure Blob Storage에서 성공적으로 다운로드했습니다.")
            return download_stream.readall().decode('utf-8')
        except Exception as e:
            logging.error(f"'{blob_name}' 파일 다운로드 오류: {e}", exc_info=True) # << 수정: exc_info=True 추가
            return None
    
    def list_definitions(self) -> list[str]:
        """포맷 정의 컨테이너에 있는 파일 목록을 가져옵니다."""
        try:
            blob_list = self.definitions_container_client.list_blobs()
            return [blob.name for blob in blob_list]
        except Exception as e:
            logging.error(f"정의 파일 목록 가져오기 오류: {e}", exc_info=True) # << 수정: exc_info=True 추가
            return []

    def upload_format_definition(self, file_name: str, content: str) -> bool:
        """CDR 포맷 정의 파일을 Blob Storage에 업로드합니다."""
        return self.upload_file(content, file_name, self.definitions_container_client)

    def get_format_definition(self, file_name: str) -> str | None:
        """CDR 포맷 정의 파일을 Blob Storage에서 다운로드합니다."""
        return self.download_file(file_name, self.definitions_container_client)
    
    def upload_cdr_file(self, file_name: str, content: str) -> bool:
        """CDR 파일을 Blob Storage에 업로드합니다. (현재는 업로드 로직은 있으나 Streamlit UI에 통합되어 있지는 않습니다. 필요 시 추가)"""
        return self.upload_file(content, file_name, self.cdrs_container_client)

    def get_cdr_file(self, file_name: str) -> str | None:
        """CDR 파일을 Blob Storage에서 다운로드합니다. (현재는 사용되지 않으나 필요 시 활용)"""
        return self.download_file(file_name, self.cdrs_container_client)