from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv

class BaseFinanceTool(ABC):
    def __init__(self, key_name: str = None):
        load_dotenv()
        self.api_key = os.getenv(key_name) if key_name else None

    @abstractmethod
    def get(self, *args, **kwargs):
        """모든 툴이 구현해야 하는 메서드. 반환 형식은 자유."""
        pass