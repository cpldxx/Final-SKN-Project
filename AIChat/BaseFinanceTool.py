from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
from typing import Any

class BaseFinanceTool(ABC):
    """
    Abstract base class for all finance-related agents/tools.
    Provides a unified get() interface for fetching data or performing actions.
    """
    def __init__(self, key_name: str = None):
        load_dotenv()
        self.api_key = os.getenv(key_name) if key_name else None

    @abstractmethod
    def get(self, *args, **kwargs) -> Any:
        """
        Abstract method to fetch or compute data. Should be implemented by all subclasses.
        """
        pass 