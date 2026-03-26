"""
Umbrella 🌂 — Backend Base
Abstract interface that all backends implement.
"""

from abc import ABC, abstractmethod
from ..schemas import ValidateResult, CompressResult


class BaseBackend(ABC):
    """All backends implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @property
    @abstractmethod
    def requires_gpu(self) -> bool: ...

    @abstractmethod
    def validate(self, bits: int, seq_len: int = 512, **kwargs) -> ValidateResult: ...

    @abstractmethod
    def compress(self, bits: int, **kwargs) -> CompressResult: ...
