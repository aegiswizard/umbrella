"""Umbrella actions."""
from .doctor   import run_doctor
from .suggest  import run_suggest
from .validate import run_validate
from .compress import run_compress
from .autotune import run_autotune

__all__ = ["run_doctor", "run_suggest", "run_validate", "run_compress", "run_autotune"]
