"""InstituteQnA package."""

__version__ = "0.1.0"

from .data_extraction import WebBasedLoader, AttachmentDownloader
# from logging_config import configure_logging
__all__ = [
    "WebBasedLoader", 
    "AttachmentDownloader",
    "configure_logging"
]
