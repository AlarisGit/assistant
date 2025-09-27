import os
import platform
import socket

import config
import logging
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_hostname():
    names = set()
    try:
        names.add(os.uname()[1])
    except:
        pass
    try:
        names.add(platform.node())
    except:
        pass
    try:
        names.add(socket.gethostname())
    except:
        pass

    return '/'.join(x.split('.')[0] for x in sorted(list(names)))

def get_pid():
    return os.getpid()

def _contains_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters."""
    cyrillic_pattern = r'[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F]'
    return bool(re.search(cyrillic_pattern, text))

def _contains_chinese(text: str) -> bool:
    """Check if text contains Chinese (CJK) characters."""
    # CJK Unified Ideographs and common Chinese punctuation
    chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f]'
    return bool(re.search(chinese_pattern, text))

def _is_cyrillic_text(text: str, min_length: int = None, min_percentage: float = None) -> bool:
    """
    Check if text is predominantly Cyrillic.
    
    Args:
        text: Text to analyze
        min_length: Minimum text length to consider (uses config if None)
        min_percentage: Minimum percentage of Cyrillic characters required (uses config if None)
    
    Returns:
        True if text meets Cyrillic criteria, False otherwise
    """
    # Use config defaults if not provided
    if min_length is None:
        import config
        min_length = config.LANG_DETECTION_MIN_LENGTH_CYRILLIC
    if min_percentage is None:
        import config
        min_percentage = config.LANG_DETECTION_MIN_PERCENTAGE_CYRILLIC
    
    if len(text) < min_length:
        return False
    
    # Count Cyrillic characters
    cyrillic_pattern = r'[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F]'
    cyrillic_chars = len(re.findall(cyrillic_pattern, text))
    
    # Calculate percentage (excluding spaces and punctuation for more accurate ratio)
    text_chars = len(re.sub(r'[\s\W\d]', '', text))  # Only count letters
    if text_chars == 0:
        return False
    
    cyrillic_ratio = cyrillic_chars / text_chars
    return cyrillic_ratio >= min_percentage

def _is_chinese_text(text: str, min_length: int = None, min_percentage: float = None) -> bool:
    """
    Check if text is predominantly Chinese.
    
    Args:
        text: Text to analyze
        min_length: Minimum text length to consider (uses config if None)
        min_percentage: Minimum percentage of Chinese characters required (uses config if None)
    
    Returns:
        True if text meets Chinese criteria, False otherwise
    """
    # Use config defaults if not provided
    if min_length is None:
        import config
        min_length = config.LANG_DETECTION_MIN_LENGTH_CHINESE
    if min_percentage is None:
        import config
        min_percentage = config.LANG_DETECTION_MIN_PERCENTAGE_CHINESE
    
    if len(text) < min_length:
        return False
    
    # Count Chinese characters
    chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]'
    chinese_chars = len(re.findall(chinese_pattern, text))
    
    # Calculate percentage (excluding spaces and punctuation for more accurate ratio)
    text_chars = len(re.sub(r'[\s\W\d]', '', text))  # Only count letters
    if text_chars == 0:
        return False
    
    chinese_ratio = chinese_chars / text_chars
    return chinese_ratio >= min_percentage

if __name__ == '__main__':
    print(get_hostname())
    print(get_pid())