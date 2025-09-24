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
    import re
    return bool(re.search(cyrillic_pattern, text))

def _contains_chinese(text: str) -> bool:
    """Check if text contains Chinese (CJK) characters."""
    # CJK Unified Ideographs and common Chinese punctuation
    chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f]'
    import re
    return bool(re.search(chinese_pattern, text))

if __name__ == '__main__':
    print(get_hostname())
    print(get_pid())