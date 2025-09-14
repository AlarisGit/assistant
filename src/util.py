import os
import platform
import socket

import config
import logging

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

if __name__ == '__main__':
    print(get_hostname())
    print(get_pid())