# import sys
# sys.path.append('/home/rcduan/utils/')
# from drc_utils import dprint
# 粘贴前两句话使用

import logging
import os
import sys

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    stream=sys.stdout,
)
# 第一步，创建一个logger
logger = logging.getLogger('drc_utils')

def dprint(*args, **kwargs):
    log = ''
    for i in args:
        log += f' {i} |'
    
    for i in kwargs.items():
        log += f' {i[0]} = {i[1]} |'
    logger.info(log)

