
import logging
import os
import sys

import torch
from fairseq import utils,search

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    stream=sys.stdout,
)
# 第一步，创建一个logger
logger = logging.getLogger('drc_utils')

def dprint(*args ,end_is_stop=True,split_all=False,**kwargs):
    if split_all:
        for i in args:
            dprint(i)
        for i in kwargs.items():
            exec(f"dprint({i[0]} = {i[1]})")
        return

    log = ''
    for i in args:
        log += f' {i} |'

    for i in kwargs.items():
        log += f' {i[0]} = {i[1]} |'
    logger.info(log)
    if end_is_stop:
        raise Exception(
            "dprint输出完毕，程序中断"
        )

def idx2word(hypo_tokens,tgt_dict):
    """
    Args:
        typo_tokens:input of index2word
        tgt_dict:the dict of input
    Return:
        str: the de-index sentence
    """
    align_dict = utils.load_align_dict(None)
    al = torch.tensor([])
    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        hypo_tokens=hypo_tokens.int().cpu(),
        src_str='src_str',
        alignment=al,
        align_dict=align_dict,
        tgt_dict=tgt_dict
    )
    return hypo_str