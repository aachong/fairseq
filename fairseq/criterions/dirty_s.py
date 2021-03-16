# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import random


def label_smoothed_nll_loss(lprobs, target, epsilon,word_ws, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if word_ws is not None:
        word_ws = word_ws.view(-1,1)
        nll_loss = nll_loss * word_ws
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('cross_entropy_dirty_s')
class CrossEntropyDirtyS(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, threshold, sensword):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.threshold = threshold
        self.sensword = sensword

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--threshold', default=1.2, type=float, metavar='D',
                            help='the threshold of dirty sentences')
        parser.add_argument('--sensword', default='sentence', type=str, metavar='D',
                            help='the threshold of dirty sentences')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        input_logits, extra = model(**sample['net_input'])
        word_ws = None
        if model.training:
            with torch.no_grad():
                # print(xx)
                # print(sample['net_input'].keys())
                # print(sample['net_input']['prev_output_tokens'][0])
                # print(self.task.target_dictionary.string(sample['net_input']['prev_output_tokens'][0]))
                # print(sample['net_input']['prev_output_tokens'].shape)
                # print(sample['target'][0])
                # print(sample['target'].shape)
                # assert False
                b, s = input_logits.shape[:2]

                noised_logits, noised_extra = model(**sample["net_input"])
                input_logits1, extra1 = model(**sample['net_input'])
                mask_dirty = torch.ones(b, s).to(input_logits)

                word_w = []
                for i in range(noised_logits.shape[0]):
                    klkl = self._get_symm_kl_except_first_from_three(
                        noised_logits[i, :, :], input_logits[i, :, :].detach(), input_logits1[i, :, :])

                    word_num = noised_logits.size(1) - 1
                    sen_kl = klkl[1:, :].sum().data / word_num
                    word_kl = klkl.sum(-1)
                    # print(sen_kl)
                    # print(word_kl)
                    # print(word_kl)
                    if self.sensword == 'sentence':
                        if sen_kl.data > self.threshold:
                            mask_dirty[i, :] = 0

                    elif self.sensword == 'word':
                        mask_dirty[i] = (word_kl < 3).float()

                    word_kl[word_kl > sen_kl] = 1 + word_kl[word_kl > sen_kl]
                    word_kl[word_kl<=sen_kl] = 1
                    word_w.append(word_kl[None,:])

                    # src_str, tgt_str = self.get_src_tgt_str(
                    #     sample['net_input']['src_tokens'][i], sample['target'][i])

                    # print(src_str)
                    # print(tgt_str)
                    # assert False

                    # if mask_dirty[i].min() == 0:

                    #     self.print_masked_word(tgt_str, mask_dirty[i])
                word_ws = torch.cat(word_w,0).to(input_logits)
                # print(word_ws.shape)

            input_logits = mask_dirty[:, :, None] * input_logits

        # assert False
        # print(input_logits.shape)
        loss, nll_loss = self.compute_loss(
            model, (input_logits, extra), sample, word_ws,reduce=reduce)
        sample_size = sample['target'].size(
            0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def print_masked_word(self, tgt_str, mask_dirty_i):
        tgt_l = tgt_str.split(' ')
        ss = ''
        for i, j in zip(tgt_l, mask_dirty_i[:-1]):
            if j == 0:
                ss += ' --' + i + '--'
            else:
                ss += ' '+i
        print(ss)

    def get_src_tgt_str(self, src_token, tgt_token):
        return self.task.source_dictionary.string(src_token), self.task.target_dictionary.string(tgt_token)

    def _get_symm_kl_except_first_from_three(self, noised_logits: torch.Tensor, input_logits: torch.Tensor, input_logits1: torch.Tensor):
        word_num = noised_logits.size(0) - 1
        klkl0 = self._get_symm_kl_list(noised_logits, input_logits)
        klkl00 = klkl0[1:, :].sum().data / word_num
        klkl1 = self._get_symm_kl_list(noised_logits, input_logits1)
        klkl2 = self._get_symm_kl_list(input_logits, input_logits1)
        kk = (klkl0+klkl1+klkl2)/3
        return kk

    def _get_symm_kl_list(self, noised_logits: torch.Tensor, input_logits: torch.Tensor):
        # noised_logits, logits_index = noised_logits.topk(25,-1)
        # input_logits = input_logits.gather(1,logits_index)

        # input_logits = input_logits.topk(1000,-1)[0]

        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                None,
                False,
                None
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                None,
                False,
                None
            )
        )

    def _get_symm_kl(self, noised_logits, input_logits):
        # print(noised_logits.shape,input_logits.shape)
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum"
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum"
            )
        ) / noised_logits.size(0)

    def compute_loss(self, model, net_output, sample,word_ws=None, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, word_ws,ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size /
                           math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum /
                           ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived(
            'ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
