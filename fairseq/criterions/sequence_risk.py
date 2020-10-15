# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import math
import torch.nn.functional as F

from fairseq import utils,metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.fairseq_criterion import FairseqSequenceCriterion
# from ..tasks.translation_struct import TranslationStructuredPredictionTask
from fairseq.drc_utils import dprint
from fairseq import drc_utils

@register_criterion('sequence_risk')
class SequenceRiskCriterion(FairseqSequenceCriterion):

    def __init__(self,task):
        super().__init__(task)

        from fairseq.tasks.translation_struct import TranslationStructuredPredictionTask
        if not isinstance(task, TranslationStructuredPredictionTask):
            raise Exception(
                'sequence_risk criterion requires `--task=translation_struct`'
            )

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--normalize-costs', action='store_true',
                            help='normalize costs within each hypothesis')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        """
            sample (dict): a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """

        bsz = len(sample['hypos'])
        nhypos = len(sample['hypos'][0])
 
        sample: dict


        # get costs for hypotheses using --seq-scorer (defaults to 1. - BLEU)
        #计算每个句子中每个预测的cost   batch size * beam size
        costs = self.task.get_costs(sample)
        # costs = costs*0.1

        #读取不到这个参数，所以直接设置为True，我也不知道它读取参数是个什么机制
        self.normalize_costs = False
        if self.normalize_costs:
            unnormalized_costs = costs.clone()
            max_costs = costs.max(dim=1, keepdim=True)[0]
            min_costs = costs.min(dim=1, keepdim=True)[0]
            costs = (costs - min_costs) / \
                (max_costs - min_costs).clamp_(min=1e-6)
        else:
            unnormalized_costs = None

        # generate a new sample from the given hypotheses
        # 把每个源句子翻译的多个句子b，n 差分成一维b*n
        new_sample = self.task.get_new_sample_for_hypotheses(sample)
        hypotheses = new_sample['target'].view(bsz, nhypos, -1, 1)#bsz,hpsz,seq_len,1
        
        hypolen = hypotheses.size(2)
        pad_mask = hypotheses.ne(self.task.target_dictionary.pad()) #bsz,hpsz,seq_len,1
        lengths = pad_mask.sum(dim=2).float() #bsz,hpsz,1

        #maxtokens 被乘以12了？设置为1000，现在有12000个，不算pad
        #dprint(lengths=lengths,end_is_stop=True,shape=lengths.shape,sum=lengths.sum())

        net_output = model(**new_sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(bsz, nhypos, hypolen, -1)

        scores = lprobs.gather(3, hypotheses) #bsz,hpsz,seq_len,1
        scores *= pad_mask.float()

        avg_scores = scores.sum(dim=2) / lengths
        # avg_scores = avg_scores*0.005
        probs = F.softmax(avg_scores, dim=1).squeeze(-1)
        #porbs.shape=batch size,beam size
 
        loss = (probs * costs).sum()

        sample_size = bsz
        assert bsz == utils.item(costs.size(dim=0))
        logging_output = {
            'loss': utils.item(loss.data),
            'num_cost': costs.numel(),
            'ntokens': sample['ntokens'],
            'nsentences': bsz,
            'sample_size': sample_size,
            'htokens':lengths.sum()
        }

        def add_cost_stats(costs, prefix=''):
            logging_output.update({
                prefix + 'sum_cost': utils.item(costs.sum()),
                prefix + 'min_cost': utils.item(costs.min(dim=1)[0].sum()),
                prefix + 'cost_at_1': utils.item(costs[:, 0].sum()),
            })

        add_cost_stats(costs)
        if unnormalized_costs is not None:
            add_cost_stats(unnormalized_costs, 'unnormalized_')


        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        num_costs = sum(log.get('num_cost', 0) for log in logging_outputs)
        agg_outputs = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }
        all_loss = sum(log.get('loss', 0) for log in logging_outputs)
        metrics.log_scalar('loss', all_loss/sample_size , sample_size, round=3)

        def add_cost_stats(prefix=''):
            agg_outputs.update({
                prefix + 'avg_cost': sum(log.get(prefix + 'sum_cost', 0) for log in logging_outputs) / num_costs,
                prefix + 'min_cost': sum(log.get(prefix + 'min_cost', 0) for log in logging_outputs) / nsentences,
                prefix + 'cost_at_1': sum(log.get(prefix + 'cost_at_1', 0) for log in logging_outputs) / nsentences,
            })

        add_cost_stats()
        if any('unnormalized_sum_cost' in log for log in logging_outputs):
            add_cost_stats('unnormalized_')

        return agg_outputs
