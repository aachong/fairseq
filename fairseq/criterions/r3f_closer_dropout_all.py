# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 使用kl散度训练，后边加入了js散度

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


@register_criterion('r3f_closer_dropout_all')
class R3fCloserDropoutAll(FairseqCriterion):
    def __init__(
        self, task, sentence_avg, label_smoothing, eps, r3f_lambda, noise_type, layer_choice,noised_with_grad
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.r3f_lambda = r3f_lambda
        self.noise_type = noise_type
        self.layer_choice = layer_choice
        self.noised_with_grad = noised_with_grad
        if self.noise_type in {"normal"}:
            self.noise_sampler = torch.distributions.normal.Normal(
                loc=0.0, scale=self.eps
            )
        elif self.noise_type == "uniform":
            self.noise_sampler = torch.distributions.uniform.Uniform(
                low=-self.eps, high=self.eps
            )
        else:
            raise Exception(f"unrecognized noise type {self.noise_type}")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--eps', type=float, default=1e-5,
                            help='noise eps')
        parser.add_argument('--r3f-lambda', type=float, default=1.0,
                            help='lambda for combining logistic loss and noisy KL loss')
        parser.add_argument('--noise-type', type=str, default='normal',
                            choices=['normal', 'uniform'],
                            help='type of noises')
        # fmt: on
        parser.add_argument('--layer-choice', type=str, default='normal',
                            choices=['normal', 'allone'])
        parser.add_argument('--noised-with-grad', action='store_true')        

    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
        ) 
        #/ noised_logits.size(0)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        token_embeddings = model.encoder.embed_tokens(
            sample["net_input"]["src_tokens"])
        input_logits, extra, input_feature = model(**sample["net_input"])

        loss, nll_loss = self.compute_loss(
            model, (input_logits, extra), sample, reduce=reduce
        )

        sample_size = (
            sample["target"].size(
                0) if self.sentence_avg else sample["ntokens"]
        )

        if model.training:
            if self.noised_with_grad:
                noised_logits, noised_extra, noised_feature = model(
                    **sample["net_input"]
                )
                print(111)
            else:
                with torch.no_grad():
                    noised_logits, noised_extra, noised_feature = model(
                        **sample["net_input"]
                    )
            symm_mse = 0
            if self.layer_choice == 'allone':
                for i in range(6):
                    symm_mse += F.mse_loss(noised_extra['inner_states'][i+1],
                                           extra['inner_states'][i+1])
                symm_mse += self._get_symm_kl(noised_logits, input_logits)
            elif self.layer_choice == 'normal':
                for i in range(6):
                    lambda_x = (i/10)**2
                    symm_mse += lambda_x * F.mse_loss(noised_extra['inner_states']
                                                      [i+1], extra['inner_states'][i+1])
                symm_mse += 0.57*self._get_symm_kl(noised_logits, input_logits)
        if model.training:
            #symm_mse = symm_mse * sample_size
            loss = loss + self.r3f_lambda * symm_mse

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if model.training:
            logging_output.update(
                symm_mse=utils.item(symm_mse.data) if reduce else symm_mse.data
            )

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.label_smoothing,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        symm_mse_sum = sum(log.get("symm_mse", 0) for log in logging_outputs)

        metrics.log_scalar("symm_mse", symm_mse_sum /
                           sample_size, sample_size, round=3)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
