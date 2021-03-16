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
import random


@register_criterion("label_smoothed_cross_entropy_r3f_noised_input")
class NoisedInputCriterion(FairseqCriterion):
    def __init__(
        self, task, sentence_avg, label_smoothing, eps, r3f_lambda, cv_lambda, noise_type,
        noised_no_grad, noised_eval_model, self_training_drc
    ):
        super().__init__(task)
        self.task = task
        self.sentence_avg = sentence_avg
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.r3f_lambda = r3f_lambda
        self.cv_lambda = cv_lambda
        self.noise_type = noise_type
        self.noised_no_grad = noised_no_grad
        self.noised_eval_model = noised_eval_model
        self.self_training_drc = self_training_drc
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
        parser.add_argument('--eps', type=float, default=1e-6,
                            help='noise eps')
        parser.add_argument('--r3f-lambda', type=float, default=1.0,
                            help='lambda for combining logistic loss and noisy KL loss')
        parser.add_argument('--cv-lambda', type=float, default=0.0,
                            help='lambda for combining logistic loss and noisy KL loss')
        parser.add_argument('--noise-type', type=str, default='normal',
                            choices=['normal', 'uniform'],
                            help='type of noises')
        parser.add_argument('--noised-no-grad', action='store_true')
        parser.add_argument('--noised-eval-model', action='store_true')
        parser.add_argument('--self-training-drc', action='store_true')

        # fmt: on

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
        )   / noised_logits.size(0)

    def show_sec(self,t:torch.tensor):
        def f(x):
            return format(float(x),'.3f')
        t = t.detach()
        for i in range(min(t.shape[0],20)):
            a,b,c = t[i,:,:].std(),t[i,:,:].mean(),self._get_symm_kl(t[i,:,:])
            print(f(a), '  ',f(b))

    def show_pic(self,t:torch.tensor):
        t = t.detach()
        # print('------------所有数据-------------')
        # print(t.mean(),t.std())
        # print(t.shape)
        # print('------------固定batchSize-------------')
        x = torch.tensor([t[i,:,:].std() for i in range(min(t.shape[0],20))])
        # print(x)
        # print(x.mean(),x.std())
        a = x.std()
        aa = x.mean()
        # print('------------固定sentenseSize-------------')
        x = torch.tensor([t[:,i,:].std() for i in range(min(t.shape[1],20))])
        # print(x)
        # print(x.mean(),x.std())
        b = x.std()
        bb = x.mean()
        # print('------------固定hiddenSize-------------')
        x = torch.tensor([t[:,:,64*i:64*(i+1)].std() for i in range(8)])
        # print(x)
        # print(x.mean(),x.std())
        c = x.std()
        cc = x.mean()
        return a,b,c,aa,bb,cc

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        noised_inputs = None
        symm_kl = torch.tensor(0)
        sample_size = (
            sample["target"].size(0)
            if self.sentence_avg else sample["ntokens"]
        )
        def f(x):
            return format(float(x),'.3f')

        if model.training:
            input_logits, extra = model(**sample["net_input"])
            # print(extra['inner_states'][-1].shape)
            a,b,c,aa,bb,cc = self.show_pic(extra['inner_states'][-1].transpose(0, 1))
            # print(f(a),' ',f(b),' ',f(c),' ',f(aa),' ',f(bb),' ',f(cc))
            # if random.randint(0,3)==1:
            #     assert False
            with torch.no_grad():
                noised_logits, noised_extra = model(**sample["net_input"])
                noised_inputs = (
                    noised_extra['inner_states'][-1].detach()-extra['inner_states'][-1].detach()).transpose(0, 1)
                # print(noised_logits.shape)
                for i in range(noised_logits.shape[0]):
                    klkl = self._get_symm_kl(noised_logits[i,:,:],input_logits[i,:,:]) 

                    src_token = sample['net_input']['src_tokens'][i]
                    src_str = self.task.source_dictionary.string(src_token,'@@ ')

                    tgt_token = sample['target'][i]
                    tgt_str = self.task.target_dictionary.string(tgt_token,'@@ ')
                    # print(klkl.data)
                    if klkl.data>2.5:
                        print(src_str)
                        print(tgt_str)
                        # assert False
                # self._get_symm_kl
                # self.show_sec(noised_inputs)
                # print(f(a),' ',f(b),' ',f(c),' ',f(aa),' ',f(bb),' ',f(cc))
                # noised_inputs /= 2
            
            # loss_i, nll_loss_i = self.compute_loss(
            #     model, (input_logits, extra), sample, reduce=reduce
            # )

        no_logits, no_extra = model(
            **sample["net_input"], noised_inputs=noised_inputs)
        loss, nll_loss = self.compute_loss(
            model, (no_logits, no_extra), sample, reduce=reduce
        )

        # if model.training:
        #     loss = loss + loss_i
        # if model.training:
        #     # symm_kl = symm_kl * sample_size
        #     loss = loss + self.r3f_lambda * symm_kl
        #     loss = loss*0.7 + no_loss*0.3

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if model.training:
            logging_output.update(
                symm_kl=utils.item(symm_kl.data) if reduce else symm_kl.data
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
        symm_kl_sum = sum(log.get("symm_kl", 0) for log in logging_outputs)

        metrics.log_scalar("symm_kl", symm_kl_sum /
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