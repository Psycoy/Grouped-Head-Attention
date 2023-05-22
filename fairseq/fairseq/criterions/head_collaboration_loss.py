# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from torch import linalg as LA
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig_headregularization(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    use_attentionmatrix_regularization: bool = field(
        default=False,
        metadata={"help": "Combine a regularization term in loss (difference between attention matrices)."},
    )
    use_subspace_regularization: bool = field(
        default=False,
        metadata={"help": "Combine a regularization term in loss (difference between v matrices)."},
    )
    use_headoutput_regularization: bool = field(
        default=False,
        metadata={"help": "Combine a regularization term in loss (difference between headoutput matrices)."},
    )
    
    use_attentionmatrix_regularization_HWA: bool = field(
        default=False,
        metadata={"help": "Combine a regularization term in loss (difference between attention matrices)."},
    )
    use_subspace_regularization_HWA: bool = field(
        default=False,
        metadata={"help": "Combine a regularization term in loss (difference between v matrices)."},
    )
    use_headoutput_regularization_HWA: bool = field(
        default=False,
        metadata={"help": "Combine a regularization term in loss (difference between headoutput matrices)."},
    )
    
    coefficient_attentionmatrix: float = field(
        default=1.0,
        metadata={"help": "Coefficient for attention difference regularization."},
    )
    coefficient_subspace: float = field(
        default=1.0,
        metadata={"help": "Coefficient for subspace difference regularization."},
    )
    coefficient_headoutput: float = field(
        default=1.0,
        metadata={"help": "Coefficient for headoutput difference regularization."},
    )
    coefficient_attentionmatrix_HWA: float = field(
        default=1.0,
        metadata={"help": "Coefficient for attention difference regularization."},
    )
    coefficient_subspace_HWA: float = field(
        default=1.0,
        metadata={"help": "Coefficient for subspace difference regularization."},
    )
    coefficient_headoutput_HWA: float = field(
        default=1.0,
        metadata={"help": "Coefficient for headoutput difference regularization."},
    )
    en_de_attn_regularization: bool = field(
        default=False,
        metadata={"help": "Whether to add the encoder-decoder attention head difference as a regularization term."},
    )
    
    headout_regularization_afterHWA: bool = field(
        default=False,
        metadata={"help": "Whether to add the headout difference after HWA or not (see HeadCollaboration/Attention_variant_scripts/multihead_attention_efficient.py, line 400+)."},
    )
    
    HWA_let_collaborationbeingsimilar: bool = field(
        default=False,
        metadata={"help": "Whether to increase the similarity between heads of the original MHA."},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_headregularization", dataclass=LabelSmoothedCrossEntropyCriterionConfig_headregularization
)
class LabelSmoothedCrossEntropyCriterion_headregularization(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        cfg=LabelSmoothedCrossEntropyCriterionConfig_headregularization
    ):
        super().__init__(task)
        self.cfg =cfg
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        
        self.use_attentionmatrix_regularization = cfg.use_attentionmatrix_regularization
        self.use_subspace_regularization = cfg.use_subspace_regularization
        self.use_headoutput_regularization = cfg.use_headoutput_regularization
        
        self.coefficient_attentionmatrix = cfg.coefficient_attentionmatrix
        self.coefficient_subspace = cfg.coefficient_subspace
        self.coefficient_headoutput = cfg.coefficient_headoutput

        self.use_attentionmatrix_regularization_HWA = cfg.use_attentionmatrix_regularization_HWA
        self.use_subspace_regularization_HWA = cfg.use_subspace_regularization_HWA
        self.use_headoutput_regularization_HWA = cfg.use_headoutput_regularization_HWA
        
        self.coefficient_attentionmatrix_HWA = cfg.coefficient_attentionmatrix_HWA
        self.coefficient_subspace_HWA = cfg.coefficient_subspace_HWA
        self.coefficient_headoutput_HWA = cfg.coefficient_headoutput_HWA
        
        self.headout_regularization_afterHWA = cfg.headout_regularization_afterHWA
        
    def add_attention_regularization_term(self, matrices_of_layers, Regu, coefficient):
        for attention_matrices in matrices_of_layers:
            bsz, num_heads, tgt_len, src_len = attention_matrices.size()
            attention_matrices = attention_matrices.view(num_heads, bsz, tgt_len, src_len).float()
            for i in range(num_heads):
                for j in range(num_heads):
                    if i < j:
                        # Matrixproduct=(attention_matrices[i] * attention_matrices[j]).sum()
                        # Matrixproduct_normed=(attention_matrices[i] * attention_matrices[j]).sum() / (num_heads * num_heads)
                        Regu = Regu + ((attention_matrices[i] * attention_matrices[j]).sum() / (num_heads * num_heads)) * coefficient
                        # sum over the batch if the original loss function is reduced by sum.
                        # Note that the term in paper is a term in the training objective which is to be maximized, 
                        # here we combine it in the loss function and donot need the negative term.
        return Regu
        
    def add_v_regularization_term(self, matrices_of_layers, Regu, coefficient):
        for v_matrices in matrices_of_layers:
            bsz, num_heads, tgt_len, head_dim = v_matrices.size()
            v_matrices = v_matrices.view(num_heads, bsz * tgt_len * head_dim).float()
            for i in range(num_heads):
                for j in range(num_heads):
                    if i < j:
                        Regu = Regu + ((v_matrices[i] @ v_matrices[j] / (LA.vector_norm(v_matrices[i]) * LA.vector_norm(v_matrices[j]))) 
                                 / (num_heads * num_heads)) * coefficient
                        # sum over the batch if the original loss function is reduced by sum.
        return Regu
        
    def add_headout_regularization_term(self, matrices_of_layers, Regu, coefficient):
        for headout_matrices in matrices_of_layers:
            bsz, num_heads, tgt_len, head_dim = headout_matrices.size()
            headout_matrices = headout_matrices.view(num_heads, bsz * tgt_len * head_dim).float()
            for i in range(num_heads):
                for j in range(num_heads):
                    if i < j:
                        Regu = Regu + ((headout_matrices[i] @ headout_matrices[j] / (LA.vector_norm(headout_matrices[i]) * LA.vector_norm(headout_matrices[j]))) 
                                 / (num_heads * num_heads)) * coefficient
                        # sum over the batch if the original loss function is reduced by sum.
        return Regu

    def add_attention_regularization_term_HWA(self, matrices_of_layers, Regu, coefficient):
        for attention_matrices in matrices_of_layers:
            bsz_h_HWA, tgt_len, src_len = attention_matrices.size()  # Here the tgt_len and src_len is the number of head of MHA
            attention_matrices = attention_matrices.contiguous().view(tgt_len, bsz_h_HWA, src_len).contiguous().view(tgt_len, bsz_h_HWA * src_len).float()
            for i in range(tgt_len):
                for j in range(tgt_len):
                    if i < j:
                        # Matrixproduct=(attention_matrices[i] * attention_matrices[j]).sum()
                        # Matrixproduct_normed=(attention_matrices[i] * attention_matrices[j]).sum() / (num_heads * num_heads)
                        if self.cfg.HWA_let_collaborationbeingsimilar:
                            Regu = Regu - (
                                (attention_matrices[i] @ attention_matrices[j]).sum() 
                                / (LA.vector_norm(attention_matrices[i]) * LA.vector_norm(attention_matrices[j])) 
                                / (tgt_len * tgt_len)
                                ) * coefficient
                        else:
                            Regu = Regu + (
                                (attention_matrices[i] @ attention_matrices[j]).sum() 
                                / (LA.vector_norm(attention_matrices[i]) * LA.vector_norm(attention_matrices[j])) 
                                / (tgt_len * tgt_len)
                                ) * coefficient  #TODO 需要normalize一下
                        # sum over the batch if the original loss function is reduced by sum.
                        # Note that the term in paper is a term in the training objective which is to be maximized, 
                        # here we combine it in the loss function and donot need the negative term.
        return Regu
        
    def add_v_regularization_term_HWA(self, matrices_of_layers, Regu, coefficient):
        for v_matrices in matrices_of_layers:
            num_heads, bsz_h_HWA, tgt_len_head_dim_div = v_matrices.size()  # Here the num_heads is the same as the tgt_len/src_len in function "add_attention_regularization_term_HWA" defined above.
            v_matrices = v_matrices.view(num_heads, bsz_h_HWA * tgt_len_head_dim_div).float()
            for i in range(num_heads):
                for j in range(num_heads):
                    if i < j:
                        if self.cfg.HWA_let_collaborationbeingsimilar:
                            Regu = Regu - (
                                (v_matrices[i] @ v_matrices[j]) 
                                / (LA.vector_norm(v_matrices[i]) * LA.vector_norm(v_matrices[j])) 
                                / (num_heads * num_heads)
                                ) * coefficient
                        else:
                            Regu = Regu + (
                                (v_matrices[i] @ v_matrices[j]) 
                                / (LA.vector_norm(v_matrices[i]) * LA.vector_norm(v_matrices[j])) 
                                / (num_heads * num_heads)
                                ) * coefficient
                        # sum over the batch if the original loss function is reduced by sum.
        return Regu
        
    def add_headout_regularization_term_HWA(self, matrices_of_layers, Regu, coefficient):
        for headout_matrices in matrices_of_layers:
            num_heads, bsz_h_HWA, tgt_len_head_dim_div = headout_matrices.size()  # Here the num_heads is the same as the tgt_len/src_len in function "add_attention_regularization_term_HWA" defined above.
            headout_matrices = headout_matrices.view(num_heads, bsz_h_HWA * tgt_len_head_dim_div).float()
            for i in range(num_heads):
                for j in range(num_heads):
                    if i < j:
                        if self.cfg.HWA_let_collaborationbeingsimilar:
                            Regu = Regu - (
                                (headout_matrices[i] @ headout_matrices[j]) 
                                / (LA.vector_norm(headout_matrices[i]) * LA.vector_norm(headout_matrices[j])) 
                                / (num_heads * num_heads)
                                ) * coefficient
                        else:
                            Regu = Regu + (
                                (headout_matrices[i] @ headout_matrices[j]) 
                                / (LA.vector_norm(headout_matrices[i]) * LA.vector_norm(headout_matrices[j])) 
                                / (num_heads * num_heads)
                                ) * coefficient
                        # sum over the batch if the original loss function is reduced by sum.
        return Regu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        if self.use_attentionmatrix_regularization:
            attention_matrices = net_output[1]["decoder_head_matrices"]["attention_matrices"]
            loss = self.add_attention_regularization_term(attention_matrices, loss, self.coefficient_attentionmatrix)
        if self.use_subspace_regularization:
            v_matrices = net_output[1]["decoder_head_matrices"]["v_matrices"]
            loss = self.add_v_regularization_term(v_matrices, loss, self.coefficient_subspace)
        if self.use_headoutput_regularization:
            if not self.headout_regularization_afterHWA:
                headout_matrices_beforeHWA = net_output[1]["decoder_head_matrices"]["headout_matrices_beforeHWA"]
                loss = self.add_headout_regularization_term(headout_matrices_beforeHWA, loss, self.coefficient_headoutput)
            else:
                headout_matrices_afterHWA = net_output[1]["decoder_head_matrices"]["headout_matrices_afterHWA"]
                loss = self.add_headout_regularization_term(headout_matrices_afterHWA, loss, self.coefficient_headoutput)
        if self.use_attentionmatrix_regularization_HWA:
            attention_matrices_HWA = net_output[1]["decoder_head_matrices"]["attention_matrices_HWA"]
            loss = self.add_attention_regularization_term_HWA(attention_matrices_HWA, loss, self.coefficient_attentionmatrix_HWA)
        if self.use_subspace_regularization_HWA:
            v_matrices_HWA = net_output[1]["decoder_head_matrices"]["v_matrices_HWA"]
            loss = self.add_v_regularization_term_HWA(v_matrices_HWA, loss, self.coefficient_subspace_HWA)
        if self.use_headoutput_regularization_HWA:
            headout_matrices_HWA = net_output[1]["decoder_head_matrices"]["headout_matrices_HWA"]
            loss = self.add_headout_regularization_term_HWA(headout_matrices_HWA, loss, self.coefficient_headoutput_HWA)
            
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
