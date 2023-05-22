# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import DDP_BACKEND_CHOICES
from omegaconf import II
from fairseq.dataclass import ChoiceEnum

import torch
from torch import linalg as LA


@dataclass
class AdaptiveLossConfig_headclustering_prune(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    ddp_backend: DDP_BACKEND_CHOICES = II("distributed_training.ddp_backend")
    
    ########################################################################################
    
    Supervise_mode: ChoiceEnum(['reconstruction', 'classification']) = field(
        default="reconstruction",
        metadata={"help": "Which clustering supervision mode to use."},
    )
    
    cluster_matrix: ChoiceEnum(['Q', 'K', 'V', 'attn_matrix', 'headout']) = field(
        default="headout",
        metadata={"help": "Sample to cluster on."},
    )
    
    supervised_matrix: ChoiceEnum(['Q', 'K', 'V', 'attn_matrix', 'headout']) = field(
        default="headout",
        metadata={"help": "Sample to compute the distance with cluster center."},
    )
    
    cluster_loss_coefficient_inclass: float = field(
        default=1.0,
        metadata={"help": "Coefficient for the clustering loss."},
    )
    cluster_loss_coefficient_interclass: float = field(
        default=1.0,
        metadata={"help": "Coefficient for the clustering loss."},
    )
    
    N_head_clusters: int = field(
        default=2,
        metadata={"help": "How many clusters to use."},
    )

    use_interclass_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use loss between classes."},
    )
    
    use_inclass_loss: bool = field(
        default=False,
        metadata={"help": "whether to use loss within classes."},
    )


@register_criterion("adaptive_loss_headclustering_prune", dataclass=AdaptiveLossConfig_headclustering_prune)
class AdaptiveLoss_headclustering_prune(FairseqCriterion):
    """This is an implementation of the loss function accompanying the adaptive softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax approximation for GPUs"
    (http://arxiv.org/abs/1609.04309)."""

    def __init__(self, task, sentence_avg, cfg=AdaptiveLossConfig_headclustering_prune):
        super().__init__(task)
        self.cfg=cfg
        self.sentence_avg = sentence_avg
        
        self.cluster_matrix = cfg.cluster_matrix
        self.supervised_matrix = cfg.supervised_matrix
        self.cluster_loss_coefficient_inclass = cfg.cluster_loss_coefficient_inclass
        self.cluster_loss_coefficient_interclass = cfg.cluster_loss_coefficient_interclass
        self.N_head_clusters = cfg.N_head_clusters
        # raise RuntimeError

    def add_cluster_loss(self, net_output, Regu): # TODO uncompleted
        # print(f"self.cluster_matrix: {self.cluster_matrix}")
        Regu_init = Regu
        if self.cfg.Supervise_mode == "reconstruction":
            if self.cluster_matrix == "Q":
                cluster_centers_layers = net_output[1]["decoder_head_matrices"]["cluster_centers_Q"]
                head_labels_layers = net_output[1]["decoder_head_matrices"]["head_labels_Q"]
            elif self.cluster_matrix == "K":
                cluster_centers_layers = net_output[1]["decoder_head_matrices"]["cluster_centers_K"]
                head_labels_layers = net_output[1]["decoder_head_matrices"]["head_labels_K"]
            elif self.cluster_matrix == "V":
                cluster_centers_layers = net_output[1]["decoder_head_matrices"]["cluster_centers_V"]
                head_labels_layers = net_output[1]["decoder_head_matrices"]["head_labels_V"]
            elif self.cluster_matrix == "attn_matrix":
                cluster_centers_layers = net_output[1]["decoder_head_matrices"]["cluster_centers_attention_matrices"]
                head_labels_layers = net_output[1]["decoder_head_matrices"]["head_labels_attention_matrices"]
            elif self.cluster_matrix == "headout":
                cluster_centers_layers = net_output[1]["decoder_head_matrices"]["cluster_centers_headout"]
                head_labels_layers = net_output[1]["decoder_head_matrices"]["head_labels_headout"]
                
            if self.supervised_matrix == "Q":
                supervised_matrices_layers = net_output[1]["decoder_head_matrices"]["q_matrices_tobesupervised"]
            elif self.supervised_matrix == "K":
                supervised_matrices_layers = net_output[1]["decoder_head_matrices"]["k_matrices_tobesupervised"]
            elif self.supervised_matrix == "V":
                supervised_matrices_layers = net_output[1]["decoder_head_matrices"]["v_matrices_tobesupervised"]
            elif self.supervised_matrix == "attn_matrix":
                supervised_matrices_layers = net_output[1]["decoder_head_matrices"]["attention_matrices_tobesupervised"]
            elif self.supervised_matrix == "headout":
                supervised_matrices_layers = net_output[1]["decoder_head_matrices"]["headout_matrices_tobesupervised"]
            
            for cluster_centers, head_labels, supervised_matrices in zip(cluster_centers_layers, head_labels_layers, supervised_matrices_layers):
                assert cluster_centers[0].shape == supervised_matrices[0].shape
                assert head_labels.shape[0] == supervised_matrices.shape[0]

                num_heads = head_labels.shape[0]
                if self.cfg.use_inclass_loss:
                    for i in range(num_heads):
                        head_label = int(head_labels[i])
                        # supervised_matrices[i] - cluster_centers[head_label]
                        
                        Regu = Regu-(                                                      # TODO here we can use MSE or dotproduct to evaluate the distance, depending on the performance
                                    (
                                    supervised_matrices[i] @ cluster_centers[head_label] 
                                    /
                                    (LA.vector_norm(supervised_matrices[i]) * LA.vector_norm(cluster_centers[head_label]))
                                    ) 
                                    / num_heads
                                    ) * self.cluster_loss_coefficient_inclass                                       # TODO the normalization method is to be reconsidered
                if self.cfg.use_interclass_loss:
                    for i in range(cluster_centers.shape[0]):
                        for j in range(cluster_centers.shape[0]):
                            if i < j:
                                Regu = Regu+(                                                      # TODO here we can use MSE or dotproduct to evaluate the distance, depending on the performance
                                            (
                                            cluster_centers[i] @ cluster_centers[j] 
                                            /
                                            (LA.vector_norm(cluster_centers[i]) * LA.vector_norm(cluster_centers[j]))
                                            ) 
                                            / math.comb(cluster_centers.shape[0], 2)    # TODO check normalization
                                            ) * self.cluster_loss_coefficient_interclass
                                                                                    
                    
        return Regu, Regu-Regu_init
    
    @classmethod
    def build_criterion(cls, cfg: AdaptiveLossConfig_headclustering_prune, task):
        if cfg.ddp_backend in {"c10d", "pytorch_ddp"}:
            raise Exception(
                "AdaptiveLoss is not compatible with the PyTorch "
                "version of DistributedDataParallel. Please use "
                "`--ddp-backend=legacy_ddp` instead."
            )
        return cls(task, cfg.sentence_avg, cfg)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        assert (
            hasattr(model.decoder, "adaptive_softmax")
            and model.decoder.adaptive_softmax is not None
        )
        adaptive_softmax = model.decoder.adaptive_softmax

        net_output = model(**sample["net_input"])
        orig_target = model.get_targets(sample, net_output)

        nsentences = orig_target.size(0)
        orig_target = orig_target.view(-1)

        bsz = orig_target.size(0)

        logits, target = adaptive_softmax(net_output[0], orig_target)
        assert len(target) == len(logits)

        loss = net_output[0].new(1 if reduce else bsz).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= logits[i].size(1)
                loss += F.cross_entropy(
                    logits[i],
                    target[i],
                    ignore_index=self.padding_idx,
                    reduction="sum" if reduce else "none",
                )

        orig = utils.strip_pad(orig_target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        
        # assert type(model.encoder.layers[0].self_attn.prune_indicator) == type(model.decoder.layers[0].self_attn.prune_indicator)
        
        if (torch.equal(model.decoder.layers[0].self_attn.prune_indicator, 
                    torch.Tensor([-1]).repeat(model.decoder.layers[0].self_attn.num_heads).to(torch.device(model.decoder.layers[0].self_attn.prune_indicator.device))) 
            or model.cfg.keep_updating_cluster):
            loss, Cluster_loss = self.add_cluster_loss(net_output, loss)
        else:
            Cluster_loss = torch.tensor(0)
        
        logging_output = {
            "loss": loss.data,
            "cluster_loss": Cluster_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        cluster_loss_sum = sum(log.get("cluster_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "cluster_loss", cluster_loss_sum / sample_size / math.log(2), sample_size, round=9
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
