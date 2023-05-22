# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter
from fairseq.modules import LayerNorm
from kmeans_pytorch import kmeans
from fairseq.dataclass.configs import FairseqConfig



    

@with_incremental_state
class MultiheadAttention_efficient(nn.Module):
    """
    To be supervised matrices:
        q_matrices_tobesupervised
        k_matrices_tobesupervised
        v_matrices_tobesupervised
        attention_matrices_tobesupervised
        headout_matrices_tobesupervised
    
    clustered matrices:
        cluster_centers_Q; head_labels_Q
        cluster_centers_K; head_labels_K
        cluster_centers_V; head_labels_V
        cluster_centers_attention_matrices; head_labels_attention_matrices
        cluster_centers_headout; head_labels_headout
        
    """

    def __init__(
        self,
        cfg,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if not self.cfg.abandon_outputprojection:
            self.out_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )
        else:
            self.out_proj = None
        
        
        
        # self.out_proj_headwise = None 

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        
        # self.head_matrices = {}
        self.need_prune = False
        self.prune_indicator = Parameter(torch.Tensor([-1]).repeat(num_heads)).float()         #  NOTE: just a workaround for saving and loading this tensor
        self.voting_list = None

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
        if self.out_proj is not None:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
    
    
    '''每一个 matrix处调用computeandsort_distances_fromcenter， 
    得到的prune_indicater append到总的prune_indicaters list，然后用finalize_prune_indicators得到最后的indicator，转为int。
    选中所需要的head让其等于indicators中对应的数值。
最后用attention script输入Fairseq cfg测试一下
'''
    def computeandsort_distances_fromcenter(self, targets: Tensor, labels: Tensor, centers: Tensor, device, reservenum_each_center: int=1):
        # return the distance of each sample from its center, and the boolen array indicating which position shoud be pruned
        # targets: [h, dim], labels: [h, ], centers: [n, dim]
        assert targets.shape[0] == labels.shape[0]
        assert targets.shape[1] == centers.shape[1]
        
        distances = torch.zeros(labels.shape).to(device)
        for i in range(targets.shape[0]):
            if self.cfg.kmeans_distance_metric == 'euclidean':
                distances[i] = torch.sqrt(((targets[i]-centers[int(labels[i])]) ** 2).sum())
            elif self.cfg.kmeans_distance_metric == 'cosine':
                distances[i] = nn.CosineSimilarity(dim=0)(targets[i].float(), centers[int(labels[i])].float())
        
        headtokeep_index = []
        for j in range(centers.shape[0]):
            assert j in labels, f"j: {j}, labels: {labels}, centers: {centers}"
            cluster_k_indexes = (labels == j).nonzero(as_tuple=False).squeeze(1)
            # print("cluster_k_indexes: ", cluster_k_indexes, "labels: ", labels)
            distances_k = torch.take(distances, cluster_k_indexes)
            scoredir = dict(zip(cluster_k_indexes, distances_k))
            scoredir = dict(sorted(scoredir.items(), reverse=False, key=lambda item: item[1]))
            
            headtokeep_index += list(scoredir.keys())[:reservenum_each_center]
            
        headtokeep_index = [int(aa) for aa in headtokeep_index]
        
        prune_indicater = torch.zeros(labels.shape).to(device)
        
        prune_indicater[headtokeep_index[:]] = 1
        # print("prune_indicater: ", prune_indicater, "headtokeep_index: ", headtokeep_index)
        prune_indicater = prune_indicater.bool()
        
        return distances, prune_indicater
    
    def finalize_prune_indicators(self, prune_indicaters: list, device):
        last_indicator = torch.zeros_like(prune_indicaters[0]).bool().to(device)
        for indicator in prune_indicaters:
            last_indicator = last_indicator | indicator

        return last_indicator
    
    def voting_on_pruneindicators(self, voting_list: list):
        assert voting_list is not None
        voted_indicator = torch.zeros_like(voting_list[0])
        # TODO keep self.cfg.N_head_clusters heads
        votingcounter = torch.zeros_like(voting_list[0])
        for indicator in voting_list:
            votingcounter += indicator
        
        print(f"voting_list: {voting_list}; voting_list_len: {len(voting_list)}")
        print(f"votingcounter: {votingcounter}")
        
        votingdir = dict(zip(list(range(votingcounter.shape[0])), votingcounter.tolist()))
        votingdir = dict(sorted(votingdir.items(), reverse=True, key=lambda item: item[1]))
        
        print(f"votingdir: {votingdir}")
        
        headstokeep = list(votingdir.keys())[:self.cfg.N_head_clusters]
        
        print(f"headstokeep: {headstokeep}")
        
        for id in headstokeep:
            voted_indicator[id] = 1
        print(f"voted_indicator: {voted_indicator}")
        
        return voted_indicator
        
    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.cfg.voting_to_prune and self.cfg.collecting_indicators:
            with torch.no_grad():
                return self.forward1(
                                    query,
                                    key,
                                    value,
                                    key_padding_mask,
                                    incremental_state,
                                    need_weights,
                                    static_kv,
                                    attn_mask,
                                    before_softmax,
                                    need_head_weights
                                )
        else:
            return self.forward1(
                                    query,
                                    key,
                                    value,
                                    key_padding_mask,
                                    incremental_state,
                                    need_weights,
                                    static_kv,
                                    attn_mask,
                                    before_softmax,
                                    need_head_weights
                                )
        
        
    def forward1(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        head_matrices = {}
        
        if self.cfg.need_prune:
            self.need_prune = True
        
        if need_head_weights:
            need_weights = True

        if (self.need_prune or self.cfg.collecting_indicators) and self.cfg.experiment_stage == 'train':
            prune_indicaters = []
            
        if self.prune_indicator.device != query.device:
            print("query.device: ", query.device)
            print("self.prune_indicator from ", self.prune_indicator.device, "to ")
            self.prune_indicator = self.prune_indicator.to(torch.device(query.device))
            print(self.prune_indicator.device)
        
        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        # if (
        #     not self.onnx_trace
        #     and not is_tpu  # don't use PyTorch version on TPUs
        #     and incremental_state is None
        #     and not static_kv
        #     # A workaround for quantization to work. Otherwise JIT compilation
        #     # treats bias in linear module as method.
        #     and not torch.jit.is_scripting()
        # ):
        #     assert key is not None and value is not None
        #     return F.multi_head_attention_forward(
        #         query,
        #         key,
        #         value,
        #         self.embed_dim,
        #         self.num_heads,
        #         torch.empty([0]),
        #         torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
        #         self.bias_k,
        #         self.bias_v,
        #         self.add_zero_attn,
        #         self.dropout_module.p,
        #         self.out_proj.weight,
        #         self.out_proj.bias,
        #         self.training or self.dropout_module.apply_during_inference,
        #         key_padding_mask,
        #         need_weights,
        #         attn_mask,
        #         use_separate_proj_weight=True,
        #         q_proj_weight=self.q_proj.weight,
        #         k_proj_weight=self.k_proj.weight,
        #         v_proj_weight=self.v_proj.weight,
        #     )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        # head_matrices["attention_shapes"] = [(self.num_heads, self.head_dim, bsz, tgt_len, src_len)]
        head_matrices["q_matrices_tobesupervised"] = [(q.contiguous()
                .view(bsz, self.num_heads, -1, self.head_dim).contiguous().transpose(0,1).contiguous().view(self.num_heads, -1)
                )]
        if self.cfg.cluster_matrix == 'Q':
            if torch.equal(self.prune_indicator, torch.Tensor([-1]).repeat(self.num_heads).to(torch.device(q.device))) or self.cfg.keep_updating_cluster:
                q_heads = (q.contiguous()
                    .view(bsz, self.num_heads, -1, self.head_dim).contiguous().transpose(0,1).contiguous().view(self.num_heads, -1)
                    )
                # kmeans_Q = KMeans(n_clusters=self.cfg.N_head_clusters, X=q_heads, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
                labels_Q, cluster_centers_Q = kmeans(q_heads, num_clusters=self.cfg.N_head_clusters, distance=self.cfg.kmeans_distance_metric, seed=0, device=torch.device(q_heads.device))
                
                head_matrices["cluster_centers_Q"] = [cluster_centers_Q]
                head_matrices["head_labels_Q"] = [labels_Q]
                
                if (self.need_prune or self.cfg.collecting_indicators) and (self.cfg.experiment_stage == 'train'):
                    _, prune_indicator = self.computeandsort_distances_fromcenter(q_heads, labels_Q, cluster_centers_Q, device=torch.device(q_heads.device), reservenum_each_center=1)
                    prune_indicaters.append(prune_indicator)

        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            head_matrices["k_matrices_tobesupervised"] = [(k.contiguous()
                    .view(bsz, self.num_heads, -1, self.head_dim).contiguous().transpose(0,1).contiguous().view(self.num_heads, -1)
                    )]
            if self.cfg.cluster_matrix == 'K':
                if torch.equal(self.prune_indicator, torch.Tensor([-1]).repeat(self.num_heads).to(torch.device(k.device))) or self.cfg.keep_updating_cluster:
                    k_heads = (k.contiguous()
                        .view(bsz, self.num_heads, -1, self.head_dim).contiguous().transpose(0,1).contiguous().view(self.num_heads, -1)
                        )
                    # kmeans_K = KMeans(n_clusters=self.cfg.N_head_clusters, random_state=0).fit(k_heads)
                    labels_K, cluster_centers_K = kmeans(k_heads, num_clusters=self.cfg.N_head_clusters, distance=self.cfg.kmeans_distance_metric, seed=0, device=torch.device(k_heads.device))
                    head_matrices["cluster_centers_K"] = [cluster_centers_K]
                    head_matrices["head_labels_K"] = [labels_K]
                    if (self.need_prune or self.cfg.collecting_indicators) and (self.cfg.experiment_stage == 'train'):
                        _, prune_indicator = self.computeandsort_distances_fromcenter(k_heads, labels_K, cluster_centers_K, device=torch.device(k_heads.device), reservenum_each_center=1)
                        prune_indicaters.append(prune_indicator)
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            
            head_matrices["v_matrices_tobesupervised"] = [(v.contiguous()
                    .view(bsz, self.num_heads, -1, self.head_dim).contiguous().transpose(0,1).contiguous().view(self.num_heads, -1)
                    )]
            if self.cfg.cluster_matrix == 'V':
                if torch.equal(self.prune_indicator, torch.Tensor([-1]).repeat(self.num_heads).to(torch.device(v.device))) or self.cfg.keep_updating_cluster:
                    v_heads = (v.contiguous()
                        .view(bsz, self.num_heads, -1, self.head_dim).contiguous().transpose(0,1).contiguous().view(self.num_heads, -1)
                        )
                    labels_V, cluster_centers_V = kmeans(v_heads, num_clusters=self.cfg.N_head_clusters, distance=self.cfg.kmeans_distance_metric, seed=0, device=torch.device(v_heads.device))
                    head_matrices["cluster_centers_V"] = [cluster_centers_V]
                    head_matrices["head_labels_V"] = [labels_V]
                    
                    if (self.need_prune or self.cfg.collecting_indicators) and (self.cfg.experiment_stage == 'train'):
                        _, prune_indicator = self.computeandsort_distances_fromcenter(v_heads, labels_V, cluster_centers_V, device=torch.device(v_heads.device), reservenum_each_center=1)
                        prune_indicaters.append(prune_indicator)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention_efficient._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        head_matrices["attention_matrices_tobesupervised"] = [(attn_probs.contiguous()
                .view(bsz, self.num_heads, tgt_len, src_len).contiguous().transpose(0,1).contiguous().view(self.num_heads, bsz * tgt_len * src_len)
                )]
        if self.cfg.cluster_matrix == 'attn_matrix':
            if torch.equal(self.prune_indicator, torch.Tensor([-1]).repeat(self.num_heads).to(torch.device(attn_probs.device))) or self.cfg.keep_updating_cluster:
                attention_matrices_heads = (attn_probs.contiguous()
                    .view(bsz, self.num_heads, tgt_len, src_len).contiguous().transpose(0,1).contiguous().view(self.num_heads, bsz * tgt_len * src_len)
                    )
                # kmeans_attention_matrices = KMeans(n_clusters=self.cfg.N_head_clusters, random_state=0).fit(attention_matrices_heads)
                labels_attention_matrices, cluster_centers_attention_matrices = kmeans(attention_matrices_heads, num_clusters=self.cfg.N_head_clusters, distance=self.cfg.kmeans_distance_metric, seed=0, device=torch.device(attention_matrices_heads.device))
                head_matrices["cluster_centers_attention_matrices"] = [cluster_centers_attention_matrices]
                head_matrices["head_labels_attention_matrices"] = [labels_attention_matrices]
                # try:
                #     print(head_matrices["cluster_centers_attention_matrices"].device, head_matrices["head_labels_attention_matrices"].device, attention_matrices_heads.device)
                # except Exception as e: print(e)
                if (self.need_prune or self.cfg.collecting_indicators) and (self.cfg.experiment_stage == 'train'):
                    _, prune_indicator = self.computeandsort_distances_fromcenter(attention_matrices_heads, labels_attention_matrices, cluster_centers_attention_matrices, device=torch.device(attention_matrices_heads.device), reservenum_each_center=1)
                    prune_indicaters.append(prune_indicator)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        
        head_matrices["headout_matrices_tobesupervised"] = [(attn.contiguous()
                .view(bsz, self.num_heads, tgt_len, self.head_dim).contiguous().transpose(0,1).contiguous().view(self.num_heads, bsz * tgt_len * self.head_dim)
                )]
        if self.cfg.cluster_matrix == 'headout':
            if torch.equal(self.prune_indicator, torch.Tensor([-1]).repeat(self.num_heads).to(torch.device(attn.device))) or self.cfg.keep_updating_cluster:
                headout_heads = (attn.contiguous()
                    .view(bsz, self.num_heads, tgt_len, self.head_dim).contiguous().transpose(0,1).contiguous().view(self.num_heads, bsz * tgt_len * self.head_dim)
                    )
                # kmeans_headout = KMeans(n_clusters=self.cfg.N_head_clusters, random_state=0).fit(headout_heads)
                labels_headout, cluster_centers_headout = kmeans(headout_heads, num_clusters=self.cfg.N_head_clusters, distance=self.cfg.kmeans_distance_metric, seed=0, device=torch.device(headout_heads.device))
                head_matrices["cluster_centers_headout"] = [cluster_centers_headout]
                head_matrices["head_labels_headout"] = [labels_headout]
                
                if (self.need_prune or self.cfg.collecting_indicators) and (self.cfg.experiment_stage == 'train'):
                    _, prune_indicator = self.computeandsort_distances_fromcenter(headout_heads, labels_headout, cluster_centers_headout, device=torch.device(headout_heads.device), reservenum_each_center=1)
                    prune_indicaters.append(prune_indicator)
        #1. no need prune: 聚类。self.prune_indicator一直是None因为进不去下面的finalize_prune_indicators。这时候不管self.cfg.keep_updating_cluster是什么，都会执行聚类，计算并反向传播聚类的loss。
        #2. need prune，self.prune_indicator is None：聚类，各类matrix的离心计算，finalize_prune_indicators计算。
        #3. need prune，self.prune_indicator is not None, self.cfg.keep_updating_cluster is True: 聚类，各类matrix的离心计算，finalize_prune_indicators计算。
        #4. need prune，self.prune_indicator is not None, self.cfg.keep_updating_cluster is False：Do nothing。会直接复用已经生成的self.prune_indicator，来执行下面的head masking。
        
        assert ~(self.cfg.collecting_indicators & self.need_prune), f"self.cfg.collecting_indicators: {self.cfg.collecting_indicators} and self.need_prune: {self.need_prune} cannot both be true."
        
        if self.cfg.voting_to_prune and self.cfg.collecting_indicators and self.cfg.experiment_stage == 'train':
            if self.voting_list is None:
                self.voting_list = []
            temp_indicator = self.finalize_prune_indicators(prune_indicaters, torch.device(attn.device)).float()
            temp_indicator.requires_grad = False
            self.voting_list.append(temp_indicator)
            
        if self.need_prune:
            if (torch.equal(self.prune_indicator, torch.Tensor([-1]).repeat(self.num_heads).to(torch.device(attn.device))) or self.cfg.keep_updating_cluster) and attn.requires_grad:
                if not self.cfg.voting_to_prune:
                    print("Need prune: ", self.cfg.need_prune, "!")
                    print("Training stage, creating self.prune_indicator based on a single batch...")
                    self.prune_indicator.data = Parameter(self.finalize_prune_indicators(prune_indicaters, torch.device(attn.device)).float()).data
                    self.prune_indicator.requires_grad = False
                    assert not self.prune_indicator.requires_grad
                else:
                    print("Need prune: ", self.cfg.need_prune, "!")
                    print("Training stage, creating self.prune_indicator based on the voting list...")
                    self.prune_indicator.data = self.voting_on_pruneindicators(self.voting_list).data
                    self.prune_indicator.requires_grad = False
                    assert not self.prune_indicator.requires_grad
                    self.voting_list = None
            # Head masking
            attn = attn.contiguous().view(bsz, self.num_heads, tgt_len, self.head_dim)
            head_mask = torch.zeros_like(attn)
            for i in range(self.num_heads):
                head_mask[:, i, :, :] = self.prune_indicator[i]
            attn = head_mask * attn
            attn = attn.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)
            
            # print(attn)
            
            # print("supervise_device: ", head_matrices["headout_matrices_tobesupervised"][0].device, "cluster_centers_headout_device: ", head_matrices["cluster_centers_headout"][0].device)
        # attn = self.Headwise_Attention(attn, self.cfg.HWA_head_num)   # [bsz * self.num_heads, tgt_len, self.head_dim]
        # assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        # head_matrices["headout_matrices_afterHWA"] = [attn.contiguous().view(bsz, self.num_heads, tgt_len, self.head_dim)]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # print(attn)
        if self.out_proj is not None:
            attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, head_matrices

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
            

        
        
if __name__ == '__main__':
    
    class CFG():
        def __init__(self):
            self.kmeans_distance_metric = 'euclidean'
            self.need_prune = True
            self.cluster_matrix = 'Q'
            self.N_head_clusters = 2
            self.keep_updating_cluster = True
            self.abandon_outputprojection = True
            
    
    cfg = CFG()
    
    HWA = MultiheadAttention_efficient(cfg, 16, 8, self_attention = True)
    query = torch.rand(9, 3, 16)
    output, _, _ = HWA(query, query, query)

    print(output)
    print(output.shape)
        
