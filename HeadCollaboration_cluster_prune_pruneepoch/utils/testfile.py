from cProfile import label
import torch
from torch import nn
from torch import linalg as LA
from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans
import torch.nn.functional as F
from fairseq import checkpoint_utils

# A = torch.rand(1, 2, 9)
# B = A.view(1, 6, 3)

# print('A: ', A)
# print('B: ', B)
# q_proj_headwise = nn.Linear(64, 64, bias=True)
# q = q_proj_headwise(torch.rand(8*192, 21, 64)).view(8 * 192, 1344).contiguous().view(8, 192, 1344)   # [num_heads, bsz, head_dim * tgt_len]
# print(q.shape)

# print(torch.tensor(0))

# def compute_attention_regularization_term(matrices_of_layers):
#     print(matrices_of_layers.shape)
#     Regu = torch.tensor(0).float()
#     for attention_matrices in matrices_of_layers:
#         bsz, num_heads, tgt_len, src_len = attention_matrices.size()
#         attention_matrices = attention_matrices.view(num_heads, bsz, tgt_len, src_len).float()
#         for i in range(num_heads):
#             for j in range(num_heads):
#                 if i < j:
#                     Regu += (-(attention_matrices[i] * attention_matrices[j]).sum() / (num_heads * num_heads))
#     return Regu


# A = torch.tensor([[[[[1, 2], 
#                     [3, 4]],
#                     [[1, 2], 
#                     [3, 4]],
#                   [[1, 2], 
#                    [3, 4]]]]]) # -(1+4+9+16)*3/9
# print(compute_attention_regularization_term(A))

# A = torch.tensor([1, 2, 3]).float()
# print(LA.vector_norm(A))

# A = torch.tensor([1, 2, 3]).float()
# print(A @ A)

# def compute_v_regularization_term(matrices_of_layers):
#         Regu = torch.tensor(0).float()
#         for v_matrices in matrices_of_layers:
#             bsz, num_heads, tgt_len, head_dim = v_matrices.size()
#             v_matrices = v_matrices.view(num_heads, bsz * tgt_len * head_dim).float()
#             for i in range(num_heads):
#                 for j in range(num_heads):
#                     if i < j:
#                         Regu += ((v_matrices[i] @ v_matrices[j] / (LA.vector_norm(v_matrices[i]) * LA.vector_norm(v_matrices[j]))) 
#                                  / (num_heads * num_heads))
#                         # sum over the batch if the original loss function is reduced by sum.
#         return Regu
    
# A = torch.tensor([[[[[1, 2], 
#                 [3, 4]],
#                 [[1, 2], 
#                 [3, 4]],
#                 [[1, 2], 
#                 [3, 4]]]]]) # -(1+4+9+16)*3/9
# print(compute_v_regularization_term(A))

# v_heads = torch.rand(8, 192, 20, 32).contiguous().view(8, -1)
# _kmeans = KMeans(n_clusters=2, random_state=0, init='random').fit(v_heads)
# print(_kmeans.cluster_centers_)
# print(_kmeans.labels_)

# # v_heads = torch.rand(8, 192, 20, 32).contiguous().view(8, -1)
# labels_, cluster_centers_ = kmeans(v_heads, num_clusters=2, seed=0)
# print(cluster_centers_)
# print(labels_)

# print(v_heads.device)

# print(type(int(labels_[0])), type(labels_[0].data), int(labels_[0]))

# A = torch.rand(8).to(torch.device('cuda:7'))
# B = torch.rand(8).to(torch.device('cuda:7'))
# C = A @ B
# print(A.device, C.device, C)

# import math
# print(math.comb(3, 1))

# A = torch.rand(3, 5)
# B = torch.rand(1, 5)
# print(A)
# A[0] = B

# print(A, B)

# print(torch.tensor([1,2,3])**2)
# print(torch.sqrt(torch.tensor(9)))
# print(type(torch.tensor([1,2,3])))
# print(nn.CosineSimilarity(dim=0)(torch.tensor([[1,2,3]])[0].float(), torch.tensor([[3,2,1]])[0].float()))

# cluster_k_indexes = torch.tensor([2, 4, 6])
# distances_k = torch.tensor([12, 3, 7])

# scoredir = dict(zip(cluster_k_indexes, distances_k))
# scoredir = dict(sorted(scoredir.items(), reverse=False, key=lambda item: item[1]))

# print(scoredir)
# print(3 in cluster_k_indexes)

# headtokeep_index = [2, 3]
# prune_indicater = torch.zeros(5)
# prune_indicater[headtokeep_index[:]] = 1
# prune_indicater = prune_indicater.bool()
# print(prune_indicater)

# headtokeep_index1 = [0, 1, 2]
# prune_indicater1 = torch.zeros(5)
# prune_indicater1[headtokeep_index1[:]] = 1
# prune_indicater1 = prune_indicater1.bool()
# print(prune_indicater1)

# # print((prune_indicater | prune_indicater1).int())

# A = torch.rand(3, 5, 6)
# A[:, 0, :] = 0

# print(A)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         self.fc3 = nn.Linear(84, 10)
#         self.prune_indicator1 = nn.Parameter(torch.zeros(2))
#         # self.state_dict()['prune_indicator1'] = self.prune_indicator1
#         # print(self.state_dict()['prune_indicator1'])
#         # self.prune_indicator2 = nn.Parameter(torch.zeros(2))


#     def forward(self, x):
#         self.prune_indicator1.data = nn.Parameter(torch.ones(2)).data
#         self.prune_indicator1.requires_grad = False
#         # self.prune_indicator2 = torch.ones(2)
#         # self.state_dict()['prune_indicator1.weight'] = self.prune_indicator1
#         # self.state_dict()['prune_indicator2'] = self.prune_indicator2

#         x = self.fc3(x)
#         return x

# net = Net()

# net(torch.rand(10, 84))

# print("Model's state_dict:")
# for param_tensor in net.state_dict():
#     print(param_tensor, "\t", net.state_dict()[param_tensor])

# torch.save(net.state_dict(), './testmodel.pt')

# net = Net()
# net.load_state_dict(torch.load('./testmodel.pt'))

# # print(net)

# print("Model's state_dict:")
# for param_tensor in net.state_dict():
#     print(param_tensor, "\t", net.state_dict()[param_tensor])

# print(nn.Parameter(torch.Tensor([-1]).repeat(8)).detach().requires_grad)

# print(nn.Parameter(torch.Tensor([-1]).repeat(8)).data.requires_grad)

# print(nn.Parameter(torch.Tensor([-1]).repeat(8)).detach()[0])

# print(torch.equal(nn.Parameter(torch.Tensor([-2]).repeat(8)).float(), torch.Tensor([-1]).repeat(8)))


# State = checkpoint_utils.load_checkpoint_to_cpu("HeadCollaboration_cluster_prune/Experimental_Results/SweepRun_2_ClusterHead_iwslt14_use_inclassloss_v_bothinterandinclass_usevariantencdecattention_5checkpointsaveraged_compoundsplit_prunefromepoch10_noearlystop_rerun5/0/checkpoints_headcolab/checkpoint.best_bleu_35.9504.pt")
# # print(State)

# for key in State['model'].keys():
#     try:
#         for key1 in State['model'][key].keys():
#             print(key, key1)
#     except:
#         print(key)


# print([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])] == [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])])

# votingcounter = torch.tensor([1,9,2,7])

# votingdir = dict(zip(list(range(votingcounter.shape[0])), votingcounter.tolist()))
# votingdir = dict(sorted(votingdir.items(), reverse=False, key=lambda item: item[1]))

# print(votingdir)

# def voting_on_pruneindicators(voting_list: list):
#         assert voting_list is not None
#         voted_indicator = torch.zeros_like(voting_list[0])
#         # TODO keep self.cfg.N_head_clusters heads
#         votingcounter = torch.zeros_like(voting_list[0])
#         for indicator in voting_list:
#             votingcounter += indicator
        
#         votingdir = dict(zip(list(range(votingcounter.shape[0])), votingcounter.tolist()))
#         votingdir = dict(sorted(votingdir.items(), reverse=True, key=lambda item: item[1]))
        
#         headstokeep = list(votingdir.keys())[:2]
        
#         for id in headstokeep:
#             voted_indicator[id] = 1

#         return voted_indicator

# voting_list = [torch.tensor([1,9,2,7]), torch.tensor([1,9,2,7]), torch.tensor([1,9,2,7])]
# print(voting_on_pruneindicators(voting_list))

print((True or False) and False)



# 'model': Namespace(N_head_clusters=2, Supervise_mode='reconstruction', _name='efficient_transformer', activation_dropout=0.0, activation_fn='relu', adam_betas='(0.9, 0.98)', adam_eps=1e-08, adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, all_gather_list_size=16384, amp=False, amp_batch_retries=2, amp_init_scale=128, amp_scale_window=None, arch='efficient_transformer', attention_dropout=0.0, azureml_logging=False, batch_size=None, batch_size_valid=None, best_checkpoint_metric='bleu', bf16=False, bias_mode_k='bias_trainable', bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_activations=False, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.0, cluster_loss_coefficient_inclass=1.0, cluster_loss_coefficient_interclass=1.0, cluster_matrix='headout', combine_valid_subsets=None, cpu=False, cpu_offload=False, criterion='label_smoothed_cross_entropy_headclustering', cross_self_attention=False, curriculum=0, data='../data-bin/iwslt14.tokenized.de-en', data_buffer_size=10, dataset_impl=None, ddp_backend='pytorch_ddp', ddp_comm_hook='none', debug_mode=True, decoder_attention_heads=8, decoder_embed_dim=512, decoder_embed_path=None, decoder_ffn_embed_dim=2048, decoder_input_dim=512, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=False, decoder_normalize_before=False, decoder_output_dim=512, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=1, distributed_port=-1, distributed_rank=0, distributed_world_size=1, dropout=0.3, efficient_multihead_attention=True, ema_decay=0.9999, ema_fp32=False, ema_seed_model=None, ema_start_update=0, ema_update_freq=1, empty_cache_freq=0, encoder_attention_heads=8, encoder_embed_dim=512, encoder_embed_path=None, encoder_ffn_embed_dim=2048, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=False, encoder_normalize_before=False, eos=2, eval_bleu=True, eval_bleu_args='{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}', eval_bleu_detok='moses', eval_bleu_detok_args='{}', eval_bleu_print_samples=True, eval_bleu_remove_bpe='@@ ', eval_tokenized_bleu=False, experiment_stage='train', fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=1, fp16=False, fp16_adam_stats=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, fp32_reduce_scatter=False, gen_subset='test', gradient_as_bucket_view=False, heartbeat_timeout=-1, ignore_prefix_size=0, ignore_unused_valid_subsets=False, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_interval_updates_pattern=-1, keep_last_epochs=-1, label_smoothing=0.1, layernorm_embedding=False, left_pad_source=True, left_pad_target=False, load_alignments=False, load_checkpoint_on_all_dp_ranks=False, localsgd_frequency=3, log_file='./Experimental_Results/debug/logfile', log_format=None, log_interval=100, lr=[0.0005], lr_scheduler='inverse_sqrt', max_epoch=0, max_tokens=4096, max_tokens_valid=4096, max_update=0, max_valid_steps=None, maximize_best_checkpoint_metric=True, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=True, no_last_checkpoints=False, no_progress_bar=False, no_reshard_after_forward=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_batch_buckets=0, num_shards=1, num_workers=1, offload_activations=False, on_cpu_convert_precision=False, optimizer='adam', optimizer_overrides='{}', pad=1, patience=5, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, plasma_path='/tmp/plasma', profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, report_accuracy=False, required_batch_size_multiple=8, required_seq_len_multiple=1, reset_dataloader=False, reset_logging=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', save_dir='./Experimental_Results/debug/checkpoints', save_interval=1, save_interval_updates=0, scoring='bleu', seed=1, sentence_avg=False, shard_id=0, share_all_embeddings=False, share_decoder_input_output_embed=True, share_kbias_across_encoder_decoder=False, share_kbias_across_layers=False, skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, source_lang=None, stop_min_lr=-1.0, stop_time_hours=0, store_ema=False, supervised_matrix='headout', suppress_crashes=False, target_lang=None, task='translation', tensorboard_logdir='./Experimental_Results/debug/tensorboard_logdir', threshold_loss_scale=None, tie_adaptive_weights=False, tokenizer=None, tpu=False, train_subset='train', truncate_source=False, unk=3, update_freq=[1], upsample_primary=-1, use_bmuf=False, use_efficient_en_de_attn=False, use_inclass_loss=False, use_interclass_loss=False, use_old_adam=False, use_plasma_view=False, use_sharded_state=False, user_dir='./', valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, wandb_project=None, wandb_runid=None, wandb_runname=None, wandb_runnotes=None, wandb_runtags=None, warmup_init_lr=-1, warmup_updates=4000, weight_decay=0.0001, write_checkpoints_asynchronously=False, zero_sharding='none')