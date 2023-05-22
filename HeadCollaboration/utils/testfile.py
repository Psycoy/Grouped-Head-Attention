from cProfile import label
import torch
from torch import nn
from torch import linalg as LA
from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans

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


# A = torch.tensor(1.0)
# A.requires_grad=True
# print(A.requires_grad, A.data.requires_grad)
# A = torch.tensor([[1,2,3], 
#                   [4,5,6]])
# B = A.view(6,)
# print(B)

# print(B.contiguous().view(3, 2))

labels_attention_matrices, cluster_centers_attention_matrices = kmeans(torch.rand(6,10), num_clusters=6, seed=0, device=torch.device('cpu'))
print(labels_attention_matrices, cluster_centers_attention_matrices)