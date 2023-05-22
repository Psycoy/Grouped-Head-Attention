#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple
from typing import Dict, Tuple
from git import List

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output
from matplotlib import pyplot


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def buffered_read(input, buffer_size):       
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, cfg, task, max_positions, encode_fn):
    # def encode_fn_target(x):
    #     return encode_fn(x)

    # if cfg.generation.constraints:
    #     # Strip (tab-delimited) contraints, if present, from input lines,
    #     # store them in batch_constraints
    #     batch_constraints = [list() for _ in lines]
    #     for i, line in enumerate(lines):
    #         if "\t" in line:
    #             lines[i], *batch_constraints[i] = line.split("\t")

    #     # Convert each List[str] to List[Tensor]
    #     for i, constraint_list in enumerate(batch_constraints):
    #         batch_constraints[i] = [
    #             task.target_dictionary.encode_line(
    #                 encode_fn_target(constraint),
    #                 append_eos=False,
    #                 add_if_not_exist=False,
    #             )
    #             for constraint in constraint_list
    #         ]

    # if cfg.generation.constraints:
    #     constraints_tensor = pack_constraints(batch_constraints)
    # else:
    #     constraints_tensor = None

    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)
    # print("tokens: ", tokens)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )


def make_batches_enc(lines, cfg, task, max_positions, encode_fn):

    tokens = encode_fn(lines)
    # print("tokens: ", tokens, type(tokens))
    # tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    tokens = task.source_dictionary.encode_line(
                tokens,
                append_eos=True,
                add_if_not_exist=False,
            )
    tokens = torch.cat((torch.tensor([task.source_dictionary.eos_index]), tokens), dim=0) if task.source_dictionary.bos_index is None else torch.cat((torch.tensor([task.source_dictionary.bos_index]), tokens), dim=0)
    tokens = tokens.unsqueeze(0)
    assert tokens.shape[0] == 1
    lengths = [T.shape[0] for T in tokens]
    lengths = torch.tensor(lengths)
    
    return [{"src_tokens": tokens, "src_lens": lengths}]
        
def make_batches_dec(lines, cfg, task, max_positions, encode_fn):
    tokens = encode_fn(lines)
    # print("tokens: ", tokens, type(tokens))
    # tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)
    tokens = task.target_dictionary.encode_line(
                tokens,
                append_eos=True,
                add_if_not_exist=False,
            )
    tokens = torch.cat((torch.tensor([task.target_dictionary.eos_index]), tokens), dim=0) if task.target_dictionary.bos_index is None else torch.cat((torch.tensor([task.target_dictionary.bos_index]), tokens), dim=0)
    tokens = tokens.unsqueeze(0)
    assert tokens.shape[0] == 1
    lengths = [T.shape[0] for T in tokens]
    lengths = torch.tensor(lengths)
    
    return [{"tgt_tokens": tokens, "tgt_lens": lengths}]

def main(cfg: FairseqConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    
    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(cfg.common)

    if cfg.interactive.buffer_size < 1:
        cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not cfg.dataset.batch_size
        or cfg.dataset.batch_size <= cfg.interactive.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def encode_fn(x):
        if tokenizer is not None:
            # print("Before tokenize: ", x)
            x = tokenizer.encode(x)
            # print("After tokenize: ", x)
        if bpe is not None:
            # print("Before bpe: ", x)
            x = bpe.encode(x)
            # print("After bpe: ", x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if cfg.generation.constraints:
        logger.warning(
            "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
        )

    if cfg.interactive.buffer_size > 1:
        logger.info("Sentence buffer size: %s", cfg.interactive.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Type the input sentence and press return:")
    start_id = 0
    
    for inputs in buffered_read(cfg.interactive.input, cfg.interactive.buffer_size):
        # print(inputs)
        assert len(inputs) == 2
        inputs, decoder_inputs = [inputs[0]], [inputs[1]]
        results = []
        for batch, batch_decoder in zip(make_batches_enc(inputs, cfg, task, max_positions, encode_fn), make_batches_dec(decoder_inputs, cfg, task, max_positions, encode_fn)):
            print(batch["src_tokens"], batch_decoder["tgt_tokens"])
            bsz = len(batch["src_tokens"])
            bsz2 = len(batch_decoder["tgt_tokens"])
            assert bsz == bsz2 == 1
            src_tokens = batch["src_tokens"]
            src_lengths = batch["src_lens"]
            # constraints = batch.constraints
            dec_tokens = batch_decoder["tgt_tokens"]
            dec_lengths = batch_decoder["tgt_lens"]
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths
                },
                "dec_input": {
                    "dec_tokens": dec_tokens,
                    "dec_lengths": dec_lengths
                }
            }
            translate_start_time = time.time()
            # print(models[0].cfg.efficient_multihead_attention)
            head_matrices = task.inference_step_visualize(
                generator, models, sample
            )
            print(head_matrices.keys())
            layer_num = len(head_matrices["attention_matrices_tobesupervised"])
            print("Total attention layer numbers: {}.".format(layer_num))
            translate_time = time.time() - translate_start_time
            total_translate_time += translate_time
            
            matrix_heatmap_plot(head_matrices, 
                                cfg = cfg,
                                layers_idx = range(len(head_matrices["attention_shapes"])),  # which layer(s) to be visualized. Choose layer_num-1 as the index of the decoder layer with a full incremental input (if using encoder-decoder attention, it will be layer_num-1 and layer_num-2).
                                type="attention_matrix")
            
    #         list_constraints = [[] for _ in range(bsz)]
    #         if cfg.generation.constraints:
    #             list_constraints = [unpack_constraints(c) for c in constraints]
    #         for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
    #             src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
    #             constraints = list_constraints[i]
    #             results.append(
    #                 (
    #                     start_id + id,
    #                     src_tokens_i,
    #                     hypos,
    #                     {
    #                         "constraints": constraints,
    #                         "time": translate_time / len(translations),
    #                     },
    #                 )
    #             )

    #     # sort output to match input order
    #     for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
    #         src_str = ''
    #         if src_dict is not None:
    #             src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
    #             print("S-{}\t{}".format(id_, src_str))
    #             print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
    #             for constraint in info["constraints"]:
    #                 print(
    #                     "C-{}\t{}".format(
    #                         id_, tgt_dict.string(constraint, cfg.common_eval.post_process)
    #                     )
    #                 )

    #         # Process top predictions
    #         for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
    #             hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
    #                 hypo_tokens=hypo["tokens"].int().cpu(),
    #                 src_str=src_str,
    #                 alignment=hypo["alignment"],
    #                 align_dict=align_dict,
    #                 tgt_dict=tgt_dict,
    #                 remove_bpe=cfg.common_eval.post_process,
    #                 extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
    #             )
    #             detok_hypo_str = decode_fn(hypo_str)
    #             score = hypo["score"] / math.log(2)  # convert to base 2
    #             # original hypothesis (after tokenization and BPE)
    #             print("H-{}\t{}\t{}".format(id_, score, hypo_str))
    #             # detokenized hypothesis
    #             print("D-{}\t{}\t{}".format(id_, score, detok_hypo_str))
    #             print(
    #                 "P-{}\t{}".format(
    #                     id_,
    #                     " ".join(
    #                         map(
    #                             lambda x: "{:.4f}".format(x),
    #                             # convert from base e to base 2
    #                             hypo["positional_scores"].div_(math.log(2)).tolist(),
    #                         )
    #                     ),
    #                 )
    #             )
    #             if cfg.generation.print_alignment:
    #                 alignment_str = " ".join(
    #                     ["{}-{}".format(src, tgt) for src, tgt in alignment]
    #                 )
    #                 print("A-{}\t{}".format(id_, alignment_str))

    #     # update running id_ counter
    #     start_id += len(inputs)

    # logger.info(
    #     "Total time: {:.3f} seconds; translation time: {:.3f}".format(
    #         time.time() - start_time, total_translate_time
    #     )
    # )

def matrix_heatmap_plot(Matrices: Dict[str, List[torch.Tensor]], layers_idx: List[int], cfg: FairseqConfig, type: str=None):
    try:
        os.mkdir(os.path.join(cfg.common_eval.results_path, "Visualizations"))
    except FileExistsError:
        print("Folder already exists.")
    if type=="attention_matrix": 
        matrices = Matrices["attention_matrices_tobesupervised"]
        matrices_shape = Matrices["attention_shapes"]
        cluster_labels = Matrices["head_labels_attention_matrices"]
        for layer_idx in layers_idx:
            matrix_plot = matrices[layer_idx]
            matrix_plot = matrix_plot.cpu()
            num_heads, head_dim, bsz, tgt_len, src_len = matrices_shape[layer_idx]
            print("matrices_shape[{}]: ".format(layer_idx), matrices_shape[layer_idx])
            assert bsz==1
            
            matrix_plot = matrix_plot.view(num_heads, bsz, tgt_len, src_len).contiguous().view(bsz, num_heads, tgt_len, src_len)
            f_min, f_max = matrix_plot.min(), matrix_plot.max()
            matrix_plot = (matrix_plot - f_min) / (f_max - f_min)   #normalize
            # ix = 1
            print("clusterlabel: ", cluster_labels[layer_idx])
            for head_idx in range(num_heads):
                ax = pyplot.subplot(math.ceil(math.sqrt(num_heads)), math.ceil(math.sqrt(num_heads)), head_idx+1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(str(int(cluster_labels[layer_idx][head_idx])))
                # plot filter channel in grayscale
                pyplot.imshow(matrix_plot[0][head_idx][:][:], cmap='gray')
                # ix += 1
            pyplot.savefig(os.path.join(cfg.common_eval.results_path, "Visualizations", "Layer_"+str(layer_idx)+".png"))

def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
