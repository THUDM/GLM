# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""

import os
import torch
import torch.nn.functional as F
import time
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint, get_checkpoint_iteration
from data_utils import make_tokenizer
import mpu

from utils import print_rank_0
from train_utils import get_model


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    # if args.deepspeed:
    #     print_rank_0("DeepSpeed is enabled.")
    #
    #     model, _, _, _ = deepspeed.initialize(
    #         model=model,
    #         model_parameters=model.parameters(),
    #         args=args,
    #         mpu=mpu,
    #         dist_init_required=False
    #     )
    if args.load is not None:
        if args.deepspeed:
            iteration, release, success = get_checkpoint_iteration(args)
            path = os.path.join(args.load, str(iteration), "mp_rank_00_model_states.pt")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["module"])
        else:
            _ = load_checkpoint(model, None, None, args)
    # if args.deepspeed:
    #     model = model.module

    return model


def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        set_loss_mask=not args.transformer_xl,
        mem_length=args.mem_length)

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits


def generate_score(model, tokenizer, args, device, input_str, eval_str):
    context_count = 0
    model.eval()
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eval_tokens = tokenizer.EncodeAsIds(eval_str).tokenization + [tokenizer.get_command('eos').Id]
        if not context_tokens:
            context_tokens = eval_tokens[0:1]
            eval_tokens = eval_tokens[1:]
        context_length = len(context_tokens)
        eval_length = len(eval_tokens)
        if context_length >= args.seq_length:
            return "输入过长。"

        # terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
        # pad_id = tokenizer.get_command('pad').Id
        # if context_length < args.out_seq_length:
        #     context_tokens.extend([pad_id] * (args.out_seq_length - context_length))

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        eval_tokens_tensor = torch.cuda.LongTensor([eval_tokens])
        context_length_tensor = torch.cuda.LongTensor([context_length])
        eval_length_tensor = torch.cuda.LongTensor([eval_length])
        # context_length = context_length_tensor[0].item()
        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)
        # print(context_tokens)
        start_time = time.time()

        counter, mems = 0, []
        org_context_length = context_length
        sumlognum = 0
        while counter < eval_length:
            if counter == 0:
                logits, *mems = model(tokens, position_ids, attention_mask, *mems)
                logits = logits[:, -1]
            else:
                index = org_context_length + counter
                logits, *mems = model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                                      tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                      dtype=torch.float), *mems)
                logits = logits[:, 0]
            # logits = logits[:, -1]
            logits /= args.temperature
            logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
            log_probs = F.softmax(logits, dim=-1)
            log_num = torch.log(log_probs).data
            # print(log_num)
            sumlognum += log_num[0, eval_tokens[counter]]
            # print(log_probs)
            # prev = torch.multinomial(log_probs, num_samples=1)[0]
            # print(tokens,eval_tokens_tensor[counter:counter+1])
            tokens = torch.cat((tokens, eval_tokens_tensor[:, counter:counter + 1]), dim=1)
            # print(tokens,sumlognum)
            context_length += 1
            counter += 1

        # trim_decode_tokens = decode_tokens[:decode_tokens.find("<|endoftext|>")]
        sumlognum = sumlognum
        return sumlognum, eval_length


def prepare_tokenizer(args):
    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    num_tokens = tokenizer.num_tokens
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by * \
               mpu.get_model_parallel_world_size()
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(
        before, after - before, after))

    args.tokenizer_num_tokens = after
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    # after = tokenizer.num_tokens
    # while after % mpu.get_model_parallel_world_size() != 0:
    #     after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer


def set_args():
    args = get_args()
    # set up
    # print(args)
    args.deepspeed = True
    args.num_nodes = 1
    args.num_gpus = 1
    args.model_parallel_size = 1
    args.deepspeed_config = "script/config_gpt_large.json"
    args.num_layers = 32
    args.hidden_size = 2560
    args.load = "/root/data/checkpoints/txl-2.8b11-20-15-10"
    args.num_attention_heads = 32
    args.max_position_embeddings = 1024
    args.tokenizer_type = "ChineseSPTokenizer"
    args.cache_dir = "cache"
    args.fp16 = True
    args.out_seq_length = 1024
    args.seq_length = 512
    args.mem_length = 0
    args.transformer_xl = True
    args.temperature = 1.0
    args.top_k = 0
    args.top_p = 0

    return args


def prepare_model():
    """Main training program."""

    # print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = set_args()
    # print(args)
    args.mem_length = args.seq_length + args.mem_length - 1

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    # setting default batch size to 1
    args.batch_size = 1

    # generate samples
    return model, tokenizer, args


def generate(fname):
    model, tokenizer, args = prepare_model()
    import json
    json_file = open(fname, 'r')
    w = json_file.readlines()
    sumscore = 0
    sumsample = 0
    total_length = 0
    for line in w:
        json_decode = json.loads(line)
        context_str = json_decode['prompt']
        text_str = json_decode['text']
        score, length = generate_score(model, tokenizer, args, torch.cuda.current_device(), context_str, text_str)
        score = -score
        sumsample += 1
        sumscore += score
        total_length += length
        print("sample:", sumsample, " score:", torch.exp(score / length), " avg score:", torch.exp(sumscore / total_length))


generate("/root/data/test.json")
