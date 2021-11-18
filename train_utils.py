import deepspeed
import torch
from apex.optimizers import FusedAdam as Adam
from torch import distributed as dist

from SwissArmyTransformer import mpu
from learning_rates import AnnealingLR
from model import GLMFPrefixModel
from model import GLMForMultiTokenCloze, GLMForMultiTokenClozeFast, GLMForSingleTokenCloze, GLMForSequenceClassification
from utils import print_rank_0, get_checkpoint_name, get_checkpoint_iteration


def load_pretrained(model, checkpoint_path, args, task_tokens=None):
    load_dir, tag, release, success = get_checkpoint_iteration(checkpoint_path)
    checkpoint_name = get_checkpoint_name(load_dir, tag, release)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading pretrained model {}'.format(
            torch.distributed.get_rank(), checkpoint_name))
    # Load the checkpoint.
    sd = torch.load(checkpoint_name, map_location='cpu')
    if args.deepspeed:
        model = model.module

    # Model.
    def extend_embedding_weights(state_weights, model_weights):
        original_length = state_weights.shape[0]
        assert original_length <= args.max_position_embeddings + 1
        new_weights = model_weights.clone()
        new_weights[:original_length] = state_weights
        return new_weights

    if args.block_lm and args.old_checkpoint:
        sd['module']['transformer.word_embeddings.weight'] = sd['module']['word_embeddings.weight']
        del sd['module']['word_embeddings.weight']
        sd['module']['mixins.block_position_embedding.block_position_embeddings.weight'] = sd['module'][
            'transformer.block_position_embeddings.weight']
        del sd['module']['transformer.block_position_embeddings.weight']

    # TODO: Fix this with mixin
    # if args.block_lm:
    #     if "mixins.block_position_embedding.block_position_embeddings.weight" in sd["module"]:
    #         position_weights = sd['module']['mixins.block_position_embedding.block_position_embeddings.weight']
    #         if args.max_position_embeddings + 1 > position_weights.shape[0]:
    #             sd['module']["transformer.position_embeddings.weight"] = extend_embedding_weights(
    #                 position_weights, model.state_dict()["transformer.position_embeddings.weight"].data)
    #             print_rank_0(f"Extend position embedding to {args.max_position_embeddings + 1}")
    #     if "transformer.block_position_embeddings.weight" in sd["module"]:
    #         block_position_weights = sd['module']["transformer.block_position_embeddings.weight"]
    #         if args.max_position_embeddings + 1 > block_position_weights.shape[0]:
    #             sd['module']["transformer.block_position_embeddings.weight"] = extend_embedding_weights(
    #                 block_position_weights,
    #                 model.state_dict()["transformer.block_position_embeddings.weight"].data)
    #             print_rank_0(f"Extend block position embedding to {args.max_position_embeddings + 1}")
    missing_keys, unexpected_keys = model.load_state_dict(sd['module'], strict=False)
    if missing_keys or unexpected_keys:
        print_rank_0(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")
    if args.continuous_prompt and args.prompt_init:
        model.prompt_spell.init_embedding(model.word_embeddings.weight.data, task_tokens)


def get_model(args, model_type=None, multi_token=True, num_labels=None, spell_length=None):
    """Build the model."""
    print_rank_0('building GPT2 model ...')
    output_predict, paralle_output = True, True
    if (model_type == "multiple_choice" or model_type == "classification") and not args.cloze_eval:
        output_predict = False
    if model_type is not None:
        paralle_output = False
    if spell_length is not None:
        print_rank_0(f"Continuous spell length {spell_length}")
    if args.prefix_prompt:
        model = GLMFPrefixModel(num_layers=args.num_layers,
                         vocab_size=args.vocab_size,
                         hidden_size=args.hidden_size,
                         num_attention_heads=args.num_attention_heads,
                         embedding_dropout_prob=args.hidden_dropout,
                         attention_dropout_prob=args.attention_dropout,
                         output_dropout_prob=args.hidden_dropout,
                         max_sequence_length=args.max_position_embeddings,
                         max_memory_length=args.mem_length,
                         checkpoint_activations=args.checkpoint_activations,
                         checkpoint_num_layers=args.checkpoint_num_layers,
                         parallel_output=paralle_output,
                         relative_encoding=args.transformer_xl,
                         block_position_encoding=args.block_lm and not args.masked_lm,
                         output_predict=output_predict,
                         spell_length=spell_length,
                         spell_func=args.prompt_func,
                         attention_scale=args.attention_scale,
                         prefix_prompt=args.prefix_prompt)
    else:
        model = GLMModel(num_layers=args.num_layers,
                         vocab_size=args.vocab_size,
                         hidden_size=args.hidden_size,
                         num_attention_heads=args.num_attention_heads,
                         embedding_dropout_prob=args.hidden_dropout,
                         attention_dropout_prob=args.attention_dropout,
                         output_dropout_prob=args.hidden_dropout,
                         max_sequence_length=args.max_position_embeddings,
                         max_memory_length=args.mem_length,
                         checkpoint_activations=args.checkpoint_activations,
                         checkpoint_num_layers=args.checkpoint_num_layers,
                         parallel_output=paralle_output,
                         relative_encoding=args.transformer_xl,
                         block_position_encoding=args.block_lm and not args.masked_lm,
                         output_predict=output_predict,
                         spell_length=spell_length,
                         spell_func=args.prompt_func,
                         attention_scale=args.attention_scale)
    if args.freeze_transformer:
        model.freeze_transformer(tune_prefix_layers=args.tune_prefix_layers)
    if model_type is not None:
        if model_type == 'multiple_choice':
            if args.cloze_eval:
                if multi_token:
                    if args.fast_decode:
                        model = GLMForMultiTokenClozeFast(model, length_penalty=args.length_penalty)
                    else:
                        model = GLMForMultiTokenCloze(model, length_penalty=args.length_penalty)
                else:
                    model = GLMForSingleTokenCloze(model, take_softmax=args.adapet)
            else:
                model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                     num_class=num_labels)
        elif model_type == 'classification':
            model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                 num_class=num_labels)
        elif model_type == 'generation':
            pass
        else:
            raise NotImplementedError(model_type)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)
    return model
