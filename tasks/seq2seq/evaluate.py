import datetime
import torch
import torch.nn.functional as F
import mpu
from utils import print_rank_0
from generation_utils import BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, \
    NoRepeatNGramLogitsProcessor
from pretrain_gpt2 import get_batch
from rouge import Rouge


def rouge_metric(predictions, labels, examples, metric="rouge-1"):
    rouge = Rouge()
    refs = [example.meta["ref"] for example in examples]
    scores = rouge.get_scores(predictions, refs, avg=True)
    return scores[metric.lower()]["f"]


def process_batch(batch, args):
    """Process batch and produce inputs for the model."""
    tokens = batch['text'].long().cuda()
    attention_mask = batch['attention_mask'].long().cuda()
    position_ids = batch['position_id'].long().cuda()
    return tokens, attention_mask, position_ids


class DecoderEvaluater:
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.start_token = tokenizer.get_command('sop').Id
        self.end_token = tokenizer.get_command('eop').Id
        self.mask_token = tokenizer.get_command('MASK').Id
        self.pad_token = tokenizer.get_command('pad').Id
        self.processors = LogitsProcessorList()
        if args.min_tgt_length > 0:
            processor = MinLengthLogitsProcessor(args.min_tgt_length, self.end_token)
            self.processors.append(processor)
        if args.no_repeat_ngram_size > 0:
            processor = NoRepeatNGramLogitsProcessor(args.no_repeat_ngram_size)
            self.processors.append(processor)

    def evaluate(self, model, dataloader, example_dict, args):
        """Calculate correct over total answers and return prediction if the
        `output_predictions` is true."""
        model.eval()
        store = torch.distributed.TCPStore(args.master_ip, 18931, mpu.get_data_parallel_world_size(),
                                           torch.distributed.get_rank() == 0, datetime.timedelta(seconds=30))
        print_rank_0("Distributed store created")
        with torch.no_grad():
            # For all the batches in the dataset.
            for idx, data in enumerate(dataloader):
                tokens, attention_mask, position_ids = process_batch(data, args)
                batch_size = tokens.size(0)
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=args.out_seq_length,
                    num_beams=args.num_beams,
                    device=tokens.device,
                    length_penalty=args.length_penalty,
                    do_early_stopping=False,
                )
                beam_scores = torch.zeros((batch_size, args.num_beams), dtype=torch.float, device=tokens.device)
                beam_scores[:, 1:] = -1e9
                beam_scores = beam_scores.view((batch_size * args.num_beams,))
                # Run the model forward.
                counter = 0
                while counter < args.tgt_seq_length:
                    if counter == 0:
                        next_token_logits, *mems = model(tokens, position_ids, attention_mask, return_memory=True)
                        seq_length = next_token_logits.size(1)
                        next_token_logits = next_token_logits[:, -1]
                        next_token_logits = next_token_logits.unsqueeze(1).repeat(1, args.num_beams, 1).view(
                            batch_size * args.num_beams, -1)
                        mems = [mem.unsqueeze(1).repeat(1, args.num_beams, 1, 1).view(batch_size * args.num_beams,
                                                                                      seq_length, -1) for mem in mems]
                        position_ids = tokens.new_ones(batch_size, args.num_beams, 2, 1)
                        for i, text in enumerate(tokens.tolist()):
                            mask_pos = text.index(self.mask_token)
                            position_ids[i, :, 0] = mask_pos
                        position_ids = position_ids.reshape(batch_size * args.num_beams, 2, 1)
                        tokens = tokens.new_zeros(batch_size * args.num_beams, 0)
                        attention_mask = tokens.new_zeros([batch_size * args.num_beams])
                    else:
                        position_ids[:, 1] = counter + 1
                        last_token = tokens[:, -1:]
                        next_token_logits, *mems = model(last_token, position_ids, attention_mask, *mems,
                                                         return_memory=True)
                        next_token_logits = next_token_logits[:, -1]
                    next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                    next_token_scores = self.processors(tokens, next_token_scores)
                    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
                    vocab_size = next_token_scores.shape[-1]
                    next_token_scores = next_token_scores.view(batch_size, args.num_beams * vocab_size)

                    probs = F.softmax(next_token_scores, dim=-1)
                    if args.select_topk:
                        _, next_tokens = torch.topk(probs, k=2 * args.num_beams, dim=-1, largest=True)
                    else:
                        next_tokens = torch.multinomial(probs, num_samples=2 * args.num_beams)
                    next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                    next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                    next_tokens = torch.gather(next_tokens, -1, _indices)

                    next_indices = next_tokens // vocab_size
                    next_tokens = next_tokens % vocab_size
                    # stateless
                    beam_outputs = beam_scorer.process(
                        tokens,
                        next_token_scores,
                        next_tokens,
                        next_indices,
                        eos_token_id=self.end_token,
                        pad_token_id=self.pad_token
                    )
                    beam_scores = beam_outputs["next_beam_scores"]
                    beam_next_tokens = beam_outputs["next_beam_tokens"]
                    beam_idx = beam_outputs["next_beam_indices"]
                    beam_next_tokens = beam_next_tokens.unsqueeze(-1)
                    tokens = torch.cat([tokens[beam_idx, :], beam_next_tokens], dim=-1)
                    mems = [mem[beam_idx] for mem in mems] if mems else []
                    if beam_scorer.is_done:
                        break
                    counter += 1
                tokens, _ = beam_scorer.finalize(tokens, beam_scores, next_tokens, next_indices,
                                                 eos_token_id=self.end_token, pad_token_id=self.pad_token)
                predictions = []
                for text in tokens.tolist():
                    text = [token for token in text if token not in [self.end_token, self.pad_token]]
                    text = self.tokenizer.DecodeIds(text)
                    predictions.append(text)
                uid_list = data['uid']
                if isinstance(uid_list, torch.Tensor):
                    uid_list = uid_list.cpu().numpy().tolist()
                for uid, prediction in zip(uid_list, predictions):
                    store.set(uid, prediction)
                if (idx + 1) % args.log_interval == 0:
                    print_rank_0(f"Iteration {idx + 1} / {len(dataloader)}")
        model.train()
        torch.distributed.barrier()
        print_rank_0("Evaluation completed")
        predictions, examples = [], []
        for uid, example in example_dict.items():
            predictions.append(store.get(uid).decode('utf-8'))
            examples.append(example)
        torch.distributed.barrier()
        return predictions, [], examples
