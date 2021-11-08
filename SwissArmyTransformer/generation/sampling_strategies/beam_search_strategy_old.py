# -*- encoding: utf-8 -*-
'''
@File    :   base_strategy.py
@Time    :   2021/10/08 22:22:42
@Author  :   Ming Ding
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from collections import UserDict
from typing import Optional, Tuple, List, Iterable, Union


class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for :meth:`~transformers.PretrainedModel.beam_search` and
    :meth:`~transformers.PretrainedModel.beam_sample`.
    """

    @abstractmethod
    def process(
            self,
            input_ids: torch.LongTensor,
            next_scores: torch.FloatTensor,
            next_tokens: torch.LongTensor,
            next_indices: torch.LongTensor,
            **kwargs
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def finalize(
            self,
            input_ids: torch.LongTensor,
            next_scores: torch.FloatTensor,
            next_tokens: torch.LongTensor,
            next_indices: torch.LongTensor,
            **kwargs
    ) -> torch.LongTensor:
        raise NotImplementedError("This is an abstract method.")


class BeamSearchScorer(BeamScorer):
    r"""
    :class:`transformers.BeamScorer` implementing standard beam search decoding.

    Adapted in part from `Facebook's XLM beam search code
    <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

    Args:
        batch_size (:obj:`int`):
            Batch Size of :obj:`input_ids` for which beam search decoding is run in parallel.
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        num_beams (:obj:`int`):
            Number of beams for beam search.
        device (:obj:`torch.device`):
            Defines the device type (*e.g.*, :obj:`"cpu"` or :obj:`"cuda"`) on which this instance of
            :obj:`BeamSearchScorer` will be allocated.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beam_hyps_to_keep (:obj:`int`, `optional`, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            :meth:`~transformer.BeamSearchScorer.finalize`.
    """

    def __init__(
            self,
            batch_size: int,
            max_length: int,
            num_beams: int,
            device: Union[torch.device, str],
            length_penalty: Optional[float] = 1.0,
            do_early_stopping: Optional[bool] = False,
            num_beam_hyps_to_keep: Optional[int] = 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        # if not isinstance(num_beams, int) or num_beams <= 1:
        #     raise ValueError(
        #         f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
        #     )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
            self,
            input_ids: torch.LongTensor,
            next_scores: torch.FloatTensor,
            next_tokens: torch.LongTensor,
            next_indices: torch.LongTensor,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            mems=None
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.num_beams)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        device = next_scores.device
        next_beam_scores = torch.zeros((batch_size, self.num_beams), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.num_beams), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.num_beams), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                        len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                        eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.num_beams + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        mems=[mem[[next_index.item()]] for mem in mems] if mems else None
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.num_beams:
                    break

            if beam_idx < self.num_beams:
                raise ValueError(
                    f"At most {self.num_beams} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
            self,
            input_ids: torch.LongTensor,
            final_beam_scores: torch.FloatTensor,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            mems=None
    ) -> Tuple[torch.LongTensor, List[torch.Tensor], torch.FloatTensor]:
        batch_size = len(self._beam_hyps)
        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score, mems=[mem[[batch_beam_idx]] for mem in mems] if mems else None)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                score, best_hyp, mems = sorted_hyps.pop()
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append((best_hyp, mems, score))

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item(), self.max_length)
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        scores = final_beam_scores.new(batch_size * self.num_beam_hyps_to_keep)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        mems = []
        for i, (hypo, mem, score) in enumerate(best):
            scores[i] = score
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id
            mems.append(mem)
        mems = [torch.cat([mem[i] for mem in mems], dim=0) for i in range(len(mems[0]))] if mems and mems[0] else None
        return decoded, mems, scores


class BeamHypotheses:
    def __init__(self, num_beams: int, max_length: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float, mems=None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (max(hyp.shape[-1], 1) ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, mems))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class LogitsProcessor(ABC):
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of :class:`~transformers.LogitsProcessor` or
    :class:`~transformers.LogitsWarper` to subsequently process a :obj:`scores` input tensor. This class inherits from
    list and adds a specific `__call__` method to apply each :class:`~transformers.LogitsProcessor` or
    :class:`~transformers.LogitsProcessor` to the inputs.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (:obj:`int`):
            The minimum length below which the score of :obj:`eos_token_id` is set to :obj:`-float("Inf")`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length: int, eos_token_ids: Union[List[int], int]):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        for eos_token_id in eos_token_ids:
            if not isinstance(eos_token_id, int) or eos_token_id < 0:
                raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_ids = eos_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            for eos_token_id in self.eos_token_ids:
                scores[:, eos_token_id] = -float("inf")
        return scores


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` that enforces no repetition of n-grams. See `Fairseq
    <https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345>`__.

    Args:
        ngram_size (:obj:`int`):
            All ngrams of size :obj:`ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = self._calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores

    def _calc_banned_ngram_tokens(
            self, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
    ) -> List[Iterable[int]]:
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < self.ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(self.ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # Before decoding the next token, prevent decoding of ngrams that have already appeared
            start_idx = cur_len + 1 - self.ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
        return banned_tokens


class BeamSearchStrategy:
    def __init__(self, num_beams, max_length, length_penalty, end_tokens, device='cuda', no_repeat_ngram_size=0,
                 min_tgt_length=0):
        self.num_beams = num_beams
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.end_tokens = end_tokens
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.min_tgt_length = min_tgt_length
        self.processors = LogitsProcessorList()
        if min_tgt_length > 0:
            processor = MinLengthLogitsProcessor(min_tgt_length, self.end_tokens)
            self.processors.append(processor)
        if no_repeat_ngram_size > 0:
            processor = NoRepeatNGramLogitsProcessor(no_repeat_ngram_size)
            self.processors.append(processor)
        self.beam_scorer = BeamSearchScorer(
            batch_size=1,
            max_length=max_length,
            num_beams=num_beams,
            device=device,
            length_penalty=length_penalty,
            do_early_stopping=False,
        )
        self.beam_scores = torch.zeros(1, dtype=torch.float, device=device)

    @property
    def is_done(self) -> bool:
        return self.beam_scorer.is_done

    def forward(self, logits, tokens, mems):
        last_beam_num = tokens.size(0)
        logits = self.processors(tokens, logits.float())
        next_token_scores = F.log_softmax(logits, dim=-1)
        next_token_scores = next_token_scores + self.beam_scores[:, None].expand_as(next_token_scores)
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(1, last_beam_num * vocab_size)

        probs = F.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=2 * self.num_beams)
        next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
        next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
        next_tokens = torch.gather(next_tokens, -1, _indices)

        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size
        # stateless
        tokens = tokens.expand((self.num_beams, -1))
        beam_outputs = self.beam_scorer.process(
            tokens,
            next_token_scores,
            next_tokens,
            next_indices,
            eos_token_id=self.end_tokens,
            mems=mems
        )
        self.beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]
        beam_next_tokens = beam_next_tokens.unsqueeze(-1)
        tokens = torch.cat([tokens[beam_idx, :], beam_next_tokens], dim=-1)
        mems = [mem[beam_idx] for mem in mems] if mems else None
        return tokens, mems

    def finalize(self, tokens, mems):
        tokens, mems, scores = self.beam_scorer.finalize(tokens, self.beam_scores,
                                                         eos_token_id=self.end_tokens[0],
                                                         mems=mems)
        return tokens, mems
