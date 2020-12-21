import glob
import json
import os
import time

from torch.utils.data import Dataset

from utils import print_rank_0
from tasks.data_utils import build_sample
from tasks.data_utils import build_block_input_from_ids, build_bert_input_from_ids
from tasks.data_utils import clean_text

NUM_CHOICES = 4
MAX_QA_LENGTH = 128


class RaceDataset(Dataset):

    def __init__(self, dataset_name, datapaths, tokenizer, max_seq_length, max_qa_length=MAX_QA_LENGTH, is_bert=False,
                 pool_token=None, cloze_format=False):

        self.dataset_name = dataset_name
        print_rank_0(' > building RACE dataset for {}:'.format(
            self.dataset_name))

        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)

        self.samples = []
        for datapath in datapaths:
            self.samples.extend(process_single_datapath(datapath, tokenizer,
                                                        max_qa_length,
                                                        max_seq_length, is_bert=is_bert, pool_token=pool_token,
                                                        cloze_format=cloze_format))

        print_rank_0('  >> total number of samples: {}'.format(
            len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def process_single_datapath(datapath, tokenizer, max_qa_length, max_seq_length, pool_token, is_bert, cloze_format):
    """Read in RACE files, combine, clean-up, tokenize, and convert to
    samples."""

    print_rank_0('   > working on {}'.format(datapath))
    start_time = time.time()

    # Get list of files.
    filenames = glob.glob(os.path.join(datapath, '*.txt'))

    samples = []
    num_docs = 0
    num_questions = 0
    num_samples = 0
    # Load all the files
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                num_docs += 1

                context = data["article"]
                questions = data["questions"]
                choices = data["options"]
                answers = data["answers"]
                # Check the length.
                assert len(questions) == len(answers)
                assert len(questions) == len(choices)

                mask_id = tokenizer.get_command('MASK').Id
                eos_id = tokenizer.get_command('eos').Id
                sop_id = tokenizer.get_command('sop').Id
                eop_id = tokenizer.get_command('eop').Id
                cls_id = tokenizer.get_command('ENC').Id

                # Context: clean up and convert to ids.
                context = clean_text(context)
                context_ids = tokenizer.EncodeAsIds(context).tokenization

                # Loop over questions.
                for qi, question in enumerate(questions):
                    num_questions += 1
                    # Label.
                    label = ord(answers[qi]) - ord("A")
                    question = "Question: " + question
                    assert label >= 0
                    assert label < NUM_CHOICES
                    assert len(choices[qi]) == NUM_CHOICES

                    # For each question, build num-choices samples.
                    if is_bert:
                        ids_list, types_list, paddings_list = [], [], []
                    else:
                        ids_list, positions_list, sep_list = [], [], []
                        mask_list, target_list = [], []
                    for ci in range(NUM_CHOICES):
                        choice = choices[qi][ci]
                        # Merge with choice.
                        if cloze_format:
                            if "_" in question:
                                left, right = question.split('_', maxsplit=1)
                                left_ids, right_ids = tokenizer.EncodeAsIds(left).tokenization, tokenizer.EncodeAsIds(
                                    right).tokenization
                                mask_position = len(left_ids)
                                q_ids = left_ids + [mask_id] + right_ids
                            else:
                                q_ids = tokenizer.EncodeAsIds(question).tokenization
                                mask_position = len(q_ids)
                                q_ids = q_ids + [mask_id]
                            answer_ids = tokenizer.EncodeAsIds(choice).tokenization
                            answer_ids = [sop_id] + answer_ids + [eop_id]
                            if len(q_ids) + len(answer_ids) > max_qa_length + 1:
                                if max_qa_length + 1 - len(q_ids) < 2:
                                    q_ids = q_ids[:max_qa_length - 1]
                                strip_tokens = len(q_ids) + len(answer_ids) - max_qa_length - 1
                                answer_ids = answer_ids[:-strip_tokens]
                            if len(context_ids) + 1 + len(q_ids) + len(answer_ids) > max_seq_length:
                                context_ids = context_ids[:max_seq_length - len(q_ids) - len(answer_ids) - 1]
                            mask_position += len(context_ids) + 1
                            ids = [cls_id] + context_ids + q_ids + [eos_id]
                            sep = len(ids)
                            position_ids = list(range(len(ids)))
                            block_position_ids = [0] * len(ids)
                            loss_masks = [0] * len(ids)
                            target_ids = [0] * len(ids)
                            ids = ids + answer_ids[:-1]
                            position_ids = position_ids + [mask_position] * (len(answer_ids) - 1)
                            block_position_ids += range(1, len(answer_ids))
                            loss_masks += [1] * (len(answer_ids) - 1)
                            target_ids += answer_ids[1:]
                            # Padding.
                            padding_length = max_seq_length - len(ids)
                            if padding_length > 0:
                                ids.extend([eos_id] * padding_length)
                                position_ids.extend([position_ids[-1]] * padding_length)
                                block_position_ids.extend(range(2, padding_length + 2))
                                loss_masks += [0] * padding_length
                                target_ids += [0] * padding_length
                            position_ids = [position_ids, block_position_ids]
                            ids_list.append(ids)
                            positions_list.append(position_ids)
                            sep_list.append(sep)
                            mask_list.append(loss_masks)
                            target_list.append(target_ids)
                        else:
                            if "_" in question:
                                qa = question.replace("_", choice)
                            else:
                                qa = " ".join([question, choice])
                            # Clean QA.
                            qa = clean_text(qa)
                            # Tokenize.
                            qa_ids = tokenizer.EncodeAsIds(qa).tokenization
                            if len(qa_ids) > max_qa_length:
                                qa_ids = qa_ids[0:max_qa_length]
                            # Trim if needed.
                            if is_bert:
                                # Build the sample.
                                ids, types, paddings = build_bert_input_from_ids(qa_ids, context_ids, max_seq_length,
                                                                                 tokenizer.get_command('ENC').Id,
                                                                                 tokenizer.get_command('sep').Id,
                                                                                 tokenizer.get_command('pad').Id)
                                ids_list.append(ids)
                                types_list.append(types)
                                paddings_list.append(paddings)
                            else:
                                input_ids = context_ids + qa_ids
                                # Build the sample.
                                ids, position_ids, mask \
                                    = build_block_input_from_ids(input_ids, max_seq_length,
                                                                 cls_id=tokenizer.get_command('ENC').Id,
                                                                 mask_id=tokenizer.get_command('MASK').Id,
                                                                 start_id=tokenizer.get_command('sop').Id,
                                                                 pad_id=tokenizer.get_command('pad').Id,
                                                                 pool_token=pool_token)

                                ids_list.append(ids)
                                positions_list.append(position_ids)
                                sep_list.append(mask)
                    # Convert to numpy and add to samples
                    if is_bert:
                        samples.append(build_sample(ids_list, types=types_list, paddings=paddings_list, label=label,
                                                    unique_id=num_samples))
                    elif cloze_format:
                        samples.append(build_sample(ids_list, positions=positions_list, masks=sep_list, label=label,
                                                    logit_mask=mask_list, target=target_list,
                                                    unique_id=num_samples))
                    else:
                        samples.append(build_sample(ids_list, positions=positions_list, masks=sep_list, label=label,
                                                    unique_id=num_samples))
                    num_samples += 1

    elapsed_time = time.time() - start_time
    print_rank_0('    > processed {} document, {} questions, and {} samples'
                 ' in {:.2f} seconds'.format(num_docs, num_questions,
                                             num_samples, elapsed_time))

    return samples
