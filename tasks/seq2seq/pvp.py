from tasks.data_utils import InputExample


class PVP:
    def __init__(self, tokenizer, max_src_length, max_tgt_length, task_mask=False):
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.task_mask = task_mask

    @property
    def cls_id(self):
        return self.tokenizer.get_command('ENC').Id

    @property
    def mask_id(self):
        return self.tokenizer.get_command('MASK').Id

    def encode(self, example: InputExample):
        raise NotImplementedError


class SummaryPVP(PVP):
    @property
    def mask_id(self):
        mask_token = 'sMASK' if self.task_mask else 'MASK'
        return self.tokenizer.get_command(mask_token).Id

    def encode(self, example: InputExample):
        source_text, target_text = example.text_a, example.text_b
        source_tokens = self.tokenizer.EncodeAsIds(" " + source_text).tokenization
        prompt = [self.cls_id, self.mask_id] + self.tokenizer.EncodeAsIds(" Content:").tokenization
        if len(source_tokens) > self.max_src_length - len(prompt):
            source_tokens = source_tokens[:self.max_src_length - len(prompt)]
        source_tokens = prompt + source_tokens
        return source_tokens, target_text


class QuesGenPVP(PVP):
    @property
    def mask_id(self):
        mask_token = 'sMASK' if self.task_mask else 'MASK'
        return self.tokenizer.get_command(mask_token).Id

    def encode(self, example: InputExample):
        source_text = example.text_a
        target_text, answer = example.meta["question"], example.meta["answer"]
        source_tokens = self.tokenizer.EncodeAsIds(source_text.rstrip() + " Question:").tokenization
        answer_tokens = self.tokenizer.EncodeAsIds(" Answer: " + answer).tokenization
        if len(source_tokens) > self.max_src_length - len(answer_tokens) - 2:
            max_src_length = self.max_src_length - len(answer_tokens) - 2
            answer_pattern = self.tokenizer.EncodeAsIds(" " + answer).tokenization

            def sub_finder(mylist, pattern):
                matches = []
                for i in range(len(mylist)):
                    if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
                        matches.append(i)
                return matches

            answer_indices = sub_finder(source_tokens, answer_pattern)
            if len(answer_indices) == 0:
                print(f"Answer {answer} not exists in the source text")
                source_tokens = source_tokens[:max_src_length]
            else:
                start_index = max(answer_indices[0] - max_src_length // 2, 0)
                source_tokens = source_tokens[start_index: start_index + max_src_length]
        source_tokens = [self.cls_id] + source_tokens + [self.mask_id] + answer_tokens
        return source_tokens, target_text


class ChineseQAPVP(PVP):
    def encode(self, example: InputExample):
        source_text = example.text_a
        target_text = example.meta["answer"].strip()
        question = example.meta["question"].strip()
        source_tokens = self.tokenizer.EncodeAsIds(source_text.rstrip()).tokenization
        question_tokens = self.tokenizer.EncodeAsIds("问题：" + question + "答案：").tokenization
        max_src_length = self.max_src_length - len(question_tokens) - 2
        if max_src_length <= 0:
            print(question)
            question_tokens = question_tokens[self.max_src_length // 4]
        source_tokens = [self.cls_id] + question_tokens + [self.mask_id] + source_tokens[:max_src_length]
        return source_tokens, target_text

PVPS = {
    "gigaword": SummaryPVP,
    "cnn_dm": SummaryPVP,
    "cnn_dm_original": SummaryPVP,
    "xsum": SummaryPVP,
    "squad_generation": QuesGenPVP,
    "cmrc": ChineseQAPVP
}