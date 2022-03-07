import copy

from torch.utils.data import Dataset


class AlignmentGenerationDataset(Dataset):
    def __init__(self, tokenizer, examples, max_len_inp=128, max_len_out=128):

        self.alignment_pairs = examples

        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.skippedcount = 0
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "labels": labels
        }

    def _build(self):
        for inputs in self.alignment_pairs:
            premise = inputs["premise"]
            source = inputs['source']

            input_sent = f"List evidence: {premise} Using only the above description and what you know about the world, " + \
                f"{source} is definitely correct </s>"
            input_sent = f"nli: Given {premise} Should we assume that {source} is true? </s>"
            ouput_sent = inputs['target']

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_sent], max_length=self.max_len_input, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [ouput_sent], max_length=self.max_len_output, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
