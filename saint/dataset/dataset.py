import copy

from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)


class MultiTaskDataLoader(DataLoader):
    def __init__(self, dataset, is_training,
                 train_batch_size, val_batch_size):
        if is_training:
            sampler = RandomSampler(dataset)
            batch_size = train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = val_batch_size
        super(MultiTaskDataLoader, self).__init__(
            dataset, sampler=sampler, batch_size=batch_size)


class AlignmentGenerationDataset(Dataset):
    def __init__(self, tokenizer, examples, max_len_inp=128, max_len_out=128):

        self.datalist = examples

        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.skippedcount = 0
        self._build()

        self.dataloader = None

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
            "labels": labels,
            "input": self.datalist[index]["input"],
            "output": self.datalist[index]["output"]
        }

    def _build(self):
        for inputs in self.datalist:
            input_sent = inputs["input"]
            output_sent = inputs["output"]

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_sent], max_length=self.max_len_input, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [output_sent], max_length=self.max_len_output, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

    def load_dataloader(self, dataset, is_training, train_batch_size, val_batch_size):
        self.dataloader = MultiTaskDataLoader(
            dataset, is_training,
            train_batch_size, val_batch_size
        )
