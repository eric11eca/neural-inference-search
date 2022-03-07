import os
import copy
import torch
import saint.utils.py_io as py_io

from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)


class QADataset(Dataset):
    def __init__(self, inputs_tokenized, outputs_tokenized):
        
        self.inputs_tokenized = inputs_tokenized
        self.outputs_tokenized = outputs_tokenized

    def __len__(self):
        return len(self.inputs_tokenized)

    def __getitem__(self, index):
        source_ids = self.inputs_tokenized[index]["input_ids"].squeeze()
        target_ids = self.outputs_tokenized[index]["input_ids"].squeeze()

        src_mask = self.inputs_tokenized[index]["attention_mask"].squeeze()
        target_mask = self.outputs_tokenized[index]["attention_mask"].squeeze()

        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "labels": labels
        }


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


class MultiTaskDataset:
    def __init__(
        self, logger, data_path, tasks, data_split, 
        max_len_input, max_len_output, 
        train_batch_size, val_batch_size, is_training
    ):
        self.data_path = data_path
        self.data = []

        for task in sorted(tasks):
            task_dir = os.path.join(self.data_path, task)
            
            prefixes = ["nli", "evid"]

            for prefix in prefixes:
                train_file_pth = os.path.join(
                    task_dir, f"{prefix}_train.jsonl")
                train_examples = py_io.read_jsonl(path=train_file_pth)

                dev_file_pth = os.path.join(
                    task_dir, f"{prefix}_val.jsonl")
                dev_examples = py_io.read_jsonl(path=dev_file_pth)

                self.data.append({
                    "task_name": task,
                    "task_prefix": prefix,
                    "train_examples": train_examples,
                    "dev_examples": dev_examples
                })

        self.data_split = data_split
        self.is_training = is_training
        self.logger = logger

        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.load = True
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def _build_cache_path(self, tokenizer):
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        return os.path.join(
            self.data_path,
            f"{self.data_split}-multi-{postfix}.pt"
        )

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        preprocessed_path = self._build_cache_path(tokenizer)

        if self.load and os.path.exists(preprocessed_path):
            self.logger.info(
                f"Loading pre-tokenized data from {preprocessed_path}")
            tokenized = torch.load(preprocessed_path)
            inputs_tokenized = tokenized["inputs_tokenized"]
            outputs_tokenized = tokenized["outputs_tokenized"]
        else:
            self.logger.info(
                f"Start tokenizing ... {len(self.data)} instances")

            inputs = []
            outputs = []
            inputs_tokenized = []
            outputs_tokenized = []

            for task in self.data:
                if self.data_split in ["train", "all"]:
                    for dp in task["train_examples"]:
                        inputs.append(dp["prompt"])
                        outputs.append(dp["output"])
                if self.data_split in ["val", "all"]:
                    for dp in task["dev_examples"]:
                        inputs.append(dp["prompt"])
                        outputs.append(dp["output"])

            self.logger.info("Printing 3 examples")
            for i in range(3):
                self.logger.info(inputs[i])
                self.logger.info(outputs[i])

            # if self.args.do_lowercase:
            #     inputs = [input0.lower() for input0 in inputs]
            #     outputs = [output0.lower() for output0 in outputs]
            # if self.args.append_another_bos:
            #     inputs = [f"<s> {input0}" for input0 in inputs]
            #     outputs = [f"<s> {output0}" for output0 in outputs]

            self.logger.info("Tokenizing data points ...")
            for prompt, output in zip(inputs, outputs):
                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [prompt], 
                    max_length=self.max_len_input, 
                    pad_to_max_length=True, 
                    return_tensors="pt"
                )

                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [output], 
                    max_length=self.max_len_output, 
                    pad_to_max_length=True, 
                    return_tensors="pt"
                )

                inputs_tokenized.append(tokenized_inputs)
                outputs_tokenized.append(tokenized_targets)

            if self.load:
                preprocessed_data = {
                    "inputs_tokenized": inputs_tokenized,
                    "outputs_tokenized": outputs_tokenized
                }
                torch.save(preprocessed_data, preprocessed_path)

        self.dataset = QADataset(inputs_tokenized, outputs_tokenized)
        self.logger.info(
            f"Loaded {len(self.dataset)} examples from {self.data_split} data")

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MultiTaskDataLoader(
            self.dataset, self.is_training,
            self.train_batch_size, self.val_batch_size
        )
        if do_return:
            return self.dataloader

    def save_predictions(self, predictions):
        assert len(predictions) == len(self), (len(predictions), len(self))

        predictions = [
            'n/a' if len(prediction.strip()) ==
            0 else prediction for prediction in predictions
        ]
        prediction_text = [
            prediction.strip()+'\n' for prediction in predictions]
        save_path = os.path.join(
            self.output_dir,
            f"{self.prefix}_predictions.txt")
        with open(save_path, "w") as f:
            f.writelines(prediction_text)

        self.logger.info(f"Saved prediction in {save_path}")