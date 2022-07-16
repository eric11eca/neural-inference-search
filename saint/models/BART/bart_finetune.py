import os
import pytorch_lightning as pl

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration, BartTokenizer

class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

class BartFineTuner(pl.LightningModule):
    def __init__(self, args, train_dataset, val_dataset):
        super(BartFineTuner, self).__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = BartTokenizer.from_pretrained(args.model)
        self.model = BartForConditionalGeneration.from_pretrained(args.model)

    def forward(self, input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                lm_labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids = batch["target_ids"],
            decoder_attention_mask=batch["target_mask"],
            lm_labels=batch["labels"]
        )

        loss = outputs[0]
        self.log('train_loss',loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids = batch["target_ids"],
            decoder_attention_mask=batch["target_mask"],
            lm_labels=batch["labels"]
        )

        loss = outputs[0]
        self.log("val_loss",loss)
        return loss

    def save_core_model(self):
        store_path = os.path.join(
            self.args.output_dir,
            f"{self.args.task_split}-{self.args.model}")
        self.model.save_pretrained(store_path)
        self.tokenizer.save_pretrained(store_path)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            eps=self.hparams.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_epochs * len(self.train_dataset))
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_dataloader(self):
        return self.train_dataset.dataloader

    def val_dataloader(self):
        return self.val_dataset.dataloader