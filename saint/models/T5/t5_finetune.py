import os
import torch
import pytorch_lightning as pl

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5ForConditionalGeneration, T5Tokenizer

from saint.utils.metric import compute_batched_metrics


class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class T5FineTuner(pl.LightningModule):
    def __init__(self, args, train_dataset, val_dataset):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = T5Tokenizer.from_pretrained(args.model)
        self.model = T5ForConditionalGeneration.from_pretrained(args.model)

    def forward(self, input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                lm_labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch["target_mask"],
            lm_labels=batch["labels"]
        )

        loss = outputs[0]
        self.log(
            'train_loss',
            loss,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch["target_mask"],
            lm_labels=batch["labels"]
        )

        loss = outputs.loss
        ppt = torch.exp(loss)

        pred_outputs = []
        target_outpus = []

        self.model.eval()

        for i in range(self.args.val_batch_size):
            input_ids = torch.unsqueeze(batch["source_ids"][i], dim=0)
            attention_mask = torch.unsqueeze(batch["source_mask"][i], dim=0)
            beam_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=10,
                top_k=20,
                top_p=0.98,
                do_sample=True,
                early_stopping=True,
                num_return_sequences=5,
                no_repeat_ngram_size=2
            )

            preds = [self._tokenizer_decode(beam_output) for beam_output in beam_outputs]
            target_outpus.append(batch["output"][i])
            pred_outputs.append(preds)
            
        self.model.train()

        return {
            "val_loss": loss, 
            "target_outpus": target_outpus, 
            "pred_outputs": pred_outputs, 
            "ppt": ppt
        }
    
    def validation_epoch_end(self, outputs) -> None:
        """End of validation epoch

        :param outputs: the output of the validation step
        :rtype: None
        """

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)

        avg_ppt = torch.stack([x["ppt"] for x in outputs]).mean()
        self.log("val_ppt", avg_ppt, on_epoch=True, prog_bar=True)

        
        target_outpus = [output['target_outpus'] for output in outputs]
        pred_outputs = [output['pred_outputs'] for output in outputs]

        metrics_scores_avg = compute_batched_metrics(pred_outputs, target_outpus)
            
        val_f1_score = metrics_scores_avg["F1"]
        val_em_score = metrics_scores_avg["EM"]
        val_bleu_score = metrics_scores_avg["BLEU-4"]
        val_rouge_score = metrics_scores_avg["ROUGEL"]

        self.log('val_f1', val_f1_score, on_epoch=True, prog_bar=True)
        self.log('val_em', val_em_score, on_epoch=True, prog_bar=True)
        self.log('val_bleu_4', val_bleu_score, on_epoch=True, prog_bar=True)
        self.log('val_rouge_l', val_rouge_score, on_epoch=True, prog_bar=True)


    def save_core_model(self):
        store_path = os.path.join(
            self.args.output_dir,
            f"{self.args.task_split}-{self.args.model}")
        self.model.save_pretrained(store_path)
        self.tokenizer.save_pretrained(store_path)
    
    def _tokenizer_decode(self, output_ids):
        prediction = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up__tokenization_spaces=True
        )
        return prediction

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
