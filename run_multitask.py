import warnings
import os
import argparse
import logging
import wandb
import pytorch_lightning as pl

from transformers import AutoTokenizer

from saint.utils import py_io
from saint.utils.utils import set_seed
from saint.models.T5.t5_finetune import T5FineTuner, MetricsCallback
from saint.models.T5.t5_generate import T5AlignmentGenerator
from saint.dataset.multi_task_dataset import MultiTaskDataset
from saint.dataset.dataset import AlignmentGenerationDataset

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar

wandb.init(
    project="causal_scaffold_modeling", 
    name="casul_rewrite_1"
)
wandb_logger = WandbLogger(
    project="causal_scaffold_modeling", 
    name="casul_rewrite_1"
)
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

set_seed(42)

task_splits = {
    "snli": ["snli"],
    "anli": ["anli"],
    "nlsat": ["nlsat"],
    "logiqa": ["logiqa"],
    "control": ["control"],
    "analytic": ["analytic"],
    "crowd_sourced": ["snli", "anli"],
    "logic": ["nlsat", "logiqa"],
    "complex": ["analytic", "control"]
}


def build_datasets(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tasks = task_splits[args.task_split]

    train_multi_data = MultiTaskDataset(
        logger, data_path=args.data_dir,
        tasks=tasks, data_split="train",
        max_len_input=args.max_len_input,
        max_len_output=args.max_len_output,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        is_training=True
    )
    dev_multi_data = MultiTaskDataset(
        logger, data_path=args.data_dir,
        tasks=tasks, data_split="dev",
        max_len_input=args.max_len_input,
        max_len_output=args.max_len_output,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        is_training=False
    )

    train_multi_data.load_dataset(tokenizer)
    train_multi_data.load_dataloader()

    dev_multi_data.load_dataset(tokenizer)
    dev_multi_data.load_dataloader()

    return train_multi_data, dev_multi_data


def train(model, train_params, checkpoint_pth):
    trainer = pl.Trainer(**train_params)

    logger.info("Training model ...")
    trainer.fit(model)

    logger.info("Saving model ...")
    model.save_core_model()

    logger.info(f"Model Saved at {checkpoint_pth}")


def evaluate(task_split, model, predict_dir):
    t5_saint_generator = T5AlignmentGenerator(model)
    datasets = task_splits[task_split]

    for dataset in datasets:
        nli_test_data = py_io.read_jsonl(f"data/{dataset}/dev.jsonl")
        nli_output = t5_saint_generator.generate(nli_test_data)

        evid_test_data = py_io.read_jsonl(f"data/{dataset}/test.jsonl")
        evid_output = t5_saint_generator.generate(evid_test_data)

        py_io.write_jsonl(nli_output, os.path.join(
            predict_dir, "nli_pred.jsonl"))
        py_io.write_jsonl(evid_output, os.path.join(
            predict_dir, "evid_pred.jsonl"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model", default="t5-base")
    parser.add_argument("--task_split", default="causal_snli/causal_nli_entail")
    parser.add_argument("--output_dir", default="runs")

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')

    # Model parameters
    parser.add_argument("--checkpoint", action='store_true')
    parser.add_argument('--max_len_input', type=int, default=128)
    parser.add_argument('--max_len_output', type=int, default=64)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--val_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--lr", default=3e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="gradient will be accumulated over this many steps.")
    parser.add_argument("--max_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--fp_16", action='store_true')

    args = parser.parse_args()
    logger.info(args)

    pred_dir = os.path.join(args.output_dir, f"{args.task_split}-{args.model}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=pred_dir,
        filename="best-model",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    metrics_callback = MetricsCallback()

    train_params = dict(
        gpus=1,
        logger=wandb_logger,
        log_every_n_steps=50,
        max_epochs=20,
        progress_bar_refresh_rate=10,
        precision= 16 if args.fp_16 else 32,
        # gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        checkpoint_callback=True,
        callbacks=[metrics_callback, checkpoint_callback, RichProgressBar()]
    )

    train_data = py_io.read_jsonl(
        "./data/causal_snli/causal_snli_entail/train.jsonl")
    val_data = py_io.read_jsonl("./data/causal_snli/causal_snli_entail/dev.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_multi_data = AlignmentGenerationDataset(tokenizer, train_data)
    dev_multi_data = AlignmentGenerationDataset(tokenizer, val_data)

    train_multi_data.load_dataloader(train_multi_data, True, 4, 2)
    dev_multi_data.load_dataloader(dev_multi_data, False, 4, 2)

    #train_multi_data, dev_multi_data = build_datasets(args)
    model = T5FineTuner(
        args, train_dataset=train_multi_data,
        val_dataset=dev_multi_data
    )

    if args.do_train:
        train(model, train_params, pred_dir)

    if args.do_eval:
        evaluate(args.task_split, model, pred_dir)
