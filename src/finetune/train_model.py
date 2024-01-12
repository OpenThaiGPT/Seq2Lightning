import argparse
import pytorch_lightning as pl

from typing import Optional

from src.finetune import (
    create_wandb_logger,
    SaveTransformersModelCallback,
    CustomLogCallback,
    LitProgressBar,
    Seq2SeqDataModule, LightningSeq2SeqModel
)


def run_training_seq2seq(
                model_name_or_path: str,
                data_source_column: str, data_target_column: str,
                dataset_name: Optional[str] = None,
                train_data_file: Optional[str] = None,
                val_data_file: Optional[str] = None,
                test_data_file: Optional[str] = None,
                configs_path: Optional[str] = None,
                padding: Optional[str] = "max_length",
                max_source_length: Optional[int] = 512, max_target_length: Optional[int] = 512,
                train_batchsize: Optional[int]=4, test_batchsize: Optional[int]=4,
                max_samples: Optional[int]=None, do_cleaning: Optional[bool]=True,
                num_workers: Optional[int]=1,
                optimizer_name: Optional[str] = "Adam", optim_lr: Optional[float] = 1e-4,
                weight_decay: Optional[float] = 0.0,
                scheduler_name: Optional[str] = "linear", scheduler_warmup_steps: Optional[int] = 0,
                accumulate_grad_batches: Optional[int] = 1,
                accelerator: Optional[str] = "auto",
                devices: Optional[int] = 1, strategy: Optional[str] = "auto",
                max_epochs: Optional[int] = 100,
                precision: Optional[str] = "32-true",
                early_stop_patience: int=5,
                checkpoints_path: Optional[str] = None, save_every_n_steps: Optional[int] = None,
                resume_from_chkpt: Optional[str] = None,
                wandb_apikey_path: Optional[str] = None,
                wandb_project: Optional[str] = None,
                wandb_name: Optional[str] = None,
                random_seed: int = 42,
                num_nodes: int = 1,
    ) -> None:
    """
    Run training/finetuning transformers model on Seq2Seq tasks
    """
    pl.seed_everything(random_seed)
    wandb_experiment_params = {
        "model_name_or_path": model_name_or_path,
        "learning_rate": optim_lr,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "accelerator": accelerator,
    }
    wandb_logger = create_wandb_logger(wandb_apikey_path,
                                       wandb_project, wandb_name, **wandb_experiment_params
                                       )
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss",
                                                     patience=early_stop_patience, verbose=True)
    custom_logger = CustomLogCallback()
    callbacks = []
    if checkpoints_path is not None:
        checkpoint_callback = SaveTransformersModelCallback(save_dir=checkpoints_path,
                                                            every_n_train_steps=save_every_n_steps)
        callbacks.append(checkpoint_callback)
        chck = pl.callbacks.ModelCheckpoint(dirpath=checkpoints_path,
                                            save_top_k=-1,
                                            every_n_epochs=1,)
        callbacks.append(chck)
    bar = LitProgressBar()
    model = LightningSeq2SeqModel(
        model_name_or_path, configs_path,
        optimizer_name, optim_lr, weight_decay,
        scheduler_name, scheduler_warmup_steps
    )
    device_stats = pl.callbacks.DeviceStatsMonitor()
    callbacks += [custom_logger, early_stop_callback, bar, device_stats]

    datamodule = Seq2SeqDataModule(
        data_source_column, data_target_column,
        tokenizer=model_name_or_path,
        dataset_name=dataset_name,
        train_data_file=train_data_file, val_data_file=val_data_file,
        test_data_file=test_data_file, padding=padding,
        max_source_length=max_source_length, max_target_length=max_target_length,
        train_batchsize=train_batchsize, test_batchsize=test_batchsize,
        max_samples=max_samples, do_cleaning=do_cleaning,
        num_workers=num_workers,

    )
    trainer = pl.Trainer(accelerator=accelerator, devices=devices,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         accumulate_grad_batches=accumulate_grad_batches,
                         precision=precision,
                         logger=wandb_logger if wandb_logger is not None else True,
                         callbacks=callbacks,
                         num_nodes=num_nodes,
                         )
    if wandb_logger is not None:
        wandb_logger.watch(model, log="gradients")
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from_chkpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        help="Path to pretrained model or model identifier from huggingface")
    parser.add_argument("--configs_path", type=str, default=None,
                        help="Pretrained config path or name if not the same as model_name")
    parser.add_argument("--optimizer_name", type=str, default="Adam",
                        help="Name of optimizer to be used. Should be one of: SGD, Adadelta, "
                             "Adagrad, Adam, AdamW, ASGD, RMSprop")
    parser.add_argument("--optim_lr", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Optional param to control the regularization strength")
    parser.add_argument("--scheduler_name", type=str, default=None,
                        help="Optional name of scheduler from huggingface library to be used. Supported names: "
                             "'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', "
                             "'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau'")
    parser.add_argument("--scheduler_warmup_steps", type=int, default=0,
                        help="Optional number of warmup steps for scheduler")
    parser.add_argument("--data_source_column", type=str,
                        help="The name of the column in the datasets containing input texts")
    parser.add_argument("--data_target_column", type=str,
                        help="The name of the column in the datasets containing target texts")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset name from huggingface library (if a custom dataset is not being used)")
    parser.add_argument("--train_data_file", type=str, default=None, help="The input training data file")
    parser.add_argument("--val_data_file", type=str, default=None,
                        help="An optional input evaluation data file to evaluate the metrics (rouge)")
    parser.add_argument("--test_data_file", type=str, default=None,
                        help="An optional input test data file to evaluate the metrics (rouge)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="The number of processes to use for the preprocessing.")
    parser.add_argument("--max_source_length", type=int, default=400,
                        help="The maximum total input sequence length kept after tokenization")
    parser.add_argument("--max_target_length", type=int, default=400,
                        help="The maximum total sequence length for target text kept after tokenization")
    parser.add_argument("--padding", type=str, default="longest",
                        help="Padding strategy for transformers tokenizer. Supported values: None, longest, max_length")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Truncate the number of examples in dataset to this")
    parser.add_argument("--do_cleaning", type=bool, default=True,
                        help="Whether to clean data before tokenization or tokenize as it is")
    parser.add_argument("--train_batchsize", type=int, default=2, help="Batch size for training")
    parser.add_argument("--test_batchsize", type=int, default=2, help="Batch size for inference")
    parser.add_argument("--accelerator", type=str, default="auto", help="Hardware accelerator to use for training")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to be used")
    parser.add_argument("--strategy", type=str, default="auto", help="Type of PyTorch Lightning strategy to use")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Number of batches to accumulate gradients over before performing the optimization step. "
                             "This can be useful when you have limited memory and need to use larger batch sizes")
    parser.add_argument("--precision", type=str, default="32-true", help="Type of precision to use")
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Number of checks with no improvement after which training will be stopped")
    parser.add_argument("--checkpoints_path", type=str, default=None, help="Path to store model checkpoints")
    parser.add_argument("--save_every_n_steps", type=int, default=50000, help="Number of steps after which checkpoint "
                                                                              "will be saved")
    parser.add_argument("--log_refresh_rate", type=int, default=50, help="Log every n batches")
    parser.add_argument("--resume_from_ckpt", type=str, default=None, help="Path to checkpoint from which resume training")
    parser.add_argument("--wandb_apikey_path", type=str, default=None, help="Path to file with wandb api key."
                                                                            "If not provided, wandb logger won't be used")
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of the W&B project to which this run "
                                                                        "will belong")
    parser.add_argument("--wandb_name", type=str, default=None, help="Name for the W&B run")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudorandom number generator")
    parser.add_argument("--num_nodes", type=int, default=1, help="The number of nodes to compute")
    args = parser.parse_args()
    run_training_seq2seq(**vars(args))
