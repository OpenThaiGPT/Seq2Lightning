import logging
import os
import wandb
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
import transformers

from ..utils import extract_key_from_file

logger = logging.getLogger()


def create_wandb_logger(apikey_path=None, project=None,
                        run_name=None, **params):
    """
    Create wandb logger for experiment logging
    """
    wandb_logger = None
    config = {
        "model": params["model_name_or_path"],
        "learning_rate": params["learning_rate"],
        "optimizer": params["optimizer"],
        "scheduler": params["scheduler"],
        "accelerator": params["accelerator"],
    }
    if apikey_path is not None:
        try:
            apikey = extract_key_from_file(apikey_path)
            wandb.login(key=apikey)
            wandb_logger = pl.loggers.WandbLogger(project=project, name=run_name,
                                                  config=config)
        except Exception as e:
            logger.error(f"Error while loading wandb logger: {e}")
            print(f"Error while loading wandb {e}")
    return wandb_logger


class LitProgressBar(pl.callbacks.ProgressBar):
    """
    Custom Keras-like progress bar
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bar = None
        self.running_loss = 0.0
        self.steps = 0
        self.enabled = True

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if self.enabled:
            self.bar = tqdm(total=self.total_train_batches,
                            desc=f"Epoch {trainer.current_epoch + 1}",
                            position=0,
                            leave=True)
            self.running_loss = 0.0
            self.steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.bar:
            self.running_loss += outputs['loss'].item()
            self.steps += 1
            self.bar.update(1)
            avg_loss = self.running_loss / self.steps
            loss = outputs['loss'].item()
            self.bar.set_postfix(avg_loss=f'{avg_loss:.3f}', current_loss=f'{loss:.3f}')

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.bar:
            val_loss = pl_module.eval_step_losses.compute()
            loss = self.running_loss / self.steps
            self.bar.set_postfix(loss=f'{loss:.3f}', val_loss=f'{val_loss:.3f}')
            self.bar.close()
            self.bar = None

    def disable(self):
        self.bar = None


class CustomLogCallback(pl.Callback):
    """
    Callback to show some additional logs
    """
    def on_test_epoch_end(self, trainer, pl_module):
        results = {}
        outputs = pl_module.test_step_outputs
        results["test_loss"] = np.mean([output["test_loss"] for output in outputs])
        results["test_rouge-1"] = np.mean([output['rouge-1'] for output in outputs])
        results["test_rouge_2"] = np.mean([output['rouge-2'] for output in outputs])
        results["test_rouge_l"] = np.mean([output['rouge-l'] for output in outputs])
        pl_module.log_dict(results, prog_bar=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        avg_loss = pl_module.eval_step_losses.compute()
        pl_module.log("val_loss", avg_loss, prog_bar=True)


class SaveTransformersModelCallback(pl.Callback):
    """
    Callback to save transformers model after every n_steps of training
    """
    def __init__(self, save_dir: str, every_n_train_steps: int) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.every_n_train_steps = every_n_train_steps
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_batch_end(self, trainer: pl.Trainer, \
                           pl_module, outputs, batch, batch_idx) -> None:
        current_step = trainer.global_step * trainer.accumulate_grad_batches
        if current_step % self.every_n_train_steps == 0:
            self._save_model(pl_module.model, pl_module.tokenizer, trainer.global_step)

    def _save_model(self, model: transformers.PreTrainedModel,
                    tokenizer: transformers.PreTrainedTokenizer,
                    global_step: int) -> None:
        checkpoint_dir = os.path.join(self.save_dir, f"step_{global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)