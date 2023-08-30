import numpy as np
import logging
import torch
import torch.nn.functional as F
from torchmetrics.aggregation import MeanMetric

from rouge import Rouge
from typing import Any, List, Tuple, Dict
from transformers.optimization import get_scheduler, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import pytorch_lightning as pl


logger = logging.getLogger(__name__)


OPTIMIZERS_TORCH = {"SGD": torch.optim.SGD,
                    "Adadelta": torch.optim.Adadelta,
                    "Adagrad": torch.optim.Adagrad,
                    "Adam": torch.optim.Adam,
                    "AdamW": torch.optim.AdamW,
                    "ASGD": torch.optim.ASGD,
                    "RMSprop": torch.optim.RMSprop}

def get_optimizer(optimizer_name) -> torch.optim.Optimizer:
    """Retrieve the optimizer class from PyTorch"""
    try:
        return OPTIMIZERS_TORCH[optimizer_name]
    except KeyError:
        print(f"Warning: {optimizer_name} is not a valid optimizer. Using SGD instead.")
        return torch.optim.SGD


class LightningSeq2SeqModel(pl.LightningModule):
    """PytorchLightning module to train Seq2Seq transformers model"""
    def __init__(self, model_name_or_path: str,
                 configs_path: str = None,
                 optimizer_name: str = "Adam", optim_lr: float = 1e-4,
                 weight_decay: float = 0.0, scheduler_name: str = "linear",
                 scheduler_warmup_steps: int = 100,
                 ) -> None:
        super().__init__()
        configs = None
        if configs_path is not None:
            configs = AutoConfig.from_pretrained(configs_path)
        self.save_hyperparameters()
        self.rouge_metric = Rouge()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=configs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       model_max_length=self.model.config.d_model)
        self.train_step_losses = MeanMetric(compute_on_cpu=True)
        self.eval_step_losses = MeanMetric(compute_on_cpu=True)
        self.test_step_outputs = []

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Prepare optimizer and schedule"""
        model_params = self.model.parameters()
        optimizer_func = get_optimizer(self.hparams.optimizer_name)
        optimizer = optimizer_func(model_params,
                                   self.hparams.optim_lr,
                                   weight_decay=self.hparams.weight_decay)
        scheduler_name = self.hparams.scheduler_name
        if scheduler_name is not None:
            scheduler = get_scheduler(scheduler_name,
                                      optimizer=optimizer,
                                      num_warmup_steps=self.hparams.scheduler_warmup_steps,
                                      num_training_steps=self.trainer.estimated_stepping_batches
                                      )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.scheduler_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Perform one training step"""
        outputs = self(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        self.train_step_losses(loss.detach().cpu())
        self.log("train_step_loss", loss, prog_bar=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Perform one validation step"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"]
        }
        outputs = self(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        self.log("val_step_loss", loss, prog_bar=True)
        self.eval_step_losses(loss.detach().cpu())
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Perform one test step"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"]
        }
        outputs = self(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = loss.detach().cpu()
        predicted_probs = F.softmax(outputs["logits"], dim=-1)
        preds = torch.argmax(predicted_probs, dim=-1)
        refs = inputs["input_ids"]
        metrics = self.compute_metric(preds, refs)
        metrics["test_loss"] = loss
        self.test_step_outputs.append(metrics)
        return {"test_loss": loss}

    def compute_metric(self, predictions, references) -> Dict[str, Any]:
        decoded_preds = self.tokenizer.batch_decode(predictions.detach().cpu(),
                                                    skip_special_tokens=True)
        decoded_refs = self.tokenizer.batch_decode(references.detach().cpu(),
                                                   skip_special_tokens=True)
        rouge_scores = self.rouge_metric.get_scores(decoded_preds, decoded_refs, avg=True)
        metrics = {}
        metrics["rouge-1"] = round(rouge_scores["rouge-1"]["f"], 3)
        metrics["rouge-2"] = round(rouge_scores["rouge-2"]["f"], 3)
        metrics["rouge-l"] = round(rouge_scores["rouge-l"]["f"], 3)
        return metrics

    def on_validation_epoch_end(self) -> None:
        self.eval_step_losses.reset()

    def on_train_epoch_end(self) -> None:
        self.train_step_losses.reset()

    def on_test_epoch_end(self) -> None:
        self.test_step_outputs.reset()