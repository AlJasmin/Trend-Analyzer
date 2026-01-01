import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from models import CosineClassifier, LinearProjection, Adapter


def _build_metric_heads(num_classes, pos_label):
    if num_classes == 2:
        metric_kwargs = {"task": "binary"}
        auroc = lambda: torchmetrics.AUROC(task="binary")
    else:
        metric_kwargs = {"task": "multiclass", "num_classes": num_classes}
        auroc = lambda: None

    return (
        torchmetrics.Accuracy(**metric_kwargs),
        torchmetrics.F1Score(**metric_kwargs),
        auroc(),
    )


class ImageOnlyMemeCLIP(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(getattr(cfg, "num_classes", 2))
        raw_pos_label = int(getattr(cfg, "pos_label", 1))
        self.pos_label = max(0, min(self.num_classes - 1, raw_pos_label))
        self.is_binary = self.num_classes == 2

        # Metrics per phase
        (
            self.train_acc,
            self.train_f1,
            self.train_auroc,
        ) = _build_metric_heads(self.num_classes, self.pos_label)
        (
            self.val_acc,
            self.val_f1,
            self.val_auroc,
        ) = _build_metric_heads(self.num_classes, self.pos_label)
        (
            self.test_acc,
            self.test_f1,
            self.test_auroc,
        ) = _build_metric_heads(self.num_classes, self.pos_label)

        # Projection and adapter layers operating on precomputed CLIP features
        self.image_map = LinearProjection(
            input_dim=self.cfg.feature_dim,
            output_dim=self.cfg.map_dim,
            num_layers=self.cfg.num_mapping_layers,
            drop_probs=self.cfg.drop_probs,
        )
        self.img_adapter = Adapter(self.cfg.map_dim, 4)

        # Optional MLP before classifier
        pre_output_layers = [nn.Dropout(p=self.cfg.drop_probs[1])]
        pre_output_input_dim = self.cfg.map_dim
        output_input_dim = pre_output_input_dim

        if self.cfg.num_pre_output_layers >= 1:
            pre_output_layers.extend(
                [
                    nn.Linear(pre_output_input_dim, self.cfg.map_dim),
                    nn.ReLU(),
                    nn.Dropout(p=self.cfg.drop_probs[2]),
                ]
            )
            output_input_dim = self.cfg.map_dim

        for _ in range(1, self.cfg.num_pre_output_layers):
            pre_output_layers.extend(
                [
                    nn.Linear(self.cfg.map_dim, self.cfg.map_dim),
                    nn.ReLU(),
                    nn.Dropout(p=self.cfg.drop_probs[2]),
                ]
            )

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.classifier = CosineClassifier(
            feat_dim=output_input_dim,
            num_classes=self.cfg.num_classes,
            dtype=torch.float32,
            scale=getattr(self.cfg, "scale", 30),
        )

        #self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")
        smoothing = float(getattr(self.cfg, "label_smoothing", 0.0))
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            reduction="mean",                                                      # <---------- TO BE ACTIVATES TOMRROW AFTER THE RETRAINING 
            label_smoothing=smoothing,
        )


    def forward(self, image_features):
        image_projection = self.image_map(image_features)
        adapted = self.img_adapter(image_projection)
        mixed = self.cfg.ratio * adapted + (1 - self.cfg.ratio) * image_projection
        normalized = F.normalize(mixed, dim=-1)
        features_pre_output = self.pre_output(normalized)
        logits = self.classifier(features_pre_output)
        return logits

    def common_step(self, batch):
        logits = self.forward(batch["image_features"])
        loss = self.cross_entropy_loss(logits, batch["labels"])
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        pos_scores = probs[:, self.pos_label] if self.is_binary else None
        return {
            "loss": loss,
            "preds": preds,
            "labels": batch["labels"],
            "pos_scores": pos_scores,
        }

    def training_step(self, batch, batch_idx):
        out = self.common_step(batch)
        self.train_acc.update(out["preds"], out["labels"])
        self.train_f1.update(out["preds"], out["labels"])
        if self.train_auroc is not None:
            self.train_auroc.update(out["pos_scores"], out["labels"])

        self.log(
            "train/loss",
            out["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["image_features"].shape[0],
        )
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.common_step(batch)
        self.val_acc.update(out["preds"], out["labels"])
        self.val_f1.update(out["preds"], out["labels"])
        if self.val_auroc is not None:
            self.val_auroc.update(out["pos_scores"], out["labels"])
        self.log(
            "val/loss",
            out["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["image_features"].shape[0],
        )
        return out

    def test_step(self, batch, batch_idx):
        out = self.common_step(batch)
        self.test_acc.update(out["preds"], out["labels"])
        self.test_f1.update(out["preds"], out["labels"])
        if self.test_auroc is not None:
            self.test_auroc.update(out["pos_scores"], out["labels"])
        return out

    def _log_and_reset(self, prefix, acc, f1, auroc):
        self.log(f"{prefix}/accuracy", acc.compute(), prog_bar=True)
        acc.reset()

        self.log(f"{prefix}/f1", f1.compute(), prog_bar=True)
        f1.reset()

        if auroc is not None:
            self.log(f"{prefix}/auroc", auroc.compute(), prog_bar=True)
            auroc.reset()

    def on_train_epoch_end(self):
        self._log_and_reset("train", self.train_acc, self.train_f1, self.train_auroc)

    def on_validation_epoch_end(self):
        self._log_and_reset("val", self.val_acc, self.val_f1, self.val_auroc)

    def on_test_epoch_end(self):
        self._log_and_reset("test", self.test_acc, self.test_f1, self.test_auroc)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        return optimizer


def create_model(cfg):
    return ImageOnlyMemeCLIP(cfg)
