from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from lion_pytorch import Lion
from matplotlib import pyplot as plt
from torch import nn


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        assert (
            frames % frame_patch_size == 0
        ), "Frames must be divisible by frame patch size"

        num_patches = (
            (image_height // patch_height)
            * (image_width // patch_width)
            * (frames // frame_patch_size)
        )
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)",
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer3 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, video):
        visual_output = self.to_patch_embedding(video)
        b, n, _ = visual_output.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, visual_output), dim=1)
        x += self.pos_embedding[:, : (n + 3)]
        x = self.dropout(x)

        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class ViTClassifier3D(pl.LightningModule):
    def __init__(
        self,
        name: str,
        lr: float = 1e-3,
        image_patch_size: int = 50,
        frame_patch_size: int = 10,
        dim: int = 1024,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        pool: str = "cls",
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        class_weights: Optional[list] = None,
        weight_decay: float = 0.0,
        scheduler_gamma: float = 0.1,
        scheduler_step_size: int = 25,
        optimizer_alg: str = "adam",
    ):
        super().__init__()
        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.classes = ["AD", "MCI", "CN"]
        self.optimizer_alg = optimizer_alg

        if class_weights is not None:
            class_weights = torch.Tensor(class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.val_conf_matrix = ConfusionMatrixPloter(classes=self.classes)
        self.train_conf_matrix = ConfusionMatrixPloter(classes=self.classes)

        self.model = ViT(
            image_size=100,
            image_patch_size=image_patch_size,
            frames=120,
            frame_patch_size=frame_patch_size,
            num_classes=3,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            pool=pool,
            channels=1,
            dim_head=dim_head,
        )
        self.model.mlp_head = nn.Sequential(self.model.mlp_head, nn.Softmax(dim=1))

    def forward(self, image):
        x = self.model(image)
        return x

    def configure_optimizers(self):
        if self.optimizer_alg == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_alg == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_alg == "lion":
            optimizer = Lion(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        if self.scheduler_gamma is not None and self.scheduler_step_size is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)

        class_predictions = logits.argmax(dim=1)
        preds = torch.zeros_like(logits)
        preds[torch.arange(logits.shape[0]), class_predictions] = 1
        self.train_conf_matrix.update(preds, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)

        class_predictions = logits.argmax(dim=1)
        preds = torch.zeros_like(logits)
        preds[torch.arange(logits.shape[0]), class_predictions] = 1

        self.val_conf_matrix.update(preds, y)

        self.log_dict(
            {
                "val_loss": loss,
            },
        )
        self.log_images(x, y, preds)

    def on_validation_epoch_end(self) -> None:
        precision, recall, f1 = self.calculate_metrics(self.val_conf_matrix.compute())
        self.log_conf_matrix(mode="val")
        self.log_dict(
            {
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
            }
        )

    def on_train_epoch_end(self) -> None:
        precision, recall, f1 = self.calculate_metrics(self.train_conf_matrix.compute())
        self.log_conf_matrix(mode="train")
        self.log_dict(
            {
                "train_precision": precision,
                "train_recall": recall,
                "train_f1": f1,
            }
        )

    def log_conf_matrix(self, mode="val"):
        if mode == "val":
            fig = self.val_conf_matrix.plot()
            name = "Validation_Confusion_Matrix"
            self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch)
            self.val_conf_matrix.reset()
        else:
            fig = self.train_conf_matrix.plot()
            name = "Train_Confusion_Matrix"
            self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch)
            self.train_conf_matrix.reset()
        plt.close()

    def log_images(self, images, labels, preds):
        random_index = np.random.randint(0, len(images))
        image = images[random_index]
        label = self.classes[labels[random_index].argmax().item()]
        pred = self.classes[preds[random_index].argmax().item()]

        slice_vertical = image[0, :, :, 55].detach().cpu().numpy()
        slice_horizontal = image[0, 45, :, :].detach().cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(slice_vertical, cmap="gray")
        ax1.set_title(f"Label: {label}, Pred: {pred}")
        ax2.imshow(slice_horizontal, cmap="gray")
        ax2.set_title(f"Label: {label}, Pred: {pred}")

        self.logger.experiment.add_figure(
            "Random_Slices",
            fig,
            self.current_epoch,
        )
        plt.close(fig)

    def calculate_metrics(self, confusion_matrix):
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)

        for metric in [precision, recall, f1]:
            metric[np.isnan(metric)] = 0
            # Transform to torch
            metric = torch.Tensor(metric)

        return precision.mean(), recall.mean(), f1.mean()


class ConfusionMatrixPloter:
    def __init__(self, classes):
        self.num_classes = len(classes)
        self.classes = classes
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, preds, targets):
        conf_matrix = self.confusion_matrix(
            preds.detach().cpu(), targets.detach().cpu()
        ).numpy()
        self.matrix += conf_matrix

    def compute(self):
        return self.matrix

    def plot(self):
        plt.figure(figsize=(10, 10))
        normalized_matrix = self.matrix / self.matrix.sum(axis=1, keepdims=True)

        plt.imshow(normalized_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")

        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(
                    j,
                    i,
                    round(normalized_matrix[i, j], 2),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=26,
                )
        return plt.gcf()

    def reset(self):
        self.matrix *= 0

    def confusion_matrix(self, preds, target):
        matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int)
        for p, t in zip(preds, target):
            pred_class = torch.argmax(p)
            target_class = torch.argmax(t)
            matrix[target_class][pred_class] += 1
        return matrix
