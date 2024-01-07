from functools import partial
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lion_pytorch import Lion
from matplotlib import pyplot as plt


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
        n_classes=400,
    ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=2
        )
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


class Classifier3D(pl.LightningModule):
    def __init__(
        self,
        model_depth: int = 18,
        lr: float = 0.001,
        scheduler_gamma: Optional[float] = 0.1,
        scheduler_step_size: Optional[int] = 25,
        weight_decay: float = 0.0,
        optimizer_alg: str = "adam",
        class_weights: Optional[list] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.name = name
        self.lr = lr
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.weight_decay = weight_decay
        self.optimizer_alg = optimizer_alg
        self.num_classes = 3
        self.classes = ["AD", "MCI", "CN"]

        if class_weights is not None:
            class_weights = torch.Tensor(class_weights)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.model = generate_model(
            model_depth=model_depth, n_input_channels=1, n_classes=self.num_classes
        )

        self.val_conf_matrix = ConfusionMatrixPloter(classes=self.classes)
        self.train_conf_matrix = ConfusionMatrixPloter(classes=self.classes)

    def forward(self, x):
        x = self.model(x)
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
