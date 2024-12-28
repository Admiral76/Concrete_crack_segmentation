import pytorch_lightning as pl
import torch
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.metrics import f1_score, get_stats, iou_score


class ImageSegmentator(pl.LightningModule):
    """
    Модуль PyTorch Lightning для сегментации изображений.

    Этот модуль инкапсулирует в себе логику обучения, валидации и тестирования модели сегментации.
    Использует Dice Loss для обучения и метрики IOU и F1 для оценки качества.

    Args:
        model (torch.nn.Module): Модель сегментации, которая будет обучена.
        lr (float): Скорость обучения (learning rate).

    Attributes:
        model (torch.nn.Module):  Модель сегментации.
        lr (float):  Скорость обучения.
        loss_fn (segmentation_models_pytorch.losses.DiceLoss): Функция потерь Dice Loss.
    """

    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = DiceLoss(mode="binary", from_logits=True)

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits.squeeze(), masks.squeeze())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images).squeeze()
        loss = self.loss_fn(logits, masks)

        outputs = torch.sigmoid(logits)
        tp, fp, fn, tn = get_stats(
            outputs, masks.to(torch.int8), mode="binary", threshold=0.5
        )
        iou = iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = f1_score(tp, fp, fn, tn, reduction="micro")

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_iou_score", iou, prog_bar=True, on_epoch=True)
        self.log("val_f1_score", f1, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_iou_score": iou, "val_f1_score": f1}

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images).squeeze()
        outputs = torch.sigmoid(logits)

        tp, fp, fn, tn = get_stats(
            outputs, masks.to(torch.int8), mode="binary", threshold=0.5
        )
        iou = iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = f1_score(tp, fp, fn, tn, reduction="micro")

        self.log("test_iou_score", iou, prog_bar=True, on_epoch=True)
        self.log("test_f1_score", f1, prog_bar=True, on_epoch=True)
        return {"test_iou_score": iou, "test_f1_score": f1}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
