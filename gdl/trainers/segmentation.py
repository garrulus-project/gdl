# Copyright (c) Microsoft Corporation. All rights reserved.
# Maintainer and Mofidication: Mohammad Wasil @Garrulus project H-BRS
# Licensed under the MIT License.

"""Trainers for semantic segmentation."""

import os
import warnings
from typing import Any

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch import Tensor
from torchgeo.datasets.utils import unbind_samples
from torchgeo.models import FCN, get_weight
from torchgeo.trainers import utils
from torchgeo.trainers.base import BaseTask
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Dice,
    MulticlassAccuracy,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision.models._api import WeightsEnum

from gdl.models.peft import adapter_h, adapter_l, lora, sam_decoder
from gdl.models.segment_anything import sam_model_registry
from gdl.samplers.batch import DistributedRandomBatchAoiGeoSampler


class GarrulusSemanticSegmentationTask(BaseTask):
    """Semantic Segmentation."""

    def __init__(
        self,
        model: str = 'unet',
        backbone: str = 'resnet50',
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 3,
        num_classes: int = 1000,
        num_filters: int = 3,
        loss: str = 'ce',
        class_weights: Tensor | None = None,
        ignore_index: int | None = None,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        peft: str = 'adapter_h',
        sam_registry_key: str = 'vit_h',
        sam_ckpt: str = None,
        peft_ckpt: str = None,
        img_size: int = 512,
        high_res_upsampling: bool = False,
        use_dense_embeddings: bool = False,
        middle_dim: int = 32,  # adapter
        scaling_factor: float = 0.1,  # adapter
        **kwargs: Any,
    ) -> None:
        """Initialize a new SemanticSegmentationTask instance.

        Args:
            model: Name of the
                `smp <https://smp.readthedocs.io/en/latest/models.html>`__ model to use.
            backbone: Name of the `timm
                <https://smp.readthedocs.io/en/latest/encoders_timm.html>`__ or `smp
                <https://smp.readthedocs.io/en/latest/encoders.html>`__ backbone to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False or
                None for random weights, or the path to a saved model state dict. FCN
                model does not support pretrained weights. Pretrained ViT weight enums
                are not supported yet.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes.
            num_filters: Number of filters. Only applicable when model='fcn'.
            loss: Name of the loss function, currently supports
                'ce', 'jaccard' or 'focal' loss.
            class_weights: Optional rescaling weight given to each
                class and used with 'ce' loss.
            ignore_index: Optional integer class index to ignore in the loss and
                metrics.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the
                decoder and segmentation head.
            freeze_decoder: Freeze the decoder network to linear probe
                the segmentation head.

        Warns:
            UserWarning: When loss='jaccard' and ignore_index is specified.

        .. versionchanged:: 0.3
           *ignore_zeros* was renamed to *ignore_index*.

        .. versionchanged:: 0.4
           *segmentation_model*, *encoder_name*, and *encoder_weights*
           were renamed to *model*, *backbone*, and *weights*.

        .. versionadded: 0.5
            The *class_weights*, *freeze_backbone*, and *freeze_decoder* parameters.

        .. versionchanged:: 0.5
           The *weights* parameter now supports WeightEnums and checkpoint paths.
           *learning_rate* and *learning_rate_schedule_patience* were renamed to
           *lr* and *patience*.
        """
        if ignore_index is not None and loss == 'jaccard':
            warnings.warn(
                "ignore_index has no effect on training when loss='jaccard'",
                UserWarning,
            )

        self.weights = weights
        super().__init__(ignore='weights')

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams['loss']
        ignore_index = self.hparams['ignore_index']
        if loss == 'ce':
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=self.hparams['class_weights']
            )
        elif loss == 'jaccard':
            self.criterion = smp.losses.JaccardLoss(
                mode='multiclass', classes=self.hparams['num_classes']
            )
        elif loss == 'focal':
            self.criterion = smp.losses.FocalLoss(
                'multiclass', ignore_index=ignore_index, normalized=True
            )
        elif loss == 'dice':
            self.criterion = smp.losses.DiceLoss(
                'multiclass', ignore_index=ignore_index, normalized=True
            )
        # ToDo: combine loss with ce-> 80% dice + 20% ce
        # ToDo: combine loss with focal -> dice 80% focal 20%
        # elif loss == "dice_ce":
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard', 'dice' or 'focal' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.classification.MulticlassAccuracy`: Overall accuracy
          (OA) using 'micro' averaging. The number of true positives divided by the
          dataset size. Higher values are better.
        * :class:`~torchmetrics.classification.MulticlassJaccardIndex`: Intersection
          over union (IoU). Uses 'micro' averaging. Higher valuers are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not reflect
             minority class accuracy.
           * 'Macro' averaging, not used here, gives equal weight to each class, useful
             for balanced performance assessment across imbalanced classes.
        """
        num_classes: int = self.hparams['num_classes']
        ignore_index: int | None = self.hparams['ignore_index']
        metrics = MetricCollection(
            [
                # micro is good for class imbalance samples
                MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average='global',
                    average='micro',
                ),
                MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average='micro'
                ),
                Dice(num_classes=num_classes, average='micro'),
                MulticlassPrecision(num_classes=num_classes, average='micro'),
                MulticlassRecall(num_classes=num_classes, average='micro'),
            ]
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams['model']
        backbone: str = self.hparams['backbone']
        weights = self.weights
        in_channels: int = self.hparams['in_channels']
        num_classes: int = self.hparams['num_classes']
        num_filters: int = self.hparams['num_filters']

        if model == 'unet':
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights='imagenet' if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == 'deeplabv3+':
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights='imagenet' if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == 'fcn':
            self.model = FCN(
                in_channels=in_channels, classes=num_classes, num_filters=num_filters
            )
        elif model == 'sam':
            sam_registry_key = self.hparams['sam_registry_key']
            sam, img_embedding_size = sam_model_registry[sam_registry_key](
                image_size=self.hparams['img_size'],
                num_classes=self.hparams['num_classes'],
                checkpoint=self.hparams['sam_ckpt'],
                pixel_mean=[0, 0, 0],
                pixel_std=[1, 1, 1],
                high_res_upsampling=self.hparams['high_res_upsampling'],
            )

            # _de -> with dense embedding
            de = self.hparams['use_dense_embeddings']
            if self.hparams['peft'] == 'adapter_h':
                self.model = adapter_h.AdapterSAM(
                    sam,
                    self.hparams['middle_dim'],
                    self.hparams['scaling_factor'],
                    use_dense_embeddings=de,
                )
            elif self.hparams['peft'] == 'adapter_l':
                self.model = adapter_l.AdapterSAM(
                    sam,
                    self.hparams['middle_dim'],
                    self.hparams['scaling_factor'],
                    use_dense_embeddings=de,
                )
            elif self.hparams['peft'] == 'lora':
                self.model = lora.LoRASAM(
                    sam, self.hparams['rank'], use_dense_embeddings=de
                )
            elif self.hparams['peft'] == 'sam_decoder':
                self.model = sam_decoder.SAMDecoder(sam, use_dense_embeddings=de)
                print('Updating decoder only')

            if self.hparams['peft_ckpt'] is not None:
                model.load_peft_parameters(self.hparams['peft_ckpt'])

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            print('Total number of parameters: ', total_params)
            print('Total number of trainable parameters', trainable_params)

        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if model != 'fcn':
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams['freeze_backbone'] and model in ['unet', 'deeplabv3+']:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams['freeze_decoder'] and model in ['unet', 'deeplabv3+']:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch['image']
        y = batch['mask']
        if self.hparams['model'] == 'sam':
            y_hat = self(
                batched_input=x,
                multimask_output=True,
                image_size=self.hparams['img_size'],
            )['masks']
        else:
            y_hat = self(x)
        loss: Tensor = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = batch['mask']
        if self.hparams['model'] == 'sam':
            y_hat = self(
                batched_input=x,
                multimask_output=True,
                image_size=self.hparams['img_size'],
            )['masks']
        else:
            y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics)

        if (
            batch_idx < 10
            and hasattr(self.trainer, 'datamodule')
            and hasattr(self.trainer.datamodule, 'plot')
            and self.logger
            and hasattr(self.logger, 'experiment')
            and hasattr(self.logger.experiment, 'add_figure')
        ):
            try:
                datamodule = self.trainer.datamodule
                batch['prediction'] = y_hat.argmax(dim=1)
                for key in ['image', 'mask', 'prediction']:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    summary_writer.add_figure(
                        f'image/{batch_idx}', fig, global_step=self.global_step
                    )
                    plt.close()
            except ValueError:
                pass

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = batch['mask']
        if self.hparams['model'] == 'sam':
            y_hat = self(
                batched_input=x,
                multimask_output=True,
                image_size=self.hparams['img_size'],
            )['masks']
        else:
            y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch['image']
        if self.hparams['model'] == 'sam':
            y_hat = self(
                batched_input=x,
                multimask_output=True,
                image_size=self.hparams['img_size'],
            )['masks']
        else:
            y_hat = self(x)
        return y_hat

    def on_train_epoch_start(self) -> None:
        """
        Update epoch for distributed sampler.
        ToDo: This function is called twice, enable print statement to debug
        """
        if hasattr(self.trainer.datamodule, 'train_batch_sampler'):
            if isinstance(
                self.trainer.datamodule.train_batch_sampler,
                DistributedRandomBatchAoiGeoSampler,
            ):
                self.trainer.datamodule.train_batch_sampler.set_epoch(
                    self.current_epoch
                )

            # resample windows for train data
            if hasattr(
                self.trainer.datamodule.train_batch_sampler, 'sample_windows'
            ) and callable(self.trainer.datamodule.train_batch_sampler.sample_windows):
                # print(f'Re-sampling windows for batch sampler epoch: {self.current_epoch}')
                self.trainer.datamodule.train_batch_sampler.sample_windows()
