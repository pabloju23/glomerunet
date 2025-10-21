from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import torch
from torch.nn.modules.loss import _Loss

from monai.losses.focal_loss import FocalLoss
from monai.networks import one_hot

from monai.losses.tversky import TverskyLoss


class FocalTverskyLoss(_Loss):
    """
    Compute both Tversky loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Tversky loss is shown in `monai.losses.TverskyLoss`.
    The details of Focal Loss is shown in `monai.losses.FocalLoss`.
    
    `gamma`, `focal_alpha`, and `lambda_focal` are only used for the focal loss.
    `alpha` and `beta` are only used for the tversky loss.
    `include_background`, `weight`, and `reduction` are used for both losses.
    Other parameters are only used for tversky loss.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        soft_label: bool = False,
        gamma: float = 2.0,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        lambda_tversky: float = 1.0,
        lambda_focal: float = 1.0,
        focal_alpha: float | None = None,
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the `target` into the one-hot format, using the number of classes 
                inferred from input (`input.shape[1]`). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the TverskyLoss, 
                don't need to specify activation function for FocalLoss.
            softmax: if True, apply a softmax function to the prediction, only used by the TverskyLoss, 
                don't need to specify activation function for FocalLoss.
            other_act: callable function to execute other activation layers, Defaults to `None`. 
                for example: other_act = torch.tanh. only used by the TverskyLoss, not for FocalLoss.
            alpha: weight of false positives for Tversky loss.
            beta: weight of false negatives for Tversky loss.
            reduction: {`"none"`, `"mean"`, `"sum"`} Specifies the reduction to apply to the output. 
                Defaults to `"mean"`.
                - `"none"`: no reduction will be applied.
                - `"mean"`: the sum of the output will be divided by the number of elements in the output.
                - `"sum"`: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing. 
                Defaults to False, a Tversky loss value is computed independently from each item in the batch 
                before any reduction.
            soft_label: whether the target contains non-binary values (soft labels) or not. 
                If True a soft label formulation of the loss will be used.
            gamma: value of the exponent gamma in the definition of the Focal loss.
            weight: weights to apply to the voxels of each class. If None no weights are applied. 
                The input can be a single value (same weight for all classes), a sequence of values 
                (the length of the sequence should be the same as the number of classes).
            lambda_tversky: the trade-off weight value for tversky loss. The value should be no less than 0.0. 
                Defaults to 1.0.
            lambda_focal: the trade-off weight value for focal loss. The value should be no less than 0.0. 
                Defaults to 1.0.
            focal_alpha: value of the alpha in the definition of the alpha-balanced Focal loss. 
                The value should be in [0, 1]. Defaults to None.
        """
        super().__init__()
        
        self.tversky = TverskyLoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            alpha=alpha,
            beta=beta,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            soft_label=soft_label,
        )
        
        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=gamma,
            weight=weight,
            alpha=focal_alpha,
            reduction=reduction,
        )
        
        if lambda_tversky < 0.0:
            raise ValueError("lambda_tversky should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        
        self.lambda_tversky = lambda_tversky
        self.lambda_focal = lambda_focal
        self.to_onehot_y = to_onehot_y

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits 
                due to the restriction of `monai.losses.FocalLoss`.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 (without one-hot encoding) 
                nor the same as input.

        Returns:
            torch.Tensor: value of the loss.
        """
        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )

        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, to_onehot_y=True ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        tversky_loss = self.tversky(input, target)
        focal_loss = self.focal(input, target)
        total_loss: torch.Tensor = self.lambda_tversky * tversky_loss + self.lambda_focal * focal_loss

        return total_loss