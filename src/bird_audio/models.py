from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

LOCKED_EFFICIENTNET_WEIGHTS = "EfficientNet_B0_Weights.IMAGENET1K_V1"


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 wrapper that keeps frozen BatchNorm statistics immutable."""

    def __init__(
        self,
        class_count: int,
        dropout: float,
        weights: EfficientNet_B0_Weights | None,
        trainable_feature_indices: Iterable[int],
    ) -> None:
        super().__init__()
        if class_count <= 1:
            raise ValueError("class_count must be greater than one")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")
        self.network = efficientnet_b0(weights=weights)
        final_linear = self.network.classifier[1]
        if not isinstance(final_linear, nn.Linear):
            raise RuntimeError("Unexpected torchvision EfficientNet-B0 classifier layout")
        self.network.classifier[0] = nn.Dropout(p=dropout, inplace=True)
        self.network.classifier[1] = nn.Linear(final_linear.in_features, class_count)

        selected = tuple(sorted(set(int(index) for index in trainable_feature_indices)))
        if any(index < 0 or index >= len(self.network.features) for index in selected):
            raise ValueError(f"Invalid EfficientNet-B0 feature indices: {selected}")
        self.trainable_feature_indices = selected
        self.frozen_feature_indices = tuple(
            index for index in range(len(self.network.features)) if index not in selected
        )
        for parameter in self.network.features.parameters():
            parameter.requires_grad = False
        for index in selected:
            for parameter in self.network.features[index].parameters():
                parameter.requires_grad = True
        for parameter in self.network.classifier.parameters():
            parameter.requires_grad = True
        self.train(self.training)

    @property
    def features(self) -> nn.Sequential:
        return self.network.features

    @property
    def classifier(self) -> nn.Sequential:
        return self.network.classifier

    def train(self, mode: bool = True) -> EfficientNetB0Classifier:
        super().train(mode)
        if mode:
            for index in self.frozen_feature_indices:
                self.network.features[index].eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def build_efficientnet_b0_classifier(
    class_count: int = 15,
    dropout: float = 0.2,
    weights_identifier: str | None = LOCKED_EFFICIENTNET_WEIGHTS,
    trainable_feature_indices: Iterable[int] = (),
) -> nn.Module:
    """Build the one permitted classifier and apply the locked fine-tuning policy."""
    if weights_identifier is None:
        weights = None
    elif weights_identifier == LOCKED_EFFICIENTNET_WEIGHTS:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Unsupported EfficientNet-B0 weights: {weights_identifier}")

    return EfficientNetB0Classifier(
        class_count=class_count,
        dropout=dropout,
        weights=weights,
        trainable_feature_indices=trainable_feature_indices,
    )


class ConvolutionalAutoencoder(nn.Module):
    """Locked skip-free undercomplete autoencoder for 224 by 224 Mel inputs."""

    def __init__(self, latent_dimensions: int = 64) -> None:
        super().__init__()
        if latent_dimensions <= 0:
            raise ValueError("latent_dimensions must be positive")
        self.latent_dimensions = latent_dimensions
        self.encoder_convolutions = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        flattened_dimensions = 128 * 14 * 14
        self.to_latent = nn.Linear(flattened_dimensions, latent_dimensions)
        self.from_latent = nn.Linear(latent_dimensions, flattened_dimensions)
        self.decoder_convolutions = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_convolutions(inputs)
        if tuple(encoded.shape[-3:]) != (128, 14, 14):
            raise ValueError("The locked autoencoder expects inputs with spatial shape 224 by 224")
        return self.to_latent(encoded.flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.from_latent(latent).reshape(-1, 128, 14, 14)
        return self.decoder_convolutions(decoded)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(inputs)
        return self.decode(latent), latent


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}
