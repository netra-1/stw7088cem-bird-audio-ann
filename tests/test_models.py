from __future__ import annotations

import unittest

import torch
from torch import nn

from bird_audio.models import (
    ConvolutionalAutoencoder,
    build_efficientnet_b0_classifier,
    parameter_counts,
)


class ModelContractTests(unittest.TestCase):
    def test_head_only_policy_keeps_every_feature_block_frozen(self) -> None:
        model = build_efficientnet_b0_classifier(
            class_count=15,
            dropout=0.2,
            weights_identifier=None,
            trainable_feature_indices=[],
        )
        model.train()
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.features.parameters())
        )
        self.assertTrue(all(not block.training for block in model.features))
        self.assertIsInstance(model.classifier[0], nn.Dropout)
        self.assertEqual(model.classifier[0].p, 0.2)
        self.assertEqual(
            parameter_counts(model),
            {"total": 4_026_763, "trainable": 19_215},
        )

    def test_partial_policy_preserves_frozen_batchnorm_statistics(self) -> None:
        model = build_efficientnet_b0_classifier(
            class_count=15,
            dropout=0.2,
            weights_identifier=None,
            trainable_feature_indices=[6, 7, 8],
        )
        model.eval()
        model.train()
        self.assertTrue(all(not model.features[index].training for index in range(6)))
        self.assertTrue(all(model.features[index].training for index in (6, 7, 8)))
        self.assertEqual(
            parameter_counts(model),
            {"total": 4_026_763, "trainable": 3_174_955},
        )

    def test_autoencoder_contract(self) -> None:
        model = ConvolutionalAutoencoder(latent_dimensions=64).eval()
        with torch.no_grad():
            reconstruction, latent = model(torch.rand(2, 1, 224, 224))
        self.assertEqual(tuple(reconstruction.shape), (2, 1, 224, 224))
        self.assertEqual(tuple(latent.shape), (2, 64))
        self.assertTrue(bool(((reconstruction >= 0) & (reconstruction <= 1)).all()))
        self.assertEqual(
            parameter_counts(model),
            {"total": 3_581_345, "trainable": 3_581_345},
        )


if __name__ == "__main__":
    unittest.main()
