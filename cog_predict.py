# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
import tempfile

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path

from data.dataloader import ImageLoader
from lib.InSPyReNet import InSPyReNet_SwinB
from utils.misc import load_config, to_cuda, to_numpy


work_dir = os.path.split(os.path.abspath(__file__))[0]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Pre-load the model to make running multiple predictions efficient"""
        # Load config
        self.model_opt = load_config(
            "configs/extra_dataset/Plus_Ultra_inference.yaml"
        )

        # Build model
        self.model = InSPyReNet_SwinB(**self.model_opt.Model)

        # Load checkpoint
        self.model.load_state_dict(
            torch.load('/checkpoints/InSPyReNet.pth',
                       map_location=torch.device('cpu')),
            strict=True)

        # Move to GPU if possible
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Set to eval mode
        self.model.eval()

    def predict(
        self,
        image_path: Path = Input(description="RGB input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        torch.cuda.empty_cache()

        # Load input image via `ImageLoader`
        samples = ImageLoader(str(image_path),
                              self.model_opt.Test.Dataset.transforms)

        for sample in samples:
            # Move to GPU if possible
            if torch.cuda.is_available():
                sample = to_cuda(sample)

            # Run inference
            with torch.no_grad():
                out = self.model(sample)

            # Convert output to bytes
            pred = to_numpy(out['pred'], sample['shape'])
            pred = (pred * 255).astype(np.uint8)

        # Save image to output path
        _, output_path = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(output_path, pred)

        return Path(output_path)
