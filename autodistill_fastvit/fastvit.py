import json
import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import timm
import torch
from typing import Any

from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get classes.json in folder relative to this file
with open(os.path.join(os.path.dirname(__file__), "classes.json")) as f:
    FASTVIT_IMAGENET_1K_CLASSES = json.load(f)

FASTVIT_IMAGENET_1K_CLASSES_LENGTH = len(FASTVIT_IMAGENET_1K_CLASSES)


@dataclass
class FastViT(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology

        model = timm.create_model("hf_hub:timm/fastvit_s12.apple_in1k", pretrained=True)
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        model.eval()

        self.model = model
        self.transforms = transforms

    def predict(self, input: Any, confidence: int = 0.5) -> sv.Detections:
        img = load_image(input, return_format="PIL")

        output = self.model(self.transforms(img).unsqueeze(0))

        prompts = self.ontology.prompts() if self.ontology is not None else []

        if len(prompts) == 0:
            probs, class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

            if confidence:
                class_indices = [
                    class_indices[0][i].tolist()
                    for i in range(len(class_indices[0]))
                    if probs[0][i] >= confidence
                ]
                probs = [
                    probs[0][i].tolist()
                    for i in range(len(probs[0]))
                    if probs[0][i] >= confidence
                ]
            else:
                class_indices = [
                    class_indices[0][i].tolist() for i in range(len(class_indices[0]))
                ]
                probs = [probs[0][i].tolist() for i in range(len(probs[0]))]

            # divide probs by 100
            probs = [p / 100 for p in probs]

            return sv.Classifications(
                class_id=np.array(class_indices), confidence=np.array(probs)
            )
        else:
            probs, class_indices = torch.topk(
                output.softmax(dim=1) * 100, k=FASTVIT_IMAGENET_1K_CLASSES_LENGTH
            )

        prompt_idx = [FASTVIT_IMAGENET_1K_CLASSES.index(p) for p in prompts]

        prompt_probs = [
            probs[0][class_indices[0].tolist().index(i)].tolist() for i in prompt_idx
        ]

        if confidence:
            prompt_idx = [
                prompt_idx[i]
                for i in range(len(prompt_idx))
                if prompt_probs[i] >= confidence
            ]
            prompt_probs = [
                prompt_probs[i]
                for i in range(len(prompt_probs))
                if prompt_probs[i] >= confidence
            ]

        prompt_probs = [p / 100 for p in prompt_probs]

        return sv.Classifications(
            class_id=np.array(prompt_idx),
            confidence=np.array(prompt_probs),
        )
