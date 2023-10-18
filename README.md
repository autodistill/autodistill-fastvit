<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill FastViT Module

This repository contains the code supporting the FastViT base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[FastViT](https://github.com/apple/ml-fastvit), developed by Apple, is a classification model that supports zero-shot classification.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [FastViT Autodistill documentation](https://autodistill.github.io/autodistill/base_models/fastvit/).

## Installation

To use FastViT with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-fastvit
```

## Quickstart

FastViT works using the ImageNet-1k class list. This class list is available in the `FASTVIT_IMAGENET_1K_CLASSES` variable.

You can provide classes from the list to retrieve predictions for a specific class in the list. You can also provide a custom ontology to map classes from the list to your own classes.

```python
from autodistill_fastvit import FastViT, FASTVIT_IMAGENET_1K_CLASSES
from autodistill.detection import CaptionOntology

# zero shot with no prompts
base_model = FastViT(None)

# zero shot with prompts from FASTVIT_IMAGENET_1K_CLASSES
base_model = FastViT(
    ontology=CaptionOntology(
        {
            "coffeemaker": "coffeemaker",
            "ice cream": "ice cream"
        }
    )
)

predictions = base_model.predict("./example.png")

labels = [FASTVIT_IMAGENET_1K_CLASSES[i] for i in predictions.class_id.tolist()]

print(labels)
```


## License

See [LICENSE](LICENSE) for the model license.

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
