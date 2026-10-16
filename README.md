# AI Models from Scratch

Hands-on projects for learning how neural networks work: generating datasets, building models, training from random weights, and running inference locally.

The projects use PyTorch building blocks. “From scratch” means training without pretrained weights; it does not mean every mathematical operation or attention layer is implemented manually.

## Projects

| Project | Task | What you learn |
|---|---|---|
| [Mini Transformer Uppercase](mini-transformer-uppercase/README.md) | Generate uppercase text from lowercase input | Character tokenization, encoder-decoder Transformers, teacher forcing, training, validation, and autoregressive inference |

## Get started

Run these commands from this directory. Python 3.10 or newer is recommended.

```bash
cd mini-transformer-uppercase
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python dataset.py
python test_project.py
python train.py --epochs 30 --batch-size 64
python inference.py --text "hello danish how are you"
```

On Windows, activate the virtual environment with `.venv\Scripts\activate` instead.

Training selects CUDA, Apple Silicon MPS, or CPU automatically, depending on availability. The smoke test checks model execution and gradients; training is needed before expecting useful predictions.

## Learning resources

- [Project guide](mini-transformer-uppercase/README.md): setup, model components, commands, and limitations.
- [Training concepts](mini-transformer-uppercase/TRAINING_CONCEPTS.md): epochs, batch size, weight updates, learning rate, dataset expansion, and generalization.

## Repository layout

```text
ai-models-from-scratch/
├── README.md
└── mini-transformer-uppercase/
    ├── README.md
    ├── TRAINING_CONCEPTS.md
    ├── requirements.txt
    ├── tokenizer.py
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── inference.py
    └── test_project.py
```

Virtual environments, generated datasets, checkpoints, and Python caches are excluded through `.gitignore`. Generate the dataset and train locally to create those artifacts.

These are educational experiments. For reliable uppercase conversion in an application, use Python's `str.upper()`; this project uses that task to explore how a model learns a text transformation.

## Author

Danish Ashraf · danish.ashraf@codehills.net · https://codehills.net
