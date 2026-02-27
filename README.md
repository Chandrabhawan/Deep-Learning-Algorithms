# Deep-Learning-NLP-Vision-for-Data-Scientists-Masterclass

A comprehensive, hands-on deep learning course covering **fundamentals, PyTorch, computer vision, NLP, Transformers, and generative AI** -- designed for data scientists who want to go from theory to production-ready skills.

---

## Who This Is For

- Data scientists / ML engineers who know Python and basic ML (linear regression, classification, scikit-learn)
- Anyone who wants a structured, notebook-first path through deep learning
- Practitioners who need both **theory intuition** and **working code**

## What You Will Learn

| Module | Topics |
|--------|--------|
| **DL100** | Neural network fundamentals -- neurons, backprop, loss functions, optimizers, regularization |
| **DL150** | PyTorch foundations -- tensors, autograd, nn.Module, DataLoader, training loops, GPU |
| **DL200** | Practical MLP -- regression, classification, tuning, data leakage |
| **DL300** | Computer vision -- CNNs, augmentation, transfer learning, detection/segmentation concepts |
| **DL400** | Sequence models -- RNN, LSTM, GRU, attention basics |
| **DL500** | NLP foundations -- preprocessing, tokenization, BoW/TF-IDF, postprocessing, evaluation |
| **DL600** | Transformers & modern NLP -- self-attention, Transformer architecture, HuggingFace (optional) |
| **DL700** | Generative models (optional) -- autoencoders, VAEs, GANs, diffusion concepts |
| **DL800** | MLOps lite -- experiment tracking, checkpointing, profiling, reproducibility |

## Recommended Learning Path

```
DL050 (Prerequisites Quick Ref)
  |
DL100 (Neural Network Fundamentals)
  |
DL150 (PyTorch Foundations)
  |
DL200 (MLP Practical)
  |
  +--- DL300 (Computer Vision / CNN)
  |         |
  |         +--- Project 01 (Image Classifier)
  |
  +--- DL400 (RNN / LSTM / GRU)
  |         |
  |         +--- DL500 (NLP Foundations)
  |                   |
  |                   +--- DL600 (Transformers)
  |                             |
  |                             +--- Project 02 (Text Classifier)
  |                             +--- Project 03 (Transformer Fine-Tuning)
  |
  +--- DL700 (Generative Models - Optional)
  |
  +--- DL800 (MLOps Lite)
```

## Setup Instructions

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate dl-masterclass
```

### Option B: pip + venv

```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### GPU Notes (optional)

- PyTorch with CUDA is **optional** -- all notebooks run on CPU
- For GPU support, install the appropriate PyTorch CUDA build:
  ```bash
  # Example for CUDA 12.1:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- The utility `src/utils/device.py` auto-detects GPU availability

## How to Run Notebooks

```bash
cd notebooks
jupyter lab
# or
jupyter notebook
```

All notebooks are self-contained. Utility imports use relative paths:
```python
import sys; sys.path.insert(0, "..")
from src.utils.seed import set_global_seed
```

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/
│   ├── raw/          # Downloaded datasets
│   └── processed/    # Preprocessed data
├── assets/
│   ├── images/
│   └── diagrams/
├── src/
│   └── utils/        # Shared helper modules
├── notebooks/
│   ├── 00_Course_Orientation.ipynb
│   ├── DL050_Deep_Learning_Prerequisites_Quick_Ref.ipynb
│   ├── DL100_Neural_Network_Fundamentals/   (8 notebooks)
│   ├── DL150_PyTorch_Foundations/            (8 notebooks)
│   ├── DL200_ANN_MLP_Practical/             (4 notebooks)
│   ├── DL300_Computer_Vision_CNN/           (7 notebooks)
│   ├── DL400_Sequence_Models_RNN_LSTM_GRU/  (5 notebooks)
│   ├── DL500_NLP_Foundations_and_Text_Processing/ (6 notebooks)
│   ├── DL600_Transformers_and_Modern_NLP/   (7 notebooks)
│   ├── DL700_Generative_Models_Optional/    (3 notebooks)
│   └── DL800_Deep_Learning_Operations_MLOps_Lite/ (4 notebooks)
├── projects/         # End-to-end capstone projects
└── exercises/        # Per-module exercise notebooks
```

## Dependencies

Core (required):
- Python >= 3.9
- PyTorch >= 2.0
- torchvision
- numpy, pandas, matplotlib, scikit-learn

Optional (guarded with try/except in notebooks):
- torchaudio
- transformers (HuggingFace)
- sentence-transformers
- tensorboard

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](LICENSE).

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description

For bug reports or suggestions, open an issue.

## Acknowledgments

Built with PyTorch, torchvision, scikit-learn, and the open-source ML community.
