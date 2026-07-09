# Smoothed-ModernBERT

Text classification that fuses **ModernBERT** contextual embeddings with a **smoothed Dirichlet neural topic model** via co-attention. The topic model (SMDIRICHLET) is a VAE-style module over bag-of-words inputs whose latent document–topic distribution is combined with ModernBERT's [CLS] representation through a learned gating mechanism, and the two components are trained jointly.

## How It Works

1. **ModernBERT encoder** (`answerdotai/ModernBERT-base`) encodes the token sequence; the [CLS] embedding represents the document.
2. **SMDIRICHLET topic model** encodes the document's bag-of-words vector through an MLP and produces a smoothed Dirichlet latent sample via reparameterization (`α + ε·ρ₂ + ρ₁` → softmax), along with a reconstruction loss and a Dirichlet KL divergence.
3. **Co-attention fusion**: both representations are projected into a shared space; a sigmoid gate `α = σ(⟨proj_b, proj_t⟩ + b)` decides per-document how much to trust the contextual vs. topic representation:

   ```
   joint = α · proj_BERT + (1 − α) · proj_topic
   ```

4. **Classification head** predicts the label from the fused representation. The training objective is cross-entropy plus a small KL regularizer from the topic model.

## Repository Structure

```
├── main.py                    # CLI entry point (argparse, JSON config save/load)
├── models/
│   ├── topicbert.py           # TopicBERT: ModernBERT + co-attention + classifier
│   └── smdirichlet.py         # Smoothed Dirichlet VAE topic model
├── datasets/
│   ├── reuters8.py            # Reuters-8 dataset wrapper (TSV)
│   ├── imdb.py                # IMDB dataset wrapper (auto-download)
│   ├── bow.py                 # Bag-of-words dataset wrapper
│   ├── vocab.py               # Vocabulary construction
│   ├── embed.py
│   └── utils.py               # Dataset partitioning for long documents
├── training/
│   ├── train_topicbert.py     # Training / evaluation loops
│   └── utils.py               # Checkpointing, GPU selection
├── raw_data/Reuters8/         # Reuters-8 splits (training/validation/test TSV + labels)
├── runs/                      # TensorBoard output
└── environment.yml
```

## Installation

```bash
git clone https://github.com/hormone03/Smoothed-ModernBERT.git
cd Smoothed-ModernBERT

conda env create -f environment.yml
conda activate smDirichtopicbert
```

> **Note:** ModernBERT requires `transformers >= 4.48`. If your environment has an older version, upgrade with `pip install -U transformers`.

Minimal manual setup, if you prefer pip:

```bash
pip install torch transformers scikit-learn numpy tqdm tensorboardX prefetch-generator
```

## Usage

### Train on Reuters-8 (default)

```bash
python main.py --device cuda --epochs 20 --batch-size 8 --lr 2e-5
```

Reuters-8 splits are included in `raw_data/Reuters8/`, so this runs out of the box.

### Train on IMDB

```bash
python main.py -d imdb --train-dataset-path path/to/imdb --device cuda
```

IMDB is downloaded automatically to the given directory if not present.

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `-d, --dataset` | `reuters8` | `reuters8` or `imdb` |
| `--batch-size` | 8 | Batch size |
| `--lr` | 2e-5 | Learning rate (AdamW + linear warmup schedule) |
| `--num-epochs` | 20 | Training epochs |
| `--alpha` | 0.9 | Weighting between classifier loss and topic-model loss |
| `--dropout` | 0.1 | Dropout on the fused representation |
| `--partition-factor` | 1 | Split long documents into chunks of 512/N tokens (label preserved) |
| `--val-freq` / `--test-freq` | 1 | Evaluate every N epochs |
| `--resume DIR` | — | Resume from checkpoint directory (also enables per-epoch saving) |
| `--seed` | — | Fix random seed for reproducibility |
| `--device` | `cpu` | `cpu` or `cuda` (multi-GPU handled automatically via DataParallel) |

### Config files

Save the current CLI settings as a reusable JSON config, then reload later:

```bash
python main.py --lr 1e-5 --epochs 30 -s configs/my_run.json
python main.py -l configs/my_run.json
```

### Monitoring

TensorBoard logs (loss, KLD, train/val/test accuracy and macro-F1, timing) are written per run:

```bash
tensorboard --logdir runs
```

## Evaluation

Validation and test metrics (accuracy, macro-F1, full per-class classification report) are computed during training at the frequency set by `--val-freq` / `--test-freq`. The best validation and test accuracies are reported at the end of training.

## Data Format samples

Reuters-8-style datasets are TSV files with one example per line, plus a `labels.txt` listing class names one per line. IMDB uses the standard `aclImdb` layout and is fetched automatically.

