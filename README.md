# NovoTempEstimate

A Python project for protein temperature estimation using LSTM networks with PyTorch.

## Features

- **Protein Sequence Encoding**: Multiple encoding strategies for protein sequences
  - One-hot encoding
  - Physicochemical property-based encoding
  - Learned embeddings
- **LSTM Network Optimization**: Optuna-based hyperparameter optimization
- **Data Pipeline**: Automated data download from Zenodo
- **Apple Silicon Support**: Optimized for M4 chips with PyTorch MPS backend

## Project Structure

```
NovoTempEstimate/
├── src/novotempestimate/          # Main package
│   ├── __init__.py
│   ├── encoding.py                # Protein sequence encoding
│   └── optimization.py            # Optuna LSTM optimization
├── scripts/                       # Utility scripts
│   ├── download_data.py           # Download data from Zenodo
│   └── download_paper.py          # Download research paper
├── data/                          # Data directory
├── papers/                        # Research papers
├── requirements.txt               # Dependencies
├── pyproject.toml                 # Project configuration
├── .python-version               # Python version for pyenv
└── README.md                     # This file
```

## Setup

### Prerequisites

- Python 3.11+ (managed with pyenv)
- PyTorch with Apple Silicon support

### Installation

1. **Set up Python environment with pyenv:**
   ```bash
   pyenv install 3.11.0
   pyenv local 3.11.0
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download data:**
   ```bash
   python scripts/download_data.py
   ```

4. **Download research paper:**
   ```bash
   python scripts/download_paper.py
   ```

## Usage

### Protein Sequence Encoding

```python
from novotempestimate import ProteinEncoder

# Initialize encoder
encoder = ProteinEncoder(encoding_type="onehot")

# Encode a single sequence
sequence = "MKFLVLLFNILCLFPVLAADNHGVGPQGASGILKTLLKQIGDLQAGLQGVQAGVWPAAVRESVPSLL"
encoded = encoder.encode_sequence(sequence)

# Encode multiple sequences
sequences = ["MKFLVLL...", "MKALIVL..."]
batch_encoded = encoder.encode_batch(sequences)
```

### LSTM Optimization with Optuna

```python
from novotempestimate import OptunaLSTMOptimizer
from torch.utils.data import DataLoader

# Create data loaders (your data)
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize optimizer
optimizer = OptunaLSTMOptimizer(
    input_dim=20,  # Encoding dimension
    train_loader=train_loader,
    val_loader=val_loader
)

# Run hyperparameter optimization
study = optimizer.optimize(n_trials=100)

# Train best model
best_model = optimizer.create_best_model()
history = optimizer.train_best_model(num_epochs=200)
```

## Data Source

This project uses data from:
- **Zenodo Record**: [10463156](https://zenodo.org/records/10463156)
- **Research Paper**: [DOI: 10.1101/2023.03.27.534365](https://doi.org/10.1101/2023.03.27.534365)

## Development

### Code Formatting

This project uses Black for code formatting:

```bash
black src/ scripts/
```

### Testing

Run the built-in tests:

```bash
python src/novotempestimate/encoding.py
python src/novotempestimate/optimization.py
```

## Requirements

- PyTorch >= 2.1.0 (with Apple Silicon support)
- Optuna >= 3.4.0
- BioPython >= 1.81
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Scikit-learn >= 1.3.0

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and formatting
5. Submit a pull request

## TODO

- [ ] Implement actual physicochemical properties for amino acids
- [ ] Add more sophisticated LSTM architectures
- [ ] Implement cross-validation
- [ ] Add model interpretability features
- [ ] Create visualization tools for results
