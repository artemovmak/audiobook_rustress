# Speech Stress Analyzer

MLOps project for Russian speech stress analysis and temporal word segmentation using PyTorch Lightning and advanced machine learning techniques.

## Project Description

This project implements an automated system for identifying stressed syllables and performing temporal word segmentation in Russian speech. The system combines audio analysis with linguistic processing to provide accurate stress detection and timing information.

### Key Features

- **Stress Detection**: Identifies stressed syllables in Russian speech using audio analysis
- **Temporal Segmentation**: Provides precise timing information for word boundaries
- **Audio Processing**: Supports various audio formats including MP3 through librosa
- **Syllable Analysis**: Uses rusyllab for accurate Russian syllable splitting
- **Deep Learning**: Implements neural networks with PyTorch Lightning
- **MLOps Pipeline**: Complete workflow with DVC, MLflow, and Hydra configuration

### Technical Components

- **Audio Encoder**: Processes speech signals for feature extraction
- **Stress Classifier**: Binary classification for stressed/unstressed syllables
- **Timing Autoencoder**: Reconstructs temporal information for segmentation
- **Configuration Management**: Hydra-based YAML configurations
- **Experiment Tracking**: MLflow integration for model versioning
- **Data Versioning**: DVC for dataset and model artifact management

## Setup

### Prerequisites

- Python 3.9-3.11
- CUDA-compatible GPU (recommended)
- Git
- Poetry

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd speech-stress-analyzer
```

2. Install Poetry if not already installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:
```bash
poetry install
```

4. Activate the virtual environment:
```bash
poetry shell
```

5. Install pre-commit hooks:
```bash
pre-commit install
```

6. Verify installation:
```bash
pre-commit run --all-files
```

### DVC Setup

Initialize DVC for data management:
```bash
dvc init
```

## Train

### Data Management with DVC

**This project uses DVC (Data Version Control) for data management. Data is automatically downloaded when needed.**

#### Automatic Data Download

Data is automatically downloaded during training and inference:
- **Training**: Set `dvc.pull_data_on_train=true` in config (default)
- **Inference**: Set `dvc.pull_model_on_infer=true` in config (default)

#### Manual Data Download

To download data manually:
```bash
poetry run python -m speech_stress_analyzer.commands download_data
```

Or using DVC directly:
```bash
poetry run dvc pull
```

### Data Preparation

**If setting up data from scratch:**

1. **Create data directories:**
```bash
mkdir -p raw_data
```

2. **Add your audio and text files:**
   - Place your Russian speech audio file as `raw_data/audio.mp3`
   - Create `raw_data/text.txt` with the corresponding Russian transcript
   - Example transcript format:
   ```
   Привет, меня зовут Анна.
   Сегодня хорошая погода.
   Давайте поговорим о машинном обучении.
   ```

3. **Add files to DVC:**
```bash
poetry run dvc add raw_data/audio.mp3 raw_data/text.txt
git add raw_data/*.dvc .gitignore
poetry run dvc push  # Upload to remote storage
```

4. **Verify data structure:**
```bash
raw_data/
├── audio.mp3.dvc    # DVC metadata file (tracked by Git)
├── text.txt.dvc     # DVC metadata file (tracked by Git)
├── audio.mp3        # Actual data (managed by DVC)
└── text.txt         # Actual data (managed by DVC)
```

**Note:** Only `.dvc` files are tracked by Git. Actual data files are managed by DVC and stored in remote storage.

### Preprocessing

Run data preprocessing to prepare training data:
```bash
poetry run python -m speech_stress_analyzer.commands preprocess
```

This will:
- Load and resample your audio to 16kHz
- Split Russian text into syllables using `rusyllab`
- Apply basic stress detection rules
- Create temporal alignments between audio and text
- Generate training metadata in JSON format
- Save processed files to `processed_data/` directory

Optional overrides:
```bash
poetry run python -m speech_stress_analyzer.commands preprocess audio.sample_rate=22050
```

**Expected output:**
```
processed_data/
├── audio_processed.wav      # Resampled audio
└── audio_training_data.json # Syllable alignments and stress annotations
```

### Model Training

Start training with default configuration:
```bash
poetry run python -m speech_stress_analyzer.commands train
```

Training with custom parameters:
```bash
poetry run python -m speech_stress_analyzer.commands train train.num_epochs=10 train.batch_size=32
```

Training with GPU (recommended):
```bash
poetry run python -m speech_stress_analyzer.commands train train.accelerator=cuda train.devices=1
```

### Monitoring

Monitor training progress with MLflow:
```bash
mlflow ui --host 127.0.0.1 --port 8080
```

Access the interface at http://127.0.0.1:8080

## Production Preparation

### Model Export to ONNX

Export trained model to ONNX format:
```bash
poetry run python -m speech_stress_analyzer.commands export_onnx
```

### TensorRT Conversion

Convert ONNX model to TensorRT for optimized inference:
```bash
poetry run python -m speech_stress_analyzer.commands export_tensorrt
```

### Model Artifacts

The following artifacts are generated for production deployment:

- **PyTorch Checkpoint**: `outputs/checkpoints/best_model.ckpt`
- **ONNX Model**: `models/speech_analyzer.onnx`
- **TensorRT Engine**: `models/speech_analyzer.plan`
- **Configuration**: All YAML configs in `configs/`
- **Preprocessing Metadata**: Stored in `processed_data/`

### Dependencies for Production

Minimal dependencies for inference:
- `torch`
- `onnxruntime` (for ONNX inference)
- `librosa` (for audio processing)
- `rusyllab` (for syllable processing)

## Infer

### Running Inference

Run inference on new audio data:
```bash
poetry run python -m speech_stress_analyzer.commands infer path/to/audio.wav
```

With custom model:
```bash
poetry run python -m speech_stress_analyzer.commands infer path/to/audio.wav infer.model_checkpoint_path=path/to/model.ckpt
```

### Input Format

- **Audio Format**: WAV, MP3, or other formats supported by librosa
- **Sample Rate**: Automatically resampled to 16kHz
- **Duration**: Any length (processed in segments if needed)

### Output Format

The inference produces:
- **Stress Predictions**: Binary labels for each syllable
- **Timing Information**: Start/end times for each word/syllable
- **Confidence Scores**: Prediction confidence values
- **Processed Audio**: Resampled and normalized audio features

### Example Output

```json
{
  "predictions": [
    {
      "word": "привет",
      "syllables": ["при", "вет"],
      "stress": [false, true],
      "timings": [
        {"start": 0.0, "end": 0.3},
        {"start": 0.3, "end": 0.7}
      ]
    }
  ]
}
```

## Development

### Code Quality

Run all quality checks:
```bash
pre-commit run --all-files
```

Individual tools:
```bash
black speech_stress_analyzer/
isort speech_stress_analyzer/
flake8 speech_stress_analyzer/
```

### Configuration Management

All configurations are managed through Hydra YAML files in the `configs/` directory:

- `config.yaml`: Main configuration
- `train.yaml`: Training parameters
- `model.yaml`: Model architecture
- `data_preprocessing.yaml`: Data processing settings
- `logging.yaml`: MLflow and logging setup

### Data Management

Add new data to DVC tracking:
```bash
dvc add data/new_dataset.wav
git add data/new_dataset.wav.dvc
git commit -m "Add new dataset"
dvc push
```

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Run pre-commit checks
5. Submit a pull request 