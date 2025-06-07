# speech_stress_analyzer/train.py

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset #, random_split # If splitting in DataModule
from omegaconf import DictConfig
import torch # For dummy data
import os
import mlflow
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import numpy as np

# Import your model from models.py
from .models import SpeechStressAnalysisModel

class SyllableStressDataset(Dataset):
    """Dataset for syllable-level stress detection."""
    
    def __init__(self, config: DictConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.audio_segments = []
        self.stress_labels = []
        self.syllable_info = []
        
        # Load processed data
        processed_data_dir = Path(config.processed_data_dir)
        
        # Find training data files
        training_files = list(processed_data_dir.glob("*_training_data.json"))
        
        if not training_files:
            raise FileNotFoundError(f"No training data found in {processed_data_dir}")
        
        print(f"Loading {len(training_files)} training data files...")
        
        for file_path in training_files:
            self._load_training_file(file_path, processed_data_dir)
        
        print(f"Loaded {len(self.audio_segments)} syllable samples for {split}")
        print(f"Stress distribution: {np.mean(self.stress_labels):.2%} stressed syllables")
    
    def _load_training_file(self, metadata_file: Path, data_dir: Path):
        """Load training data from a metadata file."""
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load audio file
        audio_file = data_dir / metadata['audio_filepath']
        if not audio_file.exists():
            print(f"Warning: Audio file not found: {audio_file}")
            return
        
        # Load full audio
        waveform, sr = librosa.load(audio_file, sr=self.config.audio.sample_rate)
        
        # Process each syllable
        for line_data in metadata['syllable_alignments']:
            for syllable_data in line_data['syllables']:
                # Extract audio segment for this syllable
                start_time = syllable_data['start_time']
                end_time = syllable_data['end_time']
                
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Ensure we have valid segment
                if start_sample >= len(waveform) or end_sample <= start_sample:
                    continue
                
                audio_segment = waveform[start_sample:end_sample]
                
                # Pad or truncate to fixed length
                target_length = int(0.3 * sr)  # 300ms per syllable
                if len(audio_segment) < target_length:
                    # Pad with zeros
                    audio_segment = np.pad(audio_segment, (0, target_length - len(audio_segment)))
                else:
                    # Truncate
                    audio_segment = audio_segment[:target_length]
                
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio_segment).float()
                
                # Get stress label
                stress_label = int(syllable_data['is_stressed'])
                
                self.audio_segments.append(audio_tensor)
                self.stress_labels.append(stress_label)
                self.syllable_info.append(syllable_data)
    
    def __len__(self):
        return len(self.audio_segments)
    
    def __getitem__(self, idx):
        audio_segment = self.audio_segments[idx]
        stress_label = self.stress_labels[idx]
        
        # Add batch dimension for audio: [1, seq_len]
        if audio_segment.dim() == 1:
            audio_segment = audio_segment.unsqueeze(0)
        
        return {
            'audio_waveform': audio_segment,
            'stress_labels': torch.tensor(stress_label, dtype=torch.long),
            'syllable_info': self.syllable_info[idx]
        }

class SpeechDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for speech stress analysis."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.get("num_workers", 2)
        
        # Data will be split from single dataset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        if stage == "fit" or stage is None:
            # Load full dataset
            full_dataset = SyllableStressDataset(self.config, split="full")
            
            # Split into train/val (80/20)
            dataset_size = len(full_dataset)
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.config.get("random_seed", 42))
            )
            
            print(f"Training set size: {len(self.train_dataset)}")
            print(f"Validation set size: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            # Use validation set as test set for now
            if self.val_dataset is None:
                self.setup("fit")
            self.test_dataset = self.val_dataset
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )

def run_training(config: DictConfig) -> None:
    """
    Run model training with PyTorch Lightning.
    """
    print("Starting model training...")
    
    # Set random seed for reproducibility
    if hasattr(config, 'random_seed'):
        pl.seed_everything(config.random_seed, workers=True)
    
    # Ensure checkpoint directory exists
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # 1. Data Module
    data_module = SpeechDataModule(config)
    
    # 2. Initialize Model
    model = SpeechStressAnalysisModel(config)
    
    # 3. Setup Callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename='stress-model-{epoch:02d}-{val/stress_f1:.3f}',
        monitor=config.callbacks.model_checkpoint.monitor,
        mode=config.callbacks.model_checkpoint.mode,
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        save_last=config.callbacks.model_checkpoint.save_last,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        patience=config.callbacks.early_stopping.patience,
        verbose=config.callbacks.early_stopping.verbose,
        mode=config.callbacks.early_stopping.mode
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # 4. Setup Logger
    mlf_logger = MLFlowLogger(
        experiment_name=config.mlflow_experiment_name,
        tracking_uri=config.mlflow_tracking_uri,
        run_name=config.run_name
    )
    
    # 5. Configure Trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", "auto"),
        logger=mlf_logger,
        callbacks=callbacks,
        deterministic=config.get("deterministic_ops", False),
        precision=config.get("precision", 32),
        gradient_clip_val=config.get("gradient_clip_val", 0.0),
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        log_every_n_steps=50,
        val_check_interval=config.get("val_check_interval", 1.0),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1)
    )
    
    print(f"Trainer configured:")
    print(f"  Max epochs: {config.num_epochs}")
    print(f"  Accelerator: {config.get('accelerator', 'auto')}")
    print(f"  Devices: {config.get('devices', 'auto')}")
    print(f"  Precision: {config.get('precision', 32)}")
    
    # 6. Start Training
    try:
        trainer.fit(model, data_module)
        
        # 7. Test the model
        print("Running final evaluation...")
        trainer.test(model, data_module)
        
        # 8. Save best model info
        best_model_path = checkpoint_callback.best_model_path
        print(f"Best model saved at: {best_model_path}")
        
        # Log final metrics
        mlflow.log_param("best_model_path", best_model_path)
        mlflow.log_param("final_epoch", trainer.current_epoch)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    return trainer, model

if __name__ == "__main__":
    print("This script is designed to be run via `python -m speech_stress_analyzer.commands train`")
    # Example of how it might be called with a dummy config (for illustration):
    # from omegaconf import OmegaConf
    # # Create a minimal dummy config similar to what hydra would provide
    # dummy_hydra_cfg = OmegaConf.create({
    #     "train": {
    #         "random_seed": 42,
    #         "batch_size": 2, # Small for dummy data
    #         "num_workers": 1,
    #         "num_epochs": 1, # Minimal epochs for test
    #         "accelerator": "cpu",
    #         "devices": 1,
    #         "checkpoint_dir": "outputs/checkpoints_test",
    #         "callbacks": {
    #             "model_checkpoint": {"save_top_k": 1, "monitor": "val/total_loss", "mode": "min"},
    #             "early_stopping": {"monitor": "val/total_loss", "patience": 3, "mode": "min"}
    #         }
    #     },
    #     "model": { # Matching what SpeechAnalysisModel and YourDataset expect
    #         "sdhubert_embedding_dim": 256,
    #         "classifier": {"num_classes": 2, "hidden_dim": 128},
    #         "autoencoder": {"encoding_dim": 64},
    #         "timing_loss_weight": 0.5,
    #         "optimizer": {"name": "Adam", "lr": 1e-4},
    #         # "scheduler": { "name": "StepLR", "step_size": 10, "gamma": 0.1 }
    #     },
    #     "logging": {
    #         "mlflow_tracking_uri": "file:./mlruns_test", # Local test mlflow
    #         "mlflow_experiment_name": "Test_Experiment"
    #     },
    #     "data": {
    #         "processed_data_dir": "data/processed_test" # Dummy path
    #     }
    # })
    # # Ensure dummy directories exist for the test run
    # os.makedirs(dummy_hydra_cfg.train.checkpoint_dir, exist_ok=True)
    # os.makedirs(dummy_hydra_cfg.data.processed_data_dir, exist_ok=True)
    # # Mock an active MLflow run if commands.py isn't setting it up
    # import mlflow
    # if not mlflow.active_run():
    #     mlflow.set_tracking_uri(dummy_hydra_cfg.logging.mlflow_tracking_uri)
    #     mlflow.set_experiment(dummy_hydra_cfg.logging.mlflow_experiment_name)
    #     mlflow.start_run()
    # run_training(dummy_hydra_cfg)
    # if mlflow.active_run():
    #    mlflow.end_run()
    pass 