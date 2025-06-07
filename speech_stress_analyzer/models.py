# speech_stress_analyzer/models.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import Dict, Any, Tuple
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

try:
    from transformers import AutoModel, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Using dummy audio encoder.")

class AudioEncoder(nn.Module):
    """Audio encoder using SdHUBERT or fallback dummy encoder."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Try to use SdHUBERT or similar audio model
                # Note: Replace with actual SdHUBERT model path when available
                model_name = config.get("sdhubert_model_name", "microsoft/wavlm-base-plus")
                self.audio_model = AutoModel.from_pretrained(model_name)
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.feature_dim = self.audio_model.config.hidden_size
                self.use_pretrained = True
                print(f"Using pretrained audio model: {model_name}")
            except Exception as e:
                print(f"Could not load pretrained model: {e}. Using dummy encoder.")
                self.use_pretrained = False
        else:
            self.use_pretrained = False
        
        if not self.use_pretrained:
            # Dummy encoder for testing - use smaller kernels for short audio segments
            self.feature_dim = config.sdhubert.dummy_sdhubert_embedding_dim
            self.dummy_conv = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),  # Smaller kernel
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # Smaller kernel
                nn.ReLU(),
                nn.Conv1d(128, self.feature_dim, kernel_size=3, stride=2, padding=1),  # Smaller kernel
                nn.AdaptiveAvgPool1d(1)  # This will work with any input size
            )
    
    def forward(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through audio encoder.
        
        Args:
            audio_waveform: [batch_size, 1, seq_len] or [batch_size, seq_len]
            
        Returns:
            audio_features: [batch_size, feature_dim]
        """
        if self.use_pretrained:
            # Process with pretrained model
            if audio_waveform.dim() == 3:
                audio_waveform = audio_waveform.squeeze(1)  # Remove channel dim for processor
            
            # Note: This is a simplified approach. In practice, you'd need proper
            # segmentation and windowing for long audio files
            with torch.no_grad():
                inputs = self.processor(
                    audio_waveform.cpu().numpy(), 
                    sampling_rate=16000, 
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(audio_waveform.device) for k, v in inputs.items()}
                outputs = self.audio_model(**inputs)
                # Use mean pooling over sequence dimension
                audio_features = outputs.last_hidden_state.mean(dim=1)
        else:
            # Dummy processing
            if audio_waveform.dim() == 2:
                audio_waveform = audio_waveform.unsqueeze(1)  # Add channel dim
            audio_features = self.dummy_conv(audio_waveform).squeeze(-1)
        
        return audio_features

class StressClassifier(nn.Module):
    """Classifier for predicting stress on syllables."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        
        audio_feature_dim = config.sdhubert.dummy_sdhubert_embedding_dim
        hidden_dim = config.classifier.hidden_dim
        num_classes = config.classifier.num_classes  # 2 for binary (stressed/unstressed)
        
        self.classifier = nn.Sequential(
            nn.Linear(audio_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Predict stress for syllables.
        
        Args:
            audio_features: [batch_size, feature_dim]
            
        Returns:
            stress_logits: [batch_size, num_classes]
        """
        return self.classifier(audio_features)

class SyllableAlignmentHead(nn.Module):
    """Head for predicting syllable timing alignment (optional)."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        
        audio_feature_dim = config.sdhubert.dummy_sdhubert_embedding_dim
        encoding_dim = config.autoencoder.encoding_dim
        
        # Simple timing prediction head
        self.timing_head = nn.Sequential(
            nn.Linear(audio_feature_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, 2)  # start_time, duration
        )
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Predict syllable timing.
        
        Args:
            audio_features: [batch_size, feature_dim]
            
        Returns:
            timing_predictions: [batch_size, 2] (start_time, duration)
        """
        return self.timing_head(audio_features)

class SpeechStressAnalysisModel(pl.LightningModule):
    """Main PyTorch Lightning model for speech stress analysis."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Model components
        self.audio_encoder = AudioEncoder(config)
        self.stress_classifier = StressClassifier(config)
        
        # Optional timing prediction
        self.use_timing_prediction = config.get("use_timing_prediction", False)
        if self.use_timing_prediction:
            self.timing_head = SyllableAlignmentHead(config)
        
        # Loss weights
        self.stress_loss_weight = 1.0
        self.timing_loss_weight = config.get("timing_loss_weight", 0.5)
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, audio_waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            audio_waveform: [batch_size, 1, seq_len] audio tensor
            
        Returns:
            Dictionary with model outputs
        """
        # Extract audio features
        audio_features = self.audio_encoder(audio_waveform)
        
        # Predict stress
        stress_logits = self.stress_classifier(audio_features)
        
        outputs = {
            "stress_logits": stress_logits,
            "audio_features": audio_features
        }
        
        # Optional timing prediction
        if self.use_timing_prediction:
            timing_predictions = self.timing_head(audio_features)
            outputs["timing_predictions"] = timing_predictions
        
        return outputs
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for training."""
        losses = {}
        
        # Stress classification loss
        stress_logits = outputs["stress_logits"]
        stress_labels = batch["stress_labels"]  # [batch_size]
        
        stress_loss = F.cross_entropy(stress_logits, stress_labels)
        losses["stress_loss"] = stress_loss
        
        # Optional timing loss
        total_loss = self.stress_loss_weight * stress_loss
        
        if self.use_timing_prediction and "timing_labels" in batch:
            timing_predictions = outputs["timing_predictions"]
            timing_labels = batch["timing_labels"]  # [batch_size, 2]
            
            timing_loss = F.mse_loss(timing_predictions, timing_labels)
            losses["timing_loss"] = timing_loss
            total_loss += self.timing_loss_weight * timing_loss
        
        losses["total_loss"] = total_loss
        return losses
    
    def compute_metrics(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        # Stress classification metrics
        stress_logits = outputs["stress_logits"]
        stress_labels = labels["stress_labels"]
        
        stress_preds = torch.argmax(stress_logits, dim=1)
        
        # Convert to numpy for sklearn metrics
        stress_preds_np = stress_preds.cpu().numpy()
        stress_labels_np = stress_labels.cpu().numpy()
        
        # Accuracy
        accuracy = accuracy_score(stress_labels_np, stress_preds_np)
        metrics["stress_accuracy"] = accuracy
        
        # Precision, Recall, F1 for stressed class (class 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            stress_labels_np, stress_preds_np, average='binary', pos_label=1, zero_division=0
        )
        
        metrics["stress_precision"] = precision
        metrics["stress_recall"] = recall
        metrics["stress_f1"] = f1
        
        return metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        audio_waveform = batch["audio_waveform"]
        outputs = self(audio_waveform)
        
        loss_dict = self.compute_loss(batch, outputs)
        total_loss = loss_dict["total_loss"]
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train/{loss_name}", loss_value, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        audio_waveform = batch["audio_waveform"]
        outputs = self(audio_waveform)
        
        loss_dict = self.compute_loss(batch, outputs)
        
        # Compute metrics
        metrics = self.compute_metrics(outputs, batch)
        
        # Store for epoch-end aggregation
        step_output = {
            **loss_dict,
            **metrics,
            "stress_logits": outputs["stress_logits"].detach(),
            "stress_labels": batch["stress_labels"].detach()
        }
        
        self.validation_step_outputs.append(step_output)
        return step_output
    
    def on_validation_epoch_end(self) -> None:
        """Aggregate validation results."""
        if not self.validation_step_outputs:
            return
        
        # Average losses and metrics
        avg_loss = torch.stack([x["total_loss"] for x in self.validation_step_outputs]).mean()
        avg_stress_loss = torch.stack([x["stress_loss"] for x in self.validation_step_outputs]).mean()
        
        avg_accuracy = np.mean([x["stress_accuracy"] for x in self.validation_step_outputs])
        avg_precision = np.mean([x["stress_precision"] for x in self.validation_step_outputs])
        avg_recall = np.mean([x["stress_recall"] for x in self.validation_step_outputs])
        avg_f1 = np.mean([x["stress_f1"] for x in self.validation_step_outputs])
        
        # Log aggregated metrics
        self.log("val/total_loss", avg_loss, prog_bar=True)
        self.log("val/stress_loss", avg_stress_loss)
        self.log("val/stress_accuracy", avg_accuracy, prog_bar=True)
        self.log("val/stress_precision", avg_precision)
        self.log("val/stress_recall", avg_recall)
        self.log("val/stress_f1", avg_f1, prog_bar=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer_config = self.config.optimizer
        
        if optimizer_config.name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.get("weight_decay", 0.01)
            )
        elif optimizer_config.name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.get("weight_decay", 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.name}")
        
        # Scheduler
        if "scheduler" in self.config and self.config.scheduler.name:
            scheduler_config = self.config.scheduler
            
            if scheduler_config.name == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config.step_size,
                    gamma=scheduler_config.gamma
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch"
                    }
                }
            elif scheduler_config.name == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_config.get("factor", 0.5),
                    patience=scheduler_config.get("patience", 5),
                    verbose=True
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/total_loss",
                        "interval": "epoch"
                    }
                }
        
        return optimizer
    
    def predict_stress(self, audio_waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict stress for given audio.
        
        Args:
            audio_waveform: [1, seq_len] or [seq_len] audio tensor
            
        Returns:
            stress_probabilities: [num_syllables, 2] probabilities
            stress_predictions: [num_syllables] binary predictions
        """
        self.eval()
        
        if audio_waveform.dim() == 1:
            audio_waveform = audio_waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
        elif audio_waveform.dim() == 2:
            audio_waveform = audio_waveform.unsqueeze(0)  # [1, channels, seq_len]
        
        with torch.no_grad():
            outputs = self(audio_waveform)
            stress_logits = outputs["stress_logits"]
            stress_probs = F.softmax(stress_logits, dim=1)
            stress_preds = torch.argmax(stress_logits, dim=1)
        
        return stress_probs, stress_preds

if __name__ == "__main__":
    print("This script defines the PyTorch Lightning model.")
    print("It should be imported and used by the training script (train.py) via commands.py.")
    # Example instantiation (requires a dummy config):
    # from omegaconf import OmegaConf
    # dummy_model_config = OmegaConf.create({
    #     "model": {
    #         "sdhubert_embedding_dim": 256,
    #         "classifier": {"num_classes": 2, "hidden_dim": 128},
    #         "autoencoder": {"encoding_dim": 64},
    #         "timing_loss_weight": 0.5
    #     },
    #     "train": {
    #         "optimizer": {"name": "Adam", "lr": 1e-3},
    #         # "scheduler": { "name": "StepLR", "step_size": 10, "gamma": 0.1 }
    #     }
    # })
    # model = SpeechAnalysisModel(dummy_model_config)
    # print("SpeechAnalysisModel instantiated successfully with dummy config.")
    # print(model)
    pass 