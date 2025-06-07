# speech_stress_analyzer/infer.py

import torch
import torch.nn.functional as F
import librosa
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from typing import List, Dict, Tuple

from .models import SpeechStressAnalysisModel

def load_model_from_checkpoint(checkpoint_path: str, config: DictConfig) -> SpeechStressAnalysisModel:
    """Load trained model from checkpoint."""
    model = SpeechStressAnalysisModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config
    )
    model.eval()
    return model

def segment_audio_by_syllables(audio_path: str, syllable_timings: List[Dict], sample_rate: int = 16000) -> List[torch.Tensor]:
    """Segment audio file into syllables based on timing information."""
    # Load audio
    waveform, sr = librosa.load(audio_path, sr=sample_rate)
    
    syllable_segments = []
    
    for syllable_data in syllable_timings:
        start_time = syllable_data.get('start_time', 0)
        end_time = syllable_data.get('end_time', start_time + 0.3)
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        if start_sample >= len(waveform) or end_sample <= start_sample:
            continue
        
        audio_segment = waveform[start_sample:end_sample]
        
        # Pad or truncate to fixed length (300ms)
        target_length = int(0.3 * sr)
        if len(audio_segment) < target_length:
            audio_segment = np.pad(audio_segment, (0, target_length - len(audio_segment)))
        else:
            audio_segment = audio_segment[:target_length]
        
        # Convert to tensor and add channel dimension
        audio_tensor = torch.from_numpy(audio_segment).float().unsqueeze(0)  # [1, seq_len]
        syllable_segments.append(audio_tensor)
    
    return syllable_segments

def predict_stress_for_syllables(model: SpeechStressAnalysisModel, audio_segments: List[torch.Tensor]) -> List[Dict]:
    """Predict stress for each syllable audio segment."""
    predictions = []
    
    with torch.no_grad():
        for i, audio_segment in enumerate(audio_segments):
            # Add batch dimension: [1, 1, seq_len]
            audio_batch = audio_segment.unsqueeze(0)
            
            # Get model outputs
            outputs = model(audio_batch)
            stress_logits = outputs["stress_logits"]
            
            # Get probabilities and prediction
            stress_probs = F.softmax(stress_logits, dim=1)
            stress_pred = torch.argmax(stress_logits, dim=1)
            
            # Extract values
            unstressed_prob = stress_probs[0, 0].item()
            stressed_prob = stress_probs[0, 1].item()
            is_stressed = bool(stress_pred[0].item())
            
            predictions.append({
                'syllable_index': i,
                'is_stressed': is_stressed,
                'stress_probability': stressed_prob,
                'unstressed_probability': unstressed_prob,
                'confidence': max(stressed_prob, unstressed_prob)
            })
    
    return predictions

def create_dummy_syllable_timings(text_lines: List[str], audio_duration: float) -> List[Dict]:
    """Create dummy syllable timings for inference when no timing data is available."""
    syllable_timings = []
    total_syllables = 0
    
    # Count syllables (simplified vowel counting)
    vowels = 'аеёиоуыэюя'
    for line in text_lines:
        words = line.strip().split()
        for word in words:
            syllable_count = max(1, sum(1 for char in word.lower() if char in vowels))
            total_syllables += syllable_count
    
    # Distribute time across syllables
    avg_syllable_duration = audio_duration / max(1, total_syllables)
    current_time = 0.0
    
    for line_idx, line in enumerate(text_lines):
        words = line.strip().split()
        for word_idx, word in enumerate(words):
            # Simple syllable segmentation
            syllable_count = max(1, sum(1 for char in word.lower() if char in vowels))
            
            for syll_idx in range(syllable_count):
                syllable_timings.append({
                    'start_time': current_time,
                    'end_time': current_time + avg_syllable_duration,
                    'syllable_text': f"{word}_syll_{syll_idx}",
                    'word': word,
                    'line_index': line_idx,
                    'word_index': word_idx,
                    'syllable_index': syll_idx
                })
                current_time += avg_syllable_duration
    
    return syllable_timings

def run_inference(config: DictConfig, input_audio: str) -> Dict:
    """
    Run stress detection inference on new audio data.
    """
    print(f"Starting stress detection inference on: {input_audio}")
    
    # 1. Load model
    checkpoint_path = config.infer.model_checkpoint_path
    if not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using dummy inference...")
        return run_dummy_inference(config, input_audio)
    
    try:
        model = load_model_from_checkpoint(checkpoint_path, config)
        print(f"Model loaded from: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using dummy inference...")
        return run_dummy_inference(config, input_audio)
    
    # 2. Load and preprocess audio
    input_audio_path = Path(input_audio)
    if not input_audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_audio}")
    
    # Get audio duration
    waveform, sr = librosa.load(input_audio_path, sr=None)
    audio_duration = len(waveform) / sr
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # 3. Load text for syllable timing (if available)
    # Look for corresponding text file
    text_file = input_audio_path.with_suffix('.txt')
    if text_file.exists():
        print(f"Loading text from: {text_file}")
        with open(text_file, 'r', encoding='utf-8') as f:
            text_lines = [line.strip() for line in f if line.strip()]
    else:
        print("No text file found. Using dummy text.")
        text_lines = ["Пример текста для анализа ударений."]
    
    # 4. Create syllable timings (simplified approach)
    print("Creating syllable alignments...")
    syllable_timings = create_dummy_syllable_timings(text_lines, audio_duration)
    
    # 5. Segment audio
    print(f"Segmenting audio into {len(syllable_timings)} syllables...")
    audio_segments = segment_audio_by_syllables(
        str(input_audio_path), 
        syllable_timings, 
        sample_rate=config.audio.sample_rate
    )
    
    # 6. Predict stress
    print("Predicting stress for each syllable...")
    stress_predictions = predict_stress_for_syllables(model, audio_segments)
    
    # 7. Combine results
    results = {
        'audio_file': str(input_audio_path),
        'audio_duration': audio_duration,
        'total_syllables': len(syllable_timings),
        'stressed_syllables': sum(1 for pred in stress_predictions if pred['is_stressed']),
        'text_lines': text_lines,
        'predictions': []
    }
    
    # Combine timing and prediction data
    for timing, prediction in zip(syllable_timings, stress_predictions):
        combined = {
            **timing,
            **prediction
        }
        results['predictions'].append(combined)
    
    # 8. Print summary
    stress_ratio = results['stressed_syllables'] / results['total_syllables']
    print(f"\n=== Inference Results ===")
    print(f"Total syllables: {results['total_syllables']}")
    print(f"Stressed syllables: {results['stressed_syllables']}")
    print(f"Stress ratio: {stress_ratio:.2%}")
    
    # Show some examples
    print(f"\nTop 5 most confident stress predictions:")
    confident_preds = sorted(results['predictions'], key=lambda x: x['confidence'], reverse=True)[:5]
    for pred in confident_preds:
        stress_status = "STRESSED" if pred['is_stressed'] else "unstressed"
        print(f"  {pred['syllable_text']}: {stress_status} (confidence: {pred['confidence']:.3f})")
    
    return results

def run_dummy_inference(config: DictConfig, input_audio: str) -> Dict:
    """Dummy inference when model is not available."""
    print("Running dummy stress inference...")
    
    input_audio_path = Path(input_audio)
    if not input_audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_audio}")
    
    # Get audio duration
    waveform, sr = librosa.load(input_audio_path, sr=None)
    audio_duration = len(waveform) / sr
    
    # Create dummy results
    dummy_predictions = []
    for i in range(10):  # Dummy 10 syllables
        dummy_predictions.append({
            'syllable_index': i,
            'syllable_text': f"dummy_syll_{i}",
            'start_time': i * 0.5,
            'end_time': (i + 1) * 0.5,
            'is_stressed': i % 3 == 0,  # Every 3rd syllable is stressed
            'stress_probability': 0.7 if i % 3 == 0 else 0.3,
            'confidence': 0.7
        })
    
    results = {
        'audio_file': str(input_audio_path),
        'audio_duration': audio_duration,
        'total_syllables': len(dummy_predictions),
        'stressed_syllables': sum(1 for pred in dummy_predictions if pred['is_stressed']),
        'text_lines': ["Dummy text for inference"],
        'predictions': dummy_predictions,
        'note': "This is dummy inference - train a model first!"
    }
    
    print(f"Dummy inference completed with {len(dummy_predictions)} syllables")
    return results

if __name__ == "__main__":
    print("This script is designed to be run via `python -m speech_stress_analyzer.commands infer <audio_file>`") 