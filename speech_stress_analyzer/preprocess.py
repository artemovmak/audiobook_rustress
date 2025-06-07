# speech_stress_analyzer/preprocess.py

from omegaconf import DictConfig
import os
import torchaudio
import librosa
import torch
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

try:
    from rusyllab import split_word, split_words
    RUSYLLAB_AVAILABLE = True
except ImportError:
    RUSYLLAB_AVAILABLE = False
    print("Warning: rusyllab not available. Using placeholder stress detection.")

def get_stress_marked_text_real(text_lines: List[str]) -> List[Dict]:
    """Real syllable splitting using rusyllab with simple stress detection."""
    if not RUSYLLAB_AVAILABLE:
        print("Falling back to placeholder stress detection.")
        return get_stress_marked_text_placeholder(text_lines)
    
    stressed_lines = []
    
    for line_idx, line in enumerate(text_lines):
        words = line.strip().split()
        stressed_words_info = []
        
        try:
            # Use rusyllab to split words into syllables
            syllables_with_spaces = split_words(words)
            
            # Process the syllables to reconstruct word information
            current_word_idx = 0
            current_syllables = []
            
            for item in syllables_with_spaces:
                if item == ' ':  # Space separator between words
                    if current_syllables and current_word_idx < len(words):
                        word = words[current_word_idx]
                        syllables = current_syllables
                        
                        # Simple stress detection: stress the first vowel-containing syllable
                        stress_index = None
                        vowels = 'аеёиоуыэюя'
                        for i, syllable in enumerate(syllables):
                            if any(char.lower() in vowels for char in syllable):
                                stress_index = i
                                break
                        
                        # Create stressed display
                        stressed_word_display = "".join(syllables)
                        if stress_index is not None:
                            syllables_copy = list(syllables)
                            syllables_copy[stress_index] += "'"
                            stressed_word_display = "".join(syllables_copy)
                        
                        # Find stress character positions
                        stress_char_positions = []
                        if stress_index is not None:
                            stressed_syllable = syllables[stress_index]
                            preceding_len = sum(len(s) for s in syllables[:stress_index])
                            for i, char in enumerate(stressed_syllable):
                                if char.lower() in vowels:
                                    stress_char_positions.append(preceding_len + i)
                                    break
                        
                        stressed_words_info.append({
                            'original': word,
                            'clean': re.sub(r'[^\w\s-]', '', word),
                            'stressed_display': stressed_word_display,
                            'syllables': syllables,
                            'stress_syllable_index': stress_index,
                            'stress_char_positions': stress_char_positions
                        })
                        
                        current_word_idx += 1
                        current_syllables = []
                else:
                    current_syllables.append(item)
            
            # Handle the last word if there's no trailing space
            if current_syllables and current_word_idx < len(words):
                word = words[current_word_idx]
                syllables = current_syllables
                
                # Simple stress detection
                stress_index = None
                vowels = 'аеёиоуыэюя'
                for i, syllable in enumerate(syllables):
                    if any(char.lower() in vowels for char in syllable):
                        stress_index = i
                        break
                
                stressed_word_display = "".join(syllables)
                if stress_index is not None:
                    syllables_copy = list(syllables)
                    syllables_copy[stress_index] += "'"
                    stressed_word_display = "".join(syllables_copy)
                
                stress_char_positions = []
                if stress_index is not None:
                    stressed_syllable = syllables[stress_index]
                    preceding_len = sum(len(s) for s in syllables[:stress_index])
                    for i, char in enumerate(stressed_syllable):
                        if char.lower() in vowels:
                            stress_char_positions.append(preceding_len + i)
                            break
                
                stressed_words_info.append({
                    'original': word,
                    'clean': re.sub(r'[^\w\s-]', '', word),
                    'stressed_display': stressed_word_display,
                    'syllables': syllables,
                    'stress_syllable_index': stress_index,
                    'stress_char_positions': stress_char_positions
                })
                
        except Exception as e:
            print(f"Warning: Could not process line '{line}' with rusyllab: {e}")
            # Fall back to word-by-word processing
            for word in words:
                try:
                    syllables = split_word(word)
                    if not syllables:
                        syllables = [word]
                    
                    # Simple stress detection
                    stress_index = None
                    vowels = 'аеёиоуыэюя'
                    for i, syllable in enumerate(syllables):
                        if any(char.lower() in vowels for char in syllable):
                            stress_index = i
                            break
                    
                    stressed_word_display = "".join(syllables)
                    if stress_index is not None:
                        syllables_copy = list(syllables)
                        syllables_copy[stress_index] += "'"
                        stressed_word_display = "".join(syllables_copy)
                    
                    stress_char_positions = []
                    if stress_index is not None:
                        stressed_syllable = syllables[stress_index]
                        preceding_len = sum(len(s) for s in syllables[:stress_index])
                        for i, char in enumerate(stressed_syllable):
                            if char.lower() in vowels:
                                stress_char_positions.append(preceding_len + i)
                                break
                    
                    stressed_words_info.append({
                        'original': word,
                        'clean': re.sub(r'[^\w\s-]', '', word),
                        'stressed_display': stressed_word_display,
                        'syllables': syllables,
                        'stress_syllable_index': stress_index,
                        'stress_char_positions': stress_char_positions
                    })
                except Exception as word_e:
                    print(f"Warning: Could not process word '{word}': {word_e}")
                    # Ultimate fallback
                    clean_word = re.sub(r'[^\w\s]', '', word)
                    stressed_words_info.append({
                        'original': word,
                        'clean': clean_word,
                        'stressed_display': clean_word + "'",
                        'syllables': [clean_word],
                        'stress_syllable_index': 0,
                        'stress_char_positions': [0]
                    })
        
        stressed_lines.append({
            'line_index': line_idx,
            'original_text': line,
            'words': stressed_words_info
        })
    
    return stressed_lines

def get_stress_marked_text_placeholder(text_lines: List[str]) -> List[Dict]:
    """Placeholder for stress detection when libraries are not available."""
    # This function is now a fallback
    # ... (implementation remains similar, generating dummy data)
    stressed_lines = []
    
    for line_idx, line in enumerate(text_lines):
        words = line.strip().split()
        stressed_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word)
            vowels = 'аеёиоуыэюя'
            stress_char_positions = []
            first_vowel_pos = -1
            for i, char in enumerate(clean_word.lower()):
                if char in vowels:
                    first_vowel_pos = i
                    break
            if first_vowel_pos != -1:
                stress_char_positions.append(first_vowel_pos)
            
            stressed_words.append({
                'original': word,
                'clean': clean_word,
                'stressed_display': clean_word + "'",
                'syllables': [clean_word],
                'stress_syllable_index': 0 if first_vowel_pos != -1 else None,
                'stress_char_positions': stress_char_positions
            })
        
        stressed_lines.append({
            'line_index': line_idx,
            'original_text': line,
            'words': stressed_words
        })
    
    return stressed_lines

def create_syllable_alignments(stressed_text_data: List[Dict], audio_duration: float) -> List[Dict]:
    """Create syllable-level alignments for training data."""
    syllable_data = []
    total_syllables = sum(
        len(word_data['syllables']) for line_data in stressed_text_data for word_data in line_data['words']
    ) or 1 # Avoid division by zero
    
    avg_syllable_duration = audio_duration / total_syllables
    current_time = 0.0
    
    for line_data in stressed_text_data:
        line_syllables = []
        
        for word_idx, word_data in enumerate(line_data['words']):
            for syll_idx, syllable in enumerate(word_data['syllables']):
                start_time = current_time
                end_time = current_time + avg_syllable_duration
                is_stressed = (syll_idx == word_data['stress_syllable_index'])
                
                line_syllables.append({
                    'syllable_text': syllable,
                    'word_original': word_data['original'],
                    'start_time': round(start_time, 3),
                    'end_time': round(end_time, 3),
                    'is_stressed': is_stressed,
                    'syllable_index_in_word': syll_idx,
                    'word_index_in_line': word_idx
                })
                
                current_time = end_time
        
        syllable_data.append({
            'line_index': line_data['line_index'],
            'original_text': line_data['original_text'],
            'syllables': line_syllables
        })
    
    return syllable_data

def run_preprocessing(config: DictConfig) -> None:
    """
    Runs data preprocessing steps, now using rusyllab for stress annotations.
    """
    print("Starting data preprocessing...")
    print(f"rusyllab available: {RUSYLLAB_AVAILABLE}")
    
    input_audio_path_str = config.get("input_audio_file", "raw_data/audio.mp3")
    input_text_path_str = config.get("input_text_file", "raw_data/text.txt")
    
    project_root = Path.cwd()
    input_audio_file = project_root / input_audio_path_str
    input_text_file = project_root / input_text_path_str
    
    output_dir = project_root / Path(config.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    target_sample_rate = config.audio.sample_rate

    print(f"Input audio: {input_audio_file}")
    print(f"Input text: {input_text_file}")
    print(f"Output directory: {output_dir}")
    print(f"Target sample rate: {target_sample_rate}")

    if not input_audio_file.exists():
        print(f"ERROR: Input audio file not found: {input_audio_file}")
        return
    if not input_text_file.exists():
        print(f"ERROR: Input text file not found: {input_text_file}")
        return

    # 1. Load and resample audio
    try:
        waveform_np, sample_rate = librosa.load(input_audio_file, sr=None)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        
        audio_duration = waveform.shape[1] / target_sample_rate
        print(f"Audio loaded and resampled. Shape: {waveform.shape}, Duration: {audio_duration:.2f}s")
    except Exception as e:
        print(f"ERROR: Could not load or resample audio file {input_audio_file}: {e}")
        return

    # 2. Load text
    try:
        with open(input_text_file, 'r', encoding='utf-8') as f:
            text_lines = [line.strip() for line in f if line.strip()]
        print(f"Text loaded. Number of lines: {len(text_lines)}")
        if not text_lines:
            print("ERROR: Text file is empty.")
            return
    except Exception as e:
        print(f"ERROR: Could not read text file {input_text_file}: {e}")
        return

    # 3. Get stress-marked text using rusyllab
    print("Detecting stress positions with rusyllab...")
    stressed_text_data = get_stress_marked_text_real(text_lines)
    
    # 4. Create syllable-level alignments for training
    print("Creating syllable alignments...")
    syllable_alignments = create_syllable_alignments(stressed_text_data, audio_duration)
    
    # 5. Save processed audio
    processed_audio_filename = f"{input_audio_file.stem}_processed.wav"
    processed_audio_path = output_dir / processed_audio_filename
    
    try:
        torchaudio.save(processed_audio_path, waveform, target_sample_rate)
        print(f"Processed audio saved to: {processed_audio_path}")
    except Exception as e:
        print(f"ERROR: Could not save processed audio file {processed_audio_path}: {e}")
        return

    # 6. Prepare training metadata
    training_metadata = {
        "audio_filepath": processed_audio_filename,
        "audio_duration_seconds": round(audio_duration, 3),
        "target_sample_rate": target_sample_rate,
        "original_text_lines": text_lines,
        "stressed_text_data": stressed_text_data, # Now from rusyllab
        "syllable_alignments": syllable_alignments,
        "total_syllables": sum(len(line['syllables']) for line in syllable_alignments),
        "total_stressed_syllables": sum(
            sum(1 for syll in line['syllables'] if syll['is_stressed']) 
            for line in syllable_alignments
        ),
        "preprocessing_config": {
            "rusyllab_available": RUSYLLAB_AVAILABLE,
            "target_sample_rate": target_sample_rate
        }
    }
    
    metadata_filename = f"{input_audio_file.stem}_training_data.json"
    metadata_path = output_dir / metadata_filename
    
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(training_metadata, f, ensure_ascii=False, indent=2)
        print(f"Training metadata saved to: {metadata_path}")
    except Exception as e:
        print(f"ERROR: Could not save metadata file {metadata_path}: {e}")
        return

    total_syllables = training_metadata["total_syllables"]
    total_stressed = training_metadata["total_stressed_syllables"]
    stress_ratio = total_stressed / total_syllables if total_syllables > 0 else 0
    
    print(f"\n=== Processing Statistics ===")
    print(f"Total syllables: {total_syllables}")
    print(f"Stressed syllables: {total_stressed}")
    print(f"Stress ratio: {stress_ratio:.2%}")
    print(f"Average syllable duration: {audio_duration/total_syllables:.3f} seconds")

    print("Data preprocessing finished.")
    print(f"Training data ready in: {output_dir}")
    print(f"Remember to `dvc add {output_dir}` and `dvc push` if this is the final processed data.")

if __name__ == "__main__":
    print("This script is designed to be run via `python -m speech_stress_analyzer.commands preprocess`")
    # For standalone testing, you would mock a Hydra config.
    # Example:
    # from omegaconf import OmegaConf
    # dummy_config = OmegaConf.create({
    #     "data_preprocessing": {
    #         "input_audio_file": "raw_data/audio.mp3", # Ensure this path is correct relative to where you run this test
    #         "input_text_file": "raw_data/text.txt",   # Ensure this path is correct
    #         "output_dir": "processed_data_test",
    #         "audio": {"sample_rate": 16000}
    #     },
    #     "hydra": { # Mock hydra runtime context
    #         "runtime": {"cwd": os.getcwd()} # Assumes script is run from project root for test
    #     }
    # })
    # # Create dummy raw_data files for testing if they don't exist
    # # Path(dummy_config.hydra.runtime.cwd / "raw_data").mkdir(exist_ok=True)
    # # with open(dummy_config.hydra.runtime.cwd / dummy_config.data_preprocessing.input_text_file, 'w') as f:
    # #     f.write("Это тестовая строка.\\nЕще одна строка.")
    # # dummy_waveform = torch.randn(1, 16000 * 5) # 5 seconds dummy audio at 16kHz
    # # torchaudio.save(dummy_config.hydra.runtime.cwd / dummy_config.data_preprocessing.input_audio_file, dummy_waveform, 16000)
    #
    # run_preprocessing(dummy_config)
    pass 