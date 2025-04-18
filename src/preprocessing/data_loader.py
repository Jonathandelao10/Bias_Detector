import pandas as pd
import re
from typing import Optional, Dict, Tuple
from pathlib import Path
from collections import Counter

def read_txt_file(file_path: str, is_bias_file: bool = False) -> Optional[list]:
    """
    Read a text file where lines are 'source_id sentence' or '0 sentence' for bias files.
    
    Args:
        file_path (str): Path to the text file.
        is_bias_file (bool): True for conservative.txt/liberal.txt, False for train/test.
    
    Returns:
        list or None: List of tuples (source, sentence, bias_label) if successful, None otherwise.
    """
    try:
        data = []
        source_counts = Counter()
        raw_ids = Counter()
        skipped_lines = 0
        invalid_ids = Counter()
        bias_label = 'conservative' if Path(file_path).stem == 'conservative' else 'liberal' if Path(file_path).stem == 'liberal' else None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    skipped_lines += 1
                    continue
                # Handle tabs or multiple spaces
                parts = re.split(r'\s+', line, 1)
                if len(parts) != 2:
                    print(f"Skipping malformed line {i+1} in {file_path}: {line}")
                    skipped_lines += 1
                    continue
                label, sentence = parts
                # Log raw ID
                raw_ids[label] += 1
                # Normalize label
                label = label.strip()
                if is_bias_file:
                    data.append(('unknown', sentence, bias_label))
                    source_counts['unknown'] += 1
                else:
                    # Ensure string comparison
                    if label not in source_mapping:
                        invalid_ids[label] += 1
                        print(f"Invalid source_id {repr(label)} in {file_path}, line {i+1}: {line}")
                        data.append(('unknown', sentence, 'unknown'))
                        source_counts['unknown'] += 1
                    else:
                        source, bias = source_mapping[label]
                        data.append((source, sentence, bias))
                        source_counts[source] += 1
        
        print(f"Read {len(data)} lines from {file_path}, skipped {skipped_lines} lines")
        print(f"Raw ID distribution: {dict(raw_ids)}")
        print(f"Source distribution: {dict(source_counts)}")
        if invalid_ids and not is_bias_file:
            print(f"Invalid source IDs: {dict(invalid_ids)}")
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Source-to-bias mapping with string keys
source_mapping: Dict[str, Tuple[str, str]] = {
    '0': ('Newsday', 'liberal'),
    '1': ('New York Times', 'liberal'),
    '2': ('Cable News Network (CNN)', 'liberal'),
    '3': ('Los Angeles Times', 'liberal'),
    '4': ('Washington Post', 'liberal'),
    '5': ('Politico', 'neutral'),
    '6': ('Wall Street Journal', 'conservative'),
    '7': ('New York Post', 'conservative'),
    '8': ('Daily Press', 'conservative'),
    '9': ('Daily Herald', 'conservative'),
    '10': ('Chicago Tribune', 'conservative')
}

def load_and_combine_datasets(
    train_path: str,
    test_path: str,
    conservative_path: str,
    liberal_path: str
) -> Optional[pd.DataFrame]:
    """
    Load and combine all dataset files into a single DataFrame.
    
    Args:
        train_path (str): Path to train_orig.txt.
        test_path (str): Path to test.txt.
        conservative_path (str): Path to conservative.txt.
        liberal_path (str): Path to liberal.txt.
    
    Returns:
        pd.DataFrame or None: Combined DataFrame with columns ['sentence', 'source', 'bias_label'].
    """
    # Load files
    train_data = read_txt_file(train_path, is_bias_file=False)
    test_data = read_txt_file(test_path, is_bias_file=False)
    cons_data = read_txt_file(conservative_path, is_bias_file=True)
    lib_data = read_txt_file(liberal_path, is_bias_file=True)
    
    if not any([train_data, test_data, cons_data, lib_data]):
        print("Error: Failed to load all files.")
        return None
    
    # Combine data
    all_data = (train_data or [])
    if not all_data:
        print("Error: No valid data loaded.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_data, columns=['source', 'sentence', 'bias_label'])
    print(f"Combined dataset has {len(df)} rows before cleaning.")
    
    return df[['sentence', 'source', 'bias_label']]

def clean_text(text: str) -> str:
    """
    Clean a single text string by removing unwanted characters and normalizing.
    
    Args:
        text (str): Input text to clean.
    
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text

def preprocess_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset.
    
    Args:
        data (pd.DataFrame): Input DataFrame with columns ['sentence', 'source', 'bias_label'].
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if data is None or not all(col in data.columns for col in ['sentence', 'source', 'bias_label']):
        print("Error: Invalid data or missing required columns.")
        return None
    
    cleaned_data = data.copy()
    
    # Clean sentences
    print("Cleaning sentences...")
    cleaned_data['sentence'] = cleaned_data['sentence'].apply(clean_text)
    
    # Remove empty or invalid rows
    initial_len = len(cleaned_data)
    cleaned_data = cleaned_data[cleaned_data['sentence'].str.len() > 0].dropna(subset=['sentence', 'bias_label'])
    print(f"Removed {initial_len - len(cleaned_data)} invalid or empty rows.")
    
    # Filter short sentences
    initial_len = len(cleaned_data)
    cleaned_data = cleaned_data[cleaned_data['sentence'].str.split().str.len() >= 3]
    print(f"Removed {initial_len - len(cleaned_data)} sentences with <3 words.")
    
    # Validate bias labels
    valid_labels = ['liberal', 'neutral', 'conservative', 'unknown']
    cleaned_data = cleaned_data[cleaned_data['bias_label'].isin(valid_labels)]
    print(f"Kept {len(cleaned_data)} rows with valid bias labels.")
    
    # Deduplicate by sentence, source, and bias
    initial_len = len(cleaned_data)
    cleaned_data = cleaned_data.drop_duplicates(subset=['sentence', 'source', 'bias_label'])
    print(f"Removed {initial_len - len(cleaned_data)} duplicate rows.")
    
    # Log distributions
    print("Final source distribution:")
    print(cleaned_data['source'].value_counts().to_dict())
    print("Final bias distribution:")
    print(cleaned_data['bias_label'].value_counts().to_dict())
    
    return cleaned_data

def save_preprocessed_data(data: pd.DataFrame, output_path: str) -> bool:
    """
    Save preprocessed dataset to CSV.
    
    Args:
        data (pd.DataFrame): DataFrame to save.
        output_path (str): Path to save CSV.
    
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved preprocessed data to data/processed/preprocessed_data.csv")
        return True
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    train_path = "data/raw/train_orig.txt"
    test_path = "data/raw/test.txt"
    cons_path = "data/raw/conservative.txt"
    lib_path = "data/raw/liberal.txt"
    output_path = "data/processed/preprocessed_data.csv"
    
    # Load and combine datasets
    data = load_and_combine_datasets(train_path, test_path, cons_path, lib_path)
    
    # Preprocess dataset
    if data is not None:
        cleaned_data = preprocess_dataset(data)
        
        # Save cleaned data
        if cleaned_data is not None:
            save_preprocessed_data(cleaned_data, output_path)