# feature_extraction.py

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from typing import Tuple, Optional
from tqdm import tqdm
from pathlib import Path

def initialize_bert(model_name: str = 'bert-base-uncased') -> Tuple[BertTokenizer, BertModel]:
    """
    Initialize BERT tokenizer and model.
    
    Args:
        model_name (str): Name of the pretrained BERT model.
    
    Returns:
        Tuple[BertTokenizer, BertModel]: Initialized tokenizer and model.
    """
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        print(f"Initialized {model_name} tokenizer and model.")
        return tokenizer, model
    except Exception as e:
        print(f"Error initializing BERT: {e}")
        return None, None

def extract_bert_features(
    sentences: list,
    tokenizer: BertTokenizer,
    model: BertModel,
    max_length: int = 128,
    batch_size: int = 32,
    device: str = None
) -> Optional[np.ndarray]:
    """
    Extract BERT feature vectors from sentences.
    
    Args:
        sentences (list): List of sentences.
        tokenizer (BertTokenizer): BERT tokenizer.
        model (BertModel): BERT model.
        max_length (int): Maximum sequence length.
        batch_size (int): Batch size for processing.
        device (str, optional): Device to run model ('cuda' or 'cpu').
    
    Returns:
        np.ndarray or None: Array of feature vectors.
    """
    if not sentences or tokenizer is None or model is None:
        print("Error: Invalid inputs for feature extraction.")
        return None
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    feature_vectors = []
    
    # Process sentences in batches
    for i in tqdm(range(0, len(sentences), batch_size), desc="Extracting BERT features"):
        batch_sentences = sentences[i:i + batch_size]
        
        # Filter out invalid sentences
        batch_sentences = [s if isinstance(s, str) else "" for s in batch_sentences]
        
        try:
            # Tokenize
            inputs = tokenizer(
                batch_sentences,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            feature_vectors.append(embeddings.cpu().numpy())
        
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            # Append zero vectors for failed batch to maintain alignment
            feature_vectors.append(np.zeros((len(batch_sentences), model.config.hidden_size)))
    
    # Concatenate feature vectors
    try:
        feature_vectors = np.concatenate(feature_vectors, axis=0)
        print(f"Extracted features for {len(feature_vectors)} sentences.")
        return feature_vectors
    except Exception as e:
        print(f"Error concatenating feature vectors: {e}")
        return None

def save_features(
    features: np.ndarray,
    data: pd.DataFrame,
    output_path: str,
    feature_prefix: str = 'feat'
) -> bool:
    """
    Save feature vectors with corresponding labels to CSV.
    
    Args:
        features (np.ndarray): Feature vectors.
        data (pd.DataFrame): Original DataFrame with labels.
        output_path (str): Path to save CSV.
        feature_prefix (str): Prefix for feature column names.
    
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        # Create feature DataFrame
        feature_columns = [f"{feature_prefix}_{i}" for i in range(features.shape[1])]
        feature_df = pd.DataFrame(features, columns=feature_columns)
        
        # Combine with labels
        label_columns = ['source', 'bias_label']
        output_df = pd.concat(
            [data[label_columns].reset_index(drop=True), feature_df],
            axis=1
        )
        
        # Save to CSV
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved features to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving features to {output_path}: {e}")
        return False

if __name__ == "__main__":
    # Example 
    input_path = "../../data/processed/augmented_data.csv"
    output_path = "../../data/processed/bert_features.csv"
    
    # Load data
    data = pd.read_csv(input_path)
    
    # Initialize BERT
    tokenizer, model = initialize_bert()
    
    if tokenizer and model:
        # Extract features for original sentences
        features = extract_bert_features(
            data['sentence'].tolist(),
            tokenizer,
            model
        )
        
        # Save features
        if features is not None:
            save_features(features, data, output_path)
        
        # Extract features for augmented sentences
        if 'augmented_sentence' in data.columns:
            aug_features = extract_bert_features(
                data['augmented_sentence'].tolist(),
                tokenizer,
                model
            )
            if aug_features is not None:
                save_features(
                    aug_features,
                    data,
                    "../../data/processed/augmented_bert_features.csv",
                    feature_prefix='aug_feat'
                )