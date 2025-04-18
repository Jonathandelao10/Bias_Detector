import pandas as pd
import nltk
from typing import List, Optional
from tqdm import tqdm

try:
    from textattack.augmentation import WordNetAugmenter
except ImportError as e:
    print(f"Error importing WordNetAugmenter from textattack: {e}")
    print("Please ensure textattack is installed correctly: pip install textattack")
    exit(1)

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

def initialize_augmenters(
    synonym_pct: float = 0.4
) -> List:
    """
    Initialize text augmentation strategy using WordNetAugmenter.
    
    Args:
        synonym_pct (float): Percentage of words to replace with synonyms.
    
    Returns:
        List: List containing WordNetAugmenter.
    """
    try:
        augmenter = WordNetAugmenter(
            pct_words_to_swap=synonym_pct,
            transformations_per_example=1
        )
        print("Initialized WordNetAugmenter for synonym replacement.")
        return [augmenter]
    except Exception as e:
        print(f"Error initializing WordNetAugmenter: {e}")
        return []

def augment_text(
    sentences: List[str],
    augmenters: List,
    augment_prob: float = 0.6
) -> List[str]:
    """
    Apply augmentation to a list of sentences.
    
    Args:
        sentences (List[str]): List of sentences to augment.
        augmenters (List): List of augmenter objects.
        augment_prob (float): Probability of applying augmentation to each sentence.
    
    Returns:
        List[str]: List of augmented sentences.
    """
    if not augmenters:
        print("No augmenters available. Returning original sentences.")
        return sentences[:]
    
    augmented_sentences = []
    for sentence in tqdm(sentences, desc="Augmenting text"):
        if pd.isna(sentence) or not isinstance(sentence, str) or len(sentence.strip()) == 0:
            augmented_sentences.append(sentence)
            continue
        
        import random
        if random.random() < augment_prob:
            augmenter = random.choice(augmenters)
            try:
                aug_sentence = augmenter.augment(sentence)[0]
                augmented_sentences.append(aug_sentence)
            except Exception as e:
                print(f"Error augmenting sentence '{sentence}': {e}")
                augmented_sentences.append(sentence)
        else:
            augmented_sentences.append(sentence)
    
    return augmented_sentences

def apply_augmentation(
    data: pd.DataFrame,
    text_column: str = 'sentence',
    output_column: str = 'augmented_sentence'
) -> Optional[pd.DataFrame]:
    """
    Apply text augmentation to a DataFrame column.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
        text_column (str): Column containing text to augment.
        output_column (str): Name of the new column for augmented text.
    
    Returns:
        pd.DataFrame or None: DataFrame with augmented text column.
    """
    if data is None or text_column not in data.columns:
        print(f"Error: Invalid data or missing {text_column} column.")
        return None
    
    # Initialize augmenters
    augmenters = initialize_augmenters()
    
    # Create a copy to avoid modifying original
    augmented_data = data.copy()
    
    # Apply augmentation
    print(f"Applying augmentation to column '{text_column}'...")
    augmented_data[output_column] = augment_text(
        augmented_data[text_column].tolist(),
        augmenters
    )
    
    return augmented_data

def save_augmented_data(data: pd.DataFrame, output_path: str) -> bool:
    """
    Save augmented dataset to CSV.
    
    Args:
        data (pd.DataFrame): DataFrame to save.
        output_path (str): Path to save CSV.
    
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        from pathlib import Path
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved augmented data to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    input_path = "data/processed/preprocessed_data.csv"
    output_path = "data/processed/augmented_data.csv"
    
    # Load preprocessed data
    try:
        data = pd.read_csv(input_path)
        print(f"Loaded {len(data)} sentences from {input_path}")
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        exit(1)
    
    # Apply augmentation
    augmented_data = apply_augmentation(data, text_column='sentence')
    
    # Save augmented data
    if augmented_data is not None:
        save_augmented_data(augmented_data, output_path)