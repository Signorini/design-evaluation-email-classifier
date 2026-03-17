# Email preprocessing functions
import pandas as pd
import re
import config

def load_email_data():
    try:
        df = pd.read_csv(config.DATA_PATH)
        print(f"Loaded {len(df)} email records from {config.DATA_PATH}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_email_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def preprocess_email_texts(df):
    print("Starting text preprocessing...")
    
    # Clean email texts
    df[config.TEXT_COLUMN + '_cleaned'] = df[config.TEXT_COLUMN].apply(clean_email_text)
    
    # Remove empty texts
    original_length = len(df)
    df = df[df[config.TEXT_COLUMN + '_cleaned'].str.len() > 0]
    
    if len(df) < original_length:
        print(f"Removed {original_length - len(df)} empty email texts")
    
    print(f"Preprocessing completed. {len(df)} emails processed")
    return df

def filter_rare_classes(df):
    print(f"Filtering classes with less than {config.MIN_CLASS_SAMPLES} samples...")
    
    original_length = len(df)
    
    # Filter based on each target column
    for col in config.TARGET_COLUMNS:
        if col in df.columns:
            value_counts = df[col].value_counts()
            valid_classes = value_counts[value_counts >= config.MIN_CLASS_SAMPLES].index
            df = df[df[col].isin(valid_classes)]
    
    print(f"Filtered dataset: {original_length} -> {len(df)} emails")
    return df

def prepare_target_labels(df):
    print("Preparing target labels...")
    
    targets = {}
    
    # Extract target columns
    for i, col in enumerate(config.TARGET_COLUMNS):
        if col in df.columns:
            targets[config.TARGET_NAMES[i]] = df[col].values
            print(f"   {config.TARGET_NAMES[i]}: {len(df[col].unique())} unique classes")
    
    return targets

def get_cleaned_texts(df):
    return df[config.TEXT_COLUMN + '_cleaned'].values

def validate_data_quality(df):
    print("Validating data quality...")
    
    # Check for missing values
    missing_text = df[config.TEXT_COLUMN + '_cleaned'].isna().sum()
    if missing_text > 0:
        print(f"Warning: {missing_text} missing text values found")
    
    # Check target columns
    for col in config.TARGET_COLUMNS:
        if col in df.columns:
            missing_targets = df[col].isna().sum()
            if missing_targets > 0:
                print(f"Warning: {missing_targets} missing values in {col}")
    
    print("Data quality validation completed")
    return True
