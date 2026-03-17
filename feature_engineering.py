# Convert textual data into numeric representation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp
import pandas as pd
import numpy as np
import config

# Global vectorizer to maintain consistency
vectorizer = None

# Global label encoders, one per target level
label_encoders = {}

def create_tfidf_vectorizer():
    """Create TF-IDF vectorizer with configuration parameters"""
    global vectorizer
    
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF,
        stop_words=config.STOP_WORDS,
        lowercase=True,
        token_pattern=r'\b\w+\b'
    )
    
    print(f"Created TF-IDF vectorizer with {config.MAX_FEATURES} max features")
    return vectorizer

def convert_text_to_features(texts, is_training=True):
    global vectorizer
    
    if vectorizer is None:
        create_tfidf_vectorizer()
    
    print(f"Converting {len(texts)} texts to numerical features...")
    
    if is_training:
        # Fit and transform for training data
        feature_matrix = vectorizer.fit_transform(texts)
        print(f"Training features created: {feature_matrix.shape}")
    else:
        # Transform only for test data
        feature_matrix = vectorizer.transform(texts)
        print(f"Test features created: {feature_matrix.shape}")
    
    return feature_matrix

def get_feature_names():
    """Get feature names from the fitted vectorizer"""
    global vectorizer
    
    if vectorizer is None:
        print("Vectorizer not fitted yet")
        return None
    
    try:
        return vectorizer.get_feature_names_out()
    except:
        return vectorizer.get_feature_names()

def split_data_and_targets(features, targets):
    print(f"Splitting data with test_size={config.TEST_SIZE}, random_state={config.RANDOM_STATE}")
    
    # Use first target for stratification
    first_target = targets[config.TARGET_NAMES[0]]
    
    # Split features
    X_train, X_test = train_test_split(
        features, 
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=first_target
    )
    
    # Split all targets
    y_train = {}
    y_test = {}
    
    for target_name in config.TARGET_NAMES:
        if target_name in targets:
            y_train[target_name], y_test[target_name] = train_test_split(
                targets[target_name],
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE,
                stratify=first_target
            )
    
    print(f"Data split completed:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]} dimensions")
    
    return X_train, X_test, y_train, y_test

def create_data_encapsulation(X_train, X_test, y_train, y_test):
    """Create encapsulated data structure for model training"""
    
    data_encapsulation = {
        'train_features': X_train,
        'test_features': X_test,
        'train_targets': y_train,
        'test_targets': y_test,
        'feature_names': get_feature_names(),
        'target_names': config.TARGET_NAMES,
        'data_shape': {
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features': X_train.shape[1],
            'target_levels': len(config.TARGET_NAMES)
        }
    }
    
    print("Data encapsulation created successfully")
    print(f"   Encapsulation contains: {list(data_encapsulation.keys())}")
    
    return data_encapsulation

def get_feature_statistics(features):
    """Get statistics about the feature matrix"""
    stats = {
        'shape': features.shape,
        'density': features.nnz / (features.shape[0] * features.shape[1]),
        'max_features_per_sample': features.getnnz(axis=1).max(),
        'min_features_per_sample': features.getnnz(axis=1).min(),
        'avg_features_per_sample': features.getnnz(axis=1).mean()
    }
    
    print("Feature Statistics:")
    print(f"   Shape: {stats['shape']}")
    print(f"   Density: {stats['density']:.4f}")
    print(f"   Avg features per email: {stats['avg_features_per_sample']:.1f}")
    
    return stats

def validate_feature_consistency(X_train, X_test):
    if X_train.shape[1] != X_test.shape[1]:
        print(f"Feature dimension mismatch: Train={X_train.shape[1]}, Test={X_test.shape[1]}")
        return False
    
    print("Feature dimensions are consistent between train and test sets")
    return True

def encode_labels(target_name, labels, fit=True):
    """Encode string labels to integers so they can be appended as chain features."""
    global label_encoders

    if fit:
        le = LabelEncoder()
        encoded = le.fit_transform(labels)
        label_encoders[target_name] = le
    else:
        le = label_encoders[target_name]
        encoded = le.transform(labels)

    return encoded

def augment_features_with_predictions(features, encoded_labels_list):
    extra = sp.csr_matrix(np.column_stack(encoded_labels_list))
    return sp.hstack([features, extra])
