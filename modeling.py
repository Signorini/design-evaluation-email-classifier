# Machine learning training and testing functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import config
import feature_engineering

# Global model storage
trained_models = {}

def create_model(target_name):
    """Create a Random Forest model for specific target"""
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    print(f"Created Random Forest model for {target_name}")
    return model

def train_single_model(X_train, y_train, target_name):
    """Train a single model for one target"""
    global trained_models
    
    print(f"Training model for {target_name}...")
    
    # Create and train model
    model = create_model(target_name)
    model.fit(X_train, y_train)
    
    # Store trained model
    trained_models[target_name] = model
    
    print(f"Model training completed for {target_name}")
    return model

def train_all_models(data_encapsulation):
    """Train models for all target levels using a classifier chain.

    Each model from level 2 onward receives the TF-IDF features augmented
    with the ground-truth encoded labels of all prior levels.
    """
    print("Starting chained training for all models...")

    X_train = data_encapsulation['train_features']
    y_train = data_encapsulation['train_targets']

    training_results = {}
    accumulated_encoded = []  # grows with each level

    for i, target_name in enumerate(config.TARGET_NAMES):
        if target_name not in y_train:
            continue

        if i == 0:
            X_current = X_train
        else:
            X_current = feature_engineering.augment_features_with_predictions(
                X_train, accumulated_encoded
            )

        model = train_single_model(X_current, y_train[target_name], target_name)
        training_results[target_name] = {
            'model': model,
            'classes': len(np.unique(y_train[target_name])),
            'samples': len(y_train[target_name])
        }

        # Fit encoder and accumulate for the next level
        encoded = feature_engineering.encode_labels(
            target_name, y_train[target_name], fit=True
        )
        accumulated_encoded.append(encoded.reshape(-1, 1))

    print(f"Chained training completed for all {len(training_results)} models")
    return training_results

def predict_single_model(X_test, target_name):
    """Make predictions using a single trained model"""
    global trained_models
    
    if target_name not in trained_models:
        print(f"Model for {target_name} not found. Train the model first.")
        return None
    
    model = trained_models[target_name]
    predictions = model.predict(X_test)
    
    print(f"Predictions completed for {target_name}: {len(predictions)} samples")
    return predictions

def predict_all_models(data_encapsulation):
    """Make predictions using chained models.

    Each model from level 2 onward receives the TF-IDF features augmented
    with the predicted (not ground-truth) encoded labels of all prior levels.
    """
    print("Making chained predictions for all models...")

    X_test = data_encapsulation['test_features']
    predictions = {}
    accumulated_encoded = []  # grows with each predicted level

    for i, target_name in enumerate(config.TARGET_NAMES):
        if i == 0:
            X_current = X_test
        else:
            X_current = feature_engineering.augment_features_with_predictions(
                X_test, accumulated_encoded
            )

        pred = predict_single_model(X_current, target_name)
        if pred is None:
            continue
        predictions[target_name] = pred

        # Encode predicted labels using the already-fitted encoder
        encoded = feature_engineering.encode_labels(target_name, pred, fit=False)
        accumulated_encoded.append(encoded.reshape(-1, 1))

    print(f"Chained predictions completed for all {len(predictions)} models")
    return predictions

def evaluate_single_model(y_true, y_pred, target_name):
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate correct predictions count
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    
    result = {
        'accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'unique_classes': len(np.unique(y_true))
    }
    
    print(f"{target_name} Results:")
    print(f"   Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    print(f"   Classes: {result['unique_classes']}")
    
    return result

def evaluate_all_models(data_encapsulation, predictions):
    print("Evaluating all models...")
    
    y_test = data_encapsulation['test_targets']
    evaluation_results = {}
    
    for target_name in config.TARGET_NAMES:
        if target_name in predictions and target_name in y_test:
            result = evaluate_single_model(
                y_test[target_name], 
                predictions[target_name], 
                target_name
            )
            evaluation_results[target_name] = result
    
    return evaluation_results

def calculate_exact_match_accuracy(data_encapsulation, predictions):
    print("Calculating exact match accuracy...")
    
    y_test = data_encapsulation['test_targets']
    
    # Check if all targets have predictions
    available_targets = []
    for target_name in config.TARGET_NAMES:
        if target_name in predictions and target_name in y_test:
            available_targets.append(target_name)
    
    if len(available_targets) < len(config.TARGET_NAMES):
        print(f"Warning: Only {len(available_targets)}/{len(config.TARGET_NAMES)} targets available")
    
    # Calculate exact matches
    exact_matches = np.ones(len(y_test[available_targets[0]]), dtype=bool)
    
    for target_name in available_targets:
        matches = (y_test[target_name] == predictions[target_name])
        exact_matches = exact_matches & matches
    
    exact_match_accuracy = exact_matches.mean()
    correct_exact_matches = exact_matches.sum()
    total_samples = len(exact_matches)
    
    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f} ({correct_exact_matches}/{total_samples})")
    
    return {
        'exact_match_accuracy': exact_match_accuracy,
        'correct_exact_matches': correct_exact_matches,
        'total_samples': total_samples
    }

def get_model_summary():
    global trained_models
    
    summary = {
        'total_models': len(trained_models),
        'model_names': list(trained_models.keys()),
        'model_types': [type(model).__name__ for model in trained_models.values()]
    }
    
    print(f"Model Summary:")
    print(f"   Total models: {summary['total_models']}")
    print(f"   Model names: {summary['model_names']}")
    
    return summary

def save_model_results(evaluation_results, exact_match_result, filename="model_results.txt"):
    with open(filename, 'w') as f:
        f.write("=== MODEL EVALUATION RESULTS ===\n\n")
        
        for target_name, result in evaluation_results.items():
            f.write(f"{target_name}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Correct: {result['correct_predictions']}/{result['total_predictions']}\n")
            f.write(f"  Classes: {result['unique_classes']}\n\n")
        
        f.write(f"Exact Match Accuracy: {exact_match_result['exact_match_accuracy']:.4f}\n")
        f.write(f"Exact Matches: {exact_match_result['correct_exact_matches']}/{exact_match_result['total_samples']}\n")
    
    print(f"Results saved to {filename}")

def get_feature_importance(target_name, top_n=10):
    global trained_models
    
    if target_name not in trained_models:
        print(f"Model for {target_name} not found")
        return None
    
    model = trained_models[target_name]
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Get top N features
        top_indices = np.argsort(importances)[-top_n:][::-1]
        
        result = {
            'feature_indices': top_indices,
            'importances': importances[top_indices]
        }
        
        print(f"Top {top_n} features for {target_name}:")
        for i, (idx, importance) in enumerate(zip(top_indices, importances[top_indices])):
            print(f"   {i+1}. Feature {idx}: {importance:.4f}")
        
        return result
    
    print(f"Model {target_name} does not support feature importance")
    return None
