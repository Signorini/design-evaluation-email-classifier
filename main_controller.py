# Implementing Main Program and Subroutine Architecture
import sys
from datetime import datetime

# Import all component modules
import config
import preprocessing
import feature_engineering  
import modeling

def print_header():
    print("=" * 80)
    print("     EMAIL CLASSIFICATION SYSTEM - DESIGN CHOICE 1")
    print("          Chained Multi-outputs Architecture")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_footer(start_time):
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    print()
    print("=" * 80)
    print("            PIPELINE EXECUTION COMPLETED")
    print("=" * 80)
    print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Execution time: {execution_time}")
    print("=" * 80)

def data_loading_and_preprocessing():
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("-" * 50)
    
    # Load data
    df = preprocessing.load_email_data()
    if df is None:
        print("Data loading failed. Exiting...")
        return None
    
    # Preprocess texts
    df = preprocessing.preprocess_email_texts(df)
    
    # Filter rare classes
    df = preprocessing.filter_rare_classes(df)
    
    # Validate data quality
    preprocessing.validate_data_quality(df)
    
    # Prepare target labels
    targets = preprocessing.prepare_target_labels(df)
    
    # Get cleaned texts
    texts = preprocessing.get_cleaned_texts(df)
    
    print("Data loading and preprocessing completed\n")
    return texts, targets

def feature_engineering_pipeline(texts, targets):
    print("STEP 2: FEATURE ENGINEERING")
    print("-" * 50)
    
    # Convert texts to features
    features = feature_engineering.convert_text_to_features(texts, is_training=True)
    
    # Get feature statistics
    feature_engineering.get_feature_statistics(features)
    
    # Split data
    X_train, X_test, y_train, y_test = feature_engineering.split_data_and_targets(features, targets)
    
    # Validate feature consistency
    feature_engineering.validate_feature_consistency(X_train, X_test)
    
    # Create data encapsulation
    data_encapsulation = feature_engineering.create_data_encapsulation(X_train, X_test, y_train, y_test)
    
    print("Feature engineering completed\n")
    return data_encapsulation

def model_training_pipeline(data_encapsulation):
    """Execute model training pipeline"""
    print("STEP 3: MODEL TRAINING")
    print("-" * 50)
    
    # Train all models
    training_results = modeling.train_all_models(data_encapsulation)
    
    # Get model summary
    modeling.get_model_summary()
    
    print("Model training completed\n")
    return training_results

def model_evaluation_pipeline(data_encapsulation):
    """Execute model evaluation pipeline"""
    print("STEP 4: MODEL EVALUATION")
    print("-" * 50)
    
    # Make predictions
    predictions = modeling.predict_all_models(data_encapsulation)
    
    # Evaluate all models
    evaluation_results = modeling.evaluate_all_models(data_encapsulation, predictions)
    
    # Calculate exact match accuracy
    exact_match_result = modeling.calculate_exact_match_accuracy(data_encapsulation, predictions)
    
    # Save results
    modeling.save_model_results(evaluation_results, exact_match_result)
    
    print("Model evaluation completed\n")
    return evaluation_results, exact_match_result, predictions

def generate_final_report(evaluation_results, exact_match_result, data_encapsulation):
    """Generate final performance report"""
    print("STEP 5: FINAL REPORT GENERATION")
    print("-" * 50)
    
    print("FINAL PERFORMANCE SUMMARY:")
    print()
    
    # Individual model performance
    for target_name, result in evaluation_results.items():
        accuracy = result['accuracy']
        correct = result['correct_predictions']
        total = result['total_predictions']
        classes = result['unique_classes']
        
        print(f"{target_name}:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Correct Predictions: {correct}/{total}")
        print(f"   Unique Classes: {classes}")
        print()
    
    # Exact match performance
    exact_acc = exact_match_result['exact_match_accuracy']
    exact_correct = exact_match_result['correct_exact_matches']
    exact_total = exact_match_result['total_samples']
    
    print(f"HIERARCHICAL PERFORMANCE:")
    print(f"   Exact Match Accuracy: {exact_acc:.4f} ({exact_acc*100:.2f}%)")
    print(f"   Exact Matches: {exact_correct}/{exact_total}")
    print()
    
    # Data summary
    data_shape = data_encapsulation['data_shape']
    print(f"DATASET SUMMARY:")
    print(f"   Training Samples: {data_shape['train_samples']}")
    print(f"   Test Samples: {data_shape['test_samples']}")
    print(f"   Feature Dimensions: {data_shape['features']}")
    print(f"   Target Levels: {data_shape['target_levels']}")
    print()
    
    print("Final report generation completed\n")

def run_complete_pipeline():
    """Run the complete email classification pipeline"""
    start_time = datetime.now()
    
    try:
        # Print header
        print_header()
        
        # Step 1: Data Loading and Preprocessing
        result = data_loading_and_preprocessing()
        if result is None:
            return False
        texts, targets = result
        
        # Step 2: Feature Engineering
        data_encapsulation = feature_engineering_pipeline(texts, targets)
        
        # Step 3: Model Training
        model_training_pipeline(data_encapsulation)
        
        # Step 4: Model Evaluation
        evaluation_results, exact_match_result, predictions = model_evaluation_pipeline(data_encapsulation)
        
        # Step 5: Final Report
        generate_final_report(evaluation_results, exact_match_result, data_encapsulation)
        
        # Print footer
        print_footer(start_time)
        
        return True
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function - Entry point of the application"""
    success = run_complete_pipeline()
    
    if success:
        print("Email classification pipeline executed successfully!")
        return 0
    else:
        print("Email classification pipeline failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
