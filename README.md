# Design choice 1: Chained multi-outputs

## How to run this code.

1. Install the required Python packages:

```bash
pip install pandas numpy scikit-learn scipy
```

3. Make sure the dataset file exists at `data/out.csv`.
4. Run the main pipeline:

```bash
python3 main_controller.py
```

This will:

- load and preprocess the email dataset
- convert text into TF-IDF features
- train the chained Random Forest models
- evaluate the predictions
- save the results to `model_results.txt`

## Architecture Sketch

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                            ┌─────────▼─────────┐
                            │   Raw Email Data  │
                            │  (CSV with text)  │
                            └─────────┬─────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PREPROCESSING LAYER                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                            ┌─────────▼─────────┐
                            │  DataPreprocessor │
                            │ • Text cleaning   │
                            │ • Remove empties  │
                            │ • Filter rare cls │
                            └─────────┬─────────┘
                                      │
                            ┌─────────▼─────────┐
                            │  TextVectorizer   │
                            │ • TF-IDF vectors  │
                            │ • 1000 features   │
                            │ • Fit/transform   │
                            └─────────┬─────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                            ┌─────────▼─────────┐
                            │ DataEncapsulator  │
                            │ • train_x, train_y│
                            │ • test_x, test_y  │
                            │ • All 3 targets   │
                            └─────────┬─────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODELING LAYER                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                     ┌────────────────▼────────────────┐
                     │   ChainedMultiLabelModel        │
                     │                                 │
                     │  ┌─────────────────────────┐    │
                     │  │     Type2 Model         │    │
                     │  │   (Random Forest)       │    │
                     │  │   input: TF-IDF         │    │
                     │  └────────────┬────────────┘    │
                     │               │ enc(pred_Type2) │
                     │  ┌────────────▼────────────┐    │
                     │  │   Type2+Type3 Model     │    │
                     │  │   (Random Forest)       │    │
                     │  │   input: TF-IDF +       │    │
                     │  │   enc(pred_Type2)       │    │
                     │  └────────────┬────────────┘    │
                     │               │ enc(pred_Type3) │
                     │  ┌────────────▼────────────┐    │
                     │  │ Type2+Type3+Type4 Model │    │
                     │  │   (Random Forest)       │    │
                     │  │   input: TF-IDF +       │    │
                     │  │   enc(pred_Type2) +     │    │
                     │  │   enc(pred_Type3)       │    │
                     │  └─────────────────────────┘    │
                     │                                 │
                     │  ┌─────────────────────────┐    │
                     │  │   LabelEncoder          │    │
                     │  │ • encode prior labels   │    │
                     │  │ • fit on train / reuse  │    │
                     │  └─────────────────────────┘    │
                     │                                 │
                     │  ┌─────────────────────────┐    │
                     │  │   FeatureAugmentor      │    │
                     │  │ • append enc labels     │    │
                     │  │ • to TF-IDF matrix      │    │
                     │  └─────────────────────────┘    │
                     │                                 │
                     └────────────────┬────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION LAYER                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                            ┌─────────▼─────────┐
                            │ EvaluationEngine  │
                            │ • Accuracy scores │
                            │ • Exact match     │
                            │ • Results output  │
                            └─────────┬─────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONTROL LAYER                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                            ┌─────────▼─────────┐
                            │  MainController   │
                            │ • Pipeline coord. │
                            │ • Component mgmt. │
                            └─────────┬─────────┘
                                      │
                            ┌─────────▼─────────┐
                            │ ConfigurationMgr  │
                            │ • Shared variables│
                            │ • Parameters      │
                            └───────────────────┘
```

## Key caracteristics

### Components (9 Total)

- DataPreprocessor
- TextVectorizer
- LabelEncoder
- FeatureAugmentor
- DataEncapsulator
- ChainedMultiLabelModel
- EvaluationEngine
- MainController
- ConfigurationManager

### Connectors

- Method Invocation
- Object Passing
- Classifier Chain
- Import Dependencies
- Return Values
- Label Encoding Bridge

### Data Elements

- RawEmailData
- ProcessedText
- VectorizedData
- AugmentedFeatures
- EncapsulatedDataset
- ChainedPredictions
- EvaluationMetrics
- ConfigurationParameters

# Components for Design Architecture

This section describes all components of design choice one.

**`1 - ConfigManager (config.py)`**

Single source of truth for all configurations and parameters. used by any module import configurations, so changing one value propagates system-wide.

**`2 - DataPreprocessor (preprocessing.py)`**

Transform the raw CSV data into a clean Python object. Responsibilities include loading the CSV file, cleaning data (e.g., checking email strings for lowercase and whitespace), and validating null values.

**`3 - TextVectorizer (feature_engineering.py)`**

Convert each data string into a fixed numeric vector using the TF-IDF algorithm and expose the vocabulary list for interpretation.

**`4 - LabelEncoder (feature_engineering.py)`**

The label encoder acts as a bridge between the categorical model output and the numeric model input, making the classifier chain.

**`5 - FeatureAugmentor (feature_engineering.py)`**

Component to build the augmented feature matrix that defines the chained architecture, responsible for returning and reapplying the matrix for the next model chain.

**`6 - DataEncapsulator (feature_engineering.py)`**

Creates a single self-contained data object that is passed between all downstream pipeline steps, avoiding repeated parameter passing.

**`7 - ChainedClassifier (modeling.py)`**

The three classifiers that form the brain of the classifier use the randomForest algorithm to classify each data entry, operating on a progressively richer feature set.

Classifier chain flow:

```
Email Text ──► TF-IDF ──────────────────────────────────────► Model_Type2 ──► pred_Type2
                                                                                    │
              TF-IDF + enc(pred_Type2) ─────────────────────► Model_Type3 ──► pred_Type3
                                                                                    │
              TF-IDF + enc(pred_Type2) + enc(pred_Type3) ───► Model_Type4 ──► pred_Type4
```

---

**`8 - ResultsEvaluator (modeling.py)`**

The next modelling component quantifies the chain performance across all three levels, ensuring the correct predictions and match accuracy.

**`9 - MainController (main_controller.py)`**

This is the system entry point calling each component in the correct order.

End-to-End Data Flow

```
data/out.csv
     │
     ▼  C2 DataPreprocessor
  clean texts + label arrays
     │
     ▼  C3 TextVectorizer
  TF-IDF matrix  (n × 1000)
     │
     ▼  C6 DataEncapsulator
  data_encapsulation dict
  { X_train, X_test, y_train, y_test }
     │
     ├──► C7 Train Level 1  (TF-IDF)                      → Model_Type2
     │         │
     │    C4 encode_labels(Type2, y2_true)
     │         │
     ├──► C7 Train Level 2  (TF-IDF + enc(y2_true))       → Model_Type3
     │         │
     │    C4 encode_labels(Type3, y3_true)
     │         │
     └──► C7 Train Level 3  (TF-IDF + enc(y2_true) + enc(y3_true)) → Model_Type4
                                                                          │
     ┌────────────────────────────────────────────────────────────────────┘
     │
     ├──► C7 Predict Level 1  (TF-IDF)                    → pred_Type2
     │         │
     │    C4 encode_labels(Type2, pred_Type2, fit=False)
     │         │
     ├──► C7 Predict Level 2  (TF-IDF + enc(pred_Type2))  → pred_Type3
     │         │
     │    C4 encode_labels(Type3, pred_Type3, fit=False)
     │         │
     └──► C7 Predict Level 3  (TF-IDF + enc(pred_Type2) + enc(pred_Type3)) → pred_Type4
                                                                                   │
     ┌─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼  C8 ResultsEvaluator
  per-level accuracy + exact match accuracy
     │
     ▼
  model_results.txt
```
