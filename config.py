# Global configuration parameters for the email classification system

# Data Configuration
DATA_PATH = "data/out.csv"
TEXT_COLUMN = "ic_deduplicated"
TARGET_COLUMNS = ["formatted_y2", "formatted_y3", "formatted_y4"]

# Model Configuration  
RANDOM_STATE = 42
TEST_SIZE = 0.3
N_ESTIMATORS = 100

# Feature Engineering Configuration
MAX_FEATURES = 1000
MIN_DF = 1
MAX_DF = 0.95
STOP_WORDS = 'english'

# Data Filtering Configuration
MIN_CLASS_SAMPLES = 5

# Target Names
TARGET_NAMES = ['Type2', 'Type2+Type3', 'Type2+Type3+Type4']
