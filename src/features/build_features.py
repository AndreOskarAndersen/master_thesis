import preprocess_BRACE
import preprocess_penn_action
from global_variables import *
from utils import make_dir
            
if __name__ == "__main__":    
    # Making overall folder for the processed-data, in case it does not already exist.
    make_dir(OVERALL_DATA_FOLDER)
    
    # Preprocessing BRACE
    preprocess_BRACE.preprocess()
    
    # Preprocessing Penn Action
    preprocess_penn_action.preprocess()