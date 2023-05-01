import sys
import preprocess_BRACE
import preprocess_penn_action
import preprocess_ClimbAlong
from global_variables import *
from utils import make_dir
            
if __name__ == "__main__":  
    #noise_scalar = int(sys.argv[1])
      
    # Making folders for the processed-data, in case it does not already exist.
    #make_dir(OVERALL_DATA_FOLDER(noise_scalar))
    
    #for subfolder in SUBFOLDERS.values():
    #    make_dir(OVERALL_DATA_FOLDER(noise_scalar) + subfolder)
        
    make_dir(OVERALL_PROCESSED_FOLDER)
    make_dir(CA_PROCESSED_PATH)
    
    # Preprocessing BRACE
    #preprocess_BRACE.preprocess(noise_scalar)

    # Preprocessing Penn Action
    #preprocess_penn_action.preprocess(noise_scalar)
    
    # Preprocessing ClimbAlong
    preprocess_ClimbAlong.preprocess()