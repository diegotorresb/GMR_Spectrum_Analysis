import numpy as np
import os
from read_data_luke import read_data

def save_luke_data():
    """
    Read data from micrometer test folder and save as .npz files for merging
    """
    # Read data from micrometer folder
    luke_folder = 'resources/data/luke_test/try1'
    print(f"Reading micrometer data from: {os.path.abspath(luke_folder)}")
    
    # Read data from folder
    data_dict_left, data_dict_right = read_data(True)
    
    # Convert dictionary data to arrays
    def dict_to_arrays(data_dict):
        angles = sorted(list(data_dict.keys()))
        wavelengths = data_dict[angles[0]][:, 0]
        data = np.array([data_dict[angle] for angle in angles])
        return data, np.array(angles)
    

if __name__ == "__main__":
    save_luke_data() 