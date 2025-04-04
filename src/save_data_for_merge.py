import numpy as np
import os
from read_data_contour_fine_adjustment import read_data_from_folder

def save_micrometer_data():
    """
    Read data from micrometer test folder and save as .npz files for merging
    """
    # Read data from micrometer folder
    micrometer_folder = 'resources/data/tilt_test_micrometer'
    print(f"Reading micrometer data from: {os.path.abspath(micrometer_folder)}")
    
    # Read data from folder
    data_dict_left, data_dict_right = read_data_from_folder(micrometer_folder)
    
    # Convert dictionary data to arrays
    def dict_to_arrays(data_dict):
        angles = sorted(list(data_dict.keys()))
        wavelengths = data_dict[angles[0]][:, 0]
        data = np.array([data_dict[angle] for angle in angles])
        return data, np.array(angles)
    
    # Convert and save left GMR data
    left_data, left_angles = dict_to_arrays(data_dict_left)
    left_output = "left_gmr_data_micrometer.npz"
    np.savez(left_output, data=left_data, angles=left_angles)
    print(f"Saved left GMR data to {left_output}")
    print(f"Shape: {left_data.shape}, Angles range: {left_angles[0]:.2f} to {left_angles[-1]:.2f}")
    
    # Convert and save right GMR data
    right_data, right_angles = dict_to_arrays(data_dict_right)
    right_output = "right_gmr_data_micrometer.npz"
    np.savez(right_output, data=right_data, angles=right_angles)
    print(f"Saved right GMR data to {right_output}")
    print(f"Shape: {right_data.shape}, Angles range: {right_angles[0]:.2f} to {right_angles[-1]:.2f}")

if __name__ == "__main__":
    save_micrometer_data() 