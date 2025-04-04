from src.read_data import read_data
from src.read_data_contour_fine_adjustment import plot_contour
from src.read_data_contour_fine_adjustment import read_data_from_folder
import matplotlib.pyplot as plt
import numpy as np

def create_dict_from_merged(merged_file):
    """
    Convert merged npz data into the dictionary format expected by plot_contour
    """
    merged_data = np.load(merged_file)
    data = merged_data['data']
    angles = merged_data['angles']
    
    # Create dictionary with angles as keys
    data_dict = {}
    for i, angle in enumerate(angles):
        data_dict[angle] = data[i]
    
    return data_dict

def main():
    # Load and convert merged data to dictionary format
    # left_dict = create_dict_from_merged('left_gmr_merged.npz')
    # right_dict = create_dict_from_merged('right_gmr_merged.npz')
    
    # # Create a single figure with two subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # # Plot for Left GMR
    # plot_contour(left_dict, 'Left GMR', ax1)
    
    # # Plot for Right GMR
    # plot_contour(right_dict, 'Right GMR', ax2)
    
    # # Add instructions text
    # fig.text(0.5, 0.02, 
    #          'Left click: Add slice to plot\nRight click: Remove last slice', 
    #          ha='center', fontsize=10)
    
    
    # Plot micrometer test data
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Read micrometer test data
    master_folder = 'resources/data/tilt_test_micrometer'
    data_dict_left_micro, data_dict_right_micro = read_data_from_folder(master_folder)
    
    # Plot micrometer test data
    plot_contour(data_dict_left_micro, 'Left GMR (Micrometer)', ax3)
    plot_contour(data_dict_right_micro, 'Right GMR (Micrometer)', ax4)
    
    # # Add instructions text for micrometer data
    # fig2.text(0.5, 0.02, 
    #          'Left click: Add slice to plot\nRight click: Remove last slice', 
    #          ha='center', fontsize=10)


    plt.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    main()
