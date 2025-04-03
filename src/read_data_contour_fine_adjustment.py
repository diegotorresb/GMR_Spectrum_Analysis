#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:58:05 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import os  # Add import for os module
import matplotlib.widgets as widgets

def parse_spectrometer_data(file_path):
    """
    Parses a spectrometer data file to extract wavelength and sample columns.
    
    Args:
        file_path (str): Path to the .txt file.
    
    Returns:
        numpy.ndarray: A 2D NumPy array with two columns: wavelength and sample.
    """
    wavelengths = []
    samples = []

    with open(file_path, 'r') as file:
        data_started = False
        for line in file:
            line = line.strip()
            # Check if data starts
            if not data_started:
                if line.startswith('[nm]'):
                    data_started = True
                    continue  # Skip the header line with units
            else:
                if not line:
                    continue  # Skip empty lines
                parts = line.split(';')
                if len(parts) >= 2:
                    try:
                        # Extract and convert wavelength and sample values
                        wavelength = float(parts[0].strip())
                        sample = float(parts[1].strip())
                        wavelengths.append(wavelength)
                        samples.append(sample)
                    except ValueError:
                        # Skip lines that cannot be parsed
                        continue

    # Combine the wavelength and sample data into a single NumPy array
    data = np.column_stack((wavelengths, samples))
    return data


def read_data(comp_plots, mult_plots):

    # File paths
    folder_path = 'tilt_test/4cw_turns'

    file_source_right = 'resources/data/'+folder_path+'/membrane.TXT'
    file_source_left = 'resources/data/'+folder_path+'/membrane.TXT'
    file_gmr_left = 'resources/data/'+folder_path+'/gmr_left.TXT'
    file_gmr_right = 'resources/data/'+folder_path+'/gmr_right.TXT'  # Fixed variable name

    # Load the data files
    data_source_right = parse_spectrometer_data(file_source_right)
    data_source_left = parse_spectrometer_data(file_source_left)
    data_gmr_left = parse_spectrometer_data(file_gmr_left)
    data_gmr_right = parse_spectrometer_data(file_gmr_right)

    # Extract wavelength and intensity data, starting from index 300
    wl_spectrum = data_source_right[300:, 0]  # Can use either source for wavelength

    # Get raw intensity data
    R_source_right = data_source_right[300:, 1]
    R_source_left = data_source_left[300:, 1]
    R_gmr_left = data_gmr_left[300:, 1]
    R_gmr_right = data_gmr_right[300:, 1]

    # Calculate normalized reflectance relative to respective sources
    R_gmr_left_normalized = R_gmr_left / R_source_left
    R_gmr_right_normalized = R_gmr_right / R_source_right


    def get_norm_spectrum(folder_path):
        # File paths

        file_source_right = 'resources/data/'+folder_path+'/membrane.TXT'
        file_source_left = 'resources/data/'+folder_path+'/membrane.TXT'
        file_gmr_left = 'resources/data/'+folder_path+'/gmr_left.TXT'
        file_gmr_right = 'resources/data/'+folder_path+'/gmr_right.TXT'  # Fixed variable name

        # Load the data files
        data_source_right = parse_spectrometer_data(file_source_right)
        data_source_left = parse_spectrometer_data(file_source_left)
        data_gmr_left = parse_spectrometer_data(file_gmr_left)
        data_gmr_right = parse_spectrometer_data(file_gmr_right)

        # Extract wavelength and intensity data, starting from index 300
        wl_spectrum = data_source_right[300:, 0]  # Can use either source for wavelength

        # Get raw intensity data
        R_source_right = data_source_right[300:, 1]
        R_source_left = data_source_left[300:, 1]
        R_gmr_left = data_gmr_left[300:, 1]
        R_gmr_right = data_gmr_right[300:, 1]

        # Calculate normalized reflectance relative to respective sources
        R_gmr_left_normalized = R_gmr_left / R_source_left
        R_gmr_right_normalized = R_gmr_right / R_source_right

        return [wl_spectrum, R_gmr_left_normalized, R_gmr_right_normalized]

    if comp_plots:
        # Create figure with proper formatting
        fig, (((ax1), (ax2), (ax3)), ((ax4), (ax5), (ax6))) = plt.subplots(2, 3, figsize=(18, 6))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Add space between subplots

        # Plot normalized spectra
        ax1.plot(wl_spectrum, R_source_left/np.max(R_source_left), color='green', linewidth=2, label='Membrane')
        ax1.plot(wl_spectrum, R_gmr_left/np.max(R_gmr_left), color='r', linewidth=2, label='Left GMR')
        ax1.set_xlabel('Wavelength (nm)', fontsize=12)
        ax1.set_ylabel('Normalized Intensity', fontsize=12)
        ax1.set_title('Normalized Spectra', fontsize=14, pad=15)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)

        # Plot raw data
        ax2.plot(wl_spectrum, R_source_left, color='green', linewidth=2, label='Membrane')
        ax2.plot(wl_spectrum, R_gmr_left, color='r', linewidth=2, label='Left GMR')
        ax2.set_xlabel('Wavelength (nm)', fontsize=12)
        ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax2.set_title('Raw Spectra', fontsize=14, pad=15)
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=10)

        # Plot normalized data
        ax3.plot(wl_spectrum, R_gmr_left_normalized, color='r', linewidth=2, label='Left GMR')
        ax3.set_xlabel('Wavelength (nm)', fontsize=12)
        ax3.set_ylabel('Transmittance', fontsize=12)
        ax3.set_title('Grating Resonance Features', fontsize=14, pad=15)
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        ax3.set_ylim([0, 1.15])


        ax4.plot(wl_spectrum, R_source_right/np.max(R_source_right), color='orange', linewidth=2, label='Membrane')
        ax4.plot(wl_spectrum, R_gmr_right/np.max(R_gmr_right), color='b', linewidth=2, label='Right GMR')
        ax4.set_xlabel('Wavelength (nm)', fontsize=12)
        ax4.set_ylabel('Normalized Intensity', fontsize=12)
        ax4.set_title('Normalized Spectra', fontsize=14, pad=15)
        ax4.legend(frameon=True, fancybox=True, shadow=True)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', labelsize=10)

        # Plot raw data
        ax5.plot(wl_spectrum, R_source_right, color='orange', linewidth=2, label='Membrane')
        ax5.plot(wl_spectrum, R_gmr_right, color='b', linewidth=2, label='Right GMR')
        ax5.set_xlabel('Wavelength (nm)', fontsize=12)
        ax5.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax5.set_title('Raw Spectra', fontsize=14, pad=15)
        ax5.legend(frameon=True, fancybox=True, shadow=True)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='both', which='major', labelsize=10)

        # Plot normalized data
        ax6.plot(wl_spectrum, R_gmr_right_normalized, color='b', linewidth=2, label='Right GMR')
        ax6.set_xlabel('Wavelength (nm)', fontsize=12)
        ax6.set_ylabel('Transmittance', fontsize=12)
        ax6.set_title('Grating Resonance Features', fontsize=14, pad=15)
        ax6.legend(frameon=True, fancybox=True, shadow=True)
        ax6.set_ylim([0, 1.15])

    if mult_plots:
         # Create figure with proper formatting
        fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(6, 6))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Add space between subplots
        
        control_data = get_norm_spectrum('tilt_test/control')
        four_cw_data = get_norm_spectrum('tilt_test/4cw_turns') 
        eight_cw_data = get_norm_spectrum('tilt_test/8cw_turns') 

        # Plot normalized data for left gmr
        ax1.plot(wl_spectrum, control_data[1], color='r', linewidth=1, label='Control')
        #ax1.plot(wl_spectrum, one_cw_data[1], color='g', linewidth=1, label='1 Turn')
        ax1.plot(wl_spectrum, eight_cw_data[1], color='g', linewidth=1, label='8 Turns')
        ax1.plot(wl_spectrum, four_cw_data[1], color='b', linewidth=1, label='4 Turns')
        ax1.set_xlabel('Wavelength (nm)', fontsize=12)
        ax1.set_ylabel('Transmittance', fontsize=12)
        ax1.set_title('Grating Resonance Features', fontsize=14, pad=15)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.set_ylim([0, 1.15])
        ax1.set_xlim([820, 920])

        # Plot normalized data
        ax2.plot(wl_spectrum, control_data[2], color='r', linewidth=1, label='Control')
        #ax2.plot(wl_spectrum, one_cw_data[2], color='g', linewidth=1, label='1 Turn')
        ax2.plot(wl_spectrum, eight_cw_data[2], color='g', linewidth=1, label='8 Turns')
        ax2.plot(wl_spectrum, four_cw_data[2], color='b', linewidth=1, label='4 Turns')
        ax2.set_xlabel('Wavelength (nm)', fontsize=12)
        ax2.set_ylabel('Transmittance', fontsize=12)
        ax2.set_title('Grating Resonance Features', fontsize=14, pad=15)
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.set_ylim([0, 1.15])
        ax2.set_xlim([550, 650])

        # Adjust layout to prevent label clipping
        plt.tight_layout()
        plt.show()

def read_data_from_folder(master_folder):
    """
    Reads data from subfolders in the master folder, each representing a turn position.
    Converts turn angles to mrad with calibration factor.
    """
    MRAD_PER_TICK = 0.275  # Base conversion factor
    CALIBRATION_FACTOR = 437/439  # Calibration factor from membrane tilt
    MRAD_PER_TURN_CALIBRATED = MRAD_PER_TICK * CALIBRATION_FACTOR
    
    data_dict_left = {}
    data_dict_right = {}
    for subfolder_name in os.listdir(master_folder):
        subfolder_path = os.path.join(master_folder, subfolder_name)
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            membrane_file = os.path.join(subfolder_path, 'membrane_7414430SP.TXT')
            gmr_left_file = os.path.join(subfolder_path, 'gmr_left_7414430SP.TXT')
            gmr_right_file = os.path.join(subfolder_path, 'gmr_right_7414430SP.TXT')
            
            # Parse data
            membrane_data = parse_spectrometer_data(membrane_file)
            gmr_left_data = parse_spectrometer_data(gmr_left_file)
            gmr_right_data = parse_spectrometer_data(gmr_right_file)
            
            # Filter wavelength range to 400-1000nm
            wavelength_mask = (membrane_data[:, 0] >= 450) & (membrane_data[:, 0] <= 1000)
            membrane_data = membrane_data[wavelength_mask]
            gmr_left_data = gmr_left_data[wavelength_mask]
            gmr_right_data = gmr_right_data[wavelength_mask]
            
            if membrane_data.size > 0 and gmr_left_data.size > 0:
                turns = float(subfolder_name.replace('ticks', ''))
                angle_mrad = turns * MRAD_PER_TURN_CALIBRATED
                gmr_left_normalized = gmr_left_data[:, 1] / membrane_data[:, 1]
                data_dict_left[angle_mrad] = np.column_stack((gmr_left_data[:, 0], gmr_left_normalized))
            
            if membrane_data.size > 0 and gmr_right_data.size > 0:
                turns = float(subfolder_name.replace('ticks', ''))
                angle_mrad = turns * MRAD_PER_TURN_CALIBRATED
                gmr_right_normalized = gmr_right_data[:, 1] / membrane_data[:, 1]
                data_dict_right[angle_mrad] = np.column_stack((gmr_right_data[:, 0], gmr_right_normalized))
    
    if not data_dict_left and not data_dict_right:
        print("No valid data files found in the specified folder.")
    return data_dict_left, data_dict_right

class SlicePlotter:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.slices = {'vertical': [], 'horizontal': []}
        self.lines = {'vertical': [], 'horizontal': []}
        
        # Initialize plot settings
        self.ax1.set_xlabel('Angle Position (mrad)')
        self.ax1.set_ylabel('Transmittance')
        self.ax1.grid(True)
        self.ax1.set_ylim([0.0, 1.2])
        
        self.ax2.set_xlabel('Wavelength (nm)')
        self.ax2.set_ylabel('Transmittance')
        self.ax2.grid(True)
        self.ax2.set_ylim([0.0, 1.2])
        
        plt.tight_layout()
        self.fig.show()
    
    def add_slice(self, wavelengths, angles, transmittance_data, x_pos, y_pos, gmr_type):
        # Add vertical slice (T vs θ)
        idx_wl = np.abs(wavelengths - x_pos).argmin()
        line1, = self.ax1.plot(angles, transmittance_data[:, idx_wl], linewidth=1, 
                              label=f'λ = {wavelengths[idx_wl]:.1f} nm')
        self.lines['vertical'].append(line1)
        self.slices['vertical'].append((wavelengths[idx_wl], idx_wl))
        
        # Add horizontal slice (T vs λ)
        idx_angle = np.abs(angles - y_pos).argmin()
        line2, = self.ax2.plot(wavelengths, transmittance_data[idx_angle, :], linewidth=1,
                              label=f'θ = {angles[idx_angle]:.2f} mrad')
        self.lines['horizontal'].append(line2)
        self.slices['horizontal'].append((angles[idx_angle], idx_angle))
        
        # Update titles and legends
        self.ax1.set_title(f'{gmr_type} GMR: T(θ)')
        self.ax2.set_title(f'{gmr_type} GMR: T(λ)')
        self.ax1.legend()
        self.ax2.legend()
        self.fig.canvas.draw_idle()
    
    def remove_last_slice(self):
        for direction in ['vertical', 'horizontal']:
            if self.lines[direction]:
                line = self.lines[direction].pop()
                line.remove()
                self.slices[direction].pop()
        
        self.ax1.legend()
        self.ax2.legend()
        self.fig.canvas.draw_idle()

def plot_contour(data_dict, title_suffix, ax):
    """
    Plots a contour plot using imshow with interactive slice selection.
    """
    if not data_dict:
        print(f"No data available to plot for {title_suffix}.")
        return

    angles = sorted(data_dict.keys())
    wavelengths = data_dict[angles[0]][:, 0]
    transmittance_data = np.array([data_dict[angle][:, 1] for angle in angles])

    # Plot contour
    im = ax.imshow(transmittance_data, aspect='auto', 
                   extent=[wavelengths[0], wavelengths[-1], angles[-1], angles[0]], 
                   cmap='inferno', vmin=0.0, vmax=1.1)
    
    plt.colorbar(im, ax=ax, label='Transmittance')
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Relative Angle Position (mrad)')
    ax.set_title(f'Transmittance vs Wavelength and Angle ({title_suffix})')

    # Add vertical and horizontal lines for slice visualization
    vline = ax.axvline(wavelengths[0], color='k', linestyle='--', alpha=0.5)
    hline = ax.axhline(angles[0], color='k', linestyle='--', alpha=0.5)
    
    # Store the data for the callback function
    ax.wavelengths = wavelengths
    ax.angles = angles
    ax.transmittance_data = transmittance_data
    ax.slice_plotter = SlicePlotter()  # Create a single slice plotter instance
    
    def on_click(event):
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            vline.set_xdata([x, x])
            hline.set_ydata([y, y])
            gmr_type = title_suffix.split()[0]
            
            if event.button == 1:  # Left click: add slice
                ax.slice_plotter.add_slice(wavelengths, angles, transmittance_data, 
                                         x_pos=x, y_pos=y, gmr_type=gmr_type)
            elif event.button == 3:  # Right click: remove last slice
                ax.slice_plotter.remove_last_slice()
            
            ax.figure.canvas.draw_idle()

    # Connect the click event
    ax.figure.canvas.mpl_connect('button_press_event', on_click)
