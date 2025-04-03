#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:58:05 2024

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt

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