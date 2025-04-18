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


def read_data(comp_plots):

    # File paths
    folder_path = 'luke_test/try1'

    file_air = 'resources/data/'+folder_path+'/air.TXT'
    file_crystal = 'resources/data/'+folder_path+'/crystal.TXT'
    file_membrane = 'resources/data/'+folder_path+'/membrane.TXT'

    # Load the data files
    data_air = parse_spectrometer_data(file_air)
    data_crystal = parse_spectrometer_data(file_crystal)
    data_membrane = parse_spectrometer_data(file_membrane)

    # Extract wavelength and intensity data, starting from index 300
    wl_spectrum = data_air[300:, 0]  # Can use either source for wavelength

    # Get raw intensity data
    R_air = data_air[300:, 1]
    R_crystal = data_crystal[300:, 1]
    R_membrane = data_membrane[300:, 1]

    # Calculate normalized reflectance relative to respective sources
    R_crystal_normalized = R_crystal / R_air
    R_membrane_normalized = R_membrane / R_air
    R_crystal_normalized_membrane = R_crystal / R_membrane


    def get_norm_spectrum(folder_path):
        # File paths

        file_air = 'resources/data/'+folder_path+'/air.TXT'
        file_crystal = 'resources/data/'+folder_path+'/crystal.TXT'
        file_membrane = 'resources/data/'+folder_path+'/membrane.TXT'

        # Load the data files
        data_air = parse_spectrometer_data(file_air)
        data_crystal = parse_spectrometer_data(file_crystal)
        data_membrane = parse_spectrometer_data(file_membrane)

        # Extract wavelength and intensity data, starting from index 300
        wl_spectrum = data_air[300:, 0]  # Can use either source for wavelength

        # Get raw intensity data
        R_air = data_air[300:, 1]
        R_crystal = data_crystal[300:, 1]
        R_membrane = data_membrane[300:, 1]

        # Calculate normalized reflectance relative to respective sources
        R_crystal_normalized = R_crystal / R_air
        R_membrane_normalized = R_membrane / R_air

        return [wl_spectrum, R_crystal_normalized, R_membrane_normalized]

    if comp_plots:
        # Create figure with proper formatting
        fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(10, 6))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Add space between subplots

        # Plot normalized spectra
        # ax1.plot(wl_spectrum, R_crystal_normalized/np.max(R_crystal_normalized), color='green', linewidth=2, label='Crystal')
        # ax1.plot(wl_spectrum, R_membrane_normalized/np.max(R_membrane_normalized), color='r', linewidth=2, label='Membrane')
        # ax1.set_xlabel('Wavelength (nm)', fontsize=12)
        # ax1.set_ylabel('Normalized Intensity', fontsize=12)
        # ax1.set_title('Normalized Spectra', fontsize=14, pad=15)
        # ax1.legend(frameon=True, fancybox=True, shadow=True)
        # ax1.grid(True, alpha=0.3)
        # ax1.tick_params(axis='both', which='major', labelsize=10)

        # # Plot raw data
        # ax2.plot(wl_spectrum, R_crystal, color='green', linewidth=2, label='Crystal')
        # ax2.plot(wl_spectrum, R_membrane, color='r', linewidth=2, label='Membrane')
        # ax2.set_xlabel('Wavelength (nm)', fontsize=12)
        # ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
        # ax2.set_title('Raw Spectra', fontsize=14, pad=15)
        # ax2.legend(frameon=True, fancybox=True, shadow=True)
        # ax2.grid(True, alpha=0.3)
        # ax2.tick_params(axis='both', which='major', labelsize=10)

        # Plot normalized data
        ax1.plot(wl_spectrum, R_crystal_normalized, color='r', linewidth=2, label='Crystal')
        ax1.set_xlabel('Wavelength (nm)', fontsize=12)
        ax1.set_ylabel('Transmittance', fontsize=12)
        ax1.set_title('Grating Resonance Features', fontsize=14, pad=15)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.set_ylim([0, 1.15])


        # ax4.plot(wl_spectrum, R_membrane_normalized/np.max(R_membrane_normalized), color='orange', linewidth=2, label='Membrane')
        # ax4.set_xlabel('Wavelength (nm)', fontsize=12)
        # ax4.set_ylabel('Normalized Intensity', fontsize=12)
        # ax4.set_title('Normalized Spectra', fontsize=14, pad=15)
        # ax4.legend(frameon=True, fancybox=True, shadow=True)
        # ax4.grid(True, alpha=0.3)
        # ax4.tick_params(axis='both', which='major', labelsize=10)

        # # Plot raw data
        # ax5.plot(wl_spectrum, R_membrane, color='orange', linewidth=2, label='Membrane')
        # ax5.set_xlabel('Wavelength (nm)', fontsize=12)
        # ax5.set_ylabel('Intensity (a.u.)', fontsize=12)
        # ax5.set_title('Raw Spectra', fontsize=14, pad=15)
        # ax5.legend(frameon=True, fancybox=True, shadow=True)
        # ax5.grid(True, alpha=0.3)
        # ax5.tick_params(axis='both', which='major', labelsize=10)

        # Plot normalized data
        ax2.plot(wl_spectrum, R_crystal_normalized_membrane, color='b', linewidth=2, label='Crsytal normalize w/ Membrane')
        ax2.set_xlabel('Wavelength (nm)', fontsize=12)
        ax2.set_ylabel('Transmittance', fontsize=12)
        ax2.set_title('Grating Resonance Features', fontsize=14, pad=15)
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.set_ylim([0, 1.15])
        # Adjust layout to prevent label clipping
        plt.tight_layout()
        plt.show()