import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.signal import find_peaks
import pandas as pd

st.set_page_config(layout="wide", page_title="GMR Data Visualization")

# Cache the data parsing function to improve performance
@st.cache_data
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
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if data starts
            if not data_started:
                if line.startswith('[nm]'):
                    data_started = True
                continue  # Skip the header line with units
            
            # Process data lines
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

@st.cache_data
def read_data_from_folder(master_folder, calib_factor, angle_conv_factor, filename_pattern_type="ticks"):
    """
    Reads data from subfolders in the master folder, each representing a turn position.
    Converts turn angles to mrad with calibration factor.
    
    Args:
        master_folder (str): Path to the master folder containing angle subfolders
        calib_factor (float): Calibration factor to apply
        angle_conv_factor (float): Conversion factor from turns/ticks to mrad
        filename_pattern_type (str): Pattern to look for in subfolder names ("ticks" or "ccw")
    
    Returns:
        tuple: (data_dict_left, data_dict_right) dictionaries with angle keys and spectral data
    """
    # Calculate calibrated conversion factor
    CALIBRATION_FACTOR = calib_factor  # From UI
    MRAD_PER_UNIT = angle_conv_factor  # From UI
    MRAD_PER_UNIT_CALIBRATED = MRAD_PER_UNIT * CALIBRATION_FACTOR
    
    data_dict_left = {}
    data_dict_right = {}
    
    try:
        for subfolder_name in os.listdir(master_folder):
            subfolder_path = os.path.join(master_folder, subfolder_name)
            if os.path.isdir(subfolder_path):  # Check if it's a directory
                # Handle different file naming patterns
                if '_7414430SP.TXT' in ''.join(os.listdir(subfolder_path)):
                    # Micrometer test naming pattern
                    membrane_file = os.path.join(subfolder_path, 'membrane_7414430SP.TXT')
                    gmr_left_file = os.path.join(subfolder_path, 'gmr_left_7414430SP.TXT')
                    gmr_right_file = os.path.join(subfolder_path, 'gmr_right_7414430SP.TXT')
                else:
                    # Standard naming pattern
                    membrane_file = os.path.join(subfolder_path, 'membrane.TXT')
                    gmr_left_file = os.path.join(subfolder_path, 'gmr_left.TXT')
                    gmr_right_file = os.path.join(subfolder_path, 'gmr_right.TXT')
                
                # Check if files exist
                files_exist = (
                    os.path.isfile(membrane_file) and 
                    os.path.isfile(gmr_left_file) and 
                    os.path.isfile(gmr_right_file)
                )
                
                if not files_exist:
                    continue
                
                # Parse data
                membrane_data = parse_spectrometer_data(membrane_file)
                gmr_left_data = parse_spectrometer_data(gmr_left_file)
                gmr_right_data = parse_spectrometer_data(gmr_right_file)
                
                # Filter wavelength range to 450-1000nm
                wavelength_mask = (membrane_data[:, 0] >= 450) & (membrane_data[:, 0] <= 1000)
                membrane_data = membrane_data[wavelength_mask]
                gmr_left_data = gmr_left_data[wavelength_mask]
                gmr_right_data = gmr_right_data[wavelength_mask]
                
                # Extract angle from subfolder name
                if filename_pattern_type == "ticks":
                    turns = float(subfolder_name.replace('ticks', ''))
                elif filename_pattern_type == "ccw":
                    turns = float(subfolder_name.replace('ccw', ''))
                else:
                    # Try to extract any numeric value from the folder name
                    import re
                    turns_match = re.search(r'(\d+(?:\.\d+)?)', subfolder_name)
                    if turns_match:
                        turns = float(turns_match.group(1))
                    else:
                        continue
                
                angle_mrad = turns * MRAD_PER_UNIT_CALIBRATED
                
                if membrane_data.size > 0 and gmr_left_data.size > 0:
                    gmr_left_normalized = gmr_left_data[:, 1] / membrane_data[:, 1]
                    data_dict_left[angle_mrad] = np.column_stack((gmr_left_data[:, 0], gmr_left_normalized))
                
                if membrane_data.size > 0 and gmr_right_data.size > 0:
                    gmr_right_normalized = gmr_right_data[:, 1] / membrane_data[:, 1]
                    data_dict_right[angle_mrad] = np.column_stack((gmr_right_data[:, 0], gmr_right_normalized))
    
    except Exception as e:
        st.error(f"Error reading data: {str(e)}")
        return {}, {}
        
    if not data_dict_left and not data_dict_right:
        st.warning("No valid data files found in the specified folder.")
    
    return data_dict_left, data_dict_right

@st.cache_data
def find_peak_wavelength(spectrum_data, prominence=0.05):
    """Find the wavelength of the transmission peak
    spectrum_data: array of shape (n, 2) containing [wavelength, transmission] pairs
    """
    wavelengths = spectrum_data[:, 0]
    transmission = spectrum_data[:, 1]
    peaks, properties = find_peaks(-transmission, prominence=prominence)  # Looking for dips in transmission
    if len(peaks) == 0:
        return None
    # Return the most prominent peak
    strongest_peak = peaks[np.argmax(properties['prominences'])]
    return wavelengths[strongest_peak]

@st.cache_data
def align_and_merge_gmr_data(coarse_data, coarse_angles, fine_data, fine_angles):
    """
    Align and merge coarse and fine datasets for a specific GMR
    """
    # Find peak wavelength for each angle in both datasets
    coarse_peaks = [find_peak_wavelength(spectrum) for spectrum in coarse_data]
    fine_peaks = [find_peak_wavelength(spectrum) for spectrum in fine_data]
    
    # Filter out None values
    valid_coarse_peaks = [(i, peak) for i, peak in enumerate(coarse_peaks) if peak is not None]
    valid_fine_peaks = [(i, peak) for i, peak in enumerate(fine_peaks) if peak is not None]
    
    if not valid_coarse_peaks or not valid_fine_peaks:
        st.error("Could not find valid peaks for alignment")
        return None, None
    
    # Use middle of fine scan for reference
    mid_fine_idx, reference_peak = valid_fine_peaks[len(valid_fine_peaks) // 2]
    
    # Find best matching peak in coarse data
    peak_differences = [(i, abs(peak - reference_peak)) for i, peak in valid_coarse_peaks]
    closest_coarse_idx, _ = min(peak_differences, key=lambda x: x[1])
    
    # Align fine dataset with coarse dataset
    fine_center = fine_angles[mid_fine_idx]
    angle_offset = coarse_angles[closest_coarse_idx] - fine_center
    adjusted_fine_angles = fine_angles + angle_offset
    
    # Merge the datasets
    all_angles = np.sort(np.unique(np.concatenate([coarse_angles, adjusted_fine_angles])))
    merged_spectra = []
    merged_angles = []
    
    for angle in all_angles:
        # Check if angle is in fine dataset range
        if angle >= adjusted_fine_angles[0] and angle <= adjusted_fine_angles[-1]:
            # Use fine data if available
            idx = np.argmin(np.abs(adjusted_fine_angles - angle))
            if np.abs(adjusted_fine_angles[idx] - angle) < 0.05:  # threshold for matching
                merged_spectra.append(fine_data[idx])
                merged_angles.append(angle)
                continue
        
        # Use coarse data if fine data not available
        idx = np.argmin(np.abs(coarse_angles - angle))
        if np.abs(coarse_angles[idx] - angle) < 0.1:
            merged_spectra.append(coarse_data[idx])
            merged_angles.append(angle)
    
    merged_spectra = np.array(merged_spectra)
    merged_angles = np.array(merged_angles)
    
    return merged_spectra, merged_angles

def plot_contour_to_figure(data_dict, title_suffix, min_wl=None, max_wl=None, min_val=0.0, max_val=1.2, cmap='inferno', angle_range=None, zoom_bands=None):
    """
    Creates a matplotlib figure with a contour plot of the GMR data
    
    Args:
        data_dict: Dictionary with angle keys and spectral data values
        title_suffix: Title suffix for the plot
        min_wl, max_wl: Wavelength range for filtering
        min_val, max_val: Transmittance value range for colormap
        cmap: Colormap name
        angle_range: Optional tuple of (min_angle, max_angle) to filter angles
        zoom_bands: Optional list of wavelength bands to highlight
    
    Returns:
        fig: matplotlib Figure object
        xdata: wavelengths array
        ydata: angles array
        zdata: transmission data array
    """
    if not data_dict:
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"No data available to plot for {title_suffix}", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig, None, None, None
    
    # Filter angles if angle_range is provided
    if angle_range is not None:
        min_angle, max_angle = angle_range
        angles = [angle for angle in sorted(data_dict.keys()) if min_angle <= angle <= max_angle]
        if not angles:  # If no angles left after filtering
            angles = sorted(data_dict.keys())
    else:
        angles = sorted(data_dict.keys())
    
    wavelengths = data_dict[angles[0]][:, 0]
    transmittance_data = np.array([data_dict[angle][:, 1] for angle in angles])
    
    # Apply wavelength filter if provided
    if min_wl is not None and max_wl is not None:
        wl_mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
        wavelengths = wavelengths[wl_mask]
        transmittance_data = transmittance_data[:, wl_mask]
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Plot contour using pcolormesh for better performance
    mesh = ax.pcolormesh(wavelengths, angles, transmittance_data,
                     shading='auto',
                     vmin=min_val,
                     vmax=max_val,
                     cmap=cmap)
    
    cbar = fig.colorbar(mesh, ax=ax, label='Transmittance')
    
    # Add zoom bands if provided
    if zoom_bands:
        for band in zoom_bands:
            if band == "550-650 nm":
                ax.axvspan(550, 650, alpha=0.2, color='green', label='550-650 nm band')
            elif band == "820-920 nm":
                ax.axvspan(820, 920, alpha=0.2, color='blue', label='820-920 nm band')
        ax.legend(loc='upper right')
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Relative Angle Position (mrad)')
    ax.set_title(f'Transmittance vs Wavelength and Angle ({title_suffix})')
    
    # Set x-axis and y-axis limits if filters were applied
    if min_wl is not None and max_wl is not None:
        ax.set_xlim(min_wl, max_wl)
    if angle_range is not None:
        ax.set_ylim(max_angle, min_angle)  # Reversed to match pcolormesh orientation
        
    fig.tight_layout()
    
    return fig, wavelengths, angles, transmittance_data

def plot_slices(wavelengths, angles, transmittance, selected_wls, selected_angles, gmr_label, trans_range=None):
    """
    Creates plots for selected wavelength and angle slices.
    
    Args:
        wavelengths (np.array): Array of wavelength values
        angles (np.array): Array of angle values
        transmittance (np.array): 2D array of transmittance values
        selected_wls (list): List of selected wavelengths for plotting
        selected_angles (list): List of selected angles for plotting
        gmr_label (str): Label indicating "Left" or "Right" GMR
        trans_range (tuple): Optional (min, max) range for transmittance axis
    
    Returns:
        matplotlib.figure.Figure: Figure containing the slice plots
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Create two subplots side by side
    if selected_wls and selected_angles:
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
    elif selected_wls:
        ax1 = plt.subplot(111)
    elif selected_angles:
        ax2 = plt.subplot(111)
    else:
        return fig

    # Plot wavelength slices (Transmittance vs Angle)
    if selected_wls:
        for wl in selected_wls:
            wl_idx = np.abs(wavelengths - wl).argmin()
            ax1.plot(angles, transmittance[:, wl_idx], label=f'{wl:.1f} nm')
        
        ax1.set_xlabel('Angle (mrad)')
        ax1.set_ylabel('Transmittance')
        ax1.set_title(f'{gmr_label} GMR - Fixed Wavelength Slices')
        if trans_range is not None:
            ax1.set_ylim(trans_range)
        ax1.legend()
        ax1.grid(True)

    # Plot angle slices (Transmittance vs Wavelength)
    if selected_angles:
        for angle in selected_angles:
            angle_idx = np.abs(angles - angle).argmin()
            ax2.plot(wavelengths, transmittance[angle_idx, :], label=f'{angle:.2f} mrad')
        
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Transmittance')
        ax2.set_title(f'{gmr_label} GMR - Fixed Angle Slices')
        if trans_range is not None:
            ax2.set_ylim(trans_range)
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    return fig

def plot_single_spectrum(data_dict, angle, gmr_type, min_wl=None, max_wl=None):
    """
    Plots a single spectrum for a given angle
    """
    if not data_dict or angle not in data_dict:
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No data available for this angle", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    spectrum_data = data_dict[angle]
    wavelengths = spectrum_data[:, 0]
    transmittance = spectrum_data[:, 1]
    
    # Apply wavelength filter if provided
    if min_wl is not None and max_wl is not None:
        wl_mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
        wavelengths = wavelengths[wl_mask]
        transmittance = transmittance[wl_mask]
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    ax.plot(wavelengths, transmittance, linewidth=2)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmittance')
    ax.set_title(f'{gmr_type} GMR at {angle:.2f} mrad')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.2])
    
    # Find and mark resonance peaks
    peaks, properties = find_peaks(-transmittance, prominence=0.05)
    if len(peaks) > 0:
        peak_wls = wavelengths[peaks]
        peak_values = transmittance[peaks]
        ax.scatter(peak_wls, peak_values, color='red', s=50, zorder=3)
        
        # Annotate peaks
        for i, (x, y) in enumerate(zip(peak_wls, peak_values)):
            ax.annotate(f"{x:.1f} nm", 
                       (x, y), 
                       xytext=(0, 10), 
                       textcoords='offset points',
                       ha='center')
    
    fig.tight_layout()
    return fig

def plot_comparison(data_dicts, labels, min_wl=None, max_wl=None, gmr_type="Left"):
    """
    Creates comparison plots for multiple datasets
    
    Args:
        data_dicts: List of data dictionaries
        labels: List of labels for each dataset
        min_wl, max_wl: Optional wavelength range
        gmr_type: Either "Left" or "Right"
    
    Returns:
        fig: matplotlib Figure object
    """
    if not data_dicts or not all(data_dicts):
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No data available for comparison", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    # Get a common angle (closest to zero)
    common_angles = []
    for data_dict in data_dicts:
        angles = np.array(list(data_dict.keys()))
        closest_to_zero = angles[np.argmin(np.abs(angles))]
        common_angles.append(closest_to_zero)
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    for i, (data_dict, label, angle) in enumerate(zip(data_dicts, labels, common_angles)):
        wavelengths = data_dict[angle][:, 0]
        transmittance = data_dict[angle][:, 1]
        
        # Apply wavelength filter if provided
        if min_wl is not None and max_wl is not None:
            wl_mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
            wavelengths = wavelengths[wl_mask]
            transmittance = transmittance[wl_mask]
        
        ax.plot(wavelengths, transmittance, linewidth=2, label=f"{label} ({angle:.2f} mrad)")
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmittance')
    ax.set_title(f'{gmr_type} GMR Comparison')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.2])
    ax.legend()
    
    fig.tight_layout()
    return fig

def fig_to_image(fig):
    """Convert a matplotlib Figure to an image for Streamlit display"""
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    return buf

# Function to load merged NPZ data into a dictionary format
def load_merged_npz(filepath):
    """
    Load a merged NPZ file and convert it to the dictionary format used by the plotting functions
    """
    try:
        data = np.load(filepath)
        merged_data = data['data']
        merged_angles = data['angles']
        
        # Convert to dictionary format
        data_dict = {}
        for i, angle in enumerate(merged_angles):
            data_dict[angle] = merged_data[i]
        
        return data_dict
    except Exception as e:
        st.error(f"Error loading NPZ file: {str(e)}")
        return None

def main():
    st.title("GMR Data Visualization")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Add file paths settings to sidebar
    with st.sidebar.expander("File Paths", expanded=True):
        base_folder = st.text_input("Base Data Folder Path", "resources/data")
        
        # Option to load NPZ files directly
        use_npz_files = st.checkbox("Use NPZ files directly", value=False)
        if use_npz_files:
            left_npz_path = st.text_input("Left GMR NPZ File Path", "left_gmr_merged.npz")
            right_npz_path = st.text_input("Right GMR NPZ File Path", "right_gmr_merged.npz")
    
    # Check if we're using NPZ files directly
    if use_npz_files:
        # Load from NPZ files if they exist
        data_dict_left = None
        data_dict_right = None
        
        if os.path.exists(left_npz_path):
            try:
                data_dict_left = load_merged_npz(left_npz_path)
                st.sidebar.success(f"Loaded Left GMR data from {left_npz_path}")
            except Exception as e:
                st.sidebar.error(f"Error loading {left_npz_path}: {str(e)}")
        
        if os.path.exists(right_npz_path):
            try:
                data_dict_right = load_merged_npz(right_npz_path)
                st.sidebar.success(f"Loaded Right GMR data from {right_npz_path}")
            except Exception as e:
                st.sidebar.error(f"Error loading {right_npz_path}: {str(e)}")
        
        available_folders = []
    else:
        # Get list of available folders
        available_folders = []
        try:
            available_folders = [folder for folder in os.listdir(base_folder) 
                                if os.path.isdir(os.path.join(base_folder, folder))]
        except:
            st.sidebar.warning(f"Cannot read directory: {base_folder}")
    
    # Analyze mode selection
    analysis_mode = st.sidebar.selectbox("Analysis Mode", 
                                       ["Single Dataset", "Merged Datasets", "Multiple Comparison", "Advanced Analysis"])
    
    # Calibration parameters
    with st.sidebar.expander("Calibration Settings", expanded=True):
        calib_factor = st.number_input("Calibration Factor", 
                                     value=437/439, 
                                     format="%.5f",
                                     help="Calibration factor from membrane tilt")
        
        angle_conv_type = st.selectbox("Angle Conversion Type", 
                                      ["Micrometer (Ticks)", "Turns (CCW)", "Custom"])
        
        if angle_conv_type == "Micrometer (Ticks)":
            angle_conv_factor = st.number_input("mrad per Tick", 
                                              value=0.275, 
                                              format="%.5f")
            filename_pattern = "ticks"
        elif angle_conv_type == "Turns (CCW)":
            angle_conv_factor = st.number_input("mrad per Turn", 
                                              value=6.63, 
                                              format="%.5f")
            filename_pattern = "ccw"
        else:
            angle_conv_factor = st.number_input("Custom Conversion Factor", 
                                              value=1.0, 
                                              format="%.5f",
                                              help="Custom conversion factor from folder name to mrad")
            filename_pattern = st.text_input("Folder Name Pattern", 
                                           value="custom",
                                           help="Text pattern to remove from folder name to extract angle value")
    
    # Visualization settings
    with st.sidebar.expander("Visualization Settings", expanded=True):
        color_map = st.selectbox("Color Map", 
                               ["inferno", "viridis", "plasma", "magma", "cividis", "turbo"])
        
        wl_range = st.slider("Wavelength Range (nm)", 
                           min_value=450, max_value=1000, 
                           value=(450, 1000))
        
        transmittance_range = st.slider("Transmittance Range", 
                                      min_value=0.0, max_value=2.0, 
                                      value=(0.0, 1.2), step=0.1)
        
        # Add angle range control
        show_angle_range = st.checkbox("Limit Angle Range", value=False)
        if show_angle_range:
            angle_range = st.slider("Angle Range (mrad)", 
                                  min_value=-50.0, max_value=50.0, 
                                  value=(-20.0, 20.0), step=0.5)
        else:
            angle_range = None
        
        # Add zoom bands
        enable_zoom_bands = st.checkbox("Enable Zoom Bands", value=False)
        if enable_zoom_bands:
            zoom_bands = st.multiselect("Predefined Zoom Bands", 
                                       ["550-650 nm", "820-920 nm"])
        else:
            zoom_bands = []
        
        gmr_selection = st.radio("GMR Selection", ["Left", "Right", "Both"])
    
    # Main content area based on selected mode
    if analysis_mode == "Single Dataset":
        st.header("Single Dataset Analysis")
        
        # If we're using NPZ files directly, show the contour plots
        if use_npz_files and (data_dict_left or data_dict_right):
            col1, col2 = st.columns(2)
            
            if gmr_selection in ["Left", "Both"] and data_dict_left:
                with col1:
                    st.subheader("Left GMR")
                    fig_left, wl_left, angles_left, trans_left = plot_contour_to_figure(
                        data_dict_left, "Left GMR", wl_range[0], wl_range[1], 
                        transmittance_range[0], transmittance_range[1], color_map,
                        angle_range if show_angle_range else None,
                        zoom_bands if enable_zoom_bands else None)
                    st.image(fig_to_image(fig_left))
                    
                    # Create angle selection for slices
                    if angles_left is not None:
                        display_slice_analysis(wl_left, angles_left, trans_left, "Left", 
                                             trans_range=(transmittance_range[0], transmittance_range[1]))
            
            if gmr_selection in ["Right", "Both"] and data_dict_right:
                with col2:
                    st.subheader("Right GMR")
                    fig_right, wl_right, angles_right, trans_right = plot_contour_to_figure(
                        data_dict_right, "Right GMR", wl_range[0], wl_range[1],
                        transmittance_range[0], transmittance_range[1], color_map,
                        angle_range if show_angle_range else None,
                        zoom_bands if enable_zoom_bands else None)
                    st.image(fig_to_image(fig_right))
                    
                    # Create angle selection for slices
                    if angles_right is not None:
                        display_slice_analysis(wl_right, angles_right, trans_right, "Right",
                                             trans_range=(transmittance_range[0], transmittance_range[1]))
        elif not available_folders:
            st.warning("No folders found in the specified base directory.")
        else:
            selected_folder = st.selectbox("Select Data Folder", available_folders)
            full_folder_path = os.path.join(base_folder, selected_folder)
            
            if st.button("Load Data"):
                with st.spinner("Loading and processing data..."):
                    data_dict_left, data_dict_right = read_data_from_folder(
                        full_folder_path, calib_factor, angle_conv_factor, filename_pattern)
                    
                    if not data_dict_left and not data_dict_right:
                        st.error("No valid data found in the selected folder.")
                    else:
                        st.success(f"Data loaded successfully from {selected_folder}")
                        
                        col1, col2 = st.columns(2)
                        
                        if gmr_selection in ["Left", "Both"] and data_dict_left:
                            with col1:
                                st.subheader("Left GMR")
                                fig_left, wl_left, angles_left, trans_left = plot_contour_to_figure(
                                    data_dict_left, "Left GMR", wl_range[0], wl_range[1], 
                                    transmittance_range[0], transmittance_range[1], color_map,
                                    angle_range if show_angle_range else None,
                                    zoom_bands if enable_zoom_bands else None)
                                st.image(fig_to_image(fig_left))
                                
                                # Create angle selection for slices
                                if angles_left is not None:
                                    display_slice_analysis(wl_left, angles_left, trans_left, "Left", 
                                                         trans_range=(transmittance_range[0], transmittance_range[1]))
                        
                        if gmr_selection in ["Right", "Both"] and data_dict_right:
                            with col2:
                                st.subheader("Right GMR")
                                fig_right, wl_right, angles_right, trans_right = plot_contour_to_figure(
                                    data_dict_right, "Right GMR", wl_range[0], wl_range[1],
                                    transmittance_range[0], transmittance_range[1], color_map,
                                    angle_range if show_angle_range else None,
                                    zoom_bands if enable_zoom_bands else None)
                                st.image(fig_to_image(fig_right))
                                
                                # Create angle selection for slices
                                if angles_right is not None:
                                    display_slice_analysis(wl_right, angles_right, trans_right, "Right",
                                                         trans_range=(transmittance_range[0], transmittance_range[1]))

def display_slice_analysis(wl, angles, trans, gmr_label, trans_range):
    st.write("### Slice Analysis")
    
    # Initialize session state keys if they don't exist
    state_key_angles = f"selected_angles_{gmr_label.lower()}"
    state_key_wls = f"selected_wls_{gmr_label.lower()}"
    
    if state_key_angles not in st.session_state:
        st.session_state[state_key_angles] = []
    if state_key_wls not in st.session_state:
        st.session_state[state_key_wls] = []
    
    # Convert lists to numpy arrays if needed
    wl = np.array(wl) if not isinstance(wl, np.ndarray) else wl
    angles = np.array(angles) if not isinstance(angles, np.ndarray) else angles
    trans = np.array(trans) if not isinstance(trans, np.ndarray) else trans
    
    # Ensure data is sorted
    angle_options = np.sort(angles)
    wavelength_options = np.sort(wl)
    
    # Add key suffix to avoid duplicate keys between Left/Right GMR
    key_suffix = "_" + gmr_label.lower()
    
    # Use session state for the multiselect widgets
    selected_angles = st.multiselect(
        f"Select angles for {gmr_label} GMR slice plots (mrad)", 
        options=angle_options.tolist(),
        default=st.session_state[state_key_angles],
        format_func=lambda x: f"{x:.2f}",
        key=f"angles{key_suffix}",
        on_change=lambda: setattr(st.session_state, state_key_angles, 
                                st.session_state[f"angles{key_suffix}"]))
    
    selected_wls = st.multiselect(
        f"Select wavelengths for {gmr_label} GMR slice plots (nm)",
        options=wavelength_options.tolist(),
        default=st.session_state[state_key_wls],
        format_func=lambda x: f"{x:.1f}",
        key=f"wls{key_suffix}",
        on_change=lambda: setattr(st.session_state, state_key_wls, 
                                st.session_state[f"wls{key_suffix}"]))
    
    if selected_angles or selected_wls:
        try:
            slice_fig = plot_slices(wl, angles, trans, selected_wls, selected_angles, gmr_label, trans_range)
            st.pyplot(slice_fig)
        except Exception as e:
            st.error(f"Error creating slice plots: {str(e)}")
            st.write("Debug info:", {
                "Selected angles": selected_angles,
                "Selected wavelengths": selected_wls,
                "Data ranges": {
                    "Wavelength": [float(wl.min()), float(wl.max())],
                    "Angles": [float(angles.min()), float(angles.max())],
                    "Transmittance": [float(trans.min()), float(trans.max())]
                }
            })

if __name__ == "__main__":
    main()