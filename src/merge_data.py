import numpy as np
from scipy.signal import find_peaks

def find_peak_wavelength(spectrum_data, prominence=0.05):
    """Find the wavelength of the transmission peak
    spectrum_data: array of shape (961, 2) containing [wavelength, transmission] pairs
    """
    wavelengths = spectrum_data[:, 0]
    transmission = spectrum_data[:, 1]
    peaks, properties = find_peaks(-transmission, prominence=prominence)  # Looking for dips in transmission
    if len(peaks) == 0:
        return None
    # Return the most prominent peak
    strongest_peak = peaks[np.argmax(properties['prominences'])]
    return wavelengths[strongest_peak]

def find_all_peaks(spectrum_data, prominence=0.05):
    """Find all significant peaks in the spectrum"""
    wavelengths = spectrum_data[:, 0]
    transmission = spectrum_data[:, 1]
    peaks, properties = find_peaks(-transmission, prominence=prominence)
    if len(peaks) == 0:
        return None, None
    return wavelengths[peaks], properties['prominences']

def align_and_merge_gmr_data(side='left'):
    """
    Align and merge coarse and fine datasets for a specific GMR (left or right)
    """
    # Load both datasets
    coarse_file = f'{side}_gmr_data_comp.npz'
    fine_file = f'{side}_gmr_data_micrometer.npz'
    
    coarse_data = np.load(coarse_file)
    fine_data = np.load(fine_file)
    
    # Extract data and angles
    coarse_spectra = coarse_data['data']
    coarse_angles = coarse_data['angles']
    fine_spectra = fine_data['data']
    fine_angles = fine_data['angles']
    
    print(f"\nProcessing {side} GMR:")
    print(f"Coarse angles range: {coarse_angles[0]:.2f} to {coarse_angles[-1]:.2f} mrad")
    print(f"Fine angles range: {fine_angles[0]:.2f} to {fine_angles[-1]:.2f} mrad")
    
    # Find all peaks in both datasets
    coarse_peaks = []
    coarse_prominences = []
    for spectrum in coarse_spectra:
        peaks, prominences = find_all_peaks(spectrum)
        if peaks is not None:
            coarse_peaks.append(peaks)
            coarse_prominences.append(prominences)
        else:
            coarse_peaks.append(None)
            coarse_prominences.append(None)
    
    fine_peaks = []
    fine_prominences = []
    for spectrum in fine_spectra:
        peaks, prominences = find_all_peaks(spectrum)
        if peaks is not None:
            fine_peaks.append(peaks)
            fine_prominences.append(prominences)
        else:
            fine_peaks.append(None)
            fine_prominences.append(None)
    
    # Find the middle of the fine scan
    mid_fine_idx = len(fine_peaks) // 2
    
    # Get all peaks from the middle of fine scan
    if fine_peaks[mid_fine_idx] is None:
        print(f"WARNING: No peaks found in middle of fine scan for {side} GMR")
        return None, None
    
    reference_peaks = fine_peaks[mid_fine_idx]
    reference_prominences = fine_prominences[mid_fine_idx]
    
    # Find best matching peaks in coarse data
    best_matches = []
    for ref_peak, ref_prom in zip(reference_peaks, reference_prominences):
        peak_differences = []
        for i, (coarse_peak, coarse_prom) in enumerate(zip(coarse_peaks, coarse_prominences)):
            if coarse_peak is None:
                peak_differences.append(float('inf'))
                continue
            # Find closest peak in coarse data
            diffs = np.abs(coarse_peak - ref_peak)
            min_diff_idx = np.argmin(diffs)
            peak_differences.append(diffs[min_diff_idx])
        
        closest_coarse_idx = np.argmin(peak_differences)
        if peak_differences[closest_coarse_idx] < 5:  # Only consider matches within 5nm
            best_matches.append((closest_coarse_idx, peak_differences[closest_coarse_idx]))
    
    if not best_matches:
        print(f"WARNING: No good peak matches found for {side} GMR")
        return None, None
    
    # Use the match with the smallest difference
    best_match_idx, best_diff = min(best_matches, key=lambda x: x[1])
    
    print(f"Best peak match difference: {best_diff:.2f} nm")
    print(f"Coarse angle at match: {coarse_angles[best_match_idx]:.2f} mrad")
    
    # Align fine dataset with coarse dataset
    fine_center = fine_angles[mid_fine_idx]
    angle_offset = coarse_angles[best_match_idx] - fine_center
    adjusted_fine_angles = fine_angles + angle_offset
    
    print(f"Adjusted fine scan range: {adjusted_fine_angles[0]:.2f} to {adjusted_fine_angles[-1]:.2f} mrad")
    
    # Ask for user verification
    print("\nPlease verify the alignment:")
    print(f"1. Original fine range: {fine_angles[0]:.2f} to {fine_angles[-1]:.2f} mrad")
    print(f"2. Adjusted fine range: {adjusted_fine_angles[0]:.2f} to {adjusted_fine_angles[-1]:.2f} mrad")
    print(f"3. Coarse range: {coarse_angles[0]:.2f} to {coarse_angles[-1]:.2f} mrad")
    
    response = input("Does this alignment look correct? (y/n): ")
    if response.lower() != 'y':
        print("Alignment rejected. Please adjust parameters or check data.")
        return None, None
    
    # Merge the datasets
    all_angles = np.sort(np.unique(np.concatenate([coarse_angles, adjusted_fine_angles])))
    merged_spectra = []
    
    for angle in all_angles:
        # Check if angle is in fine dataset range
        if angle >= adjusted_fine_angles[0] and angle <= adjusted_fine_angles[-1]:
            # Use fine data if available
            idx = np.argmin(np.abs(adjusted_fine_angles - angle))
            if np.abs(adjusted_fine_angles[idx] - angle) < 0.05:  # threshold for matching
                merged_spectra.append(fine_spectra[idx])
                continue
        
        # Use coarse data if fine data not available
        idx = np.argmin(np.abs(coarse_angles - angle))
        if np.abs(coarse_angles[idx] - angle) < 0.1:
            merged_spectra.append(coarse_spectra[idx])
    
    merged_spectra = np.array(merged_spectra)
    
    return merged_spectra, all_angles

if __name__ == '__main__':
    # Process both GMRs
    print("Processing Left GMR...")
    left_merged, left_angles = align_and_merge_gmr_data('left')
    print("Left GMR processing complete.")
    
    print("Processing Right GMR...")
    right_merged, right_angles = align_and_merge_gmr_data('right')
    print("Right GMR processing complete.")
    
    # Save merged data
    np.savez('left_gmr_merged.npz', data=left_merged, angles=left_angles)
    np.savez('right_gmr_merged.npz', data=right_merged, angles=right_angles)
    
    # Plot using pcolormesh with inferno colormap
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create coordinate meshes for proper scaling
    left_wavelengths = left_merged[0, :, 0]
    right_wavelengths = right_merged[0, :, 0]
    
    # Plot merged data for left GMR
    im1 = ax1.pcolormesh(left_wavelengths, 
                         left_angles,
                         left_merged[:, :, 1],
                         shading='auto',
                         vmin=0.0,
                         vmax=1.2,
                         cmap='inferno')
    ax1.set_title('Left GMR Merged Data')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Angle (mrad)')
    plt.colorbar(im1, ax=ax1, label='Transmission')
    
    # Plot merged data for right GMR
    im2 = ax2.pcolormesh(right_wavelengths,
                         right_angles,
                         right_merged[:, :, 1],
                         shading='auto',
                         vmin=0.0,
                         vmax=1.2,
                         cmap='inferno')
    ax2.set_title('Right GMR Merged Data')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Angle (mrad)')
    plt.colorbar(im2, ax=ax2, label='Transmission')
    
    plt.tight_layout()
    plt.show()
