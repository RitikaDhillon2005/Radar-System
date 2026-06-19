import numpy as np
from scipy import signal
from scipy.ndimage import convolve, maximum_filter

class SignalProcessor:
    def __init__(self, radar_sim):
        self.radar_sim = radar_sim
        self.wavelength = radar_sim.wavelength
        self.B = radar_sim.B
        self.T_c = radar_sim.T_c
        self.c = radar_sim.c
        self.f_s = radar_sim.f_s
        self.N_s = radar_sim.N_s
        self.N_c = radar_sim.N_c
        
        # Grid resolutions
        self.range_res = self.c / (2 * self.B)
        self.vel_res = self.wavelength / (2 * self.N_c * self.T_c)
        
        # Ranges and velocities corresponding to FFT bins
        # Positive frequencies map to ranges [0, R_max]
        # FFT of size N_s: range bins
        self.ranges = np.arange(self.N_s) * self.range_res
        # Velocity bins (after fftshift): velocity goes from -V_max to +V_max
        self.velocities = (np.arange(self.N_c) - self.N_c // 2) * self.vel_res

    def remove_clutter(self, sig_matrix):
        """
        Moving Target Indicator (MTI) / Clutter filter.
        Subtracts the mean across the slow-time dimension (chirps) to remove static clutter.
        """
        # Static clutter has 0 Hz Doppler frequency (phase does not change chirp-to-chirp)
        clutter_free = sig_matrix - np.mean(sig_matrix, axis=0, keepdims=True)
        return clutter_free

    def apply_noise_filter(self, sig_matrix, low_cutoff_hz=1000, high_cutoff_hz=500000):
        """
        Applies a Butterworth bandpass filter to each chirp (fast-time) to remove out-of-band noise.
        """
        nyquist = self.f_s / 2
        low = low_cutoff_hz / nyquist
        high = high_cutoff_hz / nyquist
        
        # Design bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Filter each chirp (along columns / axis 1)
        filtered = np.zeros_like(sig_matrix)
        for i in range(sig_matrix.shape[0]):
            # Use filtfilt for zero-phase distortion
            filtered[i, :] = signal.filtfilt(b, a, sig_matrix[i, :])
        return filtered

    def range_doppler_processing(self, sig_matrix):
        """
        Performs 2D Range-Doppler processing (2D FFT) with windowing.
        Returns:
            rdm_power: 2D array of Range-Doppler map power (linear scale)
            rdm_db: 2D array of Range-Doppler map power (dB scale)
        """
        # Apply 2D Hann window to reduce sidelobes
        window_fast = np.hanning(self.N_s)
        window_slow = np.hanning(self.N_c)
        window_2d = np.outer(window_slow, window_fast)
        
        windowed_sig = sig_matrix * window_2d
        
        # 2D FFT
        # 1. FFT along fast-time (range)
        range_fft = np.fft.fft(windowed_sig, axis=1)
        
        # 2. FFT along slow-time (Doppler) and shift center to 0
        rdm = np.fft.fft(range_fft, axis=0)
        rdm = np.fft.fftshift(rdm, axes=0)
        
        # Compute power
        rdm_power = np.abs(rdm)**2
        
        # Convert to dB (normalized to peak)
        max_power = np.max(rdm_power)
        if max_power == 0:
            max_power = 1e-12
        rdm_db = 10 * np.log10(rdm_power / max_power + 1e-12)
        
        return rdm_power, rdm_db

    def cfar_2d(self, rdm_power, num_guard=(2, 2), num_train=(4, 4), pfa=1e-4, offset_db=12.0):
        """
        Vectorized 2D Cell Averaging CFAR (CA-CFAR) using scipy.ndimage.convolve.
        num_guard: (guard_doppler, guard_range)
        num_train: (train_doppler, train_range)
        """
        gd, gr = num_guard
        td, tr = num_train
        
        # Create kernel
        kernel_doppler_size = 2 * (td + gd) + 1
        kernel_range_size = 2 * (tr + gr) + 1
        
        kernel = np.ones((kernel_doppler_size, kernel_range_size))
        
        # Zero out the guard cells and the Cell Under Test (CUT) in the center
        kernel[td : td + 2 * gd + 1, tr : tr + 2 * gr + 1] = 0
        
        # Normalize the kernel by dividing by the number of training cells
        num_training_cells = np.sum(kernel)
        kernel = kernel / num_training_cells
        
        # Convolve with the RDM power map to estimate local noise power
        # We perform convolution in linear scale
        noise_floor = convolve(rdm_power, kernel, mode='constant', cval=np.mean(rdm_power))
        
        # Compute CFAR threshold
        # Method 1: DB offset (more robust for simulated scenarios)
        # threshold_linear = noise_floor * (10**(offset_db / 10.0))
        
        # Method 2: Analytical alpha based on PFA
        alpha = num_training_cells * (pfa**(-1.0 / num_training_cells) - 1.0)
        threshold_linear = noise_floor * alpha
        
        # Detect cells exceeding the threshold
        detections = rdm_power > threshold_linear
        
        return detections, threshold_linear

    def detect_peaks(self, rdm_power, detections, neighborhood_size=5):
        """
        Isolates individual target peaks from the CFAR detection mask using
        Non-Maximum Suppression (local maximum filtering).
        Returns a list of dictionaries with detected target coordinates and power.
        """
        # Apply local maximum filter
        local_max = maximum_filter(rdm_power, size=neighborhood_size) == rdm_power
        
        # Peaks are local maxima that were also detected by CFAR
        peaks_mask = local_max & detections
        
        # Get coordinates of detections
        doppler_indices, range_indices = np.where(peaks_mask)
        
        detected_targets = []
        for d_idx, r_idx in zip(doppler_indices, range_indices):
            # Map indices to physical values
            target_range = self.ranges[r_idx]
            target_velocity = self.velocities[d_idx]
            peak_power = rdm_power[d_idx, r_idx]
            
            # Skip ranges near zero (could be clutter leakage) or negative ranges
            if target_range < 2.0:
                continue
                
            detected_targets.append({
                'range_idx': int(r_idx),
                'doppler_idx': int(d_idx),
                'range_m': float(target_range),
                'velocity_mps': float(target_velocity),
                'power_linear': float(peak_power),
                'power_db': float(10 * np.log10(peak_power + 1e-12))
            })
            
        return detected_targets
