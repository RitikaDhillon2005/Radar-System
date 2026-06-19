import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_target_features(self, rdm_power, target, processor, window_size=15):
        """
        Extracts spectral features from the Range-Doppler Map (RDM) slice around a detected target.
        
        Parameters:
            rdm_power: 2D array of RDM power (linear scale)
            target: dictionary containing target details (range_idx, doppler_idx, etc.)
            processor: SignalProcessor instance (to get velocity scale)
            window_size: Number of bins to take on each side of the Doppler peak for localized feature calculation.
        """
        r_idx = target['range_idx']
        d_idx = target['doppler_idx']
        
        # 1. Slice Doppler profile at target range index
        doppler_profile = rdm_power[:, r_idx]
        
        # 2. Extract localized profile around the detected Doppler peak
        # This isolates the target's micro-Doppler from other targets or noise
        N_c = len(doppler_profile)
        indices = np.arange(d_idx - window_size, d_idx + window_size + 1)
        # Wrap indices using modulo to handle boundary wrapping in FFT
        wrapped_indices = np.mod(indices, N_c)
        
        local_profile = doppler_profile[wrapped_indices]
        local_velocities = processor.velocities[wrapped_indices]
        
        # Ensure profile values are positive and non-zero
        local_profile = np.maximum(local_profile, 1e-12)
        
        # 3. Calculate features
        
        # RCS proxy: Log of peak power (representing target signal strength)
        rcs_proxy = target['power_db']
        
        # Velocity: Absolute estimated speed
        speed = np.abs(target['velocity_mps'])
        
        # Range: Target range (helps model attenuation or check dependency)
        target_range = target['range_m']
        
        # Normalize local profile to act as a probability distribution for statistical features
        profile_sum = np.sum(local_profile)
        p = local_profile / profile_sum
        
        # Spectral Mean (weighted center velocity)
        spectral_mean_vel = np.sum(p * local_velocities)
        
        # Doppler Spread: Power-weighted standard deviation of velocity
        doppler_spread = np.sqrt(np.sum(p * (local_velocities - spectral_mean_vel)**2))
        
        # Spectral Entropy: Measure of modulation complexity (higher for Drones/Birds, lower for Aircraft)
        spectral_entropy = -np.sum(p * np.log2(p + 1e-12))
        
        # Peak-to-Average Ratio in local Doppler profile
        peak_to_average = np.max(local_profile) / np.mean(local_profile)
        
        # Spectral Flatness: Geometric Mean / Arithmetic Mean
        # Using log-sum-exp trick to prevent underflow in geometric mean calculation
        log_geom_mean = np.mean(np.log(local_profile))
        geom_mean = np.exp(log_geom_mean)
        arith_mean = np.mean(local_profile)
        spectral_flatness = geom_mean / (arith_mean + 1e-12)
        
        # Spectral Kurtosis: "Peakedness" of the Doppler profile
        std_vel = np.std(local_velocities) if np.std(local_velocities) > 0 else 1.0
        spectral_kurtosis = np.sum(p * (local_velocities - spectral_mean_vel)**4) / (doppler_spread**4 + 1e-12)
        
        features = {
            'rcs_proxy_db': rcs_proxy,
            'speed_mps': speed,
            'range_m': target_range,
            'doppler_spread': doppler_spread,
            'spectral_entropy': spectral_entropy,
            'peak_to_average': peak_to_average,
            'spectral_flatness': spectral_flatness,
            'spectral_kurtosis': spectral_kurtosis
        }
        
        return features

    def get_feature_vector(self, features):
        """Converts feature dict to a sorted numpy array for machine learning models."""
        keys = ['rcs_proxy_db', 'speed_mps', 'range_m', 'doppler_spread', 
                'spectral_entropy', 'peak_to_average', 'spectral_flatness', 'spectral_kurtosis']
        return np.array([features[k] for k in keys]), keys
