import unittest
import numpy as np
from radar_simulator import RadarSimulator
from signal_processor import SignalProcessor
from feature_extractor import FeatureExtractor
from classifier import RadarClassifier

class TestRadarDSP(unittest.TestCase):
    def setUp(self):
        self.sim = RadarSimulator()
        self.proc = SignalProcessor(self.sim)
        self.extractor = FeatureExtractor()

    def test_range_estimation(self):
        """Verify that the Range FFT correctly calculates the range of a static target."""
        # Create a single target with a known range, no speed, no clutter, no noise
        target_range = 65.0 # meters
        params = {
            'R_0': target_range,
            'v': 0.0,
            'A_body': 5.0,
            'has_micro': False
        }
        
        # Simulate clean signal (no clutter, no noise)
        sig = self.sim.simulate_radar_return(params, snr_db=60.0, add_clutter=False)
        
        # Process Range-Doppler FFT
        rdm_power, _ = self.proc.range_doppler_processing(sig)
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(rdm_power), rdm_power.shape)
        est_range = self.proc.ranges[peak_idx[1]]
        
        print(f"\n[Test Range] Simulated: {target_range} m, Estimated: {est_range:.2f} m")
        # Estimate should be within 1 range resolution cell (0.6 meters)
        self.assertLessEqual(abs(est_range - target_range), self.proc.range_res + 0.1)

    def test_velocity_estimation(self):
        """Verify that the Doppler FFT correctly calculates target velocity."""
        # Create a target with a known speed and range
        target_range = 45.0
        target_vel = 12.5 # m/s
        params = {
            'R_0': target_range,
            'v': target_vel,
            'A_body': 5.0,
            'has_micro': False
        }
        
        sig = self.sim.simulate_radar_return(params, snr_db=60.0, add_clutter=False)
        rdm_power, _ = self.proc.range_doppler_processing(sig)
        
        peak_idx = np.unravel_index(np.argmax(rdm_power), rdm_power.shape)
        est_vel = self.proc.velocities[peak_idx[0]]
        
        print(f"[Test Velocity] Simulated: {target_vel} m/s, Estimated: {est_vel:.2f} m/s")
        # Estimate should be within 1 velocity resolution cell (~0.49 m/s)
        self.assertLessEqual(abs(est_vel - target_vel), self.proc.vel_res + 0.1)

    def test_clutter_removal(self):
        """Verify that the Moving Target Indicator filter suppresses zero-velocity (clutter) targets."""
        # Create signal with static clutter at 45.0m and a moving target at 80.0m
        clutter_range = 45.0
        params = {
            'R_0': 80.0,
            'v': 25.0,
            'A_body': 3.0,
            'has_micro': False
        }
        
        # Simulate returns WITH static clutter
        sig = self.sim.simulate_radar_return(params, snr_db=40.0, add_clutter=True)
        
        # Process directly (no clutter removal)
        rdm_power_raw, _ = self.proc.range_doppler_processing(sig)
        
        # Process with clutter removal
        sig_filtered = self.proc.remove_clutter(sig)
        rdm_power_filtered, _ = self.proc.range_doppler_processing(sig_filtered)
        
        # Check power at 0 velocity column for the clutter range
        # Find index for clutter range and 0 velocity
        c_r_idx = np.argmin(np.abs(self.proc.ranges - clutter_range))
        zero_v_idx = np.argmin(np.abs(self.proc.velocities - 0.0))
        
        power_before = rdm_power_raw[zero_v_idx, c_r_idx]
        power_after = rdm_power_filtered[zero_v_idx, c_r_idx]
        
        attenuation_ratio = power_before / (power_after + 1e-12)
        print(f"[Test Clutter] Power before filter: {power_before:.4e}, Power after filter: {power_after:.4e}")
        print(f"[Test Clutter] Attenuation: {10*np.log10(attenuation_ratio):.2f} dB")
        
        # Clutter should be attenuated by at least 30 dB
        self.assertGreater(10*np.log10(attenuation_ratio), 30.0)

    def test_feature_extractor(self):
        """Verify feature extraction coordinates are correct and features are populated."""
        target = {
            'range_idx': 50,
            'doppler_idx': 60,
            'range_m': 30.0,
            'velocity_mps': 5.0,
            'power_db': -10.0
        }
        # Fake RDM power
        rdm_power = np.ones((self.sim.N_c, self.sim.N_s)) * 1e-5
        rdm_power[60, 50] = 1.0 # Peak
        
        features = self.extractor.extract_target_features(rdm_power, target, self.proc, window_size=5)
        self.assertIn('doppler_spread', features)
        self.assertIn('spectral_entropy', features)
        self.assertIn('rcs_proxy_db', features)
        
        f_vector, f_names = self.extractor.get_feature_vector(features)
        self.assertEqual(len(f_vector), 8)
        self.assertEqual(len(f_names), 8)

if __name__ == '__main__':
    unittest.main()
