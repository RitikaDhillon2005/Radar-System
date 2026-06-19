import numpy as np

class RadarSimulator:
    def __init__(self):
        # Radar Configuration
        self.f_c = 24.0e9        # Carrier frequency: 24 GHz
        self.c = 3.0e8           # Speed of light: 3e8 m/s
        self.wavelength = self.c / self.f_c # Wavelength: ~12.5 mm
        
        self.B = 250.0e6         # Sweep Bandwidth: 250 MHz (Range resolution ~ 0.6 m)
        self.T_c = 0.05e-3       # Chirp Sweep time: 0.05 ms (Fast sweep to increase max unambiguous velocity)
        self.N_s = 256           # Number of samples per chirp
        self.N_c = 256           # Number of chirps in CPI (Coherent Processing Interval)
        
        # Derived values
        self.f_s = self.N_s / self.T_c  # Sampling Frequency: 1.024 MHz
        self.slope = self.B / self.T_c   # Frequency sweep slope: 5e11 Hz/s
        
        # Fast-time and slow-time grids
        self.t_f = np.linspace(0, self.T_c, self.N_s, endpoint=False) # Fast-time within one chirp
        self.t_s = np.arange(self.N_c) * self.T_c                     # Slow-time across chirps
        
        # 2D Grid of time
        # Shape: (N_c, N_s) where row is chirp index, column is sample index
        self.T_F, self.T_S = np.meshgrid(self.t_f, self.t_s)
        self.T_total = self.T_S + self.T_F

    def get_target_params(self, target_type):
        """Generates random physically plausible parameters for a target class."""
        if target_type == 'aircraft':
            params = {
                'R_0': np.random.uniform(70, 150),
                'v': np.random.uniform(30, 80),
                'A_body': np.random.uniform(8.0, 12.0),
                'has_micro': False
            }
        elif target_type == 'drone':
            params = {
                'R_0': np.random.uniform(30, 80),
                'v': np.random.uniform(4, 15),
                'A_body': np.random.uniform(1.2, 2.5),
                'has_micro': True,
                'micro_type': 'drone',
                'A_prop': np.random.uniform(0.1, 0.25),
                'L_prop': np.random.uniform(0.06, 0.10),
                'f_rot': np.random.uniform(80.0, 140.0),
                'num_props': 4
            }
        elif target_type == 'bird':
            params = {
                'R_0': np.random.uniform(15, 50),
                'v': np.random.uniform(1.5, 6.0),
                'A_body': np.random.uniform(0.15, 0.4),
                'has_micro': True,
                'micro_type': 'bird',
                'A_wing': np.random.uniform(0.03, 0.08),
                'L_wing': np.random.uniform(0.12, 0.25),
                'f_flap': np.random.uniform(2.5, 7.5),
                'num_wings': 2
            }
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        return params

    def simulate_radar_return(self, params, snr_db=15.0, add_clutter=True):
        """
        Simulates the raw complex baseband IF signal returned from the target.
        Shape of returned matrix: (N_c, N_s)
        """
        R_0 = params['R_0']
        v = params['v']
        A_body = params['A_body']
        
        # 1. Main Body Signal
        # Range as a function of slow-time: R(t_s) = R_0 + v * t_s
        R_t_body = R_0 + v * self.T_S
        
        # FMCW baseband phase: 2*pi * ( (2 * B * R_t / (c * T_c)) * t_f + (2 * f_c * R_t / c) )
        # Using wavelength lambda = c / f_c, the second term is 4 * pi * R_t / lambda
        phi_body = 2 * np.pi * ( (2 * self.B * R_t_body) / (self.c * self.T_c) ) * self.T_F + \
                   (4 * np.pi * R_t_body) / self.wavelength
                   
        signal = A_body * np.exp(1j * phi_body)
        
        # 2. Micro-Doppler Components
        if params['has_micro']:
            if params['micro_type'] == 'drone':
                # Drone propellers rotation
                # Propellers add phase modulation to the return signal
                A_prop = params['A_prop']
                L_prop = params['L_prop']
                f_rot = params['f_rot']
                num_props = params['num_props']
                
                for p in range(num_props):
                    # Each prop has 2 blades, opposing phases
                    # Random phase offset for each propeller
                    phi_offset = np.random.uniform(0, 2*np.pi)
                    for b in range(2):
                        blade_phase = phi_offset + b * np.pi
                        # Propeller tip displacement along line of sight
                        R_t_prop = R_t_body + L_prop * np.cos(2 * np.pi * f_rot * self.T_S + blade_phase)
                        
                        phi_prop = 2 * np.pi * ( (2 * self.B * R_t_prop) / (self.c * self.T_c) ) * self.T_F + \
                                   (4 * np.pi * R_t_prop) / self.wavelength
                        
                        signal += A_prop * np.exp(1j * phi_prop)
                        
            elif params['micro_type'] == 'bird':
                # Bird wing flapping
                A_wing = params['A_wing']
                L_wing = params['L_wing']
                f_flap = params['f_flap']
                num_wings = params['num_wings']
                
                for w in range(num_wings):
                    # Left and right wing flap out of phase or in phase?
                    # Generally flap in phase relative to body height, but symmetric displacement along LOS
                    wing_phase = w * np.pi
                    R_t_wing = R_t_body + L_wing * np.cos(2 * np.pi * f_flap * self.T_S + wing_phase)
                    
                    phi_wing = 2 * np.pi * ( (2 * self.B * R_t_wing) / (self.c * self.T_c) ) * self.T_F + \
                               (4 * np.pi * R_t_wing) / self.wavelength
                    
                    signal += A_wing * np.exp(1j * phi_wing)

        # 3. Ground / Static Clutter (optional)
        if add_clutter:
            # Clutter is target-like reflections at 0 speed (static) at various ranges
            clutter_ranges = [10.0, 45.0, 95.0]
            clutter_amp = [0.8, 0.4, 0.2]
            for R_c, A_c in zip(clutter_ranges, clutter_amp):
                # 0 speed, so R is constant
                phi_clutter = 2 * np.pi * ( (2 * self.B * R_c) / (self.c * self.T_c) ) * self.T_F + \
                              (4 * np.pi * R_c) / self.wavelength
                signal += A_c * np.exp(1j * phi_clutter)

        # 4. Noise addition (Complex Additive White Gaussian Noise)
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Generate complex AWGN
        noise = np.sqrt(noise_power / 2.0) * (
            np.random.normal(0, 1, self.T_F.shape) + 1j * np.random.normal(0, 1, self.T_F.shape)
        )
        
        noisy_signal = signal + noise
        return noisy_signal

    def generate_labeled_dataset(self, num_samples_per_class=100, min_snr=5.0, max_snr=25.0):
        """Generates a dataset of signals and target metadata for training."""
        dataset = []
        labels = []
        metadata = []
        
        classes = ['aircraft', 'drone', 'bird']
        
        for class_idx, target_type in enumerate(classes):
            for _ in range(num_samples_per_class):
                params = self.get_target_params(target_type)
                snr = np.random.uniform(min_snr, max_snr)
                
                # Simulate return
                sig = self.simulate_radar_return(params, snr_db=snr, add_clutter=True)
                
                dataset.append(sig)
                labels.append(class_idx)  # 0: aircraft, 1: drone, 2: bird
                
                # Save metadata for feature matching
                meta = params.copy()
                meta['snr_db'] = snr
                meta['class_name'] = target_type
                metadata.append(meta)
                
        return np.array(dataset), np.array(labels), metadata
