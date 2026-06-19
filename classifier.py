import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from radar_simulator import RadarSimulator
from signal_processor import SignalProcessor
from feature_extractor import FeatureExtractor

class RadarClassifier:
    def __init__(self):
        self.simulator = RadarSimulator()
        self.processor = SignalProcessor(self.simulator)
        self.extractor = FeatureExtractor()
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.class_names = ['Aircraft', 'Drone', 'Bird']

    def generate_ml_dataset(self, samples_per_class=120, min_snr=5.0, max_snr=25.0):
        """
        Simulates signals, processes them, and extracts features to construct
        the machine learning dataset.
        """
        print(f"Generating dataset with {samples_per_class} samples per class...")
        X = []
        y = []
        
        classes = ['aircraft', 'drone', 'bird']
        
        for class_idx, target_type in enumerate(classes):
            print(f" Simulating class: {target_type}...")
            for i in range(samples_per_class):
                # Get random parameters and simulate return
                params = self.simulator.get_target_params(target_type)
                snr = np.random.uniform(min_snr, max_snr)
                
                # We simulate with clutter
                sig_matrix = self.simulator.simulate_radar_return(params, snr_db=snr, add_clutter=True)
                
                # Process signal
                clutter_removed = self.processor.remove_clutter(sig_matrix)
                filtered = self.processor.apply_noise_filter(clutter_removed)
                rdm_power, _ = self.processor.range_doppler_processing(filtered)
                
                # Detect targets using CFAR
                detections, _ = self.processor.cfar_2d(rdm_power, pfa=1e-4)
                targets = self.processor.detect_peaks(rdm_power, detections)
                
                selected_target = None
                
                if len(targets) > 0:
                    # Find detected target closest to true range
                    true_r = params['R_0']
                    distances = [abs(t['range_m'] - true_r) for t in targets]
                    best_idx = np.argmin(distances)
                    if distances[best_idx] < 5.0: # Close enough to be the true target
                        selected_target = targets[best_idx]
                
                # Fallback: if CFAR missed, locate target range/doppler directly via parameters
                if selected_target is None:
                    # Determine true range and Doppler bin indices
                    r_idx = int(np.clip(round(params['R_0'] / self.processor.range_res), 0, self.simulator.N_s - 1))
                    true_doppler_freq = 2 * params['v'] / self.simulator.wavelength
                    # Doppler frequency bin
                    d_bin = true_doppler_freq * (self.simulator.N_c * self.simulator.T_c)
                    d_idx = int(np.clip(round(d_bin + self.simulator.N_c / 2), 0, self.simulator.N_c - 1))
                    
                    selected_target = {
                        'range_idx': r_idx,
                        'doppler_idx': d_idx,
                        'range_m': params['R_0'],
                        'velocity_mps': params['v'],
                        'power_db': float(10 * np.log10(rdm_power[d_idx, r_idx] + 1e-12))
                    }
                
                # Extract features
                features = self.extractor.extract_target_features(rdm_power, selected_target, self.processor)
                f_vector, f_names = self.extractor.get_feature_vector(features)
                
                X.append(f_vector)
                y.append(class_idx)
                
                if self.feature_names is None:
                    self.feature_names = f_names
                    
        return np.array(X), np.array(y)

    def train_classifier(self, X, y, test_size=0.2, random_state=42):
        """Trains the Random Forest model and prints classification evaluation."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("Training Random Forest Classifier...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=8)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTraining Complete. Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Log feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\nFeature Importances:")
        for idx in indices:
            print(f" - {self.feature_names[idx]}: {importances[idx]:.4f}")
            
        return accuracy

    def save_model(self, filepath):
        """Saves model, scaler, and features to a file."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model is not trained yet!")
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model successfully saved to {filepath}")

    def load_model(self, filepath):
        """Loads a model, scaler, and features from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.class_names = data['class_names']
        print(f"Model loaded successfully from {filepath}")

    def predict(self, sig_matrix):
        """
        Processes a raw signal matrix, detects targets, and classifies them.
        Returns a list of detected targets with their predicted class and probabilities.
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model has not been loaded or trained!")
            
        # Process signal
        clutter_removed = self.processor.remove_clutter(sig_matrix)
        filtered = self.processor.apply_noise_filter(clutter_removed)
        rdm_power, rdm_db = self.processor.range_doppler_processing(filtered)
        
        # Target detection
        detections, threshold = self.processor.cfar_2d(rdm_power, pfa=1e-4)
        targets = self.processor.detect_peaks(rdm_power, detections)
        
        predictions = []
        for target in targets:
            # Extract features
            features = self.extractor.extract_target_features(rdm_power, target, self.processor)
            f_vector, _ = self.extractor.get_feature_vector(features)
            
            # Predict
            f_scaled = self.scaler.transform(f_vector.reshape(1, -1))
            class_idx = self.model.predict(f_scaled)[0]
            probs = self.model.predict_proba(f_scaled)[0]
            
            pred_class = self.class_names[class_idx]
            pred_conf = probs[class_idx]
            
            pred_info = target.copy()
            pred_info['predicted_class'] = pred_class
            pred_info['confidence'] = float(pred_conf)
            pred_info['probabilities'] = {self.class_names[idx]: float(probs[idx]) for idx in range(3)}
            pred_info['features'] = features
            predictions.append(pred_info)
            
        return predictions, rdm_power, rdm_db, threshold
