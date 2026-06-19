import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from radar_simulator import RadarSimulator
from signal_processor import SignalProcessor
from feature_extractor import FeatureExtractor
from classifier import RadarClassifier

# Custom styled colors for dark theme visual appeal
COLOR_AIRCRAFT = '#00D2FF'  # Ice Blue
COLOR_DRONE = '#BD00FF'     # Electric Purple
COLOR_BIRD = '#00FF66'      # Neon Green
COLOR_BG = '#121212'        # Soft Dark Background
COLOR_CARD = '#1E1E1E'      # Card background
COLOR_TEXT = '#E0E0E0'      # Off-white text

def run_simulation_and_plot(target_type, model_path='radar_classifier.pkl', output_plot='radar_dsp_output.png'):
    """Simulates a target, processes the signal, predicts target type, and plots the results."""
    print(f"\n--- Running Simulation for: {target_type.upper()} ---")
    
    # 1. Load ML Model
    classifier = RadarClassifier()
    if not os.path.exists(model_path):
        print(f"Classifier model {model_path} not found. Training a new model first...")
        X, y = classifier.generate_ml_dataset(samples_per_class=120)
        classifier.train_classifier(X, y)
        classifier.save_model(model_path)
    else:
        classifier.load_model(model_path)
        
    sim = classifier.simulator
    proc = classifier.processor
    
    # 2. Get random parameters for target and simulate raw return
    params = sim.get_target_params(target_type)
    # Ensure a reasonable SNR for visualization
    snr_db = np.random.uniform(12.0, 20.0)
    print(f"Simulated Parameters:")
    print(f" - True Range: {params['R_0']:.2f} m")
    print(f" - True Velocity: {params['v']:.2f} m/s")
    if params['has_micro']:
        if params['micro_type'] == 'drone':
            print(f" - Rotor modulation frequency: {params['f_rot']:.1f} Hz (Length: {params['L_prop']*100:.1f} cm)")
        elif params['micro_type'] == 'bird':
            print(f" - Wing flap frequency: {params['f_flap']:.1f} Hz (Length: {params['L_wing']*100:.1f} cm)")
            
    # Generate signal
    sig_matrix = sim.simulate_radar_return(params, snr_db=snr_db, add_clutter=True)
    
    # 3. Predict & Process
    predictions, rdm_power, rdm_db, threshold = classifier.predict(sig_matrix)
    
    # Find prediction for our simulated target
    # If multiple detections, find the one closest to true range
    detected_target = None
    if len(predictions) > 0:
        true_r = params['R_0']
        distances = [abs(p['range_m'] - true_r) for p in predictions]
        best_idx = np.argmin(distances)
        if distances[best_idx] < 6.0:
            detected_target = predictions[best_idx]
            
    if detected_target is None:
        print("WARNING: CFAR missed target or detected other clutter instead. Visualizing raw features.")
        # Fallback to direct range/doppler slice for visualization
        r_idx = int(np.clip(round(params['R_0'] / proc.range_res), 0, sim.N_s - 1))
        true_doppler_freq = 2 * params['v'] / sim.wavelength
        d_bin = true_doppler_freq * (sim.N_c * sim.T_c)
        d_idx = int(np.clip(round(d_bin + sim.N_c / 2), 0, sim.N_c - 1))
        
        fallback_target = {
            'range_idx': r_idx,
            'doppler_idx': d_idx,
            'range_m': params['R_0'],
            'velocity_mps': params['v'],
            'power_db': float(10 * np.log10(rdm_power[d_idx, r_idx] + 1e-12))
        }
        features = classifier.extractor.extract_target_features(rdm_power, fallback_target, proc)
        # Mock prediction probabilities for fallback
        f_vector, _ = classifier.extractor.get_feature_vector(features)
        f_scaled = classifier.scaler.transform(f_vector.reshape(1, -1))
        class_idx = classifier.model.predict(f_scaled)[0]
        probs = classifier.model.predict_proba(f_scaled)[0]
        
        detected_target = fallback_target
        detected_target['predicted_class'] = classifier.class_names[class_idx]
        detected_target['confidence'] = float(probs[class_idx])
        detected_target['probabilities'] = {classifier.class_names[idx]: float(probs[idx]) for idx in range(3)}
        detected_target['features'] = features
        
    print(f"\nDetection Results:")
    print(f" - Estimated Range: {detected_target['range_m']:.2f} m (Error: {abs(detected_target['range_m'] - params['R_0']):.2f} m)")
    print(f" - Estimated Speed: {detected_target['velocity_mps']:.2f} m/s (Error: {abs(detected_target['velocity_mps'] - params['v']):.2f} m/s)")
    print(f" - Predicted Class: {detected_target['predicted_class']} ({detected_target['confidence']*100:.1f}% confidence)")
    
    # 4. PLOTTING - Premium Dark Theme Visualization Dashboard
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), facecolor=COLOR_BG)
    fig.suptitle(f"Radar Signal Processing & Target Classification Dashboard\nTarget: {target_type.upper()} | Predicted: {detected_target['predicted_class'].upper()} ({detected_target['confidence']*100:.1f}%)", 
                 fontsize=18, fontweight='bold', color='#FFFFFF', y=0.97)
    
    # --- Subplot 1: Raw Time-Domain Beat Signal (Fast Time) ---
    ax = axs[0, 0]
    ax.set_facecolor(COLOR_CARD)
    # Plot first chirp
    chirp_to_plot = sig_matrix[0, :]
    ax.plot(sim.t_f * 1e6, np.real(chirp_to_plot), color='#FF3366', alpha=0.85, label='In-Phase (Real)')
    ax.plot(sim.t_f * 1e6, np.imag(chirp_to_plot), color='#00CCFF', alpha=0.6, label='Quadrature (Imag)')
    ax.set_title("Raw Baseband IF Signal (First Chirp)", fontsize=13, fontweight='semibold', pad=10)
    ax.set_xlabel("Fast Time (microseconds)", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.grid(True, color='#333333', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.5)
    ax.set_xlim(0, sim.T_c * 1e6 * 0.4) # Zoom in to show oscillations
    
    # --- Subplot 2: Range-Doppler Map (RDM Heatmap) ---
    ax = axs[0, 1]
    ax.set_facecolor(COLOR_CARD)
    # Create Range-Doppler Mesh for plotting
    v_mesh, r_mesh = np.meshgrid(proc.velocities, proc.ranges)
    # Transpose RDM to align (Range on Y, Velocity on X)
    rdm_to_plot = rdm_db.T
    
    im = ax.pcolormesh(v_mesh, r_mesh, rdm_to_plot, cmap='magma', shading='auto', vmin=-70, vmax=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label('Relative Power (dB)', fontsize=11)
    
    # Highlight detected target
    ax.axhline(detected_target['range_m'], color='#FFFFFF', linestyle=':', alpha=0.5)
    ax.axvline(detected_target['velocity_mps'], color='#FFFFFF', linestyle=':', alpha=0.5)
    
    # Circle target
    circle_color = COLOR_AIRCRAFT if target_type == 'aircraft' else (COLOR_DRONE if target_type == 'drone' else COLOR_BIRD)
    circle = patches.Circle((detected_target['velocity_mps'], detected_target['range_m']), 
                           radius=4, edgecolor=circle_color, facecolor='none', linewidth=2.5, linestyle='-')
    ax.add_patch(circle)
    
    # Label detected target
    ax.text(detected_target['velocity_mps'] + 4.5, detected_target['range_m'] + 2.5, 
            f"{detected_target['predicted_class']}\nR={detected_target['range_m']:.1f}m\nv={detected_target['velocity_mps']:.1f}m/s", 
            color=circle_color, fontweight='bold', fontsize=10,
            bbox=dict(facecolor=COLOR_BG, alpha=0.8, edgecolor=circle_color, boxstyle='round,pad=0.3'))
    
    # Plot static clutter areas (ground returns at 0 speed)
    ax.set_title("Range-Doppler Power Map (2D FFT)", fontsize=13, fontweight='semibold', pad=10)
    ax.set_xlabel("Velocity (m/s)", fontsize=11)
    ax.set_ylabel("Range (meters)", fontsize=11)
    ax.grid(False)
    ax.set_ylim(0, 150)
    ax.set_xlim(proc.velocities[0], proc.velocities[-1])
    
    # --- Subplot 3: Micro-Doppler Slice (Doppler Profile) ---
    ax = axs[1, 0]
    ax.set_facecolor(COLOR_CARD)
    # Get Doppler profile at target range index
    r_idx = detected_target['range_idx']
    profile_db = 10 * np.log10(rdm_power[:, r_idx] / np.max(rdm_power) + 1e-12)
    
    ax.plot(proc.velocities, profile_db, color='#FFFFFF', linewidth=1.5, label='Doppler Spectrum')
    
    # Highlight local window around target
    window_size = 15
    d_idx = detected_target['doppler_idx']
    indices = np.arange(d_idx - window_size, d_idx + window_size + 1)
    wrapped_indices = np.mod(indices, len(profile_db))
    ax.plot(proc.velocities[wrapped_indices], profile_db[wrapped_indices], 
            color=circle_color, linewidth=2.5, label='Target Signal Window')
    
    # Draw true/est peak markers
    ax.axvline(detected_target['velocity_mps'], color=circle_color, linestyle='--', alpha=0.7, 
               label=f"Est Speed: {detected_target['velocity_mps']:.2f} m/s")
    
    ax.set_title(f"Doppler Spectrum Slice (at Range = {detected_target['range_m']:.1f} m)", fontsize=13, fontweight='semibold', pad=10)
    ax.set_xlabel("Velocity (m/s)", fontsize=11)
    ax.set_ylabel("Power (dB)", fontsize=11)
    ax.set_ylim(-65, 5)
    ax.grid(True, color='#333333', linestyle='--', alpha=0.5)
    ax.legend(loc='lower center', framealpha=0.5, ncol=1)
    
    # --- Subplot 4: Machine Learning Classification Probabilities ---
    ax = axs[1, 1]
    ax.set_facecolor(COLOR_CARD)
    
    prob_dict = detected_target['probabilities']
    y_pos = np.arange(len(classifier.class_names))
    bar_colors = [COLOR_AIRCRAFT, COLOR_DRONE, COLOR_BIRD]
    
    bars = ax.barh(y_pos, [prob_dict[c]*100 for c in classifier.class_names], 
            color=bar_colors, edgecolor='#FFFFFF', height=0.5, alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classifier.class_names, fontsize=12, fontweight='bold')
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_title("ML Model Prediction Confidence", fontsize=13, fontweight='semibold', pad=10)
    ax.set_xlim(0, 105)
    ax.grid(True, color='#333333', linestyle='--', alpha=0.3)
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                va='center', ha='left', fontsize=11, fontweight='bold', color='#FFFFFF')
        
    # Text box displaying physical features extracted
    features = detected_target['features']
    feat_text = (
        f"EXTRACTED FEATURES:\n"
        f" • Estimated Range: {features['range_m']:.2f} m\n"
        f" • Estimated Speed: {features['speed_mps']:.2f} m/s\n"
        f" • Peak Intensity (RCS Proxy): {features['rcs_proxy_db']:.2f} dB\n"
        f" • Doppler Spread (σ_v): {features['doppler_spread']:.3f} m/s\n"
        f" • Spectral Entropy (H): {features['spectral_entropy']:.3f} bits\n"
        f" • Peak-to-Average Ratio: {features['peak_to_average']:.2f}\n"
        f" • Spectral Flatness: {features['spectral_flatness']:.4f}\n"
        f" • Spectral Kurtosis: {features['spectral_kurtosis']:.2f}"
    )
    ax.text(0.1, 0.45, feat_text, transform=ax.transAxes, fontsize=10, 
            fontfamily='monospace', color=COLOR_TEXT,
            bbox=dict(facecolor=COLOR_BG, alpha=0.9, edgecolor='#555555', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=120, facecolor=COLOR_BG)
    print(f"Visualization saved to: {output_plot}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Radar Signal Processing & Target Classification")
    parser.add_argument('--train', action='store_true', help="Train machine learning classifier dataset")
    parser.add_argument('--simulate', type=str, choices=['aircraft', 'drone', 'bird'], 
                        help="Simulate a radar return and run target detection & classification")
    parser.add_argument('--samples', type=int, default=120, help="Number of training samples per class")
    parser.add_argument('--model', type=str, default='radar_classifier.pkl', help="Filename to load/save model")
    parser.add_argument('--output', type=str, default='radar_dsp_output.png', help="Output plot filename")
    
    args = parser.parse_args()
    
    # Default behavior: train and then simulate a drone if no arguments are provided
    if not args.train and not args.simulate:
        print("No action specified. Defaulting to: Training classifier and simulating a Drone.")
        args.train = True
        args.simulate = 'drone'
        
    classifier = RadarClassifier()
    
    if args.train:
        # Generate and train
        X, y = classifier.generate_ml_dataset(samples_per_class=args.samples)
        classifier.train_classifier(X, y)
        classifier.save_model(args.model)
        
    if args.simulate:
        run_simulation_and_plot(args.simulate, model_path=args.model, output_plot=args.output)
