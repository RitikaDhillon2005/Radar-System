import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from radar_simulator import RadarSimulator
from signal_processor import SignalProcessor
from feature_extractor import FeatureExtractor
from classifier import RadarClassifier

# Set page configuration with a wide layout and premium design feel
st.set_page_config(
    page_title="Cognitive Radar DSP & ML Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set custom styling using CSS injections
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1A1C24;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: bold;
        color: #8D96A7;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        color: #00D2FF;
        border-bottom: 2px solid #00D2FF;
    }
    div[data-testid="metric-container"] {
        background-color: #1E212A;
        border: 1px solid #2E3244;
        padding: 15px 15px 15px 15px;
        border-radius: 10px;
        color: #E0E0E0;
    }
    .classification-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Custom colors matching main.py
COLOR_AIRCRAFT = '#00D2FF'  # Ice Blue
COLOR_DRONE = '#BD00FF'     # Electric Purple
COLOR_BIRD = '#00FF66'      # Neon Green
COLOR_BG = '#0E1117'        # Soft Dark Background
COLOR_CARD = '#161A25'      # Card background
COLOR_TEXT = '#E0E0E0'      # Off-white text

# Initialize components and cache the classifier
@st.cache_resource
def get_classifier():
    classifier = RadarClassifier()
    model_path = 'radar_classifier.pkl'
    if os.path.exists(model_path):
        try:
            classifier.load_model(model_path)
        except Exception as e:
            st.warning(f"Error loading saved model, training a new one: {e}")
            X, y = classifier.generate_ml_dataset(samples_per_class=100)
            classifier.train_classifier(X, y)
            classifier.save_model(model_path)
    else:
        X, y = classifier.generate_ml_dataset(samples_per_class=100)
        classifier.train_classifier(X, y)
        classifier.save_model(model_path)
    return classifier

# Load classifier
try:
    classifier = get_classifier()
except Exception as e:
    st.error(f"Failed to load or train classifier: {e}. Running simulation in fallback mode.")
    classifier = None

# Sidebar controls
st.sidebar.title("📡 Radar Simulation Controls")
st.sidebar.markdown("Configure target parameters and noise levels to see how it affects radar returns and ML predictions.")

target_type = st.sidebar.selectbox(
    "1. Select Target Type",
    options=['Aircraft', 'Drone', 'Bird'],
    index=1
)

snr_db = st.sidebar.slider(
    "2. Signal-to-Noise Ratio (SNR)",
    min_value=-10.0,
    max_value=30.0,
    value=15.0,
    step=1.0,
    help="Higher values represent a cleaner signal with less noise clutter."
)

add_clutter = st.sidebar.checkbox(
    "Add Static Clutter",
    value=True,
    help="Simulates reflections from static objects (ground, trees, buildings) at 0 m/s velocity."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ ML Model Management")
if st.sidebar.button("Retrain Classifier Model"):
    with st.spinner("Generating synthetic training dataset and training Random Forest model... This might take up to a minute."):
        try:
            new_classifier = RadarClassifier()
            X, y = new_classifier.generate_ml_dataset(samples_per_class=150)
            acc = new_classifier.train_classifier(X, y)
            new_classifier.save_model('radar_classifier.pkl')
            st.sidebar.success(f"Model trained successfully! Accuracy: {acc*100:.2f}%")
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error training: {e}")

# Application Main Title
st.title("📡 Cognitive Radar Signal Processing & Target Classification")
st.markdown("This dashboard demonstrates real-time FMCW radar signal processing and machine learning classification to identify Aircraft, Drones, and Birds based on physical returns and micro-Doppler signatures.")

# Simulator objects
if classifier:
    sim = classifier.simulator
    proc = classifier.processor
else:
    sim = RadarSimulator()
    proc = SignalProcessor(sim)

# Generate parameters based on user selection
@st.cache_data
def get_cached_params(target_name):
    # Retrieve base parameters to let users tweak them
    return sim.get_target_params(target_name.lower())

base_params = get_cached_params(target_type).copy()

# Add sliders in main layout to tweak parameters
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    user_range = st.slider("Target Initial Range (m)", min_value=10.0, max_value=150.0, value=float(base_params['R_0']), step=1.0)
with col_p2:
    user_vel = st.slider("Target Velocity (m/s)", min_value=0.0, max_value=60.0, value=float(base_params['v']), step=0.5)
with col_p3:
    user_rcs = st.slider("Target Amplitude (RCS Proxy)", min_value=0.1, max_value=15.0, value=float(base_params['A_body']), step=0.1)

# Apply user overrides
base_params['R_0'] = user_range
base_params['v'] = user_vel
base_params['A_body'] = user_rcs

# Run simulation
sig_matrix = sim.simulate_radar_return(base_params, snr_db=snr_db, add_clutter=add_clutter)

# Perform Signal Processing
clutter_removed = proc.remove_clutter(sig_matrix)
filtered = proc.apply_noise_filter(clutter_removed)
rdm_power, rdm_db = proc.range_doppler_processing(filtered)

# Target Detection
detections, threshold = proc.cfar_2d(rdm_power, pfa=1e-4)
detected_targets = proc.detect_peaks(rdm_power, detections)

# Classification Section
st.markdown("### 🤖 Real-Time Classifier Outputs")

col_res, col_metrics = st.columns([1, 2])

# Perform ML prediction
predicted_target = None
predictions = []

if classifier:
    for target in detected_targets:
        features = classifier.extractor.extract_target_features(rdm_power, target, proc)
        f_vector, _ = classifier.extractor.get_feature_vector(features)
        f_scaled = classifier.scaler.transform(f_vector.reshape(1, -1))
        class_idx = classifier.model.predict(f_scaled)[0]
        probs = classifier.model.predict_proba(f_scaled)[0]
        
        pred_info = target.copy()
        pred_info['predicted_class'] = classifier.class_names[class_idx]
        pred_info['confidence'] = float(probs[class_idx])
        pred_info['probabilities'] = {classifier.class_names[idx]: float(probs[idx]) for idx in range(3)}
        pred_info['features'] = features
        predictions.append(pred_info)

# Find target closest to simulated range
if len(predictions) > 0:
    distances = [abs(p['range_m'] - base_params['R_0']) for p in predictions]
    best_idx = np.argmin(distances)
    if distances[best_idx] < 10.0:
        predicted_target = predictions[best_idx]

# If target missed, run fallback extraction
if predicted_target is None and classifier:
    r_idx = int(np.clip(round(base_params['R_0'] / proc.range_res), 0, sim.N_s - 1))
    true_doppler_freq = 2 * base_params['v'] / sim.wavelength
    d_bin = true_doppler_freq * (sim.N_c * sim.T_c)
    d_idx = int(np.clip(round(d_bin + sim.N_c / 2), 0, sim.N_c - 1))
    
    fallback_target = {
        'range_idx': r_idx,
        'doppler_idx': d_idx,
        'range_m': base_params['R_0'],
        'velocity_mps': base_params['v'],
        'power_db': float(10 * np.log10(rdm_power[d_idx, r_idx] + 1e-12))
    }
    features = classifier.extractor.extract_target_features(rdm_power, fallback_target, proc)
    f_vector, _ = classifier.extractor.get_feature_vector(features)
    f_scaled = classifier.scaler.transform(f_vector.reshape(1, -1))
    class_idx = classifier.model.predict(f_scaled)[0]
    probs = classifier.model.predict_proba(f_scaled)[0]
    
    predicted_target = fallback_target
    predicted_target['predicted_class'] = classifier.class_names[class_idx]
    predicted_target['confidence'] = float(probs[class_idx])
    predicted_target['probabilities'] = {classifier.class_names[idx]: float(probs[idx]) for idx in range(3)}
    predicted_target['features'] = features
    predicted_target['missed_by_cfar'] = True

with col_res:
    if predicted_target:
        pred_class = predicted_target['predicted_class']
        conf = predicted_target['confidence']
        
        # Decide border and text color based on prediction
        c_color = COLOR_AIRCRAFT if pred_class == 'Aircraft' else (COLOR_DRONE if pred_class == 'Drone' else COLOR_BIRD)
        
        st.markdown(f"""
        <div class="classification-box" style="border-color: {c_color}; background-color: #161A25;">
            <p style="font-size: 14px; text-transform: uppercase; color: #8D96A7; margin: 0;">Predicted Classification</p>
            <h1 style="color: {c_color}; font-size: 44px; margin: 5px 0 5px 0; font-weight: 800;">{pred_class.upper()}</h1>
            <h3 style="color: #FFFFFF; margin: 0; font-weight: 600;">{conf*100:.1f}% Confidence</h3>
            {"<p style='color: #FFA500; font-size: 11px; margin-top: 5px;'>⚠️ Target extracted using fallback (CFAR threshold missed)</p>" if 'missed_by_cfar' in predicted_target else ""}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No targets detected or predicted yet. Increase SNR or adjust target values.")

with col_metrics:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            "Estimated Distance", 
            value=f"{predicted_target['range_m']:.2f} m" if predicted_target else "N/A", 
            delta=f"Error: {abs(predicted_target['range_m'] - base_params['R_0']):.2f} m" if predicted_target else None,
            delta_color="inverse"
        )
    with col_m2:
        st.metric(
            "Estimated Speed", 
            value=f"{predicted_target['velocity_mps']:.2f} m/s" if predicted_target else "N/A",
            delta=f"Error: {abs(predicted_target['velocity_mps'] - base_params['v']):.2f} m/s" if predicted_target else None,
            delta_color="inverse"
        )
    with col_m3:
        # Show doppler spread as a micro-Doppler indicator
        st.metric(
            "Doppler Spread (σ_v)", 
            value=f"{predicted_target['features']['doppler_spread']:.3f} m/s" if predicted_target else "N/A",
            help="Standard deviation of velocity slice. Higher values indicate fast rotating parts (Drones) or wing flaps (Birds)."
        )
    with col_m4:
        st.metric(
            "Spectral Entropy (H)", 
            value=f"{predicted_target['features']['spectral_entropy']:.3f} bits" if predicted_target else "N/A",
            help="Signal complexity. Single peaks (Aircraft) have low entropy, modulated returns (Drones/Birds) have higher complexity."
        )

# Layout Tabs
tab1, tab2, tab3 = st.tabs(["📊 Signal Analysis & 2D FFT Maps", "📈 Doppler Spectral Signatures", "🧠 ML Features & Probabilities"])

with tab1:
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        # 1. Raw Time domain plots
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#161A25')
        ax.set_facecolor('#0E1117')
        chirp_to_plot = sig_matrix[0, :]
        ax.plot(sim.t_f * 1e6, np.real(chirp_to_plot), color='#FF3366', alpha=0.85, label='In-Phase (Real)')
        ax.plot(sim.t_f * 1e6, np.imag(chirp_to_plot), color='#00CCFF', alpha=0.6, label='Quadrature (Imag)')
        ax.set_title("Raw Baseband IF Signal (First Chirp)", color='#FFFFFF', fontsize=12, fontweight='bold')
        ax.set_xlabel("Fast Time (μs)", color='#FFFFFF')
        ax.set_ylabel("Amplitude", color='#FFFFFF')
        ax.tick_params(colors='#FFFFFF')
        ax.grid(True, color='#2E3244', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', facecolor='#161A25', edgecolor='#2E3244')
        ax.set_xlim(0, sim.T_c * 1e6 * 0.4)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_plot2:
        # 2. Range Doppler Map
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#161A25')
        ax.set_facecolor('#0E1117')
        
        v_mesh, r_mesh = np.meshgrid(proc.velocities, proc.ranges)
        im = ax.pcolormesh(v_mesh, r_mesh, rdm_db.T, cmap='magma', shading='auto', vmin=-70, vmax=0)
        ax.set_title("2D Range-Doppler Power Map (2D FFT)", color='#FFFFFF', fontsize=12, fontweight='bold')
        ax.set_xlabel("Velocity (m/s)", color='#FFFFFF')
        ax.set_ylabel("Range (meters)", color='#FFFFFF')
        ax.tick_params(colors='#FFFFFF')
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='#FFFFFF')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#FFFFFF')
        
        # Draw target circles if detected
        if predicted_target:
            c_color = COLOR_AIRCRAFT if target_type == 'Aircraft' else (COLOR_DRONE if target_type == 'Drone' else COLOR_BIRD)
            circle = patches.Circle((predicted_target['velocity_mps'], predicted_target['range_m']), 
                                   radius=4.5, edgecolor=c_color, facecolor='none', linewidth=2, linestyle='-')
            ax.add_patch(circle)
            
            # Crosshair
            ax.axhline(predicted_target['range_m'], color='#FFFFFF', linestyle=':', alpha=0.4)
            ax.axvline(predicted_target['velocity_mps'], color='#FFFFFF', linestyle=':', alpha=0.4)
            
        ax.set_ylim(0, 150)
        ax.set_xlim(proc.velocities[0], proc.velocities[-1])
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        # Doppler Slice
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#161A25')
        ax.set_facecolor('#0E1117')
        
        if predicted_target:
            r_idx = predicted_target['range_idx']
            profile_db = 10 * np.log10(rdm_power[:, r_idx] / np.max(rdm_power) + 1e-12)
            c_color = COLOR_AIRCRAFT if target_type == 'Aircraft' else (COLOR_DRONE if target_type == 'Drone' else COLOR_BIRD)
            
            ax.plot(proc.velocities, profile_db, color='#8D96A7', linewidth=1.2, label='Doppler Profile')
            
            # Local slice window highlighted
            window_size = 15
            d_idx = predicted_target['doppler_idx']
            indices = np.arange(d_idx - window_size, d_idx + window_size + 1)
            wrapped_indices = np.mod(indices, len(profile_db))
            ax.plot(proc.velocities[wrapped_indices], profile_db[wrapped_indices], 
                    color=c_color, linewidth=2.5, label='Target Doppler Profile')
            
            ax.axvline(predicted_target['velocity_mps'], color=c_color, linestyle='--', alpha=0.6,
                       label=f"Est Speed: {predicted_target['velocity_mps']:.2f} m/s")
            
            ax.set_title(f"Doppler Spectrum Slice (at {predicted_target['range_m']:.1f} m Range)", color='#FFFFFF', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No target details to plot", color='#FFFFFF', ha='center')
            
        ax.set_xlabel("Velocity (m/s)", color='#FFFFFF')
        ax.set_ylabel("Power (dB)", color='#FFFFFF')
        ax.tick_params(colors='#FFFFFF')
        ax.grid(True, color='#2E3244', linestyle='--', alpha=0.5)
        ax.legend(loc='lower center', facecolor='#161A25', edgecolor='#2E3244')
        ax.set_ylim(-65, 5)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_d2:
        # Micro-Doppler explanation card
        st.markdown(f"### 🧬 Micro-Doppler Signature of a **{target_type}**")
        if target_type == 'Aircraft':
            st.markdown(f"""
            - **Doppler Profile Shape**: Narrow, steep single peak with high symmetry.
            - **Physical Modulations**: None. An aircraft is a rigid body traveling at high speed with very small relative physical vibrations or oscillations.
            - **RCS (Signal Amplitude)**: Very high (RCS proxy: `{predicted_target['features']['rcs_proxy_db']:.1f} dB` if simulated).
            - **Entropy & Spread**: Very low. All target return energy is concentrated in a single velocity bin.
            """)
        elif target_type == 'Drone':
            st.markdown(f"""
            - **Doppler Profile Shape**: A central peak (drone body) surrounded by symmetric, wider sidebands.
            - **Physical Modulations**: Quadcopter propeller blades. The high speed of rotating blades (approx. `{base_params.get('f_rot', 100):.1f} Hz` or `{base_params.get('f_rot', 100)*60:.0f} RPM`) causes periodic frequency modulations.
            - **RCS (Signal Amplitude)**: Moderate (RCS proxy: `{predicted_target['features']['rcs_proxy_db']:.1f} dB`).
            - **Entropy & Spread**: High. Standard deviation of Doppler spread is `{predicted_target['features']['doppler_spread']:.3f} m/s`.
            """)
        elif target_type == 'Bird':
            st.markdown(f"""
            - **Doppler Profile Shape**: Broadened central peak, sometimes showing slow-varying asymmetric offsets.
            - **Physical Modulations**: Flapping wings. Birds fly slowly (typically 2–5 m/s) and flap wings at low frequencies (`{base_params.get('f_flap', 4):.1f} Hz`). The wing tip displacement causes noticeable phase swings.
            - **RCS (Signal Amplitude)**: Very low (RCS proxy: `{predicted_target['features']['rcs_proxy_db']:.1f} dB` due to organic matter composition).
            - **Entropy & Spread**: Moderate-high. Peak-to-average ratio is lower, and spectral flatness is relatively higher because of organic returns.
            """)

with tab3:
    if predicted_target:
        col_ml1, col_ml2 = st.columns(2)
        
        with col_ml1:
            # Bar chart of probabilities
            fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#161A25')
            ax.set_facecolor('#0E1117')
            
            p_dict = predicted_target['probabilities']
            classes_list = list(p_dict.keys())
            probs_list = [p_dict[c] * 100 for c in classes_list]
            
            bar_colors = [COLOR_AIRCRAFT, COLOR_DRONE, COLOR_BIRD]
            bars = ax.bar(classes_list, probs_list, color=bar_colors, edgecolor='#FFFFFF', width=0.4, alpha=0.85)
            
            ax.set_title("ML Model Prediction Probabilities", color='#FFFFFF', fontsize=12, fontweight='bold')
            ax.set_ylabel("Confidence (%)", color='#FFFFFF')
            ax.set_ylim(0, 110)
            ax.tick_params(colors='#FFFFFF')
            ax.grid(True, color='#2E3244', linestyle='--', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, height + 2, f"{height:.1f}%", 
                        ha='center', va='bottom', color='#FFFFFF', fontweight='bold', fontsize=10)
                
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_ml2:
            st.markdown("### 🔍 Feature Space Breakdown")
            st.markdown("These are the numerical features extracted from the radar returns and fed into the Machine Learning model:")
            
            features = predicted_target['features']
            
            st.json({
                "1. Peak Intensity (RCS DB)": f"{features['rcs_proxy_db']:.2f} dB (Aircraft: High, Drones: Mid, Birds: Low)",
                "2. Target Speed (m/s)": f"{features['speed_mps']:.2f} m/s (Aircraft: Fast, Drones: Mid, Birds: Slow)",
                "3. Doppler Spread (σ_v)": f"{features['doppler_spread']:.4f} m/s (Drone propellers cause the highest spread)",
                "4. Spectral Entropy (H)": f"{features['spectral_entropy']:.4f} bits (Complexity of modulation)",
                "5. Peak-to-Average Ratio": f"{features['peak_to_average']:.2f} (Spikiness of signal profile)",
                "6. Spectral Flatness": f"{features['spectral_flatness']:.4f} (Closeness to white noise)",
                "7. Spectral Kurtosis": f"{features['spectral_kurtosis']:.2f} (Tailedness of velocity peak)",
                "8. Estimated Range": f"{features['range_m']:.2f} m"
            })
    else:
        st.info("Train a model and detect targets to inspect ML outputs.")
