import streamlit as st
import numpy as np
import torch
import threading
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Import your GAN trainer, data processor, evaluator from MLproj
from MLproj import GreenGANTrainer, CICIDSDataProcessor, SecurityEvaluator

# Import real-time defense classes (ensure this module is available)
from greengan_realtime_defense import RealTimeNetworkMonitor, CONFIG

# Cache GAN models and dataset processor loading for efficiency
@st.cache(allow_output_mutation=True)
def load_gan_and_data():
    processor = CICIDSDataProcessor()
    dataset_path = r"C:\\Users\\gunav\\Downloads\\MachineLearningCSV\\MachineLearningCVE\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    X, y, feature_names = processor.load_and_preprocess_data(dataset_path)
    attack_indices = (y != 'Benign')  # use label strings, not 1
    attack_data = X[attack_indices]
    gan = GreenGANTrainer(feature_dim=X.shape[1], noise_dim=100)
    model_path = "models/"
    if (os.path.exists(os.path.join(model_path, "green_generator.pth")) and
            os.path.exists(os.path.join(model_path, "green_discriminator.pth"))):
        gan.load_models(model_path)
    else:
        gan.train(attack_data, epochs=50, batch_size=64, save_model=True)
        gan.load_models(model_path)
    return processor, gan, attack_data, feature_names

# Initialize or reuse real-time monitor and thread in Streamlit session state
if "monitor" not in st.session_state:
    st.session_state.monitor = RealTimeNetworkMonitor()
if "monitoring_thread" not in st.session_state:
    st.session_state.monitoring_thread = None

def start_real_time_monitoring():
    if not st.session_state.monitor.is_monitoring:
        st.session_state.monitoring_thread = threading.Thread(
            target=st.session_state.monitor.start_monitoring,
            args=(CONFIG['INTERFACE'],),
            daemon=True
        )
        st.session_state.monitoring_thread.start()
        st.success("Real-time network monitoring started!")
    else:
        st.warning("Monitoring is already active.")

def stop_real_time_monitoring():
    st.session_state.monitor.stop_monitoring()
    st.success("Real-time network monitoring stopped!")

# Main Streamlit app interface
st.set_page_config(page_title="GreenGAN Integrated Cyber Defense", layout="wide")
st.title("🛡️ GreenGAN Integrated Cyber Defense System")

tabs = st.tabs(["GAN Synthetic Attack Generation", "Real-Time Network Defense"])

with tabs[0]:
    st.header("GAN Synthetic Attack Generation and Evaluation")

    processor, gan, attack_data, feature_names = load_gan_and_data()

    if 'synthetic_attacks' not in st.session_state:
        st.session_state.synthetic_attacks = None

    num_samples = st.slider("Number of Synthetic Attacks to Generate", 100, 5000, 500, 100)

    if st.button("Generate Synthetic Attacks"):
        synthetic_attacks = gan.generate_synthetic_attacks(num_samples)
        st.session_state.synthetic_attacks = synthetic_attacks
        st.success(f"Generated {num_samples} synthetic attack vectors!")

    if st.session_state.synthetic_attacks is not None:
        st.dataframe(np.array(st.session_state.synthetic_attacks)[:10], width=700, height=200)

        if st.checkbox("Show Evaluation Visualizations"):
            evaluator = SecurityEvaluator(gan.discriminator, attack_data[:1000], st.session_state.synthetic_attacks[:1000])
            results = evaluator.evaluate_detection_capability()

            st.write("Evaluation Results:", results)
            st.write(f"Discriminator Accuracy on Real Attacks: {results['real_accuracy']:.3f}")
            st.write(f"Synthetic Data Fooling Rate: {results['synthetic_fooling_rate']:.3f}")

            evaluator.visualize_results()

            st.subheader("Feature Distribution Comparison (First 10 Features)")
            fig2, ax2 = plt.subplots()
            feature_comparison = np.column_stack([
                np.mean(attack_data[:1000, :10], axis=0),
                np.mean(st.session_state.synthetic_attacks[:1000, :10], axis=0)
            ])
            ax2.plot(feature_comparison[:, 0], 'o-', label='Real Attacks')
            ax2.plot(feature_comparison[:, 1], 's-', label='Synthetic Attacks')
            ax2.set_xlabel("Feature Index")
            ax2.set_ylabel("Average Value")
            ax2.set_title("Feature Comparison")
            ax2.legend()
            st.pyplot(fig2)
    else:
        st.warning("Please generate synthetic attacks first to see evaluation.")

with tabs[1]:
    st.header("Real-Time Network Defense Monitoring")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start Monitoring"):
            start_real_time_monitoring()
    with col2:
        if st.button("Stop Monitoring"):
            stop_real_time_monitoring()
    with col3:
        if st.button("Refresh Dashboard"):
            st.experimental_rerun()

    stats = st.session_state.monitor.get_stats()
    st.subheader("Real-Time Statistics")
    st.metric("Total Packets", stats['total_packets'])
    st.metric("Attacks Detected", stats['attacks_detected'])
    st.metric("IPs Blocked", stats['ips_blocked'])
    st.metric("Packets/sec", f"{stats['packets_per_second']:.1f}")

    if stats['total_packets'] > 0:
        recent_packets = list(st.session_state.monitor.packet_queue)[-100:]
        timestamps = [datetime.fromisoformat(p['timestamp']) for p in recent_packets]
        confidences = [p['confidence'] for p in recent_packets]

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(timestamps, confidences, label="Attack Confidence")
        ax3.axhline(y=CONFIG['DETECTION_THRESHOLD'], color='red', linestyle='--', label="Detection Threshold")
        ax3.set_ylabel("Confidence")
        ax3.set_xlabel("Time")
        ax3.set_title("Real-time Attack Detection")
        ax3.legend()
        ax3.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)

    st.subheader("Recent Attack Detections")
    recent_attacks = [p for p in st.session_state.monitor.packet_queue if p['confidence'] > CONFIG['DETECTION_THRESHOLD']][-20:]
    if recent_attacks:
        attack_data_dict = {
            "Timestamp": [p['timestamp'] for p in recent_attacks],
            "Source IP": [p['src_ip'] for p in recent_attacks],
            "Destination IP": [p['dst_ip'] for p in recent_attacks],
            "Confidence": [f"{p['confidence']:.3f}" for p in recent_attacks],
            "Attack Type": [st.session_state.monitor.response_system.classify_attack_type(p['features']) for p in recent_attacks]
        }
        st.table(attack_data_dict)
    else:
        st.info("No attacks detected yet.")

    st.subheader("Currently Blocked IPs")
    blocked_ips = st.session_state.monitor.response_system.blocked_ips
    if blocked_ips:
        blocked_data = {
            "IP Address": list(blocked_ips.keys()),
            "Block Time Remaining (minutes)": [max(0, int((unblock_time - time.time()) // 60)) for unblock_time in blocked_ips.values()]
        }
        st.table(blocked_data)
    else:
        st.info("No IPs currently blocked.")

# Auto-refresh dashboard when monitoring is active
if st.session_state.monitor.is_monitoring:
    time.sleep(2)
    st.experimental_rerun()
