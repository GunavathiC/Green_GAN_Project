import streamlit as st
import numpy as np
import torch
from MLproj import GreenGANTrainer, CICIDSDataProcessor, SecurityEvaluator
import os

# Cache to load GAN and dataset processor only once
@st.cache(allow_output_mutation=True)
def load_gan_and_data():
    processor = CICIDSDataProcessor()
    # You should specify the path to your CIC-IDS2017 CSV file here
    dataset_path = r"C:\Users\gunav\Downloads\MachineLearningCSV\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv "

    X, y, feature_names = processor.load_and_preprocess_data(dataset_path)
    attack_indices = (y == 1)
    attack_data = X[attack_indices]


    gan = GreenGANTrainer(feature_dim=X.shape[1], noise_dim=100)

    # Load models if saved else train
    model_path = "models/"
    if (os.path.exists(os.path.join(model_path, "green_generator.pth")) and
        os.path.exists(os.path.join(model_path, "green_discriminator.pth"))):
        gan.load_models(model_path)
    else:
        train_stats = gan.train(attack_data, epochs=50, batch_size=64, savemodel=True)
        gan.load_models(model_path)

    return processor, gan, attack_data, feature_names

processor, gan, attack_data, feature_names = load_gan_and_data()

st.title("Green-GAN: Energy Efficient Adversarial AI for Cyberattack Generation")

st.markdown("""
This app demonstrates a GAN trained on the CIC-IDS2017 dataset to generate synthetic cyberattack vectors for security testing.
- Efficient GAN architectures reduce energy usage
- Generates realistic network intrusion patterns
- Enables proactive testing of intrusion detection systems
""")


# Section: Generate Synthetic Attacks
num_samples = st.slider("Number of Synthetic Attacks to Generate", min_value=100, max_value=5000, step=100, value=500)

if st.button("Generate Synthetic Attacks"):
    synthetic_attacks = gan.generate_synthetic_attacks(num_samples)
    st.success(f"Generated {num_samples} synthetic attack vectors!")
    st.dataframe(np.array(synthetic_attacks)[:10])  # Show first 10 synthetic vectors


# Section: Evaluate GAN Performance
if st.checkbox("Show Evaluation & Visualizations"):
    st.subheader("Evaluation of Generator & Discriminator")

    # Create an evaluator instance
    
    evaluator = SecurityEvaluator(gan.discriminator, attack_data, synthetic_attacks)


    
    results = evaluator.evaluate_detection_capability()  # Call the method and save output

    st.write("Returned results:", results)  # Display the full results to see structure

    real_acc = results['real_accuracy']   # Extract the numeric value by key (adjust key if needed)
    synthetic_fooling_rate = results['synthetic_fooling_rate']  # Extract other metric

    st.write(f"Discriminator Accuracy on Real Attacks: {float(real_acc):.3f}")
    st.write(f"Synthetic Data Fooling Rate: {float(synthetic_fooling_rate):.3f}")


    # Visualize score distributions
    fig = evaluator.visualize_results()
    st.pyplot(fig)

    # Feature comparison plots
    st.subheader("Feature Distribution Comparison (First 10 Features)")
    import matplotlib.pyplot as plt

    fig2, ax2 = plt.subplots()
    feature_comparison = np.column_stack((np.mean(attack_data[:1000, :10], axis=0), np.mean(synthetic_attacks[:1000, :10], axis=0)))
    ax2.plot(feature_comparison[:, 0], 'o-', label='Real Attacks', markersize=5)
    ax2.plot(feature_comparison[:, 1], 's-', label='Synthetic Attacks', markersize=5)
    ax2.set_xlabel("Feature Index")
    ax2.set_ylabel("Average Value")
    ax2.set_title("Feature Comparison (First 10 Features)")
    ax2.legend()
    st.pyplot(fig2)



