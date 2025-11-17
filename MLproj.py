# Green-GAN: A Low-Power Adversarial Framework for Proactive Security Audits
# Complete Implementation for CIC-IDS2017 Dataset
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Set your base folder
data_path = r"C:\Users\gunav\Downloads\MachineLearningCSV\MachineLearningCVE"

# List subfolders and CSVs (optional print)
for item in os.listdir(data_path):
    full = os.path.join(data_path, item)
    if os.path.isdir(full):
        print("Folder:", item)
    elif item.lower().endswith(".csv"):
        print("CSV File:", item)

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# 1. DATA PREPROCESSING AND LOADING
# =============================================================================

class CICIDSDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_and_preprocess_data(self, file_path):
        print("Loading CIC-IDS2017 dataset...")
        
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()  # clean column names
        
        label_col = 'Label'  # make sure exact column name
        if label_col not in df.columns:
            raise KeyError(f"Expected label column '{label_col}' not found in CSV columns: {list(df.columns)}")
        
        y = df[label_col].astype(str).str.strip().str.title()
        X = df.drop(columns=[label_col])
        
        X = X.apply(pd.to_numeric, errors='coerce')
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Option 1: Drop rows with NaN values
        X_clean = X.dropna()
        y_clean = y.loc[X_clean.index]
        
        # Normalize and map labels to simplified categories
        y_processed = y.str.title()
        
        attack_type_map = {
            'Benign': 'Benign',
            'Dos': 'DoS',
            'Ddos': 'DDoS',
            'Brute Force': 'Brute Force',
            'Botnet': 'Botnet',
            'Web Attack': 'Web Attack',
            'Service Attack': 'Service Attack',
            # Add more mappings as needed
        }
        y_mapped = y_processed.map(lambda x: attack_type_map.get(x, 'Unknown'))
        
        unknown_labels = y_processed[y_mapped == 'Unknown'].unique()
        if len(unknown_labels) > 0:
            print(f"Warning: Unknown labels found and mapped as 'Unknown': {unknown_labels}")
        
        # Scale features
        
        X_scaled = self.scaler.fit_transform(X_clean.values)
        
        print(f"Dataset loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        print(f"Attack samples: {(y_mapped != 'Benign').sum()}, Benign samples: {(y_mapped == 'Benign').sum()}")
        
        return X_scaled, np.array(y_clean), list(X.columns)
        return X_scaled, np.array(y_mapped), list(df.columns.drop(label_col))

# =============================================================================
# 2. GREEN-GAN MODEL ARCHITECTURE
# =============================================================================

class EfficientGenerator(nn.Module):
    def __init__(self, noise_dim=100, feature_dim=78):
        super(EfficientGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, feature_dim),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)

class EfficientDiscriminator(nn.Module):
    def __init__(self, feature_dim=78):
        super(EfficientDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# =============================================================================
# 3. GREEN TRAINING UTILITIES
# =============================================================================

class GreenTrainingMonitor:
    def __init__(self):
        self.start_time = None
        self.g_losses = []
        self.d_losses = []
    
    def start_training(self):
        self.start_time = time.time()
        print("🌱 Green training started - optimizing for energy efficiency")
    
    def log_epoch(self, epoch, g_loss, d_loss):
        self.g_losses.append(g_loss)
        self.d_losses.append(d_loss)
        if epoch % 10 == 0:
            elapsed = time.time() - self.start_time
            print(f"Epoch {epoch}: G_Loss={g_loss:.4f}, D_Loss={d_loss:.4f}, Time={elapsed:.1f}s 🌱")
    
    def end_training(self):
        total_time = time.time() - self.start_time
        avg_g_loss = np.mean(self.g_losses[-10:])
        avg_d_loss = np.mean(self.d_losses[-10:])
        print(f"\n🌱 Green Training Complete!")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Final Generator Loss: {avg_g_loss:.4f}")
        print(f"Final Discriminator Loss: {avg_d_loss:.4f}")
        return {
            'total_time': total_time,
            'final_g_loss': avg_g_loss,
            'final_d_loss': avg_d_loss,
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }

# =============================================================================
# 4. GREEN-GAN TRAINER CLASS
# =============================================================================

class GreenGANTrainer:
    def __init__(self, feature_dim=78, noise_dim=100, lr=0.0002):
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.generator = EfficientGenerator(noise_dim, feature_dim).to(device)
        self.discriminator = EfficientDiscriminator(feature_dim).to(device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        self.monitor = GreenTrainingMonitor()
    
    def train(self, real_data, epochs=100, batch_size=64, save_model=True):
        dataset = TensorDataset(torch.FloatTensor(real_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.monitor.start_training()
        
        for epoch in range(epochs):
            for batch_idx, (real_batch,) in enumerate(dataloader):
                batch_size_current = real_batch.size(0)
                real_batch = real_batch.to(device)
                real_labels = torch.ones(batch_size_current, 1).to(device)
                fake_labels = torch.zeros(batch_size_current, 1).to(device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                d_real_output = self.discriminator(real_batch)
                d_real_loss = self.criterion(d_real_output, real_labels)
                noise = torch.randn(batch_size_current, self.noise_dim).to(device)
                fake_data = self.generator(noise)
                d_fake_output = self.discriminator(fake_data.detach())
                d_fake_loss = self.criterion(d_fake_output, fake_labels)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
            
            self.monitor.log_epoch(epoch, g_loss.item(), d_loss.item())
            
            if epoch > 20 and self.is_converged():
                print(f"🌱 Early convergence detected at epoch {epoch}. Stopping for energy efficiency.")
                break
        
        training_stats = self.monitor.end_training()
        
        if save_model:
            self.save_models()
        
        return training_stats
    
    def is_converged(self):
        if len(self.monitor.g_losses) < 10:
            return False
        g_variance = np.var(self.monitor.g_losses[-10:])
        d_variance = np.var(self.monitor.d_losses[-10:])
        return g_variance < 0.01 and d_variance < 0.01
    
    def generate_synthetic_attacks(self, num_samples=1000):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.noise_dim).to(device)
            synthetic_attacks = self.generator(noise)
            return synthetic_attacks.cpu().numpy()
    
    def save_models(self, path="models/"):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.generator.state_dict(), f"{path}green_generator.pth")
        torch.save(self.discriminator.state_dict(), f"{path}green_discriminator.pth")
        print(f"🌱 Models saved to {path}")
    
    def load_models(self, path="models/"):
        self.generator.load_state_dict(torch.load(f"{path}green_generator.pth"))
        self.discriminator.load_state_dict(torch.load(f"{path}green_discriminator.pth"))
        print(f"🌱 Models loaded from {path}")

# =============================================================================
# 5. SECURITY EVALUATION
# =============================================================================

class SecurityEvaluator:
    def __init__(self, discriminator, real_data, synthetic_data):
        self.discriminator = discriminator.to(device)
        self.real_data = real_data
        self.synthetic_data = synthetic_data
    
    def evaluate_detection_capability(self):
        self.discriminator.eval()
        with torch.no_grad():
            real_tensor = torch.FloatTensor(self.real_data).to(device)
            synthetic_tensor = torch.FloatTensor(self.synthetic_data).to(device)
            real_logits = self.discriminator(real_tensor)
            synthetic_logits = self.discriminator(synthetic_tensor)
            real_probs = torch.sigmoid(real_logits).view(-1)
            synthetic_probs = torch.sigmoid(synthetic_logits).view(-1)
            
            real_labels = torch.ones(real_probs.size(0), device=device)
            synthetic_labels = torch.zeros(synthetic_probs.size(0), device=device)
            
            real_preds = (real_probs >= 0.5).float()
            synthetic_preds = (synthetic_probs >= 0.5).float()
            
            real_accuracy = (real_preds == real_labels).float().mean().item() if real_preds.numel() > 0 else 0.0
            synthetic_fooling_rate = (synthetic_preds == synthetic_labels).float().mean().item() if synthetic_preds.numel() > 0 else 0.0
            
            real_scores = real_probs.cpu().numpy()
            synthetic_scores = synthetic_probs.cpu().numpy()
        
        print(f"\nSecurity Evaluation Results:")
        print(f"Real attack detection accuracy: {real_accuracy:.3f}")
        print(f"Synthetic attack fooling rate: {synthetic_fooling_rate:.3f}")
        print(f"Average real score: {np.mean(real_scores):.3f}")
        print(f"Average synthetic score: {np.mean(synthetic_scores):.3f}")
        
        return {
            'real_accuracy': real_accuracy,
            'synthetic_fooling_rate': synthetic_fooling_rate,
            'real_scores': real_scores,
            'synthetic_scores': synthetic_scores
        }
    
    def visualize_results(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        with torch.no_grad():
            real_scores = self.discriminator(torch.FloatTensor(self.real_data[:1000]).to(device)).cpu()
            synthetic_scores = self.discriminator(torch.FloatTensor(self.synthetic_data[:1000]).to(device)).cpu()
        
        ax1.hist(real_scores, alpha=0.7, label='Real Attack Scores', bins=30)
        ax1.hist(synthetic_scores, alpha=0.7, label='Synthetic Attack Scores', bins=30)
        ax1.set_xlabel('Discriminator Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Score Distribution')
        ax1.legend()
        
        feature_comparison = np.column_stack([
            np.mean(self.real_data[:1000, :10], axis=0),
            np.mean(self.synthetic_data[:1000, :10], axis=0)
        ])
        
        ax2.plot(feature_comparison[:, 0], 'o-', label='Real Attacks', markersize=4)
        ax2.plot(feature_comparison[:, 1], 's-', label='Synthetic Attacks', markersize=4)
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Average Value')
        ax2.set_title('Feature Comparison (First 10 Features)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('green_gan_evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()

# =============================================================================
# 6. MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    print("🌱 GREEN-GAN: Low-Power Adversarial Framework for Security Audits")
    print("=" * 70)
    
    processor = CICIDSDataProcessor()
    
    # Update this path to your actual CIC-IDS2017 CSV file
    data_path = r"C:\Users\gunav\Downloads\MachineLearningCSV\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    
    if not os.path.exists(data_path):
        print("⚠️ CIC-IDS2017 file not found. Creating sample data for demonstration...")
        np.random.seed(42)
        n_samples = 10000
        n_features = 78
        
        X_sample = np.random.randn(n_samples, n_features)
        y_sample = np.random.choice(['Benign', 'Attack'], n_samples, p=[0.8, 0.2])
        feature_names = [f'feature_{i}' for i in range(n_features)]
    else:
        X_sample, y_sample, feature_names = processor.load_and_preprocess_data(data_path)
    
    # Extract attack samples for training
    attack_indices = (y_sample != 'Benign')
    attack_data = X_sample[attack_indices]
    print(f"Attack samples for training: {len(attack_data)}")
    
    feature_dim = X_sample.shape[1]
    gan_trainer = GreenGANTrainer(feature_dim=feature_dim, noise_dim=100, lr=0.0002)
    
    training_stats = gan_trainer.train(real_data=attack_data, epochs=50, batch_size=64, save_model=True)
    
    print("\n🔧 Generating synthetic attack vectors...")
    synthetic_attacks = gan_trainer.generate_synthetic_attacks(num_samples=1000)
    
    print("\n📊 Evaluating Green-GAN performance...")
    evaluator = SecurityEvaluator(discriminator=gan_trainer.discriminator,
                                  real_data=attack_data[:1000],
                                  synthetic_data=synthetic_attacks)
    evaluation_results = evaluator.evaluate_detection_capability()
    evaluator.visualize_results()
    
    print("\n" + "="*70)
    print("🌱 GREEN-GAN PROJECT SUMMARY REPORT")
    print("="*70)
    print(f"Dataset: CIC-IDS2017 (or sample data)")
    print(f"Total features: {feature_dim}")
    print(f"Attack samples used: {len(attack_data)}")
    print(f"Training time: {training_stats['total_time']:.2f} seconds")
    print(f"Final Generator Loss: {training_stats['final_g_loss']:.4f}")
    print(f"Final Discriminator Loss: {training_stats['final_d_loss']:.4f}")
    print(f"Synthetic samples generated: {len(synthetic_attacks)}")
    print(f"Real attack detection accuracy: {evaluation_results['real_accuracy']:.3f}")
    print(f"Synthetic fooling success rate: {evaluation_results['synthetic_fooling_rate']:.3f}")
    print("\n🎯 Green-GAN successfully completed! Ready for security testing.")
    
    np.save('synthetic_attack_vectors.npy', synthetic_attacks)
    print("💾 Synthetic attack vectors saved to 'synthetic_attack_vectors.npy'")
    
    return {
        'training_stats': training_stats,
        'evaluation_results': evaluation_results,
        'synthetic_attacks': synthetic_attacks,
        'gan_trainer': gan_trainer
    }

# =============================================================================
# 7. USAGE EXAMPLE AND TESTING
# =============================================================================

if __name__ == "__main__":
    results = main()
    
    print("\n🔬 DEMONSTRATION: Using Green-GAN for Penetration Testing")
    print("-" * 60)
    gan = results['gan_trainer']
    
    attack_batch_1 = gan.generate_synthetic_attacks(100)
    attack_batch_2 = gan.generate_synthetic_attacks(100)
    
    print(f"Generated {len(attack_batch_1)} Type-A attack vectors")
    print(f"Generated {len(attack_batch_2)} Type-B attack vectors")
    
    discriminator_scores_1 = gan.discriminator(torch.FloatTensor(attack_batch_1).to(device))
    discriminator_scores_2 = gan.discriminator(torch.FloatTensor(attack_batch_2).to(device))
    
    print(f"Type-A attacks - Avg detection score: {discriminator_scores_1.mean():.3f}")
    print(f"Type-B attacks - Avg detection score: {discriminator_scores_2.mean():.3f}")
    
    print("\n✅ Green-GAN is ready for deployment in security testing!")
    print("💡 Use the generated synthetic attacks to test your security systems.")
    print("🌱 Energy-efficient training completed successfully!")

    
