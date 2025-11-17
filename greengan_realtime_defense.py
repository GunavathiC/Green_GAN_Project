# GreenGAN Real-time Network Defense System
# Extended version with live monitoring, detection, and automated response

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import scapy.all as scapy
import threading
import time
import json
import smtplib
from collections import deque, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import platform
import sqlite3
import hashlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuration
CONFIG = {
    'INTERFACE': 'Wi-Fi',  # Network interface to monitor
    'DETECTION_THRESHOLD': 0.7,  # Confidence threshold for attack detection
    'BLOCK_DURATION': 300,  # Block duration in seconds (5 minutes)
    'MAX_QUEUE_SIZE': 1000,  # Maximum size of packet queue
    'EMAIL_ALERTS': False,
    'SMTP_SERVER': 'smtp.gmail.com',
    'SMTP_PORT': 587,
    'EMAIL_USER': 'your_email@gmail.com',
    'EMAIL_PASS': 'your_app_password',
    'ALERT_EMAIL': 'admin@company.com'
}

class NetworkPacketFeatureExtractor:
    """Extract features from network packets for ML model"""
    
    def __init__(self):
        self.flow_cache = defaultdict(dict)
        
    def extract_features(self, packet):
        """Extract features from a single packet"""
        features = {}
        
        try:
            # Basic packet information
            features['packet_length'] = len(packet)
            features['timestamp'] = float(packet.time)
            
            # IP layer features
            if packet.haslayer(scapy.IP):
                ip = packet[scapy.IP]
                features['src_ip'] = ip.src
                features['dst_ip'] = ip.dst
                features['ttl'] = ip.ttl
                features['protocol'] = ip.proto
                features['flags'] = ip.flags
                
            # TCP layer features
            if packet.haslayer(scapy.TCP):
                tcp = packet[scapy.TCP]
                features['src_port'] = tcp.sport
                features['dst_port'] = tcp.dport
                features['tcp_flags'] = tcp.flags
                features['window_size'] = tcp.window
                
            # UDP layer features
            elif packet.haslayer(scapy.UDP):
                udp = packet[scapy.UDP]
                features['src_port'] = udp.sport
                features['dst_port'] = udp.dport
                features['protocol_type'] = 'UDP'
                
            # Calculate flow-based features
            flow_key = f"{features.get('src_ip', '')}_{features.get('dst_ip', '')}_{features.get('src_port', 0)}_{features.get('dst_port', 0)}"
            
            if flow_key not in self.flow_cache:
                self.flow_cache[flow_key] = {
                    'packet_count': 0,
                    'byte_count': 0,
                    'start_time': features['timestamp'],
                    'last_time': features['timestamp']
                }
            
            flow = self.flow_cache[flow_key]
            flow['packet_count'] += 1
            flow['byte_count'] += features['packet_length']
            flow['last_time'] = features['timestamp']
            
            # Flow duration and rates
            duration = max(1, flow['last_time'] - flow['start_time'])
            features['flow_duration'] = duration
            features['packets_per_second'] = flow['packet_count'] / duration
            features['bytes_per_second'] = flow['byte_count'] / duration
            features['flow_packets'] = flow['packet_count']
            features['flow_bytes'] = flow['byte_count']
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            
        return features

class AttackDetector:
    """ML-based attack detector using your trained GAN discriminator"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.feature_columns = [
            'packet_length', 'ttl', 'protocol', 'flags', 'src_port', 'dst_port',
            'tcp_flags', 'window_size', 'flow_duration', 'packets_per_second',
            'bytes_per_second', 'flow_packets', 'flow_bytes'
        ]
        
        if model_path:
            self.load_model(model_path)
        else:
            self.create_simple_model()
    
    def create_simple_model(self):
        """Create a simple neural network for demo purposes"""
        class SimpleDetector(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 32)
                self.fc4 = nn.Linear(32, 1)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                x = torch.relu(self.fc3(x))
                x = torch.sigmoid(self.fc4(x))
                return x
        
        self.model = SimpleDetector(len(self.feature_columns))
    
    def load_model(self, path):
        """Load pre-trained model"""
        try:
            self.model = torch.load(path)
            self.model.eval()
        except Exception as e:
            print(f"Model loading error: {e}")
            self.create_simple_model()
    
    def predict(self, features):
        """Predict if packet/flow is malicious"""
        try:
            # Convert features to tensor
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0))
            
            x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                prediction = self.model(x).item()
            
            return prediction
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0

class IncidentResponseSystem:
    """Automated incident response and mitigation system"""
    
    def __init__(self):
        self.blocked_ips = {}
        self.attack_counts = defaultdict(int)
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for logging"""
        self.conn = sqlite3.connect('security_logs.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                src_ip TEXT,
                dst_ip TEXT,
                attack_type TEXT,
                confidence REAL,
                action_taken TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocked_ips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address TEXT,
                block_time DATETIME,
                unblock_time DATETIME,
                reason TEXT
            )
        ''')
        
        self.conn.commit()
    
    def log_attack(self, features, confidence, action):
        """Log detected attack to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO attacks (timestamp, src_ip, dst_ip, attack_type, confidence, action_taken)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            features.get('src_ip', 'Unknown'),
            features.get('dst_ip', 'Unknown'),
            self.classify_attack_type(features),
            confidence,
            action
        ))
        self.conn.commit()
    
    def classify_attack_type(self, features):
        """Classify type of attack based on features"""
        # Simple heuristic-based classification
        dst_port = features.get('dst_port', 0)
        packet_rate = features.get('packets_per_second', 0)
        
        if packet_rate > 100:
            return 'DDoS'
        elif dst_port in [22, 23, 3389]:  # SSH, Telnet, RDP
            return 'Brute Force'
        elif dst_port in [80, 443, 8080]:  # HTTP/HTTPS
            return 'Web Attack'
        elif dst_port in [21, 25, 53]:  # FTP, SMTP, DNS
            return 'Service Attack'
        else:
            return 'Unknown'
    
    def block_ip(self, ip_address, reason="Malicious activity detected"):
        """Block IP using iptables (Linux) or Windows Firewall"""
        try:
            current_time = datetime.now()
            unblock_time = current_time.timestamp() + CONFIG['BLOCK_DURATION']
            
            if platform.system() == "Linux":
                # Linux iptables command
                subprocess.run([
                    'sudo', 'iptables', '-A', 'INPUT', '-s', ip_address, '-j', 'DROP'
                ], check=True)
            elif platform.system() == "Windows":
                # Windows firewall command
                subprocess.run([
                    'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                    f'name=Block_{ip_address}', 'dir=in', 'action=block',
                    f'remoteip={ip_address}'
                ], check=True)
            
            self.blocked_ips[ip_address] = unblock_time
            
            # Log to database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO blocked_ips (ip_address, block_time, unblock_time, reason)
                VALUES (?, ?, ?, ?)
            ''', (ip_address, current_time.isoformat(), 
                  datetime.fromtimestamp(unblock_time).isoformat(), reason))
            self.conn.commit()
            
            print(f"✅ Blocked IP: {ip_address} - Reason: {reason}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to block IP {ip_address}: {e}")
            return False
    
    def unblock_ip(self, ip_address):
        """Unblock IP address"""
        try:
            if platform.system() == "Linux":
                subprocess.run([
                    'sudo', 'iptables', '-D', 'INPUT', '-s', ip_address, '-j', 'DROP'
                ], check=True)
            elif platform.system() == "Windows":
                subprocess.run([
                    'netsh', 'advfirewall', 'firewall', 'delete', 'rule',
                    f'name=Block_{ip_address}'
                ], check=True)
            
            if ip_address in self.blocked_ips:
                del self.blocked_ips[ip_address]
            
            print(f"✅ Unblocked IP: {ip_address}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to unblock IP {ip_address}: {e}")
            return False
    
    def check_and_unblock(self):
        """Check and unblock expired IP blocks"""
        current_time = time.time()
        to_unblock = []
        
        for ip, unblock_time in self.blocked_ips.items():
            if current_time >= unblock_time:
                to_unblock.append(ip)
        
        for ip in to_unblock:
            self.unblock_ip(ip)
    
    def send_alert_email(self, subject, message):
        """Send email alert to administrator"""
        if not CONFIG['EMAIL_ALERTS']:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = CONFIG['EMAIL_USER']
            msg['To'] = CONFIG['ALERT_EMAIL']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(CONFIG['SMTP_SERVER'], CONFIG['SMTP_PORT'])
            server.starttls()
            server.login(CONFIG['EMAIL_USER'], CONFIG['EMAIL_PASS'])
            text = msg.as_string()
            server.sendmail(CONFIG['EMAIL_USER'], CONFIG['ALERT_EMAIL'], text)
            server.quit()
            
            print("✅ Alert email sent successfully")
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")

class RealTimeNetworkMonitor:
    """Real-time network monitoring system"""
    
    def __init__(self):
        self.feature_extractor = NetworkPacketFeatureExtractor()
        self.detector = AttackDetector()
        self.response_system = IncidentResponseSystem()
        self.packet_queue = deque(maxlen=CONFIG['MAX_QUEUE_SIZE'])
        self.attack_stats = {
            'total_packets': 0,
            'attacks_detected': 0,
            'ips_blocked': 0,
            'start_time': time.time()
        }
        self.is_monitoring = False
    
    def packet_handler(self, packet):
        """Handle captured packets"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(packet)
            
            # Predict if malicious
            confidence = self.detector.predict(features)
            
            # Update statistics
            self.attack_stats['total_packets'] += 1
            
            # Store packet info
            packet_info = {
                'timestamp': datetime.now().isoformat(),
                'src_ip': features.get('src_ip', 'Unknown'),
                'dst_ip': features.get('dst_ip', 'Unknown'),
                'confidence': confidence,
                'features': features
            }
            
            self.packet_queue.append(packet_info)
            
            # Check if attack detected
            if confidence > CONFIG['DETECTION_THRESHOLD']:
                self.handle_attack(packet_info)
                
        except Exception as e:
            print(f"Packet handling error: {e}")
    
    def handle_attack(self, packet_info):
        """Handle detected attack"""
        src_ip = packet_info['src_ip']
        confidence = packet_info['confidence']
        features = packet_info['features']
        
        self.attack_stats['attacks_detected'] += 1
        
        # Log the attack
        action = "Alert Generated"
        
        # Decide on blocking based on attack severity
        if confidence > 0.9 or self.response_system.attack_counts[src_ip] > 5:
            if self.response_system.block_ip(src_ip, f"High confidence attack ({confidence:.2f})"):
                action = "IP Blocked"
                self.attack_stats['ips_blocked'] += 1
        
        # Log to database
        self.response_system.log_attack(features, confidence, action)
        
        # Increment attack count for this IP
        self.response_system.attack_counts[src_ip] += 1
        
        # Send email alert for high-confidence attacks
        if confidence > 0.9:
            subject = f"🚨 High Severity Attack Detected from {src_ip}"
            message = f"""
            Attack Details:
            - Source IP: {src_ip}
            - Confidence: {confidence:.2f}
            - Attack Type: {self.response_system.classify_attack_type(features)}
            - Action Taken: {action}
            - Timestamp: {packet_info['timestamp']}
            """
            self.response_system.send_alert_email(subject, message)
        
        print(f"🚨 ATTACK DETECTED: {src_ip} (Confidence: {confidence:.2f}) - {action}")
    
    def start_monitoring(self, interface=None):
        """Start real-time monitoring"""
        interface = interface or CONFIG['INTERFACE']
        self.is_monitoring = True
        
        print(f"🔍 Starting network monitoring on interface: {interface}")
        
        try:
            # Start packet capture
            scapy.sniff(
                iface=interface,
                prn=self.packet_handler,
                store=False,
                stop_filter=lambda x: not self.is_monitoring
            )
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            self.is_monitoring = False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        print("⏹️ Network monitoring stopped")
    
    def get_stats(self):
        """Get monitoring statistics"""
        runtime = time.time() - self.attack_stats['start_time']
        
        return {
            'total_packets': self.attack_stats['total_packets'],
            'attacks_detected': self.attack_stats['attacks_detected'],
            'ips_blocked': self.attack_stats['ips_blocked'],
            'packets_per_second': self.attack_stats['total_packets'] / max(runtime, 1),
            'attack_rate': (self.attack_stats['attacks_detected'] / max(self.attack_stats['total_packets'], 1)) * 100,
            'runtime_minutes': runtime / 60
        }

def main_streamlit_app():
    """Main Streamlit application"""
    st.set_page_config(page_title="GreenGAN Real-time Defense System", layout="wide")
    
    st.title("🛡️ GreenGAN Real-time Network Defense System")
    st.markdown("---")
    
    # Initialize monitoring system
    if 'monitor' not in st.session_state:
        st.session_state.monitor = RealTimeNetworkMonitor()
        st.session_state.monitoring_thread = None
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Monitoring", type="primary"):
            if not st.session_state.monitor.is_monitoring:
                st.session_state.monitoring_thread = threading.Thread(
                    target=st.session_state.monitor.start_monitoring,
                    daemon=True
                )
                st.session_state.monitoring_thread.start()
                st.success("Network monitoring started!")
            else:
                st.warning("Monitoring is already active!")
    
    with col2:
        if st.button("⏹️ Stop Monitoring", type="secondary"):
            st.session_state.monitor.stop_monitoring()
            st.success("Network monitoring stopped!")
    
    with col3:
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()
    
    # Display statistics
    stats = st.session_state.monitor.get_stats()
    
    st.subheader("📊 Real-time Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Packets", f"{stats['total_packets']:,}")
    
    with col2:
        st.metric("Attacks Detected", f"{stats['attacks_detected']:,}")
    
    with col3:
        st.metric("IPs Blocked", f"{stats['ips_blocked']:,}")
    
    with col4:
        st.metric("Packets/Second", f"{stats['packets_per_second']:.1f}")
    
    # Attack rate visualization
    if stats['total_packets'] > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Prepare data for visualization
        recent_packets = list(st.session_state.monitor.packet_queue)[-100:]  # Last 100 packets
        
        if recent_packets:
            timestamps = [datetime.fromisoformat(p['timestamp']) for p in recent_packets]
            confidences = [p['confidence'] for p in recent_packets]
            
            # Plot confidence scores over time
            ax.plot(timestamps, confidences, alpha=0.7, label='Attack Confidence')
            ax.axhline(y=CONFIG['DETECTION_THRESHOLD'], color='red', linestyle='--', 
                      label=f'Detection Threshold ({CONFIG["DETECTION_THRESHOLD"]})')
            ax.set_ylabel('Attack Confidence')
            ax.set_xlabel('Time')
            ax.set_title('Real-time Attack Detection')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
    
    # Recent attacks table
    st.subheader("🚨 Recent Attack Detections")
    
    recent_attacks = [
        p for p in st.session_state.monitor.packet_queue 
        if p['confidence'] > CONFIG['DETECTION_THRESHOLD']
    ][-20:]  # Last 20 attacks
    
    if recent_attacks:
        attack_df = pd.DataFrame([
            {
                'Timestamp': p['timestamp'],
                'Source IP': p['src_ip'],
                'Destination IP': p['dst_ip'],
                'Confidence': f"{p['confidence']:.3f}",
                'Attack Type': st.session_state.monitor.response_system.classify_attack_type(p['features'])
            }
            for p in recent_attacks
        ])
        
        st.dataframe(attack_df, use_container_width=True)
    else:
        st.info("No attacks detected yet.")
    
    # Blocked IPs section
    st.subheader("🚫 Currently Blocked IPs")
    
    blocked_ips = st.session_state.monitor.response_system.blocked_ips
    
    if blocked_ips:
        blocked_df = pd.DataFrame([
            {
                'IP Address': ip,
                'Block Time Remaining': f"{max(0, int((unblock_time - time.time()) / 60))} minutes"
            }
            for ip, unblock_time in blocked_ips.items()
        ])
        
        st.dataframe(blocked_df, use_container_width=True)
    else:
        st.info("No IPs currently blocked.")
    
    # Configuration section
    st.subheader("⚙️ Configuration")
    
    with st.expander("System Configuration"):
        st.json(CONFIG)
    
    # Auto-refresh
    if st.session_state.monitor.is_monitoring:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    # Check if running as Streamlit app
    try:
        main_streamlit_app()
    except Exception:
        # Running as standalone script
        print("Starting GreenGAN Real-time Defense System...")
        
        monitor = RealTimeNetworkMonitor()
        
        try:
            monitor.start_monitoring()
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitoring...")
            monitor.stop_monitoring()
        except Exception as e:
            print(f"❌ Error: {e}")