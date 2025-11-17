import requests
import threading
import time

def ddos_simulation():
    """Simulate DDoS by making rapid requests"""
    print("🚀 Starting DDoS attack simulation...")
    for i in range(50):
        try:
            response = requests.get("http://httpbin.org/delay/0.1", timeout=5)
            print(f"Attack request {i+1}: {response.status_code}")
        except Exception as e:
            print(f"Attack request {i+1}: Failed with error {e}")
        time.sleep(0.1)
    print("✅ Attack simulation completed")

# Run the attack
print("Starting attack in 3 seconds...")
time.sleep(0.3)
threading.Thread(target=ddos_simulation).start()
attack_thread = threading.Thread(target=ddos_simulation)
attack_thread.start()
attack_thread.join()  # Wait for completion

