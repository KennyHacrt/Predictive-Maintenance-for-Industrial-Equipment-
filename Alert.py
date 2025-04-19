import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import joblib
import os
from sklearn.ensemble import IsolationForest
import threading

# Set matplotlib to use 'Agg' backend to avoid threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ThingSpeak credentials
CHANNEL_ID = "2925292"
READ_API_KEY = "8M3SFRL42ZBOIGE4"
WRITE_API_KEY = "79Z8SMIMWPCJT4BM"

# Global models
turbine_model = None
compressor_model = None

# Historical data file
HISTORICAL_DATA_FILE = 'historical_data.csv'

# Check frequency (seconds)
CHECK_INTERVAL = 15

# Alert thresholds
ALERT_THRESHOLD = {
    'Turbine': {
        'temperature': 85.0,  # °C
        'vibration': 3.5,     # mm/s
    },
    'Compressor': {
        'pressure_high': 70.0,  # psi
        'pressure_low': 40.0,   # psi
    }
}

# Alert tracking
active_alerts = {
    'Turbine': False,
    'Compressor': False
}

# Alert history for plotting
alert_history = {
    'Turbine': {
        'timestamps': [],
        'anomaly_scores': [],
        'temperature': [],
        'vibration': [],
        'alerts': []
    },
    'Compressor': {
        'timestamps': [],
        'anomaly_scores': [],
        'pressure': [],
        'alerts': []
    }
}

def train_anomaly_models(df):
    """Train anomaly detection models using the historical data"""
    print("Training anomaly detection models...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Separate by equipment type
    turbine_data = df[df['equipment_type'] == 'Turbine'].copy()
    compressor_data = df[df['equipment_type'] == 'Compressor'].copy()
    
    print(f"Training Turbine model with {len(turbine_data)} samples")
    print(f"Training Compressor model with {len(compressor_data)} samples")
    
    # Turbine model (using temperature and vibration)
    X_turbine = turbine_data[['temperature', 'vibration']].values
    
    # Calculate contamination (proportion of anomalies)
    contamination_turbine = turbine_data['is_anomaly'].mean()
    if contamination_turbine == 0:
        contamination_turbine = 0.05  # Default if no anomalies in training data
    
    # Train Isolation Forest for turbine
    global turbine_model
    turbine_model = IsolationForest(
        n_estimators=100,
        contamination=float(contamination_turbine),
        random_state=42
    )
    turbine_model.fit(X_turbine)
    
    # Save turbine model
    joblib.dump(turbine_model, 'models/turbine_anomaly_model.joblib')
    
    # Compressor model (using pressure)
    X_compressor = compressor_data[['pressure']].values
    
    # Calculate contamination
    contamination_compressor = compressor_data['is_anomaly'].mean()
    if contamination_compressor == 0:
        contamination_compressor = 0.05  # Default if no anomalies in training data
    
    # Train Isolation Forest for compressor
    global compressor_model
    compressor_model = IsolationForest(
        n_estimators=100,
        contamination=float(contamination_compressor),
        random_state=42
    )
    compressor_model.fit(X_compressor)
    
    # Save compressor model
    joblib.dump(compressor_model, 'models/compressor_anomaly_model.joblib')
    
    print("Models trained and saved successfully")
    
    # Generate visualization
    visualize_model_boundaries(turbine_data, compressor_data)

def visualize_model_boundaries(turbine_data, compressor_data):
    """Visualize the decision boundaries of the anomaly detection models"""
    print("Generating model visualization...")
    
    try:
        # Set up the figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Turbine visualization (2D plot: temperature vs vibration)
        temp_min, temp_max = turbine_data['temperature'].min(), turbine_data['temperature'].max()
        vib_min, vib_max = turbine_data['vibration'].min(), turbine_data['vibration'].max()
        
        # Add some margin
        temp_range = temp_max - temp_min
        vib_range = vib_max - vib_min
        temp_min -= temp_range * 0.1
        temp_max += temp_range * 0.1
        vib_min -= vib_range * 0.1
        vib_max += vib_range * 0.1
        
        # Create a grid for visualization
        xx, yy = np.meshgrid(np.linspace(temp_min, temp_max, 100),
                            np.linspace(vib_min, vib_max, 100))
        
        # Get predictions on the grid
        Z = turbine_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary and points
        axes[0].contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
        
        # Plot the data points
        normal = turbine_data[turbine_data['is_anomaly'] == 0]
        anomaly = turbine_data[turbine_data['is_anomaly'] == 1]
        
        axes[0].scatter(normal['temperature'], normal['vibration'], 
                c='green', s=20, label='Normal', alpha=0.5)
        axes[0].scatter(anomaly['temperature'], anomaly['vibration'], 
                c='red', s=30, label='Anomaly', alpha=0.8)
        
        axes[0].set_title('Turbine Anomaly Detection Boundary')
        axes[0].set_xlabel('Temperature (°C)')
        axes[0].set_ylabel('Vibration (mm/s)')
        axes[0].legend()
        
        # Compressor visualization (1D visualization as histogram)
        # Create bins for pressure values
        bins = np.linspace(compressor_data['pressure'].min(), compressor_data['pressure'].max(), 40)
        
        # Plot normal and anomaly distributions
        normal = compressor_data[compressor_data['is_anomaly'] == 0]
        anomaly = compressor_data[compressor_data['is_anomaly'] == 1]
        
        axes[1].hist(normal['pressure'], bins=bins, alpha=0.5, 
                density=True, color='green', label='Normal')
        axes[1].hist(anomaly['pressure'], bins=bins, alpha=0.5, 
                density=True, color='red', label='Anomaly')
        
        # Generate points for decision function
        xx = np.linspace(compressor_data['pressure'].min(), compressor_data['pressure'].max(), 100)
        Z = compressor_model.decision_function(xx.reshape(-1, 1))
        
        # Plot decision function (scaled for visibility)
        scaled_Z = (Z - Z.min()) / (Z.max() - Z.min()) * 0.5
        axes[1].plot(xx, scaled_Z, color='blue', label='Decision Boundary', linewidth=2)
        
        # Calculate a threshold for visualization (approximation)
        # In Isolation Forest, negative scores are more anomalous
        # Find where predictions change from 1 to -1
        pred = compressor_model.predict(xx.reshape(-1, 1))
        threshold_indices = np.where(np.diff(pred))[0]
        
        # If we found a threshold crossover point
        if len(threshold_indices) > 0:
            threshold_idx = threshold_indices[0]
            axes[1].axvline(xx[threshold_idx], color='black', linestyle='--', 
                        alpha=0.7, label='Threshold')
        
        axes[1].set_title('Compressor Pressure Distribution and Anomaly Threshold')
        axes[1].set_xlabel('Pressure (PSI)')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('anomaly_detection_models.png')
        plt.close('all')
        
        print("Model visualization saved as 'anomaly_detection_models.png'")
    except Exception as e:
        print(f"Error in model visualization: {e}")

def load_models():
    """Load the trained anomaly detection models"""
    global turbine_model, compressor_model
    
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Check if models exist
        if (os.path.exists('models/turbine_anomaly_model.joblib') and 
            os.path.exists('models/compressor_anomaly_model.joblib')):
            
            print("Loading existing anomaly detection models...")
            turbine_model = joblib.load('models/turbine_anomaly_model.joblib')
            compressor_model = joblib.load('models/compressor_anomaly_model.joblib')
            print("Models loaded successfully")
            return True
        
        else:
            print("No existing models found. Training new models...")
            
            # Check if historical data exists
            if not os.path.exists(HISTORICAL_DATA_FILE):
                print(f"Error: Historical data file '{HISTORICAL_DATA_FILE}' not found.")
                print("Please run longterm.py first to generate historical data.")
                return False
            
            # Load historical data
            df = pd.read_csv(HISTORICAL_DATA_FILE)
            print(f"Loaded {len(df)} records from historical data")
            
            # Train models
            train_anomaly_models(df)
            return True
            
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def get_latest_data_from_thingspeak():
    """Fetch the latest data from ThingSpeak"""
    try:
        # ThingSpeak API endpoint for latest data
        url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json"
        params = {'api_key': READ_API_KEY}
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if we have entries
            if 'field1' not in data:
                print("No data available yet")
                return None
            
            # Create a safe converter to handle potential errors in type conversion
            def safe_float(value, default=None):
                if value is None or value == '':
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
                
            def safe_int(value, default=0):
                if value is None or value == '':
                    return default
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return default
            
            # Parse the data according to your ThingSpeak fields
            reading = {
                'timestamp': data['created_at'],
                'temperature': safe_float(data['field1']),
                'vibration': safe_float(data['field2']),
                'pressure': safe_float(data['field3']),
                'anomaly_status': safe_int(data['field4'], 0),
                'maintenance_performed': safe_int(data['field5'], 0),
                'maintenance_due': safe_float(data['field6'], 0),
                'equipment_code': data['field7'] if data['field7'] else ''
            }
            
            # Determine equipment type from available data
            if reading['temperature'] is not None or reading['vibration'] is not None:
                reading['equipment_type'] = 'Turbine'
            elif reading['pressure'] is not None:
                reading['equipment_type'] = 'Compressor'
            else:
                # Try to determine from equipment code
                equipment_code = str(reading['equipment_code']).upper()
                if 'TURB' in equipment_code:
                    reading['equipment_type'] = 'Turbine'
                elif 'COMP' in equipment_code:
                    reading['equipment_type'] = 'Compressor'
                else:
                    # Default
                    reading['equipment_type'] = 'Turbine'
            
            # Set status field based on anomaly_status for compatibility
            reading['status'] = "Faulty" if reading['anomaly_status'] == 1 else "Normal"
            
            return reading
        else:
            print(f"Error getting data: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching data from ThingSpeak: {e}")
        return None
    
def detect_anomalies(reading):
    """Detect anomalies in the sensor data using the trained models"""
    if reading is None:
        return None
    
    equipment_type = reading['equipment_type']
    anomaly_detected = False
    anomaly_score = 0
    
    try:
        if equipment_type == 'Turbine':
            # Check if we have valid data to analyze
            if reading['temperature'] is None or reading['vibration'] is None:
                print(f"Missing temperature or vibration data for Turbine")
                reading['anomaly_detected'] = 0
                reading['anomaly_score'] = 0.0
                return reading
            
            # Get relevant features
            features = np.array([[reading['temperature'], reading['vibration']]])
            
            # Get anomaly score (-1 for anomalies, 1 for normal)
            score = turbine_model.decision_function(features)[0]
            
            # Convert to a normalized score where higher is more anomalous
            anomaly_score = (score * -1 + 1) / 2
            
            # Flag as anomaly if predicted by model or if values exceed thresholds
            model_prediction = turbine_model.predict(features)[0] == -1
            threshold_exceeded = (
                reading['temperature'] > ALERT_THRESHOLD['Turbine']['temperature'] or
                reading['vibration'] > ALERT_THRESHOLD['Turbine']['vibration']
            )
            
            anomaly_detected = model_prediction or threshold_exceeded
            
        elif equipment_type == 'Compressor':
            # Check if we have valid data to analyze
            if reading['pressure'] is None:
                print(f"Missing pressure data for Compressor")
                reading['anomaly_detected'] = 0
                reading['anomaly_score'] = 0.0
                return reading
            
            # Get pressure feature
            features = np.array([[reading['pressure']]])
            
            # Get anomaly score
            score = compressor_model.decision_function(features)[0]
            
            # Convert to a normalized score where higher is more anomalous
            anomaly_score = (score * -1 + 1) / 2
            
            # Flag as anomaly if predicted by model or if values exceed thresholds
            model_prediction = compressor_model.predict(features)[0] == -1
            threshold_exceeded = (
                reading['pressure'] > ALERT_THRESHOLD['Compressor']['pressure_high'] or
                reading['pressure'] < ALERT_THRESHOLD['Compressor']['pressure_low']
            )
            
            anomaly_detected = model_prediction or threshold_exceeded
            
        # Update reading with anomaly result
        reading['anomaly_detected'] = 1 if anomaly_detected else 0
        reading['anomaly_score'] = round(anomaly_score, 3)
        
        # Update alert history
        update_alert_history(reading)
        
        return reading
        
    except Exception as e:
        print(f"Error detecting anomalies: {e}")
        # Add default values in case of error
        reading['anomaly_detected'] = 0
        reading['anomaly_score'] = 0.0
        return reading

def update_alert_history(reading):
    """Update the alert history for plotting"""
    equipment_type = reading['equipment_type']
    history = alert_history[equipment_type]
    
    # Add timestamp
    timestamp = pd.to_datetime(reading['timestamp'])
    history['timestamps'].append(timestamp)
    
    # Add anomaly score
    history['anomaly_scores'].append(reading.get('anomaly_score', 0))
    
    # Add parameter values
    if equipment_type == 'Turbine':
        # Use default values if None
        temp = reading['temperature'] if reading['temperature'] is not None else 0
        vib = reading['vibration'] if reading['vibration'] is not None else 0
        history['temperature'].append(temp)
        history['vibration'].append(vib)
    else:  # Compressor
        # Use default value if None
        press = reading['pressure'] if reading['pressure'] is not None else 0
        history['pressure'].append(press)
    
    # Add alert status
    history['alerts'].append(reading.get('anomaly_detected', 0))
    
    # Keep only the last 100 readings
    if len(history['timestamps']) > 100:
        history['timestamps'] = history['timestamps'][-100:]
        history['anomaly_scores'] = history['anomaly_scores'][-100:]
        history['alerts'] = history['alerts'][-100:]
        
        if equipment_type == 'Turbine':
            history['temperature'] = history['temperature'][-100:]
            history['vibration'] = history['vibration'][-100:]
        else:  # Compressor
            history['pressure'] = history['pressure'][-100:]

def send_anomaly_alert(reading):
    """Send alert when anomaly is detected"""
    equipment_type = reading['equipment_type']
    
    # Check if this is a new alert (to avoid repeated alerts)
    if not active_alerts[equipment_type]:
        active_alerts[equipment_type] = True
        
        # Format alert message
        timestamp = pd.to_datetime(reading['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        message = f"⚠️ ANOMALY ALERT: {equipment_type}\n"
        message += f"Time: {timestamp}\n"
        
        if equipment_type == 'Turbine':
            message += f"Temperature: {reading['temperature']}°C\n"
            message += f"Vibration: {reading['vibration']} mm/s\n"
        else:  # Compressor
            message += f"Pressure: {reading['pressure']} psi\n"
        
        message += f"Anomaly Score: {reading['anomaly_score']:.3f}\n"
        message += "Immediate inspection recommended."
        
        # Print alert
        print("\n" + "!"*50)
        print(message)
        print("!"*50 + "\n")
        
        # In a real system, you would send email, SMS, etc.
        # For this example, we'll just print the alert
        
        # Update live visualization if enabled
        update_visualization()
        
        return True
    
    return False

def clear_alert(equipment_type):
    """Clear an active alert"""
    if active_alerts[equipment_type]:
        active_alerts[equipment_type] = False
        print(f"Alert cleared for {equipment_type}")
        return True
    
    return False

def update_visualization():
    """Update the real-time visualization of anomaly detection"""
    try:
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Turbine plots
        if len(alert_history['Turbine']['timestamps']) > 0:
            # Temperature subplot
            ax1 = plt.subplot(3, 2, 1)
            ax1.plot(alert_history['Turbine']['timestamps'], 
                   alert_history['Turbine']['temperature'], 'b-')
            ax1.set_title('Turbine Temperature')
            ax1.set_ylabel('Temperature (°C)')
            ax1.grid(True, alpha=0.3)
            
            # Highlight anomalies
            anomaly_indices = [i for i, alert in enumerate(alert_history['Turbine']['alerts']) if alert == 1]
            if anomaly_indices:
                anomaly_times = [alert_history['Turbine']['timestamps'][i] for i in anomaly_indices]
                anomaly_temps = [alert_history['Turbine']['temperature'][i] for i in anomaly_indices]
                ax1.scatter(anomaly_times, anomaly_temps, color='red', s=50, marker='x')
            
            # Vibration subplot
            ax2 = plt.subplot(3, 2, 2)
            ax2.plot(alert_history['Turbine']['timestamps'], 
                   alert_history['Turbine']['vibration'], 'g-')
            ax2.set_title('Turbine Vibration')
            ax2.set_ylabel('Vibration (mm/s)')
            ax2.grid(True, alpha=0.3)
            
            # Highlight anomalies
            if anomaly_indices:
                anomaly_times = [alert_history['Turbine']['timestamps'][i] for i in anomaly_indices]
                anomaly_vibs = [alert_history['Turbine']['vibration'][i] for i in anomaly_indices]
                ax2.scatter(anomaly_times, anomaly_vibs, color='red', s=50, marker='x')
            
            # Anomaly score subplot
            ax3 = plt.subplot(3, 2, 3)
            ax3.plot(alert_history['Turbine']['timestamps'], 
                   alert_history['Turbine']['anomaly_scores'], 'r-')
            ax3.set_title('Turbine Anomaly Score')
            ax3.set_ylabel('Anomaly Score')
            ax3.set_xlabel('Time')
            ax3.grid(True, alpha=0.3)
            
            # Add threshold line
            ax3.axhline(y=0.5, color='k', linestyle='--', alpha=0.7)
        
        # Compressor plots
        if len(alert_history['Compressor']['timestamps']) > 0:
            # Pressure subplot
            ax4 = plt.subplot(3, 2, 4)
            ax4.plot(alert_history['Compressor']['timestamps'], 
                   alert_history['Compressor']['pressure'], 'b-')
            ax4.set_title('Compressor Pressure')
            ax4.set_ylabel('Pressure (psi)')
            ax4.grid(True, alpha=0.3)
            
            # Highlight anomalies
            anomaly_indices = [i for i, alert in enumerate(alert_history['Compressor']['alerts']) if alert == 1]
            if anomaly_indices:
                anomaly_times = [alert_history['Compressor']['timestamps'][i] for i in anomaly_indices]
                anomaly_press = [alert_history['Compressor']['pressure'][i] for i in anomaly_indices]
                ax4.scatter(anomaly_times, anomaly_press, color='red', s=50, marker='x')
            
            # Anomaly score subplot
            ax5 = plt.subplot(3, 2, 5)
            ax5.plot(alert_history['Compressor']['timestamps'], 
                   alert_history['Compressor']['anomaly_scores'], 'r-')
            ax5.set_title('Compressor Anomaly Score')
            ax5.set_ylabel('Anomaly Score')
            ax5.set_xlabel('Time')
            ax5.grid(True, alpha=0.3)
            
            # Add threshold line
            ax5.axhline(y=0.5, color='k', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('anomaly_detection_live.png')
        plt.close('all')
        
    except Exception as e:
        print(f"Error updating visualization: {e}")

def continuous_monitoring():
    """Continuously monitor equipment data for anomalies"""
    print("Starting continuous monitoring. Checking every", CHECK_INTERVAL, "seconds.")
    print("Press Ctrl+C to stop monitoring.")
    
    try:
        while True:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"\nChecking for anomalies at {current_time}...")
            
            # Get latest data
            print("Fetching latest data from ThingSpeak...")
            reading = get_latest_data_from_thingspeak()
            
            if reading is None:
                print("No data available to check")
            else:
                # Detect anomalies
                reading = detect_anomalies(reading)
                
                if reading is None:
                    print("Error processing reading data")
                    continue
                    
                if 'anomaly_detected' not in reading:
                    print("Anomaly detection failed")
                    continue
                    
                if reading['anomaly_detected'] == 1:
                    # Send alert
                    send_anomaly_alert(reading)
                else:
                    # Clear alert if it was active
                    if active_alerts[reading['equipment_type']]:
                        clear_alert(reading['equipment_type'])
                    
                    # Display normal status
                    print(f"Equipment: {reading['equipment_type']}")
                    if reading['equipment_type'] == 'Turbine':
                        print(f"Temperature: {reading['temperature']}°C")
                        print(f"Vibration: {reading['vibration']} mm/s")
                    else:  # Compressor
                        print(f"Pressure: {reading['pressure']} psi")
                    print(f"Status: Normal (Score: {reading['anomaly_score']:.3f})")
                
                # Update visualization periodically
                if 'anomaly_detected' in reading and (reading['anomaly_detected'] == 1 or 
                                                    len(alert_history[reading['equipment_type']]['timestamps']) % 10 == 0):
                    update_visualization()
            
            print(f"\nNext check in {CHECK_INTERVAL} seconds...")
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError during monitoring: {e}")

def main():
    """Main function for the anomaly detection system"""
    print("Starting ML-based equipment anomaly detection")
    print("============================================")
    
    # Load or train models
    if not load_models():
        print("Error: Failed to load or train anomaly detection models.")
        print("Please ensure you have generated historical data with longterm.py.")
        return
    
    # Start continuous monitoring
    continuous_monitoring()

if __name__ == "__main__":
    main()
