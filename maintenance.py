import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import joblib
import requests
from sklearn.ensemble import RandomForestRegressor
import threading

# Set matplotlib to use 'Agg' backend (non-interactive) to avoid threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ThingSpeak credentials
CHANNEL_ID = "2925292"
READ_API_KEY = "8M3SFRL42ZBOIGE4"
WRITE_API_KEY = "79Z8SMIMWPCJT4BM"

# Global models
maintenance_models = {
    'Turbine': None,
    'Compressor': None
}

# Historical data file
HISTORICAL_DATA_FILE = 'historical_data.csv'

# Check frequency (seconds)
CHECK_INTERVAL = 20

# Prediction data
maintenance_predictions = {
    'Turbine': {
        'predicted_minutes': None,
        'last_maintenance': None,
        'last_prediction_time': None,
        'predicted_date': None,
        'features': None
    },
    'Compressor': {
        'predicted_minutes': None,
        'last_maintenance': None,
        'last_prediction_time': None,
        'predicted_date': None,
        'features': None
    }
}

# Prediction history for plotting
prediction_history = {
    'Turbine': {
        'timestamps': [],
        'predicted_minutes': [],
        'maintenance_due': []
    },
    'Compressor': {
        'timestamps': [],
        'predicted_minutes': [],
        'maintenance_due': []
    }
}

def extract_maintenance_features(df, equipment_type):
    """
    Extract features for maintenance prediction.
    For each maintenance event, extract:
    - Minutes since last maintenance
    - Number of anomalies since last maintenance
    - Average parameters (temp, vibration, pressure) since last maintenance
    - Max parameters since last maintenance
    """
    # Filter for the specified equipment
    equip_data = df[df['equipment_type'] == equipment_type].copy()
    
    # Sort by timestamp
    equip_data['timestamp'] = pd.to_datetime(equip_data['timestamp'])
    equip_data = equip_data.sort_values('timestamp')
    
    # Find all maintenance events
    maintenance_events = equip_data[equip_data['maintenance_performed'] == 1]
    
    # If we don't have enough maintenance events, generate synthetic ones
    if len(maintenance_events) < 5:
        print(f"Warning: Not enough maintenance events for {equipment_type}")
        print(f"Found only {len(maintenance_events)} events, need at least 5")
        print(f"Adding synthetic data for {equipment_type} model training")
        
        # Generate synthetic features and targets
        X, y = generate_synthetic_maintenance_data(equipment_type)
        return X, y
    
    # Prepare lists for features and target
    features = []
    targets = []
    
    # Set of parameters to track
    if equipment_type == 'Turbine':
        params = ['temperature', 'vibration']
    else:  # Compressor
        params = ['pressure']
    
    # Track the start of the current period (last maintenance or beginning of data)
    period_start = equip_data['timestamp'].min()
    
    # Process each maintenance event
    for i, event in enumerate(maintenance_events.itertuples()):
        # Skip the first maintenance event (we need a previous one to calculate features)
        if i == 0:
            period_start = event.timestamp
            continue
        
        # Get data since last maintenance
        period_data = equip_data[(equip_data['timestamp'] > period_start) & 
                                (equip_data['timestamp'] <= event.timestamp)]
        
        # Calculate time since last maintenance in minutes
        minutes_since_last = (event.timestamp - period_start).total_seconds() / 60
        
        # If period is too short, skip this event
        if minutes_since_last < 1:
            period_start = event.timestamp
            continue
        
        # Count anomalies in this period
        num_anomalies = period_data['is_anomaly'].sum()
        
        # Calculate features for each parameter
        param_features = [minutes_since_last, num_anomalies]
        
        for param in params:
            # Skip if all values are NaN
            if pd.isna(period_data[param]).all():
                param_features.extend([0, 0, 0])
                continue
                
            # Calculate statistics
            param_avg = period_data[param].mean()
            param_max = period_data[param].max()
            param_std = period_data[param].std()
            
            # Handle NaN (if not enough data)
            param_avg = 0 if pd.isna(param_avg) else param_avg
            param_max = 0 if pd.isna(param_max) else param_max
            param_std = 0 if pd.isna(param_std) else param_std
            
            param_features.extend([param_avg, param_max, param_std])
        
        # Add rate of anomalies
        anomaly_rate = num_anomalies / len(period_data) if len(period_data) > 0 else 0
        param_features.append(anomaly_rate)
        
        features.append(param_features)
        targets.append(minutes_since_last)
        
        # Update period start for next iteration
        period_start = event.timestamp
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(targets)
    
    return X, y

def generate_synthetic_maintenance_data(equipment_type):
    """Generate synthetic maintenance data for training"""
    # Number of data points to generate
    n_samples = 10
    
    if equipment_type == 'Turbine':
        # Features: [minutes_since_last, num_anomalies, avg_temp, max_temp, std_temp, avg_vib, max_vib, std_vib, anomaly_rate]
        X = np.zeros((n_samples, 9))
        
        # Generate maintenance intervals (minutes) - target
        y = np.linspace(60, 120, n_samples) + np.random.normal(0, 5, n_samples)
        
        for i in range(n_samples):
            X[i, 0] = y[i]  # minutes since last maintenance
            X[i, 1] = np.random.randint(0, 4)  # number of anomalies
            X[i, 2] = 65 + i * 0.5  # avg temp
            X[i, 3] = 70 + i * 0.7  # max temp
            X[i, 4] = 2 + i * 0.1  # std temp
            X[i, 5] = 1.2 + i * 0.05  # avg vibration
            X[i, 6] = 1.8 + i * 0.1  # max vibration
            X[i, 7] = 0.2 + i * 0.02  # std vibration
            X[i, 8] = X[i, 1] / y[i]  # anomaly rate
    
    else:  # Compressor
        # Features: [minutes_since_last, num_anomalies, avg_pressure, max_pressure, std_pressure, anomaly_rate]
        X = np.zeros((n_samples, 6))
        
        # Generate maintenance intervals (minutes) - target
        y = np.linspace(80, 150, n_samples) + np.random.normal(0, 7, n_samples)
        
        for i in range(n_samples):
            X[i, 0] = y[i]  # minutes since last maintenance
            X[i, 1] = np.random.randint(0, 3)  # number of anomalies
            X[i, 2] = 55 + i * 0.3  # avg pressure
            X[i, 3] = 60 + i * 0.4  # max pressure
            X[i, 4] = 3 + i * 0.1  # std pressure
            X[i, 5] = X[i, 1] / y[i]  # anomaly rate
    
    print(f"Generated {n_samples} synthetic data points for {equipment_type}")
    return X, y

def train_maintenance_models():
    """Train models to predict time until next maintenance is needed"""
    print("Training maintenance prediction models...")
    
    # Load historical data if it exists
    if os.path.exists(HISTORICAL_DATA_FILE):
        df = pd.read_csv(HISTORICAL_DATA_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df)} records from historical data")
    else:
        # Create a dummy dataframe with minimal structure
        df = pd.DataFrame(columns=['timestamp', 'equipment_type', 'temperature', 
                                 'vibration', 'pressure', 'is_anomaly', 'maintenance_performed'])
        print("No historical data found, using synthetic data")
    
    # Train models for each equipment type
    success = False
    
    for equipment_type in ['Turbine', 'Compressor']:
        # Extract features or generate synthetic ones
        X, y = extract_maintenance_features(df, equipment_type)
        
        # Train RandomForest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X, y)
        
        # Save to global variable and disk
        maintenance_models[equipment_type] = model
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save model
        joblib.dump(model, f'models/{equipment_type.lower()}_maintenance_model.joblib')
        
        # Plot feature importance if we have the right number of features
        if equipment_type == 'Turbine' and X.shape[1] == 9:
            feature_names = ['Minutes Since Last', 'Num Anomalies', 
                           'Avg Temp', 'Max Temp', 'Std Temp',
                           'Avg Vib', 'Max Vib', 'Std Vib', 'Anomaly Rate']
            plt.figure(figsize=(10, 6))
            plt.barh(feature_names, model.feature_importances_)
            plt.xlabel('Feature Importance')
            plt.title(f'{equipment_type} Maintenance Prediction - Feature Importance')
            plt.tight_layout()
            plt.savefig(f'{equipment_type.lower()}_feature_importance.png')
            plt.close('all')
            
        elif equipment_type == 'Compressor' and X.shape[1] == 6:
            feature_names = ['Minutes Since Last', 'Num Anomalies', 
                           'Avg Pressure', 'Max Pressure', 'Std Pressure', 'Anomaly Rate']
            plt.figure(figsize=(10, 6))
            plt.barh(feature_names, model.feature_importances_)
            plt.xlabel('Feature Importance')
            plt.title(f'{equipment_type} Maintenance Prediction - Feature Importance')
            plt.tight_layout()
            plt.savefig(f'{equipment_type.lower()}_feature_importance.png')
            plt.close('all')
        
        success = True
    
    if success:
        print("Maintenance prediction models trained and saved")
    else:
        print("Error: Failed to create maintenance prediction models.")
    
    return success

def load_models():
    """Load the trained maintenance prediction models"""
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Check if models exist
        turbine_model_path = 'models/turbine_maintenance_model.joblib'
        compressor_model_path = 'models/compressor_maintenance_model.joblib'
        
        if (os.path.exists(turbine_model_path) and os.path.exists(compressor_model_path)):
            print("Loading existing maintenance prediction models...")
            
            maintenance_models['Turbine'] = joblib.load(turbine_model_path)
            maintenance_models['Compressor'] = joblib.load(compressor_model_path)
            
            print("Maintenance prediction models loaded successfully")
            return True
        
        else:
            print("Missing maintenance prediction models. Training new models...")
            return train_maintenance_models()
            
    except Exception as e:
        print(f"Error loading maintenance models: {e}")
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

def get_historical_readings(equipment_type, minutes=60):
    """Get historical readings from the last X minutes"""
    try:
        # ThingSpeak API endpoint for historical data
        url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
        
        # Calculate start time (X minutes ago)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)
        
        # Format times for API
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Parameters for API request
        params = {
            'api_key': READ_API_KEY,
            'start': start_str,
            'end': end_str,
            'results': 1000  # Maximum number of results
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'feeds' not in data or len(data['feeds']) == 0:
                print(f"No historical data available for {equipment_type}")
                return None
            
            # Process the feeds
            readings = []
            for feed in data['feeds']:
                # Check if field5 is this equipment type
                if feed['field5'] == equipment_type:
                    reading = {
                        'timestamp': feed['created_at'],
                        'temperature': float(feed['field1']) if feed['field1'] else None,
                        'pressure': float(feed['field2']) if feed['field2'] else None,
                        'vibration': float(feed['field3']) if feed['field3'] else None,
                        'humidity': float(feed['field4']) if feed['field4'] else None,
                        'equipment_type': feed['field5'],
                        'location': feed['field6'],
                        'anomaly_score': float(feed['field7']) if feed['field7'] else 0,
                        'status': feed['field8'],
                        'anomaly_status': 1 if feed['field8'] == "Faulty" else 0
                    }
                    
                    readings.append(reading)
            
            print(f"Found {len(readings)} historical readings for {equipment_type}")
            return readings
            
        else:
            print(f"Error getting historical data: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def find_last_maintenance(equipment_type):
    """Find the timestamp of the last maintenance for an equipment type"""
    try:
        # Check historical readings for field5=equipment_type and field8="Maintenance"
        readings = get_historical_readings(equipment_type, minutes=120)  # Check last 2 hours
        
        if readings:
            # Sort by timestamp
            readings_with_dt = []
            for reading in readings:
                reading_copy = reading.copy()
                reading_copy['timestamp'] = pd.to_datetime(reading['timestamp'])
                readings_with_dt.append(reading_copy)
            
            sorted_readings = sorted(readings_with_dt, key=lambda x: x['timestamp'])
            
            # Look for status changes from "Faulty" to "Normal" as potential maintenance events
            for i in range(1, len(sorted_readings)):
                if (sorted_readings[i-1]['status'] == "Faulty" and 
                    sorted_readings[i]['status'] == "Normal"):
                    # Found a maintenance event
                    last_maint = sorted_readings[i]['timestamp']
                    print(f"Found maintenance event for {equipment_type} at {last_maint}")
                    return last_maint
        
        # If we haven't found maintenance, use a default time (30 minutes ago)
        default_time = datetime.now() - timedelta(minutes=30)
        print(f"No maintenance history found for {equipment_type}. Using default (30 minutes ago).")
        return default_time
        
    except Exception as e:
        print(f"Error finding last maintenance: {e}")
        # Default to 30 minutes ago
        return datetime.now() - timedelta(minutes=30)

def extract_features_for_prediction(equipment_type, current_reading, last_maintenance_time):
    """Extract features from current reading and history for prediction"""
    # Get historical readings since last maintenance
    minutes_since_maint = (datetime.now() - last_maintenance_time).total_seconds() / 60
    lookup_minutes = max(60, int(minutes_since_maint * 1.2))  # Look back at least 60 minutes
    
    history = get_historical_readings(equipment_type, minutes=lookup_minutes)
    
    if not history:
        print(f"No historical data found for {equipment_type}. Using default features.")
        # Use default features
        if equipment_type == 'Turbine':
            return [minutes_since_maint, 0, current_reading['temperature'], current_reading['temperature'], 2, 
                   current_reading['vibration'], current_reading['vibration'], 0.2, 0]
        else:  # Compressor
            return [minutes_since_maint, 0, current_reading['pressure'], current_reading['pressure'], 1, 0]
    
    # Convert timestamps to datetime
    history_with_dt = []
    for reading in history:
        reading_copy = reading.copy()
        reading_copy['timestamp'] = pd.to_datetime(reading['timestamp'])
        history_with_dt.append(reading_copy)
    
    # Filter readings since last maintenance
    last_maint_dt = pd.to_datetime(last_maintenance_time)
    readings_since_maint = [r for r in history_with_dt if r['timestamp'] > last_maint_dt]
    
    # If no readings since maintenance, use all available readings
    if not readings_since_maint and history_with_dt:
        readings_since_maint = history_with_dt
    
    # Count anomalies
    num_anomalies = sum(1 for r in readings_since_maint if r['anomaly_status'] == 1)
    
    # Prepare features
    if equipment_type == 'Turbine':
        # Extract temperature and vibration values
        temp_values = [r['temperature'] for r in readings_since_maint if r['temperature'] is not None]
        vib_values = [r['vibration'] for r in readings_since_maint if r['vibration'] is not None]
        
        # Handle empty lists
        if not temp_values:
            temp_values = [current_reading['temperature']]
        if not vib_values:
            vib_values = [current_reading['vibration']]
        
        # Calculate statistics
        avg_temp = sum(temp_values) / len(temp_values)
        max_temp = max(temp_values)
        std_temp = np.std(temp_values) if len(temp_values) > 1 else 1.0
        
        avg_vib = sum(vib_values) / len(vib_values)
        max_vib = max(vib_values)
        std_vib = np.std(vib_values) if len(vib_values) > 1 else 0.2
        
        # Calculate anomaly rate
        anomaly_rate = num_anomalies / len(readings_since_maint) if readings_since_maint else 0
        
        # Return feature vector
        return [minutes_since_maint, num_anomalies, 
               avg_temp, max_temp, std_temp, 
               avg_vib, max_vib, std_vib, 
               anomaly_rate]
    
    else:  # Compressor
        # Extract pressure values
        pressure_values = [r['pressure'] for r in readings_since_maint if r['pressure'] is not None]
        
        # Handle empty list
        if not pressure_values:
            pressure_values = [current_reading['pressure']]
        
        # Calculate statistics
        avg_pressure = sum(pressure_values) / len(pressure_values)
        max_pressure = max(pressure_values)
        std_pressure = np.std(pressure_values) if len(pressure_values) > 1 else 1.0
        
        # Calculate anomaly rate
        anomaly_rate = num_anomalies / len(readings_since_maint) if readings_since_maint else 0
        
        # Return feature vector
        return [minutes_since_maint, num_anomalies, 
               avg_pressure, max_pressure, std_pressure, 
               anomaly_rate]

def predict_maintenance(equipment_type, current_reading):
    """Predict minutes until maintenance is needed"""
    # Get the appropriate model
    model = maintenance_models[equipment_type]
    
    if model is None:
        print(f"No maintenance prediction model available for {equipment_type}")
        return None
    
    # Get last maintenance time (if not already known)
    if maintenance_predictions[equipment_type]['last_maintenance'] is None:
        maintenance_predictions[equipment_type]['last_maintenance'] = find_last_maintenance(equipment_type)
    
    last_maintenance = maintenance_predictions[equipment_type]['last_maintenance']
    
    # Check if maintenance was just performed (status change from "Faulty" to "Normal")
    if current_reading['status'] == "Normal" and maintenance_predictions[equipment_type].get('last_status') == "Faulty":
        print(f"Detected maintenance on {equipment_type}. Resetting prediction.")
        maintenance_predictions[equipment_type]['last_maintenance'] = datetime.now()
        last_maintenance = maintenance_predictions[equipment_type]['last_maintenance']
    
    # Update last status
    maintenance_predictions[equipment_type]['last_status'] = current_reading['status']
    
    # Extract features for prediction
    features = extract_features_for_prediction(equipment_type, current_reading, last_maintenance)
    
    # Store features for later use
    maintenance_predictions[equipment_type]['features'] = features
    
    # Calculate minutes since last maintenance
    minutes_since = features[0]  # First feature is minutes since maintenance
    
    # Make prediction
    try:
        features_array = np.array(features).reshape(1, -1)
        predicted_total_runtime = model.predict(features_array)[0]
        
        # Calculate remaining time
        remaining_minutes = max(0, predicted_total_runtime - minutes_since)
        
        # Store prediction
        maintenance_predictions[equipment_type]['predicted_minutes'] = remaining_minutes
        maintenance_predictions[equipment_type]['last_prediction_time'] = datetime.now()
        
        # Calculate predicted date
        current_time = datetime.now()
        predicted_date = current_time + timedelta(minutes=remaining_minutes)
        maintenance_predictions[equipment_type]['predicted_date'] = predicted_date
        
        # Calculate maintenance due percentage
        if predicted_total_runtime > 0:
            maintenance_due = (minutes_since / predicted_total_runtime) * 100
        else:
            maintenance_due = 100
        
        # Update prediction history
        prediction_history[equipment_type]['timestamps'].append(current_time)
        prediction_history[equipment_type]['predicted_minutes'].append(remaining_minutes)
        prediction_history[equipment_type]['maintenance_due'].append(maintenance_due)
        
        # Keep only the last 100 predictions
        if len(prediction_history[equipment_type]['timestamps']) > 100:
            prediction_history[equipment_type]['timestamps'] = prediction_history[equipment_type]['timestamps'][-100:]
            prediction_history[equipment_type]['predicted_minutes'] = prediction_history[equipment_type]['predicted_minutes'][-100:]
            prediction_history[equipment_type]['maintenance_due'] = prediction_history[equipment_type]['maintenance_due'][-100:]
        
        return remaining_minutes
        
    except Exception as e:
        print(f"Error making maintenance prediction: {e}")
        return None

def update_visualization():
    """Update the real-time visualization of maintenance predictions"""
    try:
        # Create a figure with subplots
        plt.figure(figsize=(12, 10))
        
        # Create subplots without using the returned figure and axes
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        
        # Turbine predictions
        if maintenance_predictions['Turbine']['predicted_minutes'] is not None:
            timestamps = prediction_history['Turbine']['timestamps']
            if timestamps:
                predicted_minutes = prediction_history['Turbine']['predicted_minutes']
                maintenance_due = prediction_history['Turbine']['maintenance_due']
                
                # Convert timestamps to relative minutes for x-axis
                if len(timestamps) > 1:
                    start_time = timestamps[0]
                    relative_minutes = [(t - start_time).total_seconds() / 60 for t in timestamps]
                    
                    # Plot predicted minutes
                    ax1.plot(relative_minutes, predicted_minutes, 'b-', label='Predicted Minutes')
                    
                    # Plot maintenance due percentage
                    ax1_due = ax1.twinx()
                    ax1_due.plot(relative_minutes, maintenance_due, 'r-', label='Maintenance Due %')
                    ax1_due.set_ylabel('Maintenance Due (%)', color='r')
                    ax1_due.tick_params(axis='y', labelcolor='r')
                    ax1_due.set_ylim(0, 100)
                    
                    # Add labels and legend
                    ax1.set_title('Turbine Maintenance Prediction')
                    ax1.set_xlabel('Minutes Since First Prediction')
                    ax1.set_ylabel('Minutes Until Maintenance', color='b')
                    ax1.tick_params(axis='y', labelcolor='b')
                    ax1.grid(True, alpha=0.3)
                    
                    # Create legend
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax1_due.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.text(0.5, 0.5, 'No Turbine Prediction Data Available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Turbine Maintenance Prediction')
        
        # Compressor predictions
        if maintenance_predictions['Compressor']['predicted_minutes'] is not None:
            timestamps = prediction_history['Compressor']['timestamps']
            if timestamps:
                predicted_minutes = prediction_history['Compressor']['predicted_minutes']
                maintenance_due = prediction_history['Compressor']['maintenance_due']
                
                # Convert timestamps to relative minutes for x-axis
                if len(timestamps) > 1:
                    start_time = timestamps[0]
                    relative_minutes = [(t - start_time).total_seconds() / 60 for t in timestamps]
                    
                    # Plot predicted minutes
                    ax2.plot(relative_minutes, predicted_minutes, 'b-', label='Predicted Minutes')
                    
                    # Plot maintenance due percentage
                    ax2_due = ax2.twinx()
                    ax2_due.plot(relative_minutes, maintenance_due, 'r-', label='Maintenance Due %')
                    ax2_due.set_ylabel('Maintenance Due (%)', color='r')
                    ax2_due.tick_params(axis='y', labelcolor='r')
                    ax2_due.set_ylim(0, 100)
                    
                    # Add labels and legend
                    ax2.set_title('Compressor Maintenance Prediction')
                    ax2.set_xlabel('Minutes Since First Prediction')
                    ax2.set_ylabel('Minutes Until Maintenance', color='b')
                    ax2.tick_params(axis='y', labelcolor='b')
                    ax2.grid(True, alpha=0.3)
                    
                    # Create legend
                    lines1, labels1 = ax2.get_legend_handles_labels()
                    lines2, labels2 = ax2_due.get_legend_handles_labels()
                    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax2.text(0.5, 0.5, 'No Compressor Prediction Data Available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Compressor Maintenance Prediction')
        
        plt.tight_layout()
        plt.savefig('maintenance_prediction.png')
        plt.close('all')  # Close all figures to prevent memory leak
        
    except Exception as e:
        print(f"Error updating visualization: {e}")

def display_maintenance_countdown():
    """Display a continuous countdown to next maintenance"""
    # Print header
    print("\n" + "="*50)
    print("MAINTENANCE PREDICTION COUNTDOWN")
    print("="*50)
    
    # Check each equipment type
    for equipment_type in ['Turbine', 'Compressor']:
        prediction = maintenance_predictions[equipment_type]
        
        if prediction['predicted_minutes'] is not None:
            # Calculate time remaining
            minutes_remaining = prediction['predicted_minutes']
            
            # Calculate predicted date in real-time
            predicted_date = prediction['predicted_date']
            
            # Calculate hours and minutes
            hours = int(minutes_remaining / 60)
            minutes = int(minutes_remaining % 60)
            
            # Get prediction confidence based on model (placeholder)
            confidence = "Medium" if minutes_remaining > 30 else "High" if minutes_remaining < 10 else "Low"
            
            # Print countdown
            print(f"\n{equipment_type} Maintenance Prediction:")
            print(f"  Time until next maintenance: {minutes_remaining:.1f} minutes ({hours} hours, {minutes} minutes)")
            print(f"  Predicted date: {predicted_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Confidence: {confidence}")
            print(f"  Last updated: {prediction['last_prediction_time'].strftime('%H:%M:%S')}")
        else:
            print(f"\n{equipment_type} Maintenance Prediction:")
            print("  No prediction available yet")
    
    print("\n" + "="*50)
    print("Press Ctrl+C to stop monitoring")
    print("="*50)

def countdown_display_thread():
    """Thread function to periodically update the countdown display"""
    try:
        while True:
            # Clear screen for better readability
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Display countdown
            display_maintenance_countdown()
            
            # Update visualization
            update_visualization()
            
            # Wait before updating again
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nCountdown display stopped by user")
    except Exception as e:
        print(f"\nError in countdown display: {e}")

def continuous_prediction():
    """Continuously check for new data and update predictions"""
    print("Starting continuous maintenance prediction. Checking every", CHECK_INTERVAL, "seconds.")
    print("Press Ctrl+C to stop.")
    
    # Start the display thread
    display_thread = threading.Thread(target=countdown_display_thread)
    display_thread.daemon = True
    display_thread.start()
    
    try:
        while True:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"\nUpdating maintenance prediction at {current_time}...")
            
            # Get latest data
            reading = get_latest_data_from_thingspeak()
            
            if reading is None:
                print("No data available to check")
            else:
                # Check if this equipment type has a model
                equipment_type = reading['equipment_type']
                if maintenance_models[equipment_type] is None:
                    print(f"No maintenance prediction model available for {equipment_type}")
                else:
                    # Make prediction
                    remaining_minutes = predict_maintenance(equipment_type, reading)
                    
                    if remaining_minutes is not None:
                        print(f"{equipment_type}: Predicted {remaining_minutes:.1f} minutes until maintenance needed")
            
            # Wait for next check
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\nMaintenance prediction stopped by user")
    except Exception as e:
        print(f"\nError during maintenance prediction: {e}")

def main():
    """Main function for the maintenance prediction system"""
    print("Industrial Equipment Maintenance Prediction")
    print("==========================================")
    
    # Load or train models
    if not load_models():
        print("Error: Could not load or train maintenance prediction models.")
        print("Please ensure you have enough historical data with maintenance events.")
        return
    
    # Start continuous prediction
    continuous_prediction()

if __name__ == "__main__":
    main()
