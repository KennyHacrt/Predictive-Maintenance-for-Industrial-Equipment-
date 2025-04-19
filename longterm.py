import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest

# Historical data file
HISTORICAL_DATA_FILE = 'historical_data.csv'

def generate_historical_data(hours=24, seed=42):
    """
    Generate historical data for training models
    
    Parameters:
    - hours: Number of hours of historical data to generate
    - seed: Random seed for reproducibility
    
    Returns:
    - DataFrame with historical data
    """
    print(f"Generating {hours} hours of historical data...")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Create timestamp range
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
    
    # Initialize data lists
    data = []
    
    # Equipment parameters
    equipment_types = ['Turbine', 'Compressor']
    
    # Track maintenance and anomaly events
    last_maintenance = {
        'Turbine': start_date,
        'Compressor': start_date
    }
    
    last_anomaly = {
        'Turbine': start_date,
        'Compressor': start_date
    }
    
    # Typical maintenance intervals (minutes)
    maintenance_intervals = {
        'Turbine': 90,  # ~1.5 hours
        'Compressor': 120  # ~2 hours
    }
    
    # Base parameters
    base_params = {
        'Turbine': {
            'temperature': 65.0,  # °C
            'vibration': 1.2,     # mm/s
            'drift_rate': 0.1     # Parameter drift per minute
        },
        'Compressor': {
            'pressure': 55.0,     # psi
            'drift_rate': 0.08    # Parameter drift per minute
        }
    }
    
    # Anomaly definitions
    anomalies = {
        'Turbine': [
            {'name': 'Bearing Fault', 'temp_effect': 15, 'vib_effect': 2.5, 'duration': 15},  # 15 minutes
            {'name': 'Misalignment', 'temp_effect': 8, 'vib_effect': 1.8, 'duration': 10},
            {'name': 'Unbalance', 'temp_effect': 5, 'vib_effect': 3.0, 'duration': 12}
        ],
        'Compressor': [
            {'name': 'Valve Leak', 'pressure_effect': -10, 'duration': 12},
            {'name': 'Pipe Blockage', 'pressure_effect': 15, 'duration': 10}
        ]
    }
    
    # Active anomalies tracking
    active_anomalies = {
        'Turbine': None,
        'Compressor': None
    }
    
    # Alternate between equipment types for each timestamp
    current_equip_idx = 0
    
    # Generate data minute by minute
    for ts in timestamps:
        # Alternate between equipment types
        equip_type = equipment_types[current_equip_idx]
        current_equip_idx = (current_equip_idx + 1) % len(equipment_types)
        
        # Calculate minutes since last maintenance and anomaly
        minutes_since_maintenance = (ts - last_maintenance[equip_type]).total_seconds() / 60
        minutes_since_anomaly = (ts - last_anomaly[equip_type]).total_seconds() / 60
        
        # Initialize record
        record = {
            'timestamp': ts,
            'equipment_type': equip_type,
            'temperature': None,
            'vibration': None,
            'pressure': None,
            'is_anomaly': 0,
            'maintenance_performed': 0
        }
        
        # Check for scheduled maintenance
        maintenance_due = minutes_since_maintenance >= maintenance_intervals[equip_type]
        
        # Also add some random maintenance events
        random_maintenance = random.random() < 0.005 and minutes_since_maintenance > 30
        
        if maintenance_due or random_maintenance:
            # Perform maintenance
            record['maintenance_performed'] = 1
            last_maintenance[equip_type] = ts
            
            # Reset any active anomalies
            active_anomalies[equip_type] = None
            
            print(f"Maintenance performed on {equip_type} at {ts}")
        
        # Check for anomaly initiation
        if active_anomalies[equip_type] is None:
            # Probability increases with time since maintenance
            anomaly_prob = min(0.01, 0.002 + (minutes_since_maintenance / 500))
            
            # Reduce probability if an anomaly happened recently
            if minutes_since_anomaly < 20:
                anomaly_prob *= 0.1
            
            if random.random() < anomaly_prob:
                # Select a random anomaly type
                anomaly = random.choice(anomalies[equip_type])
                active_anomalies[equip_type] = {
                    'anomaly': anomaly,
                    'start_time': ts,
                    'remaining_minutes': anomaly['duration']
                }
                last_anomaly[equip_type] = ts
                
                print(f"Anomaly ({anomaly['name']}) started on {equip_type} at {ts}")
        
        # Generate parameter values
        if equip_type == 'Turbine':
            # Base values with drift based on time since maintenance
            drift = base_params['Turbine']['drift_rate'] * minutes_since_maintenance
            base_temp = base_params['Turbine']['temperature'] + drift * 0.5
            base_vib = base_params['Turbine']['vibration'] + drift * 0.1
            
            # Add random noise
            temp = base_temp + np.random.normal(0, 1)
            vib = base_vib + np.random.normal(0, 0.1)
            
            # Add anomaly effects if active
            if active_anomalies['Turbine'] is not None:
                anomaly_data = active_anomalies['Turbine']
                anomaly_def = anomaly_data['anomaly']
                
                # Calculate how far into the anomaly we are (0-1)
                progress = 1 - (anomaly_data['remaining_minutes'] / anomaly_def['duration'])
                
                # Apply effects that increase with time
                temp += anomaly_def['temp_effect'] * progress
                vib += anomaly_def['vib_effect'] * progress
                
                # Mark as anomaly
                record['is_anomaly'] = 1
                
                # Decrease remaining time
                anomaly_data['remaining_minutes'] -= 1
                
                # Check if anomaly is finished
                if anomaly_data['remaining_minutes'] <= 0:
                    active_anomalies['Turbine'] = None
            
            record['temperature'] = round(temp, 1)
            record['vibration'] = round(vib, 2)
        
        else:  # Compressor
            # Base values with drift
            drift = base_params['Compressor']['drift_rate'] * minutes_since_maintenance
            
            # Compressors can drift in either direction
            if not hasattr(generate_historical_data, f'{equip_type}_drift_dir'):
                setattr(generate_historical_data, f'{equip_type}_drift_dir', random.choice([-1, 1]))
            
            drift_dir = getattr(generate_historical_data, f'{equip_type}_drift_dir')
            base_pressure = base_params['Compressor']['pressure'] + drift * drift_dir
            
            # Add noise
            pressure = base_pressure + np.random.normal(0, 0.5)
            
            # Add anomaly effects if active
            if active_anomalies['Compressor'] is not None:
                anomaly_data = active_anomalies['Compressor']
                anomaly_def = anomaly_data['anomaly']
                
                # Calculate progress (0-1)
                progress = 1 - (anomaly_data['remaining_minutes'] / anomaly_def['duration'])
                
                # Apply effects
                pressure += anomaly_def['pressure_effect'] * progress
                
                # Mark as anomaly
                record['is_anomaly'] = 1
                
                # Decrease remaining time
                anomaly_data['remaining_minutes'] -= 1
                
                # Check if anomaly is finished
                if anomaly_data['remaining_minutes'] <= 0:
                    active_anomalies['Compressor'] = None
            
            record['pressure'] = round(pressure, 1)
        
        # Add record to data
        data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Print statistics
    print(f"Generated {len(df)} data points over {hours} hours")
    print(f"Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"Maintenance events: {df['maintenance_performed'].sum()}")
    
    # Count anomalies and maintenance by equipment type
    for equip in equipment_types:
        equip_data = df[df['equipment_type'] == equip]
        print(f"{equip}: {len(equip_data)} records")
        print(f"  - Anomalies: {equip_data['is_anomaly'].sum()} ({equip_data['is_anomaly'].mean()*100:.1f}%)")
        print(f"  - Maintenance events: {equip_data['maintenance_performed'].sum()}")
    
    # Save to CSV
    df.to_csv(HISTORICAL_DATA_FILE, index=False)
    print(f"Historical data saved to {HISTORICAL_DATA_FILE}")
    
    return df

def visualize_historical_data(df):
    """Create visualizations of the historical data"""
    print("Generating visualizations of historical data...")
    
    # Create directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Separate by equipment type
    turbine_data = df[df['equipment_type'] == 'Turbine'].copy()
    compressor_data = df[df['equipment_type'] == 'Compressor'].copy()
    
    # Convert timestamp to datetime if it's not already
    turbine_data['timestamp'] = pd.to_datetime(turbine_data['timestamp'])
    compressor_data['timestamp'] = pd.to_datetime(compressor_data['timestamp'])
    
    # Sort by timestamp
    turbine_data = turbine_data.sort_values('timestamp')
    compressor_data = compressor_data.sort_values('timestamp')
    
    # 1. Turbine Temperature and Vibration over time
    plt.figure(figsize=(15, 10))
    
    # Plot temperature
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(turbine_data['timestamp'], turbine_data['temperature'], 'b-', alpha=0.7)
    
    # Highlight anomalies
    anomalies = turbine_data[turbine_data['is_anomaly'] == 1]
    ax1.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', s=15, label='Anomaly')
    
    # Highlight maintenance
    maintenance = turbine_data[turbine_data['maintenance_performed'] == 1]
    for maint_time in maintenance['timestamp']:
        ax1.axvline(x=maint_time, color='green', linestyle='--', alpha=0.7)
    
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Turbine Temperature Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot vibration
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(turbine_data['timestamp'], turbine_data['vibration'], 'g-', alpha=0.7)
    
    # Highlight anomalies
    ax2.scatter(anomalies['timestamp'], anomalies['vibration'], color='red', s=15, label='Anomaly')
    
    # Highlight maintenance
    for maint_time in maintenance['timestamp']:
        ax2.axvline(x=maint_time, color='green', linestyle='--', alpha=0.7, label='Maintenance')
    
    # Only add label once
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Vibration (mm/s)')
    ax2.set_title('Turbine Vibration Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/turbine_historical.png')
    plt.close()
    
    # 2. Compressor Pressure over time
    plt.figure(figsize=(15, 6))
    
    plt.plot(compressor_data['timestamp'], compressor_data['pressure'], 'b-', alpha=0.7)
    
    # Highlight anomalies
    anomalies = compressor_data[compressor_data['is_anomaly'] == 1]
    plt.scatter(anomalies['timestamp'], anomalies['pressure'], color='red', s=15, label='Anomaly')
    
    # Highlight maintenance
    maintenance = compressor_data[compressor_data['maintenance_performed'] == 1]
    for maint_time in maintenance['timestamp']:
        plt.axvline(x=maint_time, color='green', linestyle='--', alpha=0.7)
    
    plt.xlabel('Date')
    plt.ylabel('Pressure (psi)')
    plt.title('Compressor Pressure Over Time')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(['Pressure', 'Anomaly', 'Maintenance'])
    
    plt.tight_layout()
    plt.savefig('visualizations/compressor_historical.png')
    plt.close()
    
    # 3. Scatter plot for Turbine (Temperature vs Vibration)
    plt.figure(figsize=(10, 8))
    
    plt.scatter(turbine_data[turbine_data['is_anomaly'] == 0]['temperature'], 
               turbine_data[turbine_data['is_anomaly'] == 0]['vibration'], 
               alpha=0.5, color='blue', label='Normal')
    
    plt.scatter(turbine_data[turbine_data['is_anomaly'] == 1]['temperature'], 
               turbine_data[turbine_data['is_anomaly'] == 1]['vibration'], 
               alpha=0.7, color='red', s=20, label='Anomaly')
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Vibration (mm/s)')
    plt.title('Turbine: Temperature vs Vibration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/turbine_scatter.png')
    plt.close()
    
    # 4. Distribution of normal vs anomaly values
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Temperature distribution
    ax1.hist(turbine_data[turbine_data['is_anomaly'] == 0]['temperature'], bins=20, alpha=0.5, color='blue', label='Normal')
    ax1.hist(turbine_data[turbine_data['is_anomaly'] == 1]['temperature'], bins=20, alpha=0.5, color='red', label='Anomaly')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Turbine Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Vibration distribution
    ax2.hist(turbine_data[turbine_data['is_anomaly'] == 0]['vibration'], bins=20, alpha=0.5, color='blue', label='Normal')
    ax2.hist(turbine_data[turbine_data['is_anomaly'] == 1]['vibration'], bins=20, alpha=0.5, color='red', label='Anomaly')
    ax2.set_xlabel('Vibration (mm/s)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Turbine Vibration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Pressure distribution
    ax3.hist(compressor_data[compressor_data['is_anomaly'] == 0]['pressure'], bins=20, alpha=0.5, color='blue', label='Normal')
    ax3.hist(compressor_data[compressor_data['is_anomaly'] == 1]['pressure'], bins=20, alpha=0.5, color='red', label='Anomaly')
    ax3.set_xlabel('Pressure (psi)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Compressor Pressure')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/parameter_distributions.png')
    plt.close()
    
    print("Visualizations saved to 'visualizations' directory.")

def train_anomaly_detection_models(df):
    """Train anomaly detection models using the historical data"""
    print("Training anomaly detection models...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Separate by equipment type
    turbine_data = df[df['equipment_type'] == 'Turbine'].copy()
    compressor_data = df[df['equipment_type'] == 'Compressor'].copy()
    
    # Turbine model (using temperature and vibration)
    X_turbine = turbine_data[['temperature', 'vibration']].values
    y_turbine = turbine_data['is_anomaly'].values
    
    # Calculate contamination (proportion of anomalies)
    contamination_turbine = y_turbine.mean()
    if contamination_turbine == 0:
        contamination_turbine = 0.05  # Default if no anomalies in training data
    
    # Train Isolation Forest for turbine
    turbine_model = IsolationForest(
        n_estimators=100,
        contamination=float(contamination_turbine),
        random_state=42
    )
    turbine_model.fit(X_turbine)
    
    # Save turbine model
    import joblib
    joblib.dump(turbine_model, 'models/turbine_anomaly_model.joblib')
    
    # Compressor model (using pressure)
    X_compressor = compressor_data[['pressure']].values
    y_compressor = compressor_data['is_anomaly'].values
    
    # Calculate contamination
    contamination_compressor = y_compressor.mean()
    if contamination_compressor == 0:
        contamination_compressor = 0.05  # Default if no anomalies in training data
    
    # Train Isolation Forest for compressor
    compressor_model = IsolationForest(
        n_estimators=100,
        contamination=float(contamination_compressor),
        random_state=42
    )
    compressor_model.fit(X_compressor)
    
    # Save compressor model
    joblib.dump(compressor_model, 'models/compressor_anomaly_model.joblib')
    
    print("Anomaly detection models trained and saved to 'models' directory.")

def main():
    """Main function to generate and process historical data"""
    print("Industrial Equipment Predictive Maintenance - Historical Data Generator")
    print("=====================================================================")
    
    # Ask how many hours of historical data to generate
    hours_input = input("Enter number of hours of historical data to generate (default 24): ")
    hours = int(hours_input) if hours_input.strip() else 24
    
    # Generate historical data
    df = generate_historical_data(hours=hours)
    
    # Visualize the data
    visualize_historical_data(df)
    
    # Train anomaly detection models
    train_anomaly_detection_models(df)
    
    print("\nHistorical data generation complete.")
    print("You can now run sensor.py to start the real-time simulation.")

if __name__ == "__main__":
    main()
