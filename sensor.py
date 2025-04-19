import requests
import time
import random
import math
import numpy as np
from datetime import datetime, timedelta
import os

# ThingSpeak credentials
CHANNEL_ID = "2925292"
READ_API_KEY = "8M3SFRL42ZBOIGE4"
WRITE_API_KEY = "79Z8SMIMWPCJT4BM"

# Simulation parameters
UPDATE_INTERVAL = 15  # seconds between sensor readings
SIMULATE_ANOMALIES = True

# Equipment initial state
equipment_state = {
    'Turbine': {
        'temperature': 65.0,  # °C
        'vibration': 1.2,  # mm/s
        'active_anomaly': None,
        'anomaly_countdown': 0,
        'maintenance_due': 0.0,  # percentage
    },
    'Compressor': {
        'pressure': 55.0,  # psi
        'active_anomaly': None,
        'anomaly_countdown': 0,
        'maintenance_due': 0.0,  # percentage
    }
}

# Anomaly types and their effects
anomaly_types = {
    'Turbine': [
        {
            'name': 'Bearing Fault',
            'probability': 0.02,  # chance per check
            'effects': {'temperature': 15, 'vibration': 2.5},
            'duration': 12,  # readings until critical if not fixed
        },
        {
            'name': 'Misalignment',
            'probability': 0.015,
            'effects': {'temperature': 8, 'vibration': 1.8},
            'duration': 15,
        },
        {
            'name': 'Unbalance',
            'probability': 0.01,
            'effects': {'temperature': 5, 'vibration': 3.0},
            'duration': 20,
        }
    ],
    'Compressor': [
        {
            'name': 'Valve Leak',
            'probability': 0.015,
            'effects': {'pressure': -10},
            'duration': 18,
        },
        {
            'name': 'Pipe Blockage',
            'probability': 0.01,
            'effects': {'pressure': 15},
            'duration': 14,
        }
    ]
}

# Start time for tracking
START_TIME = datetime.now()

# Initialize last maintenance times
LAST_MAINTENANCE = {
    'Turbine': START_TIME - timedelta(minutes=30),  # 30 minutes ago
    'Compressor': START_TIME - timedelta(minutes=20)  # 20 minutes ago
}

# Historical data file
HISTORICAL_DATA_FILE = 'historical_data.csv'

def maintenance_probability(minutes_since_maintenance, equipment_type):
    """Calculate probability that maintenance is needed based on time"""
    if equipment_type == 'Turbine':
        # Turbines typically need maintenance more frequently
        baseline = max(0, min(100, (minutes_since_maintenance / 120) * 100))
    else:  # Compressor
        baseline = max(0, min(100, (minutes_since_maintenance / 150) * 100))
    
    # Add randomness
    variation = random.uniform(-5, 5)
    return max(0, min(100, baseline + variation))

def generate_reading(equipment_type):
    """Generate a single sensor reading with potential anomalies"""
    state = equipment_state[equipment_type]
    minutes_since_maintenance = (datetime.now() - LAST_MAINTENANCE[equipment_type]).total_seconds() / 60
    
    # Update maintenance due percentage
    state['maintenance_due'] = maintenance_probability(minutes_since_maintenance, equipment_type)
    
    # Initialize reading with defaults
    reading = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'equipment_type': equipment_type,
        'temperature': None,
        'vibration': None,
        'pressure': None,
        'anomaly_status': 0,
        'maintenance_performed': 0,
        'maintenance_due': state['maintenance_due'],
        'anomaly_score': 0.0  # Initialize anomaly score
    }
    
    # Check if there's an active anomaly
    if state['active_anomaly'] is not None:
        anomaly = state['active_anomaly']
        state['anomaly_countdown'] -= 1
        
        # If anomaly countdown reaches zero, it becomes critical
        if state['anomaly_countdown'] <= 0:
            print(f"⚠️ CRITICAL: {anomaly['name']} on {equipment_type} has reached critical level!")
            # Reset after critical (simulates automatic emergency shutdown/reset)
            state['active_anomaly'] = None
            perform_maintenance(equipment_type)
            reading['maintenance_performed'] = 1
        else:
            # Apply intensifying anomaly effects (gets worse over time)
            progress = 1 - (state['anomaly_countdown'] / anomaly['duration'])
            reading['anomaly_status'] = 1
            reading['anomaly_score'] = min(0.99, progress * 0.9)  # Scale from 0 to 0.9
    
    # Check for new anomalies if none active and simulation allows
    elif SIMULATE_ANOMALIES:
        # Get anomaly candidates for this equipment
        anomaly_candidates = anomaly_types[equipment_type]
        
        # Check each potential anomaly type
        for anomaly in anomaly_candidates:
            # Roll for anomaly
            if random.random() < anomaly['probability']:
                # Activate the anomaly
                state['active_anomaly'] = anomaly
                state['anomaly_countdown'] = anomaly['duration']
                
                print(f"⚠️ {anomaly['name']} anomaly started on {equipment_type}!")
                print(f"  - Anomaly occurred after {minutes_since_maintenance:.1f} minutes since last maintenance")
                
                reading['anomaly_status'] = 1
                reading['anomaly_score'] = 0.1  # Starting anomaly score
                break
    
    # Generate sensor values based on equipment type and state
    if equipment_type == 'Turbine':
        # Normal ranges with random variation
        base_temp = 65 + random.uniform(-2, 2)
        base_vibration = 1.2 + random.uniform(-0.2, 0.2)
        
        # Add maintenance effects (parameters degrade over time)
        maint_factor = minutes_since_maintenance / 180  # 3 hours for full effect
        temp_drift = 10 * min(1, maint_factor)  # Max +10°C drift
        vibration_drift = 1 * min(1, maint_factor)  # Max +1mm/s drift
        
        base_temp += temp_drift
        base_vibration += vibration_drift
        
        # Add anomaly effects if active
        if state['active_anomaly'] is not None:
            anomaly = state['active_anomaly']
            progress = 1 - (state['anomaly_countdown'] / anomaly['duration'])
            
            # Effects intensify as countdown decreases
            temp_effect = anomaly['effects'].get('temperature', 0) * progress
            vib_effect = anomaly['effects'].get('vibration', 0) * progress
            
            base_temp += temp_effect
            base_vibration += vib_effect
        
        # Set final values with random noise
        reading['temperature'] = round(base_temp + random.uniform(-0.5, 0.5), 1)
        reading['vibration'] = round(base_vibration + random.uniform(-0.1, 0.1), 2)
        
    else:  # Compressor
        # Normal range with random variation
        base_pressure = 55 + random.uniform(-2, 2)
        
        # Add maintenance effects
        maint_factor = minutes_since_maintenance / 210  # 3.5 hours for full effect
        pressure_drift = 5 * min(1, maint_factor)  # Drift by up to 5 psi
        
        # Pressure can drift up or down based on compressor type (randomize)
        if not hasattr(generate_reading, 'drift_direction'):
            generate_reading.drift_direction = random.choice([-1, 1])
        
        base_pressure += pressure_drift * generate_reading.drift_direction
        
        # Add anomaly effects if active
        if state['active_anomaly'] is not None:
            anomaly = state['active_anomaly']
            progress = 1 - (state['anomaly_countdown'] / anomaly['duration'])
            
            # Effects intensify as countdown decreases
            pressure_effect = anomaly['effects'].get('pressure', 0) * progress
            base_pressure += pressure_effect
        
        # Set final value with random noise
        reading['pressure'] = round(base_pressure + random.uniform(-0.5, 0.5), 1)
    
    return reading

def perform_maintenance(equipment_type):
    """Simulate maintenance being performed on equipment"""
    state = equipment_state[equipment_type]
    
    # Reset equipment state
    if equipment_type == 'Turbine':
        state['temperature'] = 65.0 + random.uniform(-1, 1)
        state['vibration'] = 1.2 + random.uniform(-0.1, 0.1)
    else:  # Compressor
        state['pressure'] = 55.0 + random.uniform(-1, 1)
    
    # Clear any active anomalies
    state['active_anomaly'] = None
    state['anomaly_countdown'] = 0
    state['maintenance_due'] = 0.0
    
    # Update last maintenance time
    LAST_MAINTENANCE[equipment_type] = datetime.now()
    
    print(f"✅ Maintenance performed on {equipment_type}")
    print(f"  - All parameters reset to normal levels")
    
    # Create a maintenance record
    reading = generate_reading(equipment_type)
    reading['maintenance_performed'] = 1
    reading['anomaly_status'] = 0
    reading['anomaly_score'] = 0.0
    
    # Save to historical data
    append_to_historical_data(reading)
    
    return reading

def send_to_thingspeak(data):
    """Send sensor data to ThingSpeak"""
    # Format according to your ThingSpeak channel fields
    field1 = data['temperature'] if data['temperature'] is not None else ''
    field2 = data['vibration'] if data['vibration'] is not None else ''
    field3 = data['pressure'] if data['pressure'] is not None else ''
    field4 = data['anomaly_status']  # Anomaly Status (0 or 1)
    field5 = data.get('maintenance_performed', 0)  # Maintenance Performed
    field6 = data.get('maintenance_due', 0)  # Maintenance Due percentage
    field7 = data.get('equipment_code', '') or \
             ('TURB01' if data['equipment_type'] == 'Turbine' else 'COMP01')  # Equipment Code
    
    # ThingSpeak API endpoint
    url = f'https://api.thingspeak.com/update'
    
    # Payload with sensor readings
    params = {
        'api_key': WRITE_API_KEY,
        'field1': field1,
        'field2': field2,
        'field3': field3,
        'field4': field4,
        'field5': field5,
        'field6': field6,
        'field7': field7
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Response status code: {response.status_code}")
        return response
    except Exception as e:
        print(f"Error sending data to ThingSpeak: {e}")
        return None

def append_to_historical_data(reading):
    """Append a reading to the historical data CSV file"""
    # Create file with headers if it doesn't exist
    if not os.path.exists(HISTORICAL_DATA_FILE):
        with open(HISTORICAL_DATA_FILE, 'w') as f:
            f.write('timestamp,equipment_type,temperature,vibration,pressure,is_anomaly,maintenance_performed\n')
    
    # Format data for CSV
    temperature = reading['temperature'] if reading['temperature'] is not None else ''
    vibration = reading['vibration'] if reading['vibration'] is not None else ''
    pressure = reading['pressure'] if reading['pressure'] is not None else ''
    
    # Append data to file
    with open(HISTORICAL_DATA_FILE, 'a') as f:
        f.write(f"{reading['timestamp']},{reading['equipment_type']},{temperature},{vibration},{pressure},{reading['anomaly_status']},{reading['maintenance_performed']}\n")

def check_thingspeak_connectivity():
    """Test connection to ThingSpeak"""
    print("Checking ThingSpeak connectivity...")
    test_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'equipment_type': 'Turbine',
        'temperature': 65.0,
        'vibration': 1.2,
        'pressure': None,
        'anomaly_status': 0,
        'maintenance_performed': 0,
        'maintenance_due': 0.0,
        'anomaly_score': 0.0
    }
    
    response = send_to_thingspeak(test_data)
    if response and response.status_code == 200:
        print("✅ ThingSpeak connection verified!")
        return True
    else:
        print("❌ Failed to connect to ThingSpeak. Check your API keys and internet connection.")
        return False

def clear_thingspeak_data():
    """Clear all data in ThingSpeak channel"""
    print("Clearing ThingSpeak channel data...")
    try:
        url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json'
        response = requests.delete(url, params={'api_key': WRITE_API_KEY})
        
        if response.status_code == 200:
            print("✅ Successfully cleared ThingSpeak channel data")
            return True
        else:
            print(f"❌ Failed to clear ThingSpeak data: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Failed to clear ThingSpeak data: {e}")
        return False

def continuous_data_stream():
    """Continuously send data to ThingSpeak"""
    reading_count = 0
    equipment_sequence = ['Turbine', 'Compressor']
    
    print("\nStarting equipment simulation. Press Ctrl+C to stop.")
    print(f"Data will be sent every {UPDATE_INTERVAL} seconds")
    
    # Check ThingSpeak connection first
    check_thingspeak_connectivity()
    
    try:
        while True:
            # Alternate between equipment types
            equipment_type = equipment_sequence[reading_count % len(equipment_sequence)]
            reading_count += 1
            
            print(f"\nReading {reading_count} at {datetime.now().strftime('%H:%M:%S')}")
            print(f"Equipment: {equipment_type}")
            
            minutes_since_maintenance = (datetime.now() - LAST_MAINTENANCE[equipment_type]).total_seconds() / 60
            
            print(f"Minutes since maintenance: {minutes_since_maintenance:.1f}")
            
            # Generate reading
            reading = generate_reading(equipment_type)
            
            # Display reading
            if equipment_type == 'Turbine':
                print(f"Temperature: {reading['temperature']}°C")
                print(f"Vibration: {reading['vibration']} mm/s")
            else:  # Compressor
                print(f"Pressure: {reading['pressure']} psi")
            
            print(f"Status: {'FAULTY' if reading['anomaly_status'] == 1 else 'Normal'}")
            print(f"Anomaly Score: {reading['anomaly_score']:.2f}")
            print(f"Maintenance Need: {reading['maintenance_due']:.1f}%")
            
            # Send to ThingSpeak
            response = send_to_thingspeak(reading)
            
            # Get entry ID from response (if available)
            entry_id = 0
            if response and response.status_code == 200:
                try:
                    entry_id = int(response.text)
                except:
                    entry_id = 0
            
            print(f"ThingSpeak Entry ID: {entry_id}")
            
            # Save to historical data
            append_to_historical_data(reading)
            
            # Check for maintenance input
            print("Press 'm' to perform maintenance, or Enter to continue:")
            time.sleep(0.5)  # Give a short moment to press a key
            
            # Check keyboard input (non-blocking)
            maintenance_key = False
            if os.name == 'nt':  # Windows
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    if key == 'm':
                        maintenance_key = True
            else:  # Unix/Linux/Mac
                import select
                import sys
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    key = sys.stdin.read(1).lower()
                    if key == 'm':
                        maintenance_key = True
            
            # Perform maintenance if requested
            if maintenance_key:
                maintenance_reading = perform_maintenance(equipment_type)
                send_to_thingspeak(maintenance_reading)
            
            # Automatic maintenance if very critical (> 95% need)
            if reading['maintenance_due'] > 95:
                print(f"⚠️ CRITICAL: Maintenance need at {reading['maintenance_due']:.1f}%")
                print("Performing automatic emergency maintenance...")
                maintenance_reading = perform_maintenance(equipment_type)
                send_to_thingspeak(maintenance_reading)
            
            # Wait for next reading
            wait_time = random.uniform(UPDATE_INTERVAL - 2, UPDATE_INTERVAL + 2)
            print(f"\nWaiting {wait_time:.1f} seconds for next reading...")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print("\n\nSensor simulation stopped by user")
    except Exception as e:
        print(f"\n\nError in sensor simulation: {e}")
        raise

if __name__ == "__main__":
    print("Industrial Equipment Sensor Simulation")
    print("======================================")
    
    # Ask if user wants to clear ThingSpeak data first
    clear_data = input("Do you want to clear all ThingSpeak channel data before starting? (y/n): ").lower() == 'y'
    if clear_data:
        clear_thingspeak_data()
    
    print("Simulating Turbine and Compressor in lab environment")
    print("- Turbine: Monitoring temperature and vibration")
    print("- Compressor: Monitoring pressure")
    print(f"Data will be sent to ThingSpeak every {UPDATE_INTERVAL-2}-{UPDATE_INTERVAL+2} seconds")
    
    # Display initial maintenance status
    print("Maintenance history:")
    for equip in ['Turbine', 'Compressor']:
        minutes = (datetime.now() - LAST_MAINTENANCE[equip]).total_seconds() / 60
        print(f"- {equip}: Last maintenance {minutes:.1f} minutes ago")
    
    print("======================================")
    print("Starting equipment simulation. Press Ctrl+C to stop.")
    print("Press 'm' when prompted to perform maintenance")
    print("======================================")
    
    continuous_data_stream()
