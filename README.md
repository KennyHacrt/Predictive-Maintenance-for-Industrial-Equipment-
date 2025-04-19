Project Overview: Design a system that uses sensor data to predict failures in industrial equipment, reducing downtime and maintenance costs.

Data Collection & Analysis:

Live Parameters:

  Real-time vibration and temperature readings from machinery e.g. pumps, motors etc.

  Instantaneous load or pressure measurements e.g. valves, pipes etc.

Long-Term Parameters:

  Trend analysis of sensor readings over weeks

  Cumulative usage hours and historical maintenance logs

  Predictive models that correlate gradual shifts in parameters with potential faults
  
Additional Details: 
Simulate or deploy physical sensors, stream data to a cloud database, and apply machine learning techniques for anomaly detection. Short-term alerts can be generated if live readings exceed thresholds, while long-term analysis refines maintenance schedules.

REMINDER!!!!!!!!!! BEFORE USING THE CODES, PlEASE CHANGE THE THINGSPEAK CHANNEL WIRTE API and READ API 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Predictive Maintenance for Industrial Equipment - Design Report
System Overview
Our industrial equipment monitoring system provides comprehensive maintenance forecasting through a network of integrated components. The system begins with sensor monitoring that captures essential equipment parameters like temperature, vibration, and pressure. This data flows into ThingSpeak's cloud platform where it's stored and visualized. When sensor readings deviate from established normal patterns, the system triggers alerts for maintenance personnel. A dedicated maintenance scheduler tracks equipment condition and optimizes service timing. For long-term analysis, the system preserves historical performance data, enabling trend identification and improved future predictions.

Component Descriptions
Sensor Simulation (sensor.py)
The sensor component simulates physical measurements from industrial equipment, generating realistic temperature, vibration, and pressure data. It creates structured data packets that include timestamp, equipment type, and measured parameters. This component transmits data to ThingSpeak at configurable intervals, ensuring continuous monitoring regardless of equipment operating conditions.

Alert System (Alert.py)
The alert system continuously checks for abnormal equipment behavior. When measurements exceed acceptable thresholds or patterns indicate potential issues, this component generates immediate notifications. It maintains an alert log for equipment history tracking and ensures maintenance personnel receive timely information about developing problems.

Maintenance Scheduler (maintenance.py)
The maintenance component manages service scheduling based on equipment condition. It tracks operational hours, anomaly frequency, and maintenance history to determine optimal service timing. The scheduler updates CSV records with maintenance dates and activities, providing a complete service history for each equipment unit.

Long-term Analysis (longterm.py)
This component performs historical data analysis to identify gradual degradation patterns. It compiles equipment performance data into CSV files organized by time periods and equipment types. These historical datasets enable detailed trend analysis and improve the system's predictive capabilities over time.

Technical Implementation Details
Machine Learning Approach
Our system employs the Isolation Forest algorithm for anomaly detection. This unsupervised learning method identifies data points that diverge from normal patterns without requiring labeled examples of failures. The algorithm constructs random decision trees and isolates outliers based on how few splits are needed to isolate a data point. This approach is particularly effective for industrial equipment monitoring because:

It performs well with high-dimensional data (multiple sensor readings)
It requires minimal preprocessing compared to distribution-based methods
It handles varying equipment types with different normal operating parameters
It maintains high detection accuracy with relatively small training datasets
We implemented separate models for turbines and compressors to account for their distinct operational characteristics. The turbine model analyzes correlations between temperature and vibration patterns, while the compressor model focuses primarily on pressure fluctuations.

Data Processing Pipeline
The system's data flow begins with collection through simulated sensors or API connections to existing industrial sensors. This raw data undergoes preprocessing to normalize values and handle missing readings. The processed data is then evaluated against our trained Isolation Forest models, generating an anomaly score between 0 and 1 for each reading. Scores exceeding threshold values (typically 0.7) trigger the alert system. All processed data, including anomaly scores, is stored in ThingSpeak for visualization and retrieved by the long-term analysis component for trend identification.

Storage Architecture
Our data storage approach uses both cloud-based and local components. Immediate sensor readings and short-term analysis are managed through ThingSpeak's cloud platform, providing accessible visualization and real-time monitoring. For long-term storage and detailed analysis, the system maintains local CSV files organized by equipment type, time period, and data category (raw readings, maintenance records, and anomaly detections). This hybrid approach balances immediate accessibility with comprehensive historical record-keeping.

Deployment Considerations
The system is designed for flexible deployment across various industrial environments. Core components can run on standard computing hardware with Python support, while the ThingSpeak integration provides cloud-based visualization accessible from any device with internet access. For production environments, we recommend secured API connections, automated backup of CSV files, and integration with existing maintenance management systems through standard data interchange formats.
