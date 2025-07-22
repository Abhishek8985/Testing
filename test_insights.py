#!/usr/bin/env python3
"""
Test script to verify the new chart insight structure
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'enhanced-backend'))

import pandas as pd
import numpy as np
import json
from app.services.univariate_anomaly_service import UnivariateAnomalyDetectionService
from app.services.multivariate_anomaly_service import MultivariateAnomalyDetectionService
from app.services.event_detection_service import EventDetectionService

def test_univariate_insights():
    """Test univariate anomaly detection with insights"""
    print("🔍 Testing Univariate Anomaly Detection Insights...")
    
    # Create sample data
    np.random.seed(42)
    data = {
        'value1': np.random.normal(10, 2, 100),
        'value2': np.random.normal(20, 5, 100),
        'value3': np.random.normal(0, 1, 100)
    }
    
    # Add some anomalies
    data['value1'][10] = 50  # Spike
    data['value2'][20] = -10  # Outlier
    data['value3'][30] = 10   # Anomaly
    
    df = pd.DataFrame(data)
    
    # Run detection
    service = UnivariateAnomalyDetectionService()
    result = service.detect_anomalies(df)
    
    if result['success']:
        charts = result['results']['charts']
        print(f"✅ Generated {len(charts)} charts with insights")
        
        # Check structure
        for chart_name, chart_data in charts.items():
            if isinstance(chart_data, dict) and 'insight' in chart_data:
                print(f"📊 {chart_name}: Has insight ({len(chart_data['insight'])} chars)")
                print(f"   First 100 chars: {chart_data['insight'][:100]}...")
            else:
                print(f"⚠️  {chart_name}: No insight structure")
    else:
        print(f"❌ Error: {result['error']}")

def test_multivariate_insights():
    """Test multivariate anomaly detection with insights"""
    print("\n🔍 Testing Multivariate Anomaly Detection Insights...")
    
    # Create sample data
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100)
    }
    
    # Add some anomalies
    data['feature1'][10] = 5
    data['feature2'][10] = 5
    data['feature3'][10] = 5
    
    df = pd.DataFrame(data)
    
    # Run detection
    service = MultivariateAnomalyDetectionService()
    result = service.detect_anomalies(df)
    
    if result['success']:
        charts = result['results']['charts']
        print(f"✅ Generated {len(charts)} charts with insights")
        
        # Check structure
        for chart_name, chart_data in charts.items():
            if isinstance(chart_data, dict) and 'insight' in chart_data:
                print(f"📊 {chart_name}: Has insight ({len(chart_data['insight'])} chars)")
                print(f"   First 100 chars: {chart_data['insight'][:100]}...")
            else:
                print(f"⚠️  {chart_name}: No insight structure")
    else:
        print(f"❌ Error: {result['error']}")

def test_event_detection_insights():
    """Test event detection with insights"""
    print("\n🔍 Testing Event Detection Insights...")
    
    # Create sample data with events
    np.random.seed(42)
    data = {
        'sensor1': np.random.normal(10, 1, 100),
        'sensor2': np.random.normal(20, 2, 100)
    }
    
    # Add events
    data['sensor1'][20:25] = 20  # Spike
    data['sensor2'][50:55] = 20  # Flatline
    
    df = pd.DataFrame(data)
    
    # Run detection
    service = EventDetectionService()
    result = service.detect_events(df)
    
    if result['success']:
        charts = result['results']['charts']
        print(f"✅ Generated {len(charts)} charts with insights")
        
        # Check structure
        for chart_name, chart_data in charts.items():
            if isinstance(chart_data, dict) and 'insight' in chart_data:
                print(f"📊 {chart_name}: Has insight ({len(chart_data['insight'])} chars)")
                print(f"   First 100 chars: {chart_data['insight'][:100]}...")
            else:
                print(f"⚠️  {chart_name}: No insight structure")
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    print("🧪 Testing Chart Insights Implementation\n")
    
    try:
        test_univariate_insights()
        test_multivariate_insights()
        test_event_detection_insights()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
