#!/usr/bin/env python3
"""
Test script to verify chart insights functionality
"""

import sys
import os
sys.path.append('/Users/manu/Downloads/ Superhacker-v1/enhanced-backend')

import pandas as pd
import numpy as np
from app.services.univariate_anomaly_service import UnivariateAnomalyDetectionService
from app.services.multivariate_anomaly_service import MultivariateAnomalyDetectionService
from app.services.event_detection_service import EventDetectionService
from app.services.eda_service import EDAService

def test_chart_insights():
    print("🧪 Testing Chart Insights Functionality...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data with some anomalies
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.normal(-2, 1.5, n_samples)
    }
    
    # Add some anomalies
    data['feature1'][950:970] = np.random.normal(10, 1, 20)  # Spike anomalies
    data['feature2'][980:] = np.random.normal(20, 1, 20)     # Drift anomalies
    
    df = pd.DataFrame(data)
    
    print(f"📊 Test data shape: {df.shape}")
    print(f"📊 Test data columns: {list(df.columns)}")
    
    # Test 1: Univariate Anomaly Detection with Insights
    print("\n🔍 Testing Univariate Anomaly Detection...")
    try:
        univariate_service = UnivariateAnomalyDetectionService()
        result = univariate_service.detect_anomalies(df, {'method': 'all'})
        
        if result['success']:
            charts = result['results'].get('charts', {})
            print(f"✅ Univariate charts generated: {len(charts)}")
            
            # Check for insights
            insights_found = 0
            for chart_name, chart_data in charts.items():
                if isinstance(chart_data, dict) and 'insight' in chart_data:
                    insights_found += 1
                    print(f"📋 {chart_name}: Has insight ({len(chart_data['insight'])} chars)")
            
            print(f"✅ Charts with insights: {insights_found}/{len(charts)}")
        else:
            print(f"❌ Univariate anomaly detection failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Univariate test error: {str(e)}")
    
    # Test 2: Multivariate Anomaly Detection with Insights
    print("\n🔍 Testing Multivariate Anomaly Detection...")
    try:
        multivariate_service = MultivariateAnomalyDetectionService()
        result = multivariate_service.detect_anomalies(df, {'method': 'all'})
        
        if result['success']:
            charts = result['results'].get('charts', {})
            print(f"✅ Multivariate charts generated: {len(charts)}")
            
            # Check for insights
            insights_found = 0
            for chart_name, chart_data in charts.items():
                if isinstance(chart_data, dict) and 'insight' in chart_data:
                    insights_found += 1
                    print(f"📋 {chart_name}: Has insight ({len(chart_data['insight'])} chars)")
            
            print(f"✅ Charts with insights: {insights_found}/{len(charts)}")
        else:
            print(f"❌ Multivariate anomaly detection failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Multivariate test error: {str(e)}")
    
    # Test 3: Event Detection with Insights
    print("\n🔍 Testing Event Detection...")
    try:
        event_service = EventDetectionService()
        result = event_service.detect_events(df, {'method': 'all'})
        
        if result['success']:
            charts = result['results'].get('charts', {})
            print(f"✅ Event detection charts generated: {len(charts)}")
            
            # Check for insights
            insights_found = 0
            for chart_name, chart_data in charts.items():
                if isinstance(chart_data, dict) and 'insight' in chart_data:
                    insights_found += 1
                    print(f"📋 {chart_name}: Has insight ({len(chart_data['insight'])} chars)")
            
            print(f"✅ Charts with insights: {insights_found}/{len(charts)}")
        else:
            print(f"❌ Event detection failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Event detection test error: {str(e)}")
    
    # Test 4: EDA Service with Insights
    print("\n🔍 Testing EDA Service...")
    try:
        eda_service = EDAService()
        result = eda_service.generate_eda_analysis(df)
        
        if result['success']:
            charts = result['results'].get('charts', {})
            print(f"✅ EDA charts generated: {len(charts)}")
            
            # Check for insights
            insights_found = 0
            for chart_name, chart_data in charts.items():
                if isinstance(chart_data, dict) and 'insight' in chart_data:
                    insights_found += 1
                    print(f"📋 {chart_name}: Has insight ({len(chart_data['insight'])} chars)")
            
            print(f"✅ Charts with insights: {insights_found}/{len(charts)}")
        else:
            print(f"❌ EDA analysis failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ EDA test error: {str(e)}")
    
    print("\n🎉 Chart insights testing completed!")

if __name__ == "__main__":
    test_chart_insights()
