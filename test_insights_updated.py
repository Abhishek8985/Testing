#!/usr/bin/env python3
"""
Test script to verify that chart insights are generated correctly after removing 
"Business Insights:" and "Recommendations:" headings.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'enhanced-backend'))

import pandas as pd
import numpy as np
from app.services.univariate_anomaly_service import UnivariateAnomalyDetectionService
from app.services.multivariate_anomaly_service import MultivariateAnomalyDetectionService
from app.services.event_detection_service import EventDetectionService
from app.services.eda_service import EDAService

def test_insights_format():
    """Test that insights no longer contain Business Insights: or Recommendations: headings"""
    print("🔍 Testing updated insight format...")
    
    # Create test data
    np.random.seed(42)
    n_samples = 100
    
    # Generate sample data with anomalies
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='D')
    })
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, 5, replace=False)
    data.loc[anomaly_indices, 'feature1'] += 5  # Create outliers
    
    print(f"✅ Created test dataset with {len(data)} samples and {len(data.columns)} features")
    
    # Test EDA Service
    try:
        print("\n📊 Testing EDA Service insights...")
        eda_service = EDAService()
        eda_results = eda_service.generate_comprehensive_analysis(data)
        
        # Check for charts with insights
        charts_with_insights = 0
        problematic_insights = []
        
        for chart_name, chart_data in eda_results.get('charts', {}).items():
            if isinstance(chart_data, dict) and 'insight' in chart_data:
                insight = chart_data['insight']
                charts_with_insights += 1
                
                # Check for problematic headings
                if 'Business Insights:' in insight or 'Recommendations:' in insight:
                    problematic_insights.append(chart_name)
                    
                print(f"   📈 {chart_name}: {'✅ Clean' if 'Business Insights:' not in insight and 'Recommendations:' not in insight else '❌ Has problematic headings'}")
        
        print(f"✅ EDA Service: {charts_with_insights} charts with insights")
        if problematic_insights:
            print(f"❌ Found problematic headings in: {problematic_insights}")
        else:
            print("✅ No problematic headings found in EDA insights")
            
    except Exception as e:
        print(f"❌ EDA Service error: {e}")
    
    # Test Univariate Anomaly Service
    try:
        print("\n🔍 Testing Univariate Anomaly Service insights...")
        uni_service = UnivariateAnomalyDetectionService()
        uni_results = uni_service.detect_anomalies(data)
        
        charts_with_insights = 0
        problematic_insights = []
        
        charts = uni_results.get('results', {}).get('charts', {})
        for chart_name, chart_data in charts.items():
            if isinstance(chart_data, dict) and 'insight' in chart_data:
                insight = chart_data['insight']
                charts_with_insights += 1
                
                # Check for problematic headings
                if 'Business Insights:' in insight or 'Recommendations:' in insight:
                    problematic_insights.append(chart_name)
                    
                print(f"   📈 {chart_name}: {'✅ Clean' if 'Business Insights:' not in insight and 'Recommendations:' not in insight else '❌ Has problematic headings'}")
        
        print(f"✅ Univariate Service: {charts_with_insights} charts with insights")
        if problematic_insights:
            print(f"❌ Found problematic headings in: {problematic_insights}")
        else:
            print("✅ No problematic headings found in Univariate insights")
            
    except Exception as e:
        print(f"❌ Univariate Service error: {e}")
    
    # Test Multivariate Anomaly Service
    try:
        print("\n🔍 Testing Multivariate Anomaly Service insights...")
        multi_service = MultivariateAnomalyDetectionService()
        multi_results = multi_service.detect_anomalies(data)
        
        charts_with_insights = 0
        problematic_insights = []
        
        charts = multi_results.get('results', {}).get('charts', {})
        for chart_name, chart_data in charts.items():
            if isinstance(chart_data, dict) and 'insight' in chart_data:
                insight = chart_data['insight']
                charts_with_insights += 1
                
                # Check for problematic headings
                if 'Business Insights:' in insight or 'Recommendations:' in insight:
                    problematic_insights.append(chart_name)
                    
                print(f"   📈 {chart_name}: {'✅ Clean' if 'Business Insights:' not in insight and 'Recommendations:' not in insight else '❌ Has problematic headings'}")
        
        print(f"✅ Multivariate Service: {charts_with_insights} charts with insights")
        if problematic_insights:
            print(f"❌ Found problematic headings in: {problematic_insights}")
        else:
            print("✅ No problematic headings found in Multivariate insights")
            
    except Exception as e:
        print(f"❌ Multivariate Service error: {e}")
    
    # Test Event Detection Service
    try:
        print("\n📡 Testing Event Detection Service insights...")
        event_service = EventDetectionService()
        event_results = event_service.detect_events(data)
        
        charts_with_insights = 0
        problematic_insights = []
        
        charts = event_results.get('results', {}).get('charts', {})
        for chart_name, chart_data in charts.items():
            if isinstance(chart_data, dict) and 'insight' in chart_data:
                insight = chart_data['insight']
                charts_with_insights += 1
                
                # Check for problematic headings
                if 'Business Insights:' in insight or 'Recommendations:' in insight:
                    problematic_insights.append(chart_name)
                    
                print(f"   📈 {chart_name}: {'✅ Clean' if 'Business Insights:' not in insight and 'Recommendations:' not in insight else '❌ Has problematic headings'}")
        
        print(f"✅ Event Detection Service: {charts_with_insights} charts with insights")
        if problematic_insights:
            print(f"❌ Found problematic headings in: {problematic_insights}")
        else:
            print("✅ No problematic headings found in Event Detection insights")
            
    except Exception as e:
        print(f"❌ Event Detection Service error: {e}")
    
    print("\n🎉 Insight format testing completed!")

if __name__ == "__main__":
    test_insights_format()
