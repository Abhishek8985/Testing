#!/usr/bin/env python3
"""
Test script to verify that node type detection is working correctly
"""

import sys
import os
import pandas as pd

# Add the enhanced-backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'enhanced-backend'))

from app.services.workflow_service import AdvancedWorkflowService

def test_node_type_detection():
    """Test that node types are correctly identified"""
    service = AdvancedWorkflowService()
    
    # Test data source node output (direct DataFrame)
    data_source_output = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    node_type = service._determine_node_type_from_output(data_source_output)
    print(f"Data source node type: {node_type}")
    assert node_type == 'data_source', f"Expected 'data_source', got '{node_type}'"
    
    # Test data cleaning node output (dict with cleaning_summary)
    data_cleaning_output = {
        'data': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
        'cleaning_summary': {
            'original_shape': (5, 2),
            'final_shape': (3, 2),
            'rows_removed': 2,
            'columns_removed': 0,
            'operations_performed': ['Removed duplicates', 'Filled missing values'],
            'data_quality_score': 85.5
        },
        'type': 'cleaned_data'
    }
    node_type = service._determine_node_type_from_output(data_cleaning_output)
    print(f"Data cleaning node type: {node_type}")
    assert node_type == 'data_cleaning', f"Expected 'data_cleaning', got '{node_type}'"
    
    # Test descriptive stats node output (dict with basic_stats)
    descriptive_stats_output = {
        'data': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
        'basic_stats': {
            'col1': {'mean': 2.0, 'std': 1.0, 'min': 1, 'max': 3},
            'col2': {'count': 3, 'unique': 3, 'top': 'a', 'freq': 1}
        },
        'data_types': {'col1': 'int64', 'col2': 'object'},
        'missing_values': {'col1': 0, 'col2': 0}
    }
    node_type = service._determine_node_type_from_output(descriptive_stats_output)
    print(f"Descriptive stats node type: {node_type}")
    assert node_type == 'statistical_analysis', f"Expected 'statistical_analysis', got '{node_type}'"
    
    print("✅ All node type detection tests passed!")

if __name__ == "__main__":
    test_node_type_detection()
