"""Debug Node Data Flow for AI Summary"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'enhanced-backend'))

import time
import json
from app.services.ai_service_advanced import AdvancedAIInsightService
from app.services.workflow_service import AdvancedWorkflowService
import pandas as pd
import numpy as np

def debug_node_data_flow():
    """Debug the actual data flow between nodes"""
    
    # Create a workflow service instance
    workflow_service = AdvancedWorkflowService()
    
    # Create mock data that represents actual workflow execution
    mock_data_source_output = {
        'dataframe': pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [10, 20, 30, 40, 50]
        }),
        'shape': (5, 3),
        'columns': ['feature1', 'feature2', 'target'],
        'file_info': {
            'filename': 'test_data.csv',
            'size': 1024
        }
    }
    
    # Create mock data cleaning output
    cleaned_df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['a', 'b', 'c', 'd', 'e'],
        'target': [10, 20, 30, 40, 50]
    })
    
    mock_data_cleaning_output = {
        'data': cleaned_df,
        'cleaning_summary': {
            'original_shape': (5, 3),
            'final_shape': (5, 3),
            'rows_removed': 0,
            'columns_removed': 0,
            'operations_performed': ['No cleaning operations needed'],
            'data_quality_score': 100.0
        },
        'type': 'cleaned_data'
    }
    
    # Create mock statistical analysis output
    mock_stats_output = {
        'dataframe': cleaned_df,
        'statistics': {
            'mean': {'feature1': 3.0, 'target': 30.0},
            'std': {'feature1': 1.58, 'target': 15.81},
            'count': {'feature1': 5, 'target': 5}
        },
        'correlations': {
            'feature1_target': 1.0
        }
    }
    
    # Create mock connected analysis that mimics what workflow service would create
    mock_connected_analysis = {
        'total_nodes': 3,
        'node_types': ['data_source', 'data_cleaning', 'statistical_analysis'],
        'has_valid_data': True,
        'dataframes_count': 3,
        'models_count': 0,
        'statistics_count': 1,
        'charts_count': 0,
        'primary_data_shape': (5, 3),
        'total_memory_usage': 1024.0,
        'data_sources': ['data_source_1'],
        'node_outputs': {
            'data_source_1': {
                'type': 'data_source',
                'data': mock_data_source_output,
                'output_type': 'data_source_output'
            },
            'data_cleaning_1': {
                'type': 'data_cleaning',
                'data': mock_data_cleaning_output,
                'output_type': 'data_cleaning_output'
            },
            'statistical_analysis_1': {
                'type': 'statistical_analysis',
                'data': mock_stats_output,
                'output_type': 'statistical_analysis_output'
            }
        }
    }
    
    print("=== DEBUGGING NODE DATA FLOW ===")
    print(f"Connected analysis total nodes: {mock_connected_analysis['total_nodes']}")
    print(f"Node types: {mock_connected_analysis['node_types']}")
    print(f"Has valid data: {mock_connected_analysis['has_valid_data']}")
    
    # Test the comprehensive data preparation
    print("\n=== TESTING COMPREHENSIVE DATA PREPARATION ===")
    try:
        comprehensive_data = workflow_service._prepare_comprehensive_data_for_ai(mock_connected_analysis)
        print(f"Comprehensive data keys: {list(comprehensive_data.keys())}")
        print(f"Nodes in comprehensive data: {len(comprehensive_data.get('nodes', {}))}")
        print(f"Node types in comprehensive data: {[node['type'] for node in comprehensive_data.get('nodes', {}).values()]}")
        
        # Check each node's data
        for node_id, node_info in comprehensive_data.get('nodes', {}).items():
            node_type = node_info['type']
            node_data = node_info['data']
            print(f"\nNode {node_id} ({node_type}):")
            print(f"  Data keys: {list(node_data.keys()) if isinstance(node_data, dict) else 'non-dict'}")
            
            # Check for DataFrames
            for key, value in node_data.items() if isinstance(node_data, dict) else []:
                if isinstance(value, pd.DataFrame):
                    print(f"  DataFrame in '{key}': shape {value.shape}")
                elif isinstance(value, dict):
                    print(f"  Dict in '{key}': {len(value)} items")
                else:
                    print(f"  {key}: {type(value).__name__}")
    
    except Exception as e:
        print(f"❌ Error in comprehensive data preparation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test AI service with this data
    print("\n=== TESTING AI SERVICE WITH COMPREHENSIVE DATA ===")
    try:
        ai_service = AdvancedAIInsightService()
        
        # Test validation for each node
        print("Testing node validation...")
        nodes = comprehensive_data.get('nodes', {})
        for node_id, node_info in nodes.items():
            node_type = node_info['type']
            node_data = node_info['data']
            
            has_valid_data = ai_service._has_valid_data(node_data, node_type)
            print(f"  {node_id} ({node_type}): valid={has_valid_data}")
            
            if not has_valid_data:
                print(f"    Data keys: {list(node_data.keys()) if isinstance(node_data, dict) else 'non-dict'}")
                print(f"    Node data type: {type(node_data)}")
        
        # Test comprehensive workflow insights
        print("\nTesting comprehensive workflow insights...")
        result = ai_service.generate_comprehensive_workflow_insights(comprehensive_data)
        
        print(f"AI Analysis Success: {result.get('success', False)}")
        if result.get('success'):
            print("✅ AI analysis completed successfully!")
            metadata = result.get('metadata', {})
            print(f"  Nodes analyzed: {metadata.get('nodes_analyzed', 0)}")
            print(f"  Analysis type: {metadata.get('analysis_type', 'unknown')}")
            print(f"  AI model used: {metadata.get('ai_model_used', 'unknown')}")
            
            # Check for actual insights
            insights = result.get('insights', {})
            if insights:
                print(f"  Generated insights: {len(insights)} items")
            else:
                print("  ⚠️ No insights generated")
        else:
            print("❌ AI analysis failed")
            print(f"  Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error in AI service testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_node_data_flow()
