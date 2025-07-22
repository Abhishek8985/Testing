#!/usr/bin/env python3
"""
Test script to verify that AI service properly analyzes data cleaning nodes
"""

import sys
import os
import pandas as pd

# Add the enhanced-backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'enhanced-backend'))

from app.services.ai_service_advanced import AdvancedAIInsightService

def test_ai_service_data_cleaning():
    """Test that AI service properly handles data cleaning nodes"""
    
    # Create AI service instance
    ai_service = AdvancedAIInsightService()
    
    # Test data cleaning node data structure
    data_cleaning_node_data = {
        'data': pd.DataFrame({
            'col1': [1, 2, 3, 4, 5], 
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [10.1, 20.2, 30.3, 40.4, 50.5]
        }),
        'cleaning_summary': {
            'original_shape': (10, 3),
            'final_shape': (5, 3),
            'rows_removed': 5,
            'columns_removed': 0,
            'operations_performed': ['Removed duplicates', 'Filled missing values', 'Removed outliers'],
            'data_quality_score': 85.5
        },
        'type': 'cleaned_data'
    }
    
    # Test that the AI service considers this valid data
    has_valid_data = ai_service._has_valid_data(data_cleaning_node_data, 'data_cleaning')
    print(f"Data cleaning node has valid data: {has_valid_data}")
    assert has_valid_data == True, f"Expected True, got {has_valid_data}"
    
    # Test prompt generation
    try:
        prompt = ai_service.prompt_router.generate_prompt(
            node_type='data_cleaning',
            data=data_cleaning_node_data,
            node_id='test_cleaning_node',
            context={}
        )
        print(f"Generated prompt length: {len(prompt)} characters")
        print(f"Prompt starts with: {prompt[:100]}...")
        
        # Check that it's not an error message
        assert "CRITICAL ERROR" not in prompt, "Prompt generation failed"
        assert "DATA CLEANING ANALYSIS" in prompt, "Prompt doesn't contain expected content"
        
        print("✅ Data cleaning node prompt generated successfully!")
        
    except Exception as e:
        print(f"❌ Error generating prompt: {str(e)}")
        raise
    
    # Test with a comprehensive workflow scenario
    test_workflow_data = {
        'nodes': {
            'data_source_node': {
                'type': 'data_source',
                'data': {
                    'dataframe': pd.DataFrame({
                        'col1': [1, 2, 3, 4, 5], 
                        'col2': ['a', 'b', 'c', 'd', 'e']
                    })
                }
            },
            'data_cleaning_node': {
                'type': 'data_cleaning',
                'data': data_cleaning_node_data
            }
        },
        'workflow_context': {
            'total_nodes': 2,
            'has_data': True
        }
    }
    
    # Test workflow analysis
    workflow_analysis = ai_service._analyze_workflow_structure(
        test_workflow_data['nodes'], 
        test_workflow_data['workflow_context']
    )
    
    print(f"Workflow analysis - total analyzed nodes: {workflow_analysis.get('total_analyzed_nodes', 0)}")
    print(f"Cleaning nodes found: {len(workflow_analysis.get('analysis_categories', {}).get('cleaning_nodes', []))}")
    
    # Check that cleaning node is included
    cleaning_nodes = workflow_analysis.get('analysis_categories', {}).get('cleaning_nodes', [])
    assert 'data_cleaning_node' in cleaning_nodes, f"Data cleaning node not found in analysis: {cleaning_nodes}"
    
    print("✅ All AI service data cleaning tests passed!")

if __name__ == "__main__":
    test_ai_service_data_cleaning()
