"""Debug AI Summary Issue"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'enhanced-backend'))

from app.services.ai_service_advanced import AdvancedAIInsightService
from app.services.workflow_service import AdvancedWorkflowService
import pandas as pd
import numpy as np

def debug_ai_summary():
    """Debug why AI summary isn't working"""
    
    # Create mock data similar to what the user would have
    print("Creating mock data for debugging...")
    
    # Create a mock data cleaning node result
    mock_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, None],
        'B': ['a', 'b', 'c', 'd', 'e', 'f'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    })
    
    # Create cleaning summary
    cleaning_summary = {
        'original_shape': (6, 3),
        'final_shape': (5, 3),
        'rows_removed': 1,
        'columns_removed': 0,
        'operations_performed': [
            'Removed 1 rows with missing values',
            'Cleaned text in column B'
        ],
        'data_quality_score': 85.5
    }
    
    # Create mock workflow data
    mock_workflow_data = {
        'nodes': {
            'data_source_1': {
                'type': 'data_source',
                'data': {
                    'dataframe': mock_df,
                    'shape': (6, 3),
                    'columns': ['A', 'B', 'C']
                }
            },
            'data_cleaning_1': {
                'type': 'data_cleaning',
                'data': {
                    'data': mock_df.dropna(),
                    'cleaning_summary': cleaning_summary,
                    'type': 'cleaned_data'
                }
            },
            'statistical_analysis_1': {
                'type': 'statistical_analysis',
                'data': {
                    'dataframe': mock_df.dropna(),
                    'statistics': {
                        'mean': mock_df.dropna().select_dtypes(include=[np.number]).mean().to_dict(),
                        'std': mock_df.dropna().select_dtypes(include=[np.number]).std().to_dict()
                    }
                }
            }
        },
        'workflow_context': {
            'analysis_type': 'data_processing',
            'user_intent': 'data_cleaning_and_analysis'
        }
    }
    
    print("Mock workflow data created successfully!")
    print(f"Number of nodes: {len(mock_workflow_data['nodes'])}")
    print(f"Node types: {[node['type'] for node in mock_workflow_data['nodes'].values()]}")
    
    # Test AI service
    ai_service = AdvancedAIInsightService()
    
    # Test node validation
    print("\nTesting node validation...")
    for node_id, node_info in mock_workflow_data['nodes'].items():
        node_type = node_info['type']
        node_data = node_info['data']
        
        has_valid_data = ai_service._has_valid_data(node_data, node_type)
        print(f"Node {node_id} (type: {node_type}) has valid data: {has_valid_data}")
        
        if has_valid_data:
            # Try to generate prompt
            try:
                prompt = ai_service.prompt_router.generate_prompt(
                    node_type=node_type,
                    data=node_data,
                    node_id=node_id,
                    context=mock_workflow_data['workflow_context']
                )
                print(f"Generated prompt for {node_id}: {len(prompt)} characters")
            except Exception as e:
                print(f"Error generating prompt for {node_id}: {e}")
    
    # Test comprehensive workflow insights
    print("\nTesting comprehensive workflow insights...")
    try:
        result = ai_service.generate_comprehensive_workflow_insights(mock_workflow_data)
        print(f"AI Analysis Result: {result.get('success', False)}")
        
        if result.get('success'):
            print("✅ AI analysis completed successfully!")
            metadata = result.get('metadata', {})
            print(f"Nodes analyzed: {metadata.get('nodes_analyzed', 0)}")
            print(f"Analysis type: {metadata.get('analysis_type', 'unknown')}")
        else:
            print("❌ AI analysis failed")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error in comprehensive workflow insights: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ai_summary()
