"""Debug AI Summary Node Processing Issue"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'enhanced-backend'))

from app.services.ai_service_advanced import AdvancedAIInsightService
import pandas as pd
import numpy as np

def debug_node_processing_issue():
    """Debug why only one node is being processed in AI summary"""
    
    # Create data that matches the user's actual workflow
    # Based on the AI response, they have a dataset with 9357 rows, 15 columns
    # with columns like Date, Time, CO(GT), PT08.S1(CO), etc.
    
    # Create mock data that represents the actual user data
    np.random.seed(42)  # For reproducible results
    
    # Create a realistic air quality dataset
    air_quality_df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=9357, freq='H'),
        'Time': pd.date_range('2023-01-01', periods=9357, freq='H').strftime('%H:%M:%S'),
        'CO(GT)': np.random.normal(2.5, 0.5, 9357),
        'PT08.S1(CO)': np.random.normal(1000, 200, 9357),
        'NMHC(GT)': np.random.normal(150, 50, 9357),
        'C6H6(GT)': np.random.normal(8, 2, 9357),
        'PT08.S2(NMHC)': np.random.normal(800, 150, 9357),
        'NOx(GT)': np.random.normal(250, 100, 9357),
        'PT08.S3(NOx)': np.random.normal(700, 100, 9357),
        'NO2(GT)': np.random.normal(80, 30, 9357),
        'PT08.S4(NO2)': np.random.normal(1200, 300, 9357),
        'PT08.S5(O3)': np.random.normal(1000, 200, 9357),
        'T': np.random.normal(20, 10, 9357),
        'RH': np.random.normal(50, 20, 9357),
        'AH': np.random.normal(1.2, 0.5, 9357)
    })
    
    print("=== DEBUGGING AI SUMMARY NODE PROCESSING ===")
    print(f"Created air quality dataset: {air_quality_df.shape}")
    print(f"Columns: {list(air_quality_df.columns)}")
    
    # Create the exact node structure that might be causing the issue
    # This simulates what the user's actual workflow might be generating
    
    # Data source node - this one seems to be working
    data_source_node = {
        'type': 'data_source',
        'data': {
            'dataframe': air_quality_df,
            'shape': air_quality_df.shape,
            'columns': list(air_quality_df.columns),
            'file_info': {
                'filename': 'air_quality_data.csv',
                'size': 1024 * 1024  # 1MB
            }
        }
    }
    
    # Data cleaning node - this might be the issue
    # Let's test different possible data structures
    
    # Version 1: Standard structure
    data_cleaning_v1 = {
        'type': 'data_cleaning',
        'data': {
            'data': air_quality_df,  # cleaned data
            'cleaning_summary': {
                'original_shape': air_quality_df.shape,
                'final_shape': air_quality_df.shape,
                'rows_removed': 0,
                'columns_removed': 0,
                'operations_performed': ['No cleaning operations needed'],
                'data_quality_score': 95.0
            },
            'type': 'cleaned_data'
        }
    }
    
    # Version 2: Alternative structure (dataframe key)
    data_cleaning_v2 = {
        'type': 'data_cleaning',
        'data': {
            'dataframe': air_quality_df,  # cleaned data
            'cleaning_summary': {
                'original_shape': air_quality_df.shape,
                'final_shape': air_quality_df.shape,
                'rows_removed': 0,
                'columns_removed': 0,
                'operations_performed': ['No cleaning operations needed'],
                'data_quality_score': 95.0
            },
            'type': 'cleaned_data'
        }
    }
    
    # Version 3: Missing cleaning summary
    data_cleaning_v3 = {
        'type': 'data_cleaning',
        'data': {
            'data': air_quality_df,  # cleaned data
            'type': 'cleaned_data'
        }
    }
    
    # Statistical analysis node
    stats_node = {
        'type': 'statistical_analysis',
        'data': {
            'dataframe': air_quality_df,
            'statistics': {
                'mean': air_quality_df.select_dtypes(include=[np.number]).mean().to_dict(),
                'std': air_quality_df.select_dtypes(include=[np.number]).std().to_dict(),
                'count': air_quality_df.select_dtypes(include=[np.number]).count().to_dict()
            },
            'correlations': {
                'CO_NOx': 0.65,
                'temp_humidity': -0.45
            }
        }
    }
    
    # Test different scenarios
    test_scenarios = [
        ("Standard cleaning structure", data_cleaning_v1),
        ("Alternative cleaning structure", data_cleaning_v2),
        ("Missing cleaning summary", data_cleaning_v3)
    ]
    
    ai_service = AdvancedAIInsightService()
    
    for scenario_name, cleaning_node in test_scenarios:
        print(f"\n=== TESTING SCENARIO: {scenario_name} ===")
        
        # Create nodes dict
        nodes = {
            'data_source_1': data_source_node,
            'data_cleaning_1': cleaning_node,
            'statistical_analysis_1': stats_node
        }
        
        # Test validation for each node
        print("Testing node validation...")
        all_valid = True
        for node_id, node_info in nodes.items():
            node_type = node_info['type']
            node_data = node_info['data']
            
            # Check if it's an AI node
            if ai_service._is_ai_node(node_type, node_id):
                print(f"  {node_id} ({node_type}): SKIPPED (AI node)")
                continue
            
            # Check if it has valid data
            has_valid_data = ai_service._has_valid_data(node_data, node_type)
            print(f"  {node_id} ({node_type}): valid={has_valid_data}")
            
            if not has_valid_data:
                all_valid = False
                print(f"    ❌ INVALID DATA - Keys: {list(node_data.keys()) if isinstance(node_data, dict) else 'non-dict'}")
                
                # Debug why it's invalid
                if node_type == 'data_cleaning':
                    print(f"    Data cleaning debug:")
                    expected_keys = ['data', 'dataframe', 'cleaning_summary', 'cleaning_stats', 'before_cleaning', 'after_cleaning']
                    for key in expected_keys:
                        if key in node_data:
                            value = node_data[key]
                            if isinstance(value, pd.DataFrame):
                                print(f"      {key}: DataFrame shape {value.shape}, empty={value.empty}")
                            else:
                                print(f"      {key}: {type(value).__name__}")
                        else:
                            print(f"      {key}: MISSING")
            else:
                # Try to generate prompt
                try:
                    prompt = ai_service.prompt_router.generate_prompt(
                        node_type=node_type,
                        data=node_data,
                        node_id=node_id,
                        context={}
                    )
                    
                    if "CRITICAL ERROR" in prompt or "No data available" in prompt:
                        print(f"    ❌ PROMPT ERROR: {prompt[:100]}...")
                        all_valid = False
                    else:
                        print(f"    ✅ PROMPT OK: {len(prompt)} characters")
                
                except Exception as e:
                    print(f"    ❌ PROMPT EXCEPTION: {e}")
                    all_valid = False
        
        print(f"Scenario result: {'✅ ALL VALID' if all_valid else '❌ SOME INVALID'}")
        
        if all_valid:
            print("  This scenario should work correctly in AI summary!")
        else:
            print("  This scenario would cause nodes to be skipped in AI summary!")

if __name__ == "__main__":
    debug_node_processing_issue()
