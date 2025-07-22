"""
Test AI Node Data Flow - Test sending collected node outputs to AI Summary
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'enhanced-backend'))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from app.services.workflow_service import AdvancedWorkflowService
from app.services.ai_service_advanced import AdvancedAIInsightService

def print_dict_structure(data, name="Data", indent=0, max_depth=2):
    """Print dictionary structure with controlled depth"""
    prefix = "  " * indent
    
    if indent > max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    if isinstance(data, dict):
        print(f"{prefix}{name} (Dict): {len(data)} keys")
        for key, value in list(data.items())[:8]:  # Limit to first 8 items
            if isinstance(value, (dict, list)):
                print_dict_structure(value, f"'{key}'", indent + 1, max_depth)
            elif isinstance(value, pd.DataFrame):
                print(f"{prefix}  '{key}' (DataFrame): shape {value.shape}")
            elif isinstance(value, np.ndarray):
                print(f"{prefix}  '{key}' (Array): shape {value.shape}")
            else:
                value_str = str(value)[:80] + "..." if len(str(value)) > 80 else str(value)
                print(f"{prefix}  '{key}' ({type(value).__name__}): {value_str}")
        
        if len(data) > 8:
            print(f"{prefix}  ... and {len(data) - 8} more items")
    
    elif isinstance(data, list):
        print(f"{prefix}{name} (List): {len(data)} items")
        if data and indent < max_depth:
            print_dict_structure(data[0], "First Item", indent + 1, max_depth)
    
    elif isinstance(data, pd.DataFrame):
        print(f"{prefix}{name} (DataFrame): shape {data.shape}")
        print(f"{prefix}  Columns: {list(data.columns)[:5]}...")
    
    else:
        value_str = str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
        print(f"{prefix}{name} ({type(data).__name__}): {value_str}")

def create_test_workflow_data():
    """Create comprehensive test workflow data similar to real execution"""
    print("🔧 Creating comprehensive test workflow data...")
    
    # Create realistic test dataset
    test_df = pd.DataFrame({
        'id': range(1, 201),
        'feature_a': np.random.randn(200),
        'feature_b': np.random.randn(200) * 2 + 5,
        'feature_c': np.random.exponential(2, 200),
        'category': np.random.choice(['Type_A', 'Type_B', 'Type_C'], 200),
        'target': np.random.randint(0, 2, 200),
        'date': pd.date_range('2024-01-01', periods=200, freq='D')
    })
    
    # Create workflow service and populate with realistic node outputs
    workflow_service = AdvancedWorkflowService()
    
    # 1. Data Source Node Output
    print("  ✅ Adding Data Source Node...")
    workflow_service.data_cache['data_source_1'] = test_df
    
    # 2. Data Cleaning Node Output  
    print("  ✅ Adding Data Cleaning Node...")
    cleaned_df = test_df.dropna()
    cleaning_result = {
        'data': cleaned_df,
        'cleaning_summary': {
            'original_shape': test_df.shape,
            'final_shape': cleaned_df.shape,
            'rows_removed': len(test_df) - len(cleaned_df),
            'columns_removed': 0,
            'operations_performed': [
                'Removed missing values',
                'Validated data types',
                'Checked for duplicates'
            ],
            'data_quality_score': 96.8,
            'issues_found': ['No major issues detected'],
            'cleaning_time': 0.45
        },
        'type': 'cleaned_data'
    }
    workflow_service.data_cache['data_cleaning_1'] = cleaning_result
    
    # 3. Statistical Analysis Node Output
    print("  ✅ Adding Statistical Analysis Node...")
    numeric_cols = cleaned_df.select_dtypes(include=[np.number])
    stats_result = {
        'basic_stats': {
            'count': numeric_cols.describe().loc['count'].to_dict(),
            'mean': numeric_cols.describe().loc['mean'].to_dict(),
            'std': numeric_cols.describe().loc['std'].to_dict(),
            'min': numeric_cols.describe().loc['min'].to_dict(),
            'max': numeric_cols.describe().loc['max'].to_dict(),
            'quantiles': {
                '25%': numeric_cols.describe().loc['25%'].to_dict(),
                '50%': numeric_cols.describe().loc['50%'].to_dict(),
                '75%': numeric_cols.describe().loc['75%'].to_dict()
            }
        },
        'correlations': numeric_cols.corr().to_dict(),
        'data_types': cleaned_df.dtypes.astype(str).to_dict(),
        'missing_values': cleaned_df.isnull().sum().to_dict(),
        'unique_values': {
            col: cleaned_df[col].nunique() for col in cleaned_df.columns
        },
        'data': cleaned_df,
        'charts_generated': 5,
        'analysis_summary': {
            'total_features': len(cleaned_df.columns),
            'numeric_features': len(numeric_cols.columns),
            'categorical_features': len(cleaned_df.select_dtypes(include=['object']).columns),
            'datetime_features': len(cleaned_df.select_dtypes(include=['datetime']).columns)
        }
    }
    workflow_service.data_cache['statistical_analysis_1'] = stats_result
    
    # 4. Machine Learning Node Output
    print("  ✅ Adding Machine Learning Node...")
    ml_result = {
        'algorithm': 'random_forest_classifier',
        'model_info': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'class_weight': 'balanced'
        },
        'training_results': {
            'train_accuracy': 0.892,
            'validation_accuracy': 0.847,
            'cv_scores': [0.85, 0.88, 0.84, 0.86, 0.83],
            'cv_mean': 0.852,
            'cv_std': 0.018
        },
        'predictions': {
            'test_predictions': np.random.randint(0, 2, 40).tolist(),
            'prediction_probabilities': np.random.rand(40, 2).tolist()
        },
        'feature_importance': {
            'feature_a': 0.28,
            'feature_b': 0.35,
            'feature_c': 0.22,
            'category_encoded': 0.15
        },
        'confusion_matrix': [[18, 2], [3, 17]],
        'classification_report': {
            'precision': {'0': 0.86, '1': 0.89},
            'recall': {'0': 0.90, '1': 0.85},
            'f1-score': {'0': 0.88, '1': 0.87}
        },
        'data_split': {
            'training_size': 160,
            'testing_size': 40,
            'features_used': ['feature_a', 'feature_b', 'feature_c', 'category_encoded']
        }
    }
    workflow_service.data_cache['classification_1'] = ml_result
    
    # 5. Add some charts to chart cache
    print("  ✅ Adding Chart Cache...")
    workflow_service.chart_cache['statistical_analysis_1'] = [
        {
            'type': 'correlation_heatmap',
            'title': 'Feature Correlation Matrix',
            'data': 'base64_encoded_heatmap_data_placeholder',
            'description': 'Correlation analysis between numeric features'
        },
        {
            'type': 'distribution_plot',
            'title': 'Feature Distributions',
            'data': 'base64_encoded_distribution_data_placeholder',
            'description': 'Distribution plots for all numeric features'
        },
        {
            'type': 'box_plot',
            'title': 'Outlier Detection',
            'data': 'base64_encoded_boxplot_data_placeholder',
            'description': 'Box plots showing potential outliers'
        }
    ]
    
    workflow_service.chart_cache['classification_1'] = [
        {
            'type': 'confusion_matrix',
            'title': 'Model Performance Matrix',
            'data': 'base64_encoded_confusion_matrix_placeholder',
            'description': 'Confusion matrix showing classification results'
        },
        {
            'type': 'feature_importance',
            'title': 'Feature Importance Ranking',
            'data': 'base64_encoded_feature_importance_placeholder',
            'description': 'Ranking of features by importance in the model'
        }
    ]
    
    print(f"  📊 Created workflow with {len(workflow_service.data_cache)} nodes")
    print(f"  📈 Created {sum(len(charts) for charts in workflow_service.chart_cache.values())} charts")
    
    return workflow_service

def test_ai_summary_data_preparation():
    """Test how data is prepared for AI Summary node"""
    print("\n" + "=" * 80)
    print("TESTING AI SUMMARY DATA PREPARATION")
    print("=" * 80)
    
    workflow_service = create_test_workflow_data()
    
    # Simulate what happens when AI Summary node needs data
    print("\n1. COLLECTING ALL NODE OUTPUTS FOR AI SUMMARY")
    print("-" * 60)
    
    all_node_outputs = {}
    for executed_node_id, node_output in workflow_service.data_cache.items():
        all_node_outputs[executed_node_id] = node_output
        print(f"✅ Collected: {executed_node_id} ({type(node_output).__name__})")
    
    print(f"\n📊 AI Summary Input Structure:")
    print_dict_structure(all_node_outputs, "AI Summary Input Data", max_depth=2)
    
    # Test the AI service data validation
    print("\n2. TESTING AI SERVICE DATA VALIDATION")
    print("-" * 60)
    
    ai_service = AdvancedAIInsightService()
    
    # Test each node's data validation
    for node_id, node_data in all_node_outputs.items():
        # Determine node type from data structure
        node_type = determine_node_type(node_data)
        print(f"\n🔍 Testing {node_id} (detected type: {node_type})")
        
        try:
            has_valid_data = ai_service._has_valid_data(node_data, node_type)
            print(f"   Valid data: {has_valid_data}")
            
            if has_valid_data:
                # Try to generate a prompt
                try:
                    prompt = ai_service.prompt_router.generate_prompt(
                        node_type=node_type,
                        data=node_data,
                        node_id=node_id,
                        context={'analysis_type': 'comprehensive_workflow'}
                    )
                    print(f"   Prompt generated: {len(prompt)} characters")
                except Exception as e:
                    print(f"   ❌ Prompt generation failed: {e}")
            else:
                print(f"   ❌ Data validation failed")
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
    
    # Test comprehensive workflow data preparation
    print("\n3. TESTING COMPREHENSIVE WORKFLOW DATA PREPARATION")
    print("-" * 60)
    
    try:
        # Create mock connected analysis structure
        connected_analysis = {
            'total_nodes': len(all_node_outputs),
            'node_types': [determine_node_type(data) for data in all_node_outputs.values()],
            'has_valid_data': True,
            'dataframes_count': sum(1 for data in all_node_outputs.values() 
                                  if isinstance(data, pd.DataFrame) or 
                                  (isinstance(data, dict) and 'data' in data and isinstance(data['data'], pd.DataFrame))),
            'node_outputs': all_node_outputs
        }
        
        print(f"✅ Connected Analysis Structure:")
        print(f"   Total nodes: {connected_analysis['total_nodes']}")
        print(f"   Node types: {connected_analysis['node_types']}")
        print(f"   DataFrames found: {connected_analysis['dataframes_count']}")
        
        # Test comprehensive data preparation
        comprehensive_data = workflow_service._prepare_comprehensive_data_for_ai(connected_analysis)
        
        print(f"\n📋 Comprehensive Data Structure:")
        print_dict_structure(comprehensive_data, "Comprehensive Data", max_depth=2)
        
        # Test AI insights generation
        print("\n4. TESTING AI INSIGHTS GENERATION")
        print("-" * 60)
        
        result = ai_service.generate_comprehensive_workflow_insights(comprehensive_data)
        
        print(f"🤖 AI Analysis Result:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Task ID: {result.get('task_id', 'None')}")
        
        if result.get('success'):
            metadata = result.get('metadata', {})
            print(f"   Nodes analyzed: {metadata.get('nodes_analyzed', 0)}")
            print(f"   Analysis type: {metadata.get('analysis_type', 'unknown')}")
            print(f"   Background processing: {result.get('background_processing', False)}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()

def determine_node_type(node_data):
    """Determine node type from data structure"""
    if isinstance(node_data, pd.DataFrame):
        return 'data_source'
    elif isinstance(node_data, dict):
        if 'cleaning_summary' in node_data:
            return 'data_cleaning'
        elif 'basic_stats' in node_data or 'correlations' in node_data:
            return 'statistical_analysis'
        elif 'algorithm' in node_data or 'model_info' in node_data:
            return 'classification'
        elif 'anomalies' in node_data:
            return 'anomaly_detection'
        else:
            return 'unknown'
    else:
        return 'unknown'

def test_ai_summary_node_simulation():
    """Simulate the actual AI Summary node execution"""
    print("\n" + "=" * 80)
    print("SIMULATING AI SUMMARY NODE EXECUTION")
    print("=" * 80)
    
    workflow_service = create_test_workflow_data()
    
    # Create AI Summary node configuration
    ai_summary_node = {
        'id': 'ai_summary_1',
        'type': 'ai_summary',
        'name': 'AI Workflow Summary',
        'config': {
            'analysis_depth': 'comprehensive',
            'include_insights': True,
            'include_recommendations': True,
            'output_format': 'detailed'
        }
    }
    
    print(f"🤖 AI Summary Node Configuration:")
    print_dict_structure(ai_summary_node, "AI Node Config")
    
    # Simulate input data preparation (what _get_node_inputs would do for AI Summary)
    print(f"\n📥 Preparing AI Summary Node Inputs...")
    
    # AI Summary gets ALL executed node outputs
    all_node_outputs = {}
    for executed_node_id, node_output in workflow_service.data_cache.items():
        all_node_outputs[executed_node_id] = node_output
        print(f"   📊 Including: {executed_node_id}")
    
    ai_summary_input = {'default': all_node_outputs}
    
    print(f"\n📊 AI Summary Input Data:")
    print_dict_structure(ai_summary_input, "AI Summary Input", max_depth=2)
    
    # Test the actual AI Summary processor
    print(f"\n🚀 Executing AI Summary Processor...")
    
    try:
        result = workflow_service._process_ai_summary(ai_summary_node, ai_summary_input)
        
        print(f"✅ AI Summary Execution Complete!")
        print(f"📋 Result Structure:")
        print_dict_structure(result, "AI Summary Result", max_depth=2)
        
        # Check if background processing was initiated
        if 'task_id' in result:
            print(f"\n🔄 Background Processing:")
            print(f"   Task ID: {result['task_id']}")
            print(f"   Status: {result.get('status', 'unknown')}")
            
            # Check if we can get streaming results
            task_id = result['task_id']
            if task_id in workflow_service.streaming_results:
                streaming_result = workflow_service.streaming_results[task_id]
                print(f"   Streaming result available: {len(str(streaming_result))} characters")
            
    except Exception as e:
        print(f"❌ AI Summary execution failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test execution"""
    print("🚀 Starting AI Node Data Flow Testing...")
    print("=" * 80)
    
    try:
        # Test 1: Data preparation for AI Summary
        test_ai_summary_data_preparation()
        
        # Test 2: Simulate actual AI Summary node execution
        test_ai_summary_node_simulation()
        
        print("\n" + "=" * 80)
        print("✅ ALL AI NODE TESTS COMPLETED!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
