"""
Test Node Outputs - Print and Examine Node Results Structure
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'enhanced-backend'))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from app.services.workflow_service import AdvancedWorkflowService

def print_dict_structure(data, name="Data", indent=0, max_depth=3):
    """Recursively print dictionary structure with controlled depth"""
    prefix = "  " * indent
    
    if indent > max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    if isinstance(data, dict):
        print(f"{prefix}{name} (Dict): {len(data)} keys")
        for key, value in list(data.items())[:10]:  # Limit to first 10 items
            if isinstance(value, (dict, list)):
                print_dict_structure(value, f"'{key}'", indent + 1, max_depth)
            elif isinstance(value, pd.DataFrame):
                print(f"{prefix}  '{key}' (DataFrame): shape {value.shape}, columns: {list(value.columns)[:5]}...")
            elif isinstance(value, np.ndarray):
                print(f"{prefix}  '{key}' (Array): shape {value.shape}, dtype: {value.dtype}")
            else:
                value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                print(f"{prefix}  '{key}' ({type(value).__name__}): {value_str}")
        
        if len(data) > 10:
            print(f"{prefix}  ... and {len(data) - 10} more items")
    
    elif isinstance(data, list):
        print(f"{prefix}{name} (List): {len(data)} items")
        if data and indent < max_depth:
            print_dict_structure(data[0], "First Item", indent + 1, max_depth)
            if len(data) > 1:
                print(f"{prefix}  ... and {len(data) - 1} more items")
    
    elif isinstance(data, pd.DataFrame):
        print(f"{prefix}{name} (DataFrame): shape {data.shape}")
        print(f"{prefix}  Columns: {list(data.columns)}")
        print(f"{prefix}  Sample data:")
        print(data.head(2).to_string(max_cols=5).replace('\n', f'\n{prefix}    '))
    
    elif isinstance(data, np.ndarray):
        print(f"{prefix}{name} (Array): shape {data.shape}, dtype: {data.dtype}")
        if data.size < 20:
            print(f"{prefix}  Values: {data.flatten()}")
    
    else:
        value_str = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
        print(f"{prefix}{name} ({type(data).__name__}): {value_str}")

def test_workflow_execution_and_outputs():
    """Test workflow execution and examine all node outputs"""
    print("=" * 80)
    print("TESTING WORKFLOW NODE OUTPUTS")
    print("=" * 80)
    
    # Create workflow service instance
    workflow_service = AdvancedWorkflowService()
    
    # Create test DataFrame for mock data source
    test_df = pd.DataFrame({
        'id': range(1, 101),
        'feature_a': np.random.randn(100),
        'feature_b': np.random.randn(100) * 2 + 5,
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100),
        'date': pd.date_range('2024-01-01', periods=100, freq='D')
    })
    
    # Manually populate data cache with mock node outputs
    print("\n1. SIMULATING NODE EXECUTION RESULTS")
    print("-" * 50)
    
    # Data Source Node
    workflow_service.data_cache['data_source_1'] = test_df
    print("✅ Data Source Node - Added test DataFrame")
    
    # Data Cleaning Node
    cleaned_data = {
        'data': test_df.dropna(),
        'cleaning_summary': {
            'original_shape': test_df.shape,
            'final_shape': test_df.dropna().shape,
            'rows_removed': 0,
            'columns_removed': 0,
            'operations_performed': ['No missing values found'],
            'data_quality_score': 98.5
        },
        'type': 'cleaned_data'
    }
    workflow_service.data_cache['data_cleaning_1'] = cleaned_data
    print("✅ Data Cleaning Node - Added cleaned data with summary")
    
    # Descriptive Stats Node
    numeric_cols = test_df.select_dtypes(include=[np.number])
    stats_result = {
        'basic_stats': {
            'count': numeric_cols.describe().loc['count'].to_dict(),
            'mean': numeric_cols.describe().loc['mean'].to_dict(),
            'std': numeric_cols.describe().loc['std'].to_dict(),
            'min': numeric_cols.describe().loc['min'].to_dict(),
            'max': numeric_cols.describe().loc['max'].to_dict()
        },
        'data_types': test_df.dtypes.astype(str).to_dict(),
        'missing_values': test_df.isnull().sum().to_dict(),
        'correlations': numeric_cols.corr().to_dict(),
        'charts_generated': 3,
        'data': test_df  # Include the data for downstream processing
    }
    workflow_service.data_cache['stats_1'] = stats_result
    print("✅ Descriptive Stats Node - Added statistical analysis")
    
    # Classification Node
    ml_result = {
        'algorithm': 'random_forest',
        'model': 'MockRandomForestModel',  # In real scenario, this would be the actual model
        'predictions': np.random.randint(0, 2, 20),
        'metrics': {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1_score': 0.85
        },
        'feature_importance': {
            'feature_a': 0.35,
            'feature_b': 0.42,
            'category_encoded': 0.23
        },
        'training_data_shape': (80, 3),
        'test_data_shape': (20, 3)
    }
    workflow_service.data_cache['classification_1'] = ml_result
    print("✅ Classification Node - Added ML model results")
    
    # Chart Cache for visualization nodes
    workflow_service.chart_cache['stats_1'] = [
        {
            'type': 'histogram',
            'title': 'Feature A Distribution',
            'data': 'base64_encoded_chart_data_here'
        },
        {
            'type': 'correlation_heatmap',
            'title': 'Feature Correlations',
            'data': 'base64_encoded_heatmap_data_here'
        }
    ]
    print("✅ Chart Cache - Added visualization data")
    
    print(f"\nTotal nodes in data cache: {len(workflow_service.data_cache)}")
    print(f"Total charts in chart cache: {len(workflow_service.chart_cache)}")
    
    # 2. EXAMINE DATA CACHE CONTENTS
    print("\n2. EXAMINING DATA CACHE CONTENTS")
    print("-" * 50)
    
    for node_id, node_output in workflow_service.data_cache.items():
        print(f"\n🔍 NODE: {node_id}")
        print("=" * 60)
        print_dict_structure(node_output, f"Node Output", max_depth=2)
    
    # 3. EXAMINE CHART CACHE CONTENTS
    print("\n3. EXAMINING CHART CACHE CONTENTS")
    print("-" * 50)
    
    for node_id, charts in workflow_service.chart_cache.items():
        print(f"\n📊 CHARTS FOR NODE: {node_id}")
        print("=" * 60)
        print_dict_structure(charts, f"Charts", max_depth=2)
    
    # 4. TEST AI SUMMARY DATA COLLECTION
    print("\n4. TESTING AI SUMMARY DATA COLLECTION")
    print("-" * 50)
    
    # Simulate what AI Summary node would receive
    all_node_outputs = {}
    for executed_node_id, node_output in workflow_service.data_cache.items():
        all_node_outputs[executed_node_id] = node_output
        print(f"✅ Collected data from {executed_node_id}: {type(node_output).__name__}")
    
    print(f"\nAI Summary would receive data from {len(all_node_outputs)} nodes:")
    print_dict_structure(all_node_outputs, "AI Summary Input", max_depth=1)
    
    # 5. TEST OUTPUT SUMMARY GENERATION
    print("\n5. TESTING OUTPUT SUMMARY GENERATION")
    print("-" * 50)
    
    for node_id, node_output in workflow_service.data_cache.items():
        try:
            summary = workflow_service._get_output_summary(node_output)
            print(f"\n📋 SUMMARY FOR {node_id}:")
            print_dict_structure(summary, "Output Summary", max_depth=2)
        except Exception as e:
            print(f"❌ Error generating summary for {node_id}: {e}")
    
    # 6. TEST SERIALIZATION
    print("\n6. TESTING DATA SERIALIZATION")
    print("-" * 50)
    
    for node_id, node_output in workflow_service.data_cache.items():
        try:
            # Test conversion to JSON-serializable format
            serializable_data = workflow_service.convert_numpy_types(node_output)
            
            # Try to serialize to JSON (this will fail if not properly converted)
            json_str = json.dumps(serializable_data, default=str)
            print(f"✅ {node_id}: Successfully serialized ({len(json_str)} characters)")
            
        except Exception as e:
            print(f"❌ {node_id}: Serialization failed - {e}")
    
    # 7. TEST WORKFLOW RESULTS FORMAT
    print("\n7. TESTING FRONTEND RESULTS FORMAT")
    print("-" * 50)
    
    # Simulate what gets sent to frontend
    frontend_results = {}
    for node_id, node_output in workflow_service.data_cache.items():
        try:
            result_summary = workflow_service._get_output_summary(node_output)
            
            frontend_result = {
                'status': 'completed',
                'result_type': result_summary.get('type'),
                'result_summary': result_summary,
                'execution_time': 1.25,  # Mock execution time
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add charts if available
            if node_id in workflow_service.chart_cache:
                frontend_result['charts'] = workflow_service.chart_cache[node_id]
                frontend_result['chart_count'] = len(workflow_service.chart_cache[node_id])
            
            frontend_results[node_id] = frontend_result
            print(f"✅ Frontend result for {node_id} prepared")
            
        except Exception as e:
            print(f"❌ Error preparing frontend result for {node_id}: {e}")
    
    print(f"\nFrontend Results Structure:")
    print_dict_structure(frontend_results, "Frontend Results", max_depth=2)
    
    # 8. MEMORY USAGE ANALYSIS
    print("\n8. MEMORY USAGE ANALYSIS")
    print("-" * 50)
    
    total_memory = 0
    for node_id, node_output in workflow_service.data_cache.items():
        try:
            if isinstance(node_output, pd.DataFrame):
                memory_mb = node_output.memory_usage(deep=True).sum() / 1024**2
                total_memory += memory_mb
                print(f"📊 {node_id}: {memory_mb:.2f} MB (DataFrame)")
            elif isinstance(node_output, dict):
                # Estimate memory for dict (rough approximation)
                memory_estimate = sys.getsizeof(str(node_output)) / 1024**2
                total_memory += memory_estimate
                print(f"📋 {node_id}: ~{memory_estimate:.2f} MB (Dict)")
            else:
                memory_estimate = sys.getsizeof(node_output) / 1024**2
                total_memory += memory_estimate
                print(f"🔧 {node_id}: ~{memory_estimate:.2f} MB ({type(node_output).__name__})")
        except Exception as e:
            print(f"❌ Error calculating memory for {node_id}: {e}")
    
    print(f"\n💾 Total estimated memory usage: {total_memory:.2f} MB")
    
    return workflow_service

def test_specific_node_output_types():
    """Test specific node output types and their structures"""
    print("\n" + "=" * 80)
    print("TESTING SPECIFIC NODE OUTPUT TYPES")
    print("=" * 80)
    
    # Test different data types that nodes might output
    test_outputs = {
        'dataframe_output': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}),
        'numpy_array_output': np.array([[1, 2], [3, 4]]),
        'list_output': [1, 2, 3, {'nested': 'value'}],
        'complex_dict_output': {
            'statistics': {'mean': 1.5, 'std': 0.5},
            'data': pd.DataFrame({'X': [1, 2], 'Y': [3, 4]}),
            'metadata': {
                'created_at': datetime.now(),
                'version': 1.0
            }
        },
        'ml_model_output': {
            'model_type': 'RandomForest',
            'parameters': {'n_estimators': 100, 'max_depth': 10},
            'performance': {
                'train_score': 0.95,
                'test_score': 0.87,
                'cv_scores': [0.85, 0.88, 0.90, 0.86, 0.89]
            },
            'predictions': np.random.rand(50),
            'feature_names': ['feature_1', 'feature_2', 'feature_3']
        }
    }
    
    workflow_service = AdvancedWorkflowService()
    
    for output_name, output_data in test_outputs.items():
        print(f"\n🧪 TESTING: {output_name}")
        print("-" * 60)
        
        # Test output summary generation
        try:
            summary = workflow_service._get_output_summary(output_data)
            print("Output Summary:")
            print_dict_structure(summary, "Summary")
        except Exception as e:
            print(f"❌ Summary generation failed: {e}")
        
        # Test serialization
        try:
            serialized = workflow_service.convert_numpy_types(output_data)
            json.dumps(serialized, default=str)
            print("✅ Serialization: SUCCESS")
        except Exception as e:
            print(f"❌ Serialization failed: {e}")
        
        # Show structure
        print("Data Structure:")
        print_dict_structure(output_data, "Output Data", max_depth=2)

if __name__ == "__main__":
    print("🚀 Starting Node Output Testing...")
    
    try:
        # Run main workflow test
        workflow_service = test_workflow_execution_and_outputs()
        
        # Run specific output type tests
        test_specific_node_output_types()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print(f"\n📊 Final Cache Status:")
        print(f"   Data Cache: {len(workflow_service.data_cache)} nodes")
        print(f"   Chart Cache: {len(workflow_service.chart_cache)} nodes")
        print(f"   Memory Threshold: {workflow_service.memory_threshold / 1024**3:.1f} GB")
        print(f"   Max Cache Size: {workflow_service.max_cache_size} items")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
