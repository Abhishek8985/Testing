"""
Test AI Summary Charts Display - Verify that charts from connected nodes are included in AI Summary results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'enhanced-backend'))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from app.services.workflow_service import AdvancedWorkflowService

def create_workflow_with_charts():
    """Create a workflow with multiple nodes that generate charts"""
    print("🔧 Creating workflow with chart-generating nodes...")
    
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
    
    workflow_service = AdvancedWorkflowService()
    
    # 1. Data Source Node
    print("  ✅ Adding Data Source Node...")
    workflow_service.data_cache['data_source_1'] = test_df
    
    # 2. Statistical Analysis Node (generates charts)
    print("  ✅ Adding Statistical Analysis Node with charts...")
    numeric_cols = test_df.select_dtypes(include=[np.number])
    stats_result = {
        'basic_stats': {
            'count': numeric_cols.describe().loc['count'].to_dict(),
            'mean': numeric_cols.describe().loc['mean'].to_dict(),
            'std': numeric_cols.describe().loc['std'].to_dict(),
        },
        'correlations': numeric_cols.corr().to_dict(),
        'data': test_df,
        'charts_generated': 3
    }
    workflow_service.data_cache['statistical_analysis_1'] = stats_result
    
    # Add charts to chart cache (simulating chart generation)
    workflow_service.chart_cache['statistical_analysis_1'] = {
        'correlation_heatmap': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
        'distribution_plot': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
        'box_plot': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
    }
    
    # 3. EDA Analysis Node (generates charts)
    print("  ✅ Adding EDA Analysis Node with charts...")
    eda_result = {
        'eda_results': {
            'summary_statistics': {
                'total_features': len(test_df.columns),
                'numeric_features': len(numeric_cols.columns),
                'missing_values': test_df.isnull().sum().to_dict()
            },
            'insights': [
                'Dataset contains 200 observations across 7 features',
                'No missing values detected',
                'Balanced distribution of categorical variables'
            ]
        },
        'charts_generated': 4
    }
    workflow_service.data_cache['eda_analysis_1'] = eda_result
    
    # Add EDA charts to chart cache
    workflow_service.chart_cache['eda_analysis_1'] = {
        'feature_distributions': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
        'missing_values_heatmap': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
        'correlation_matrix': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
        'categorical_plots': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
    }
    
    # 4. Basic Plots Node (generates charts)
    print("  ✅ Adding Basic Plots Node with charts...")
    basic_plots_result = {
        'charts': {
            'scatter_plot_feature_a': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
            'histogram_feature_b': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
        },
        'chart_count': 2,
        'plot_type': 'scatter',
        'data': test_df
    }
    workflow_service.data_cache['basic_plots_1'] = basic_plots_result
    
    # Add basic plots charts to chart cache
    workflow_service.chart_cache['basic_plots_1'] = basic_plots_result['charts']
    
    print(f"  📊 Created workflow with {len(workflow_service.data_cache)} nodes")
    print(f"  📈 Created {sum(len(charts) for charts in workflow_service.chart_cache.values())} total charts")
    
    return workflow_service

def test_ai_summary_charts_inclusion():
    """Test that AI Summary includes charts from all connected nodes"""
    print("\n" + "=" * 80)
    print("TESTING AI SUMMARY CHARTS INCLUSION")
    print("=" * 80)
    
    workflow_service = create_workflow_with_charts()
    
    # Show current chart cache state
    print("\n1. CHART CACHE STATE BEFORE AI SUMMARY")
    print("-" * 60)
    for node_id, charts in workflow_service.chart_cache.items():
        if isinstance(charts, dict):
            chart_names = list(charts.keys())
            print(f"📊 {node_id}: {len(charts)} charts - {chart_names}")
        elif isinstance(charts, list):
            print(f"📊 {node_id}: {len(charts)} charts (list format)")
        else:
            print(f"📊 {node_id}: 1 chart (single format)")
    
    total_charts_before = sum(len(charts) if isinstance(charts, (dict, list)) else 1 
                             for charts in workflow_service.chart_cache.values())
    print(f"\n📈 Total charts in cache: {total_charts_before}")
    
    # Create AI Summary node configuration
    ai_summary_node = {
        'id': 'ai_summary_1',
        'type': 'ai_summary',
        'name': 'AI Workflow Summary with Charts',
        'config': {
            'analysis_depth': 'comprehensive',
            'include_insights': True,
            'include_recommendations': True,
            'output_format': 'detailed'
        }
    }
    
    # Prepare AI Summary input (all connected node outputs)
    ai_summary_input = {'default': workflow_service.data_cache}
    
    print(f"\n2. EXECUTING AI SUMMARY NODE")
    print("-" * 60)
    print(f"🤖 AI Summary processing {len(ai_summary_input['default'])} connected nodes...")
    
    # Execute AI Summary processor
    try:
        result = workflow_service._process_ai_summary(ai_summary_node, ai_summary_input)
        
        print(f"✅ AI Summary execution completed!")
        
        # Check if charts are included in result
        print(f"\n3. ANALYZING AI SUMMARY RESULT FOR CHARTS")
        print("-" * 60)
        
        # Check for charts in the result
        charts_in_result = result.get('charts', {})
        chart_count_in_result = result.get('chart_count', 0)
        
        print(f"📊 Charts found in AI Summary result: {len(charts_in_result)} collections")
        print(f"📈 Total chart count in result: {chart_count_in_result}")
        
        if charts_in_result:
            print(f"\n📋 Chart Collections in AI Summary:")
            for collection_name, charts in charts_in_result.items():
                if isinstance(charts, dict):
                    chart_names = list(charts.keys())
                    print(f"  🎨 {collection_name}: {len(charts)} charts")
                    for chart_name in chart_names:
                        chart_data = charts[chart_name]
                        chart_type = "base64 image" if chart_data.startswith('data:image') else "unknown format"
                        print(f"     • {chart_name}: {chart_type}")
                elif isinstance(charts, list):
                    print(f"  🎨 {collection_name}: {len(charts)} charts (list format)")
                else:
                    print(f"  🎨 {collection_name}: 1 chart")
        else:
            print(f"❌ No charts found in AI Summary result!")
        
        # Check AI analysis content
        ai_analysis = result.get('ai_analysis', {})
        print(f"\n4. AI ANALYSIS CONTENT")
        print("-" * 60)
        print(f"✅ AI Analysis Success: {ai_analysis.get('success', False)}")
        print(f"🔄 Processing Mode: {ai_analysis.get('source', 'unknown')}")
        print(f"📝 Insights Available: {'insights' in ai_analysis}")
        
        # Check workflow summary
        workflow_summary = result.get('workflow_summary', {})
        print(f"\n5. WORKFLOW SUMMARY")
        print("-" * 60)
        print(f"🔗 Connected Nodes: {workflow_summary.get('total_connected_nodes', 0)}")
        print(f"📊 DataFrames Found: {workflow_summary.get('dataframes_found', 0)}")
        print(f"📈 Models Found: {workflow_summary.get('models_found', 0)}")
        print(f"📋 Statistics Found: {workflow_summary.get('statistics_found', 0)}")
        
        # Verify chart inclusion
        print(f"\n6. CHART INCLUSION VERIFICATION")
        print("-" * 60)
        
        expected_charts = total_charts_before
        actual_charts = chart_count_in_result
        
        print(f"📊 Expected charts: {expected_charts}")
        print(f"📈 Actual charts in result: {actual_charts}")
        
        if actual_charts == expected_charts:
            print(f"✅ SUCCESS: All charts correctly included in AI Summary!")
        elif actual_charts > 0:
            print(f"⚠️  PARTIAL: Some charts included ({actual_charts}/{expected_charts})")
        else:
            print(f"❌ FAILURE: No charts included in AI Summary result")
        
        return result
        
    except Exception as e:
        print(f"❌ AI Summary execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_result_structure(result, max_depth=2, current_depth=0):
    """Print the structure of the AI Summary result"""
    if current_depth > max_depth:
        return
    
    indent = "  " * current_depth
    
    if isinstance(result, dict):
        for key, value in result.items():
            if key == 'charts' and isinstance(value, dict):
                print(f"{indent}{key}: {len(value)} chart collections")
                for chart_collection, charts in value.items():
                    if isinstance(charts, dict):
                        print(f"{indent}  📊 {chart_collection}: {len(charts)} charts")
                    else:
                        print(f"{indent}  📊 {chart_collection}: 1 chart")
            elif isinstance(value, (dict, list)) and current_depth < max_depth:
                print(f"{indent}{key}:")
                print_result_structure(value, max_depth, current_depth + 1)
            else:
                value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"{indent}{key}: {value_str}")
    elif isinstance(result, list):
        print(f"{indent}List with {len(result)} items")

def main():
    """Main test execution"""
    print("🚀 Starting AI Summary Charts Test...")
    print("=" * 80)
    
    try:
        result = test_ai_summary_charts_inclusion()
        
        if result:
            print("\n" + "=" * 80)
            print("✅ AI SUMMARY CHARTS TEST COMPLETED!")
            print("=" * 80)
            
            print(f"\n📋 Final Result Structure:")
            print_result_structure(result)
        else:
            print("\n" + "=" * 80)
            print("❌ AI SUMMARY CHARTS TEST FAILED!")
            print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
