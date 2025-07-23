#!/usr/bin/env python3
"""
Test the prompt router directly with actual anomaly data
"""

import sys
import os

# Add paths for imports
sys.path.append('enhanced-backend/app/api/ai')

def test_prompt_router():
    """Test the NodePromptRouter directly"""
    print("🧪 Testing NodePromptRouter")
    print("=" * 50)
    
    try:
        from node_prompt_router import NodePromptRouter
        print("✅ Successfully imported NodePromptRouter")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return
    
    # Create router instance
    router = NodePromptRouter()
    
    # Test data from our diagnostic
    test_data = {
        'anomaly_results': {
            'CO(GT)': {
                'anomalies_detected': 45,
                'anomaly_indices': [123, 456, 789, 1234, 5678],
                'threshold_lower': -2.5,
                'threshold_upper': 8.9,
                'method': 'IQR',
                'anomaly_percentage': 0.048
            },
            'C6H6(GT)': {
                'anomalies_detected': 32,
                'anomaly_indices': [234, 567, 890, 2345],
                'threshold_lower': -1.2,
                'threshold_upper': 15.6,
                'method': 'IQR',
                'anomaly_percentage': 0.034
            },
            'NOx(GT)': {
                'anomalies_detected': 67,
                'anomaly_indices': [345, 678, 901, 3456, 6789, 8901],
                'threshold_lower': -50.2,
                'threshold_upper': 425.8,
                'method': 'IQR',
                'anomaly_percentage': 0.072
            }
        },
        'total_anomalies': 144,
        'detection_method': 'IQR (Interquartile Range)',
        'detection_settings': {
            'iqr_factor': 1.5,
            'columns_analyzed': 13,
            'exclude_categorical': True
        },
        'summary_statistics': {
            'total_data_points': 93572,
            'total_anomalies_found': 144,
            'overall_anomaly_rate': 0.154,
            'most_anomalous_column': 'NOx(GT)',
            'least_anomalous_column': 'RH'
        }
    }
    
    print("\n📊 Test Data:")
    print(f"  - Node type: univariate_anomaly_detection")
    print(f"  - Has anomaly_results: {'anomaly_results' in test_data}")
    print(f"  - Total anomalies: {test_data.get('total_anomalies', 'N/A')}")
    
    # Test the router generation
    try:
        print("\n🎯 Testing prompt router...")
        
        prompt = router.generate_prompt(
            node_type='univariate_anomaly_detection',
            data=test_data,
            node_id='test_anomaly_node',
            context={}
        )
        
        print(f"✅ SUCCESS: Prompt generated ({len(prompt)} characters)")
        
        # Check for detailed prompt indicators
        checks = [
            ("ANOMALY DETECTION INTELLIGENCE CENTER", "Enhanced prompt header"),
            ("DETAILED ANOMALY BREAKDOWN", "Column breakdown section"),
            ("CO(GT)", "Column-specific data"),
            ("threshold", "Threshold information"),
            ("IQR", "Detection method"),
            ("144", "Total anomalies count"),
            ("Advanced prompt generation temporarily unavailable", "Fallback prompt indicator")
        ]
        
        print(f"\n🔍 Content Checks:")
        for check_text, description in checks:
            found = check_text in prompt
            status = "✅" if found else "❌"
            is_bad = "Fallback prompt indicator" in description
            if is_bad and found:
                status = "🚨"
            print(f"  {status} {description}: {'Found' if found else 'Missing'}")
        
        # Show prompt preview
        print(f"\n📝 Prompt Preview (first 800 characters):")
        print("-" * 80)
        print(prompt[:800] + "..." if len(prompt) > 800 else prompt)
        print("-" * 80)
        
        # Save full prompt for inspection
        with open('test_router_prompt_output.txt', 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"\n💾 Full prompt saved to: test_router_prompt_output.txt")
        
        # Final assessment
        if "ANOMALY DETECTION INTELLIGENCE CENTER" in prompt:
            print(f"\n🎉 SUCCESS: Enhanced anomaly prompt generated via router!")
            return True
        elif "Advanced prompt generation temporarily unavailable" in prompt:
            print(f"\n🚨 FALLBACK: Router is using fallback prompt instead of enhanced prompt")
            return False
        else:
            print(f"\n❓ UNKNOWN: Unexpected prompt format")
            return False
        
    except Exception as e:
        print(f"❌ ERROR generating prompt via router: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting NodePromptRouter Test")
    print("=" * 60)
    
    router_success = test_prompt_router()
    
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    print(f"Router Generation: {'✅ PASS' if router_success else '❌ FAIL'}")
    
    if router_success:
        print(f"\n🎉 ROUTER WORKING!")
        print(f"   The NodePromptRouter is correctly generating enhanced prompts.")
    else:
        print(f"\n❌ ROUTER ISSUE!")
        print(f"   The NodePromptRouter is falling back to basic prompts.")
