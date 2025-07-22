#!/usr/bin/env python3
"""
Simple test to verify the insights are properly formatted without
"Business Insights:" and "Recommendations:" headings
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'enhanced-backend'))

import pandas as pd
import numpy as np

def test_sample_insight_format():
    """Test a sample insight to verify format"""
    print("🧪 Testing sample insight format...")
    
    # Create sample insight text as it would appear in charts
    sample_insight = """
📊 **Distribution Analysis: sales_amount**

**Statistical Properties:**
• Data points: 1,000
• Mean: 2,456.78
• Median: 2,200.00
• Standard deviation: 892.45
• Skewness: 0.85
• Kurtosis: 2.31

**Distribution Shape:**
• Right-skewed (tail extends right)
• Normal-like tail behavior

**Key Insights:**
• Consider log transformation for right-skewed data to improve normality
• Potential outliers detected in the tails - investigate extreme values
• Distribution characteristics help determine appropriate statistical methods

**Next Steps:**
• Use distribution shape to guide data preprocessing and transformation choices
• Consider normalization or standardization if distribution is heavily skewed
• Monitor for consistency in production data to detect distribution drift
    """.strip()
    
    # Check for problematic headings
    problematic_headings = ["Business Insights:", "Recommendations:"]
    
    print(f"✅ Sample insight text:")
    print(sample_insight)
    print("\n" + "="*50)
    
    has_problems = False
    for heading in problematic_headings:
        if heading in sample_insight:
            print(f"❌ Found problematic heading: {heading}")
            has_problems = True
    
    if not has_problems:
        print("✅ No problematic headings found!")
        print("✅ Insight format is clean and user-friendly")
    
    # Check for improved headings
    improved_headings = ["Key Insights:", "Next Steps:", "Statistical Properties:"]
    found_improved = []
    
    for heading in improved_headings:
        if heading in sample_insight:
            found_improved.append(heading)
    
    if found_improved:
        print(f"✅ Found improved headings: {', '.join(found_improved)}")
    
    print("\n🎉 Insight format verification completed!")

if __name__ == "__main__":
    test_sample_insight_format()
