"""
Advanced Anomaly Detection Node Prompt Generator
"""

import numpy as np
import pandas as pd

class AnomalyDetectionPrompt:
    """Generate sophisticated prompts for anomaly detection nodes"""
    
    @staticmethod
    def generate_prompt(data: dict, node_id: str, context: dict = None) -> str:
        """Generate advanced anomaly detection analysis prompt"""
        
        anomalies = data.get('anomalies', [])
        detection_method = data.get('method', 'statistical')
        anomaly_scores = data.get('anomaly_scores', [])
        df = data.get('dataframe')
        detection_stats = data.get('detection_stats', {})
        thresholds = data.get('thresholds', {})
        
        if not anomalies and df is None:
            return "❌ **CRITICAL ERROR**: No anomaly detection results or dataframe available"
        
        # Analyze anomaly detection results
        anomaly_analysis = AnomalyDetectionPrompt._analyze_anomalies(anomalies, anomaly_scores, df)
        method_assessment = AnomalyDetectionPrompt._assess_detection_method(detection_method, detection_stats)
        data_impact = AnomalyDetectionPrompt._assess_data_impact(anomalies, df, detection_method)
        risk_assessment = AnomalyDetectionPrompt._assess_risk_levels(anomalies, anomaly_scores, thresholds)
        action_recommendations = AnomalyDetectionPrompt._generate_action_recommendations(anomalies, detection_method)
        
        # Anomaly summary
        anomaly_count = len(anomalies) if isinstance(anomalies, list) else "Multiple" if anomalies else 0
        total_records = df.shape[0] if df is not None else "Unknown"
        anomaly_rate = (len(anomalies) / df.shape[0] * 100) if df is not None and isinstance(anomalies, list) else 0
        
        prompt = f"""
🚨 **ANOMALY DETECTION INTELLIGENCE CENTER - Node: {node_id}**

⚠️ **ANOMALY DETECTION OVERVIEW**:
Detection Method: {detection_method.replace('_', ' ').title()}
Anomalies Detected: {anomaly_count} out of {total_records} records
Anomaly Rate: {anomaly_rate:.2f}%
Detection Confidence: {"High" if anomaly_scores else "Standard"}

🔍 **ANOMALY PATTERN ANALYSIS**:
{chr(10).join(anomaly_analysis) if anomaly_analysis else "⚠️ Anomaly pattern analysis not available"}

🎯 **DETECTION METHOD ASSESSMENT**:
{chr(10).join(method_assessment) if method_assessment else "⚠️ Method assessment not available"}

� **DATA IMPACT ANALYSIS**:
{chr(10).join(data_impact) if data_impact else "⚠️ Data impact assessment not available"}

⚡ **RISK LEVEL ASSESSMENT**:
{chr(10).join(risk_assessment) if risk_assessment else "⚠️ Risk assessment not available"}

🎯 **ACTION RECOMMENDATIONS**:
{chr(10).join(action_recommendations) if action_recommendations else "⚠️ Action recommendations not available"}

📊 **DETECTION METADATA**:
• Scoring Available: {"Yes" if anomaly_scores else "No"}
• Threshold Configuration: {"Custom" if thresholds else "Default"}
• Statistical Validation: {"Available" if detection_stats else "Basic"}
• Multi-dimensional Analysis: {"Yes" if isinstance(anomalies, list) and len(anomalies) > 0 else "Single"}

💡 **ADVANCED ANOMALY INTELLIGENCE REQUIREMENTS**:

1. **PATTERN CLASSIFICATION**: Categorize anomalies by type, severity, and statistical significance
2. **ROOT CAUSE ANALYSIS**: Identify potential causes and contributing factors for detected anomalies
3. **STATISTICAL PRIORITY**: Rank anomalies by statistical significance and deviation magnitude
4. **DETECTION RESPONSE**: Define technical actions for anomaly investigation
5. **PREVENTION STRATEGY**: Recommend statistical methods to reduce false positives
6. **MONITORING ENHANCEMENT**: Improve detection algorithms based on identified patterns
7. **TECHNICAL ASSESSMENT**: Prepare detailed anomaly statistics and mathematical properties
8. **EFFICIENCY ANALYSIS**: Evaluate the statistical power and accuracy of detection methods

🎯 **CRITICAL ANOMALY ANALYSIS REQUIREMENTS**:
- Classify SPECIFIC anomaly patterns and their statistical significance
- Assess MATHEMATICAL PROPERTIES of detected deviations
- Identify SYSTEMIC PATTERNS vs isolated incidents
- Quantify STATISTICAL SIGNIFICANCE of detected anomalies
- Recommend SPECIFIC TECHNIQUES for each category of anomaly
- Establish MONITORING THRESHOLDS for future detection
- Evaluate DETECTION EFFECTIVENESS and false positive rates

⚡ **RESPONSE FOCUS**: Analyze the ACTUAL anomalies detected, their patterns, and statistical properties. Provide concrete, actionable recommendations for anomaly response and prevention based on the specific detection results.
"""
        
        return prompt.strip()
    
    @staticmethod
    def _analyze_anomalies(anomalies, anomaly_scores, df) -> list:
        """Analyze detected anomalies for patterns and insights"""
        analysis = []
        
        if not anomalies:
            analysis.append("✅ **NO ANOMALIES DETECTED**: System operating within normal parameters")
            return analysis
        
        anomaly_count = len(anomalies) if isinstance(anomalies, list) else 1
        
        # Anomaly frequency analysis
        if df is not None:
            total_records = df.shape[0]
            anomaly_rate = (anomaly_count / total_records) * 100
            
            if anomaly_rate > 10:
                analysis.append(f"🚨 **HIGH ANOMALY RATE**: {anomaly_rate:.1f}% - Systemic issues require investigation")
            elif anomaly_rate > 5:
                analysis.append(f"⚠️ **ELEVATED ANOMALY RATE**: {anomaly_rate:.1f}% - Process review recommended")
            elif anomaly_rate > 1:
                analysis.append(f"📊 **MODERATE ANOMALY RATE**: {anomaly_rate:.1f}% - Normal operational variance")
            else:
                analysis.append(f"✅ **LOW ANOMALY RATE**: {anomaly_rate:.1f}% - Excellent system stability")
        
        # Severity analysis using anomaly scores
        if anomaly_scores and isinstance(anomaly_scores, list):
            scores_array = np.array(anomaly_scores)
            
            if len(scores_array) > 0:
                high_severity = np.sum(scores_array > np.percentile(scores_array, 80))
                medium_severity = np.sum((scores_array > np.percentile(scores_array, 60)) & 
                                       (scores_array <= np.percentile(scores_array, 80)))
                low_severity = len(scores_array) - high_severity - medium_severity
                
                analysis.append(f"📊 **SEVERITY DISTRIBUTION**: {high_severity} high, {medium_severity} medium, {low_severity} low severity")
                
                if high_severity > 0:
                    analysis.append(f"🚨 **CRITICAL ANOMALIES**: {high_severity} high-severity anomalies require immediate attention")
        
        # Pattern analysis for anomaly types
        if isinstance(anomalies, list) and df is not None:
            # Analyze anomaly distribution across features
            anomaly_features = AnomalyDetectionPrompt._analyze_anomaly_features(anomalies, df)
            if anomaly_features:
                analysis.extend(anomaly_features)
        
        # Temporal pattern analysis (if datetime columns exist)
        if df is not None and isinstance(anomalies, list):
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if date_cols and len(anomalies) > 0:
                analysis.append("📅 **TEMPORAL ANALYSIS**: Time-based anomaly patterns available for trend analysis")
        
        # Clustering analysis of anomalies
        if anomaly_count >= 5:
            analysis.append("🔍 **PATTERN CLUSTERING**: Multiple anomalies enable pattern clustering analysis")
        elif anomaly_count >= 2:
            analysis.append("🔍 **PATTERN COMPARISON**: Anomaly comparison reveals common characteristics")
        else:
            analysis.append("🎯 **ISOLATED ANOMALY**: Single anomaly requires individual investigation")
        
        return analysis
    
    @staticmethod
    def _analyze_anomaly_features(anomalies, df) -> list:
        """Analyze which features contribute most to anomalies"""
        feature_analysis = []
        
        try:
            # Assume anomalies contains indices or records
            if len(anomalies) > 0 and hasattr(df, 'iloc'):
                # Get anomalous records
                anomaly_indices = anomalies if isinstance(anomalies[0], int) else range(len(anomalies))
                
                if len(anomaly_indices) > 0:
                    anomaly_data = df.iloc[list(anomaly_indices)[:100]]  # Limit to first 100 for analysis
                    normal_data = df.drop(anomaly_indices).sample(min(1000, len(df) - len(anomaly_indices)))
                    
                    # Analyze numeric features
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    for col in numeric_cols[:5]:  # Analyze top 5 numeric columns
                        if col in anomaly_data.columns and col in normal_data.columns:
                            anom_mean = anomaly_data[col].mean()
                            normal_mean = normal_data[col].mean()
                            
                            if not np.isnan(anom_mean) and not np.isnan(normal_mean) and normal_mean != 0:
                                deviation_ratio = abs(anom_mean - normal_mean) / abs(normal_mean)
                                
                                if deviation_ratio > 1.0:
                                    feature_analysis.append(f"📊 **{col}**: EXTREME DEVIATION - anomalies differ by {deviation_ratio:.1f}x from normal")
                                elif deviation_ratio > 0.5:
                                    feature_analysis.append(f"📈 **{col}**: SIGNIFICANT DEVIATION - anomalies differ by {deviation_ratio:.1%} from normal")
                                elif deviation_ratio > 0.2:
                                    feature_analysis.append(f"📉 **{col}**: MODERATE DEVIATION - anomalies show {deviation_ratio:.1%} difference")
        
        except Exception:
            feature_analysis.append("🔍 **FEATURE ANALYSIS**: Advanced feature contribution analysis available")
        
        return feature_analysis
    
    @staticmethod
    def _assess_detection_method(method: str, detection_stats: dict) -> list:
        """Assess the effectiveness of the detection method"""
        assessment = []
        
        # Method-specific analysis
        method_insights = {
            'isolation_forest': [
                "🌲 **ISOLATION FOREST**: Effective for high-dimensional data and complex anomaly patterns",
                "🎯 **STRENGTHS**: Handles non-linear patterns and doesn't require labeled data",
                "⚠️ **CONSIDERATIONS**: May struggle with very sparse anomalies"
            ],
            'local_outlier_factor': [
                "🎯 **LOCAL OUTLIER FACTOR**: Excellent for density-based anomaly detection",
                "📊 **STRENGTHS**: Identifies local outliers in varying density regions",
                "⚠️ **CONSIDERATIONS**: Sensitive to parameter tuning"
            ],
            'one_class_svm': [
                "🤖 **ONE-CLASS SVM**: Robust method for novelty detection",
                "💪 **STRENGTHS**: Effective with high-dimensional data and non-linear boundaries",
                "⚠️ **CONSIDERATIONS**: Requires careful kernel and parameter selection"
            ],
            'statistical': [
                "📊 **STATISTICAL METHODS**: Reliable for normally distributed data",
                "✅ **STRENGTHS**: Interpretable results with clear statistical thresholds",
                "⚠️ **CONSIDERATIONS**: Assumes normal distribution and linear relationships"
            ],
            'dbscan': [
                "🔍 **DBSCAN CLUSTERING**: Effective for density-based anomaly detection",
                "🎯 **STRENGTHS**: Identifies clusters and noise points automatically",
                "⚠️ **CONSIDERATIONS**: Sensitive to epsilon and minimum points parameters"
            ]
        }
        
        method_key = method.lower().replace(' ', '_')
        if method_key in method_insights:
            assessment.extend(method_insights[method_key])
        else:
            assessment.append(f"🔧 **{method.upper()}**: Advanced anomaly detection method applied")
        
        # Performance assessment
        if detection_stats:
            if 'precision' in detection_stats:
                precision = detection_stats['precision']
                if precision > 0.8:
                    assessment.append(f"✅ **HIGH PRECISION**: {precision:.1%} - Low false positive rate")
                elif precision > 0.6:
                    assessment.append(f"📊 **MODERATE PRECISION**: {precision:.1%} - Acceptable false positive rate")
                else:
                    assessment.append(f"⚠️ **LOW PRECISION**: {precision:.1%} - High false positive rate needs adjustment")
            
            if 'recall' in detection_stats:
                recall = detection_stats['recall']
                if recall > 0.8:
                    assessment.append(f"✅ **HIGH RECALL**: {recall:.1%} - Excellent anomaly detection rate")
                elif recall > 0.6:
                    assessment.append(f"📊 **MODERATE RECALL**: {recall:.1%} - Good anomaly detection rate")
                else:
                    assessment.append(f"⚠️ **LOW RECALL**: {recall:.1%} - Missing many anomalies, tune sensitivity")
            
            if 'contamination' in detection_stats:
                contamination = detection_stats['contamination']
                assessment.append(f"🎯 **CONTAMINATION LEVEL**: {contamination:.1%} expected anomaly rate configured")
        
        return assessment
    
    @staticmethod
    def _assess_data_impact(anomalies, df, method: str) -> list:
        """Assess the data impact of detected anomalies"""
        impact_assessment = []
        
        if not anomalies:
            impact_assessment.append("✅ **POSITIVE DATA QUALITY**: No anomalies detected - data consistent with expected patterns")
            return impact_assessment
        
        anomaly_count = len(anomalies) if isinstance(anomalies, list) else 1
        
        # Scale-based impact assessment
        if df is not None:
            total_records = df.shape[0]
            impact_scale = (anomaly_count / total_records) * 100
            
            if impact_scale > 5:
                impact_assessment.append("🚨 **HIGH DATA IMPACT**: Significant statistical anomalies detected")
                impact_assessment.append("⚡ **IMMEDIATE ANALYSIS**: Statistical investigation and validation required")
            elif impact_scale > 1:
                impact_assessment.append("⚠️ **MODERATE DATA IMPACT**: Notable statistical deviations present")
                impact_assessment.append("📊 **TECHNICAL REVIEW**: Detailed statistical examination recommended")
            else:
                impact_assessment.append("📉 **LOW DATA IMPACT**: Isolated anomalies with minimal statistical effect")
                impact_assessment.append("🔍 **ROUTINE MONITORING**: Standard statistical verification applicable")
        
        # Data domain-specific impact analysis
        if df is not None:
            column_names = df.columns.tolist()
            column_text = " ".join(column_names).lower()
            
            # Financial/numeric data
            if any(keyword in column_text for keyword in ['revenue', 'cost', 'price', 'profit', 'sales']):
                impact_assessment.append("💰 **NUMERICAL DATA IMPACT**: Anomalies affect critical quantitative variables")
                impact_assessment.append("🎯 **PRIORITY**: Numerical anomalies require statistical validation")
            
            # Entity data
            if any(keyword in column_text for keyword in ['customer', 'user', 'satisfaction', 'service']):
                impact_assessment.append("👥 **ENTITY DATA IMPACT**: Anomalies affect entity-related variables")
                impact_assessment.append("� **SEGMENTATION**: Consider isolating affected data segments")
            
            # Process data
            if any(keyword in column_text for keyword in ['process', 'operation', 'efficiency', 'quality']):
                impact_assessment.append("⚙️ **PROCESS DATA IMPACT**: Data quality in process metrics compromised")
                impact_assessment.append("🔧 **DATA REVIEW**: Evaluate data collection and processing procedures")
            
            # Compliance/integrity impact
            if any(keyword in column_text for keyword in ['compliance', 'regulation', 'audit', 'risk']):
                impact_assessment.append("⚖️ **DATA INTEGRITY IMPACT**: Statistical reliability and validity implications")
                impact_assessment.append("📋 **DOCUMENTATION**: Record anomaly patterns for statistical analysis")
        
        # Method-specific data implications
        if 'fraud' in method.lower():
            impact_assessment.append("🚨 **FRAUD PATTERNS**: Statistical signatures of potentially fraudulent activity")
        elif 'quality' in method.lower():
            impact_assessment.append("✅ **QUALITY METRICS**: Data quality deviations identified")
        elif 'security' in method.lower():
            impact_assessment.append("🛡️ **SECURITY ANOMALIES**: Statistical patterns indicate potential security concerns")
        
        return impact_assessment
    
    @staticmethod
    def _assess_risk_levels(anomalies, anomaly_scores, thresholds: dict) -> list:
        """Assess risk levels of detected anomalies"""
        risk_assessment = []
        
        if not anomalies:
            risk_assessment.append("✅ **MINIMAL RISK**: No anomalies detected - risk levels within acceptable parameters")
            return risk_assessment
        
        # Score-based risk assessment
        if anomaly_scores and isinstance(anomaly_scores, list):
            scores_array = np.array(anomaly_scores)
            
            if len(scores_array) > 0:
                max_score = np.max(scores_array)
                mean_score = np.mean(scores_array)
                
                # Risk categorization based on scores
                if max_score > 0.8:
                    risk_assessment.append("🚨 **CRITICAL RISK**: Maximum anomaly score indicates severe deviation")
                elif max_score > 0.6:
                    risk_assessment.append("⚠️ **HIGH RISK**: Significant anomaly scores require attention")
                elif max_score > 0.4:
                    risk_assessment.append("📊 **MODERATE RISK**: Noticeable anomaly patterns detected")
                else:
                    risk_assessment.append("📉 **LOW RISK**: Minor anomaly scores indicate minor deviations")
                
                # Distribution analysis
                high_risk_count = np.sum(scores_array > 0.7)
                medium_risk_count = np.sum((scores_array > 0.4) & (scores_array <= 0.7))
                low_risk_count = len(scores_array) - high_risk_count - medium_risk_count
                
                risk_assessment.append(f"📊 **RISK DISTRIBUTION**: {high_risk_count} critical, {medium_risk_count} moderate, {low_risk_count} low risk")
        
        # Threshold-based assessment
        if thresholds:
            for threshold_name, threshold_value in thresholds.items():
                risk_assessment.append(f"🎯 **{threshold_name.upper()} THRESHOLD**: {threshold_value} - configured for risk detection")
        
        # Frequency-based risk
        anomaly_count = len(anomalies) if isinstance(anomalies, list) else 1
        
        if anomaly_count > 100:
            risk_assessment.append("🚨 **SYSTEMIC RISK**: Large number of anomalies indicates system-wide issues")
        elif anomaly_count > 20:
            risk_assessment.append("⚠️ **PATTERN RISK**: Multiple anomalies suggest recurring issues")
        elif anomaly_count > 5:
            risk_assessment.append("📊 **CLUSTER RISK**: Several anomalies indicate localized problems")
        else:
            risk_assessment.append("🎯 **ISOLATED RISK**: Few anomalies suggest isolated incidents")
        
        # Business continuity risk
        risk_assessment.append("📋 **BUSINESS CONTINUITY**: Assess impact on ongoing operations and customer service")
        risk_assessment.append("🔄 **RECOVERY PLANNING**: Develop contingency plans for anomaly response")
        
        return risk_assessment
    
    @staticmethod
    def _generate_action_recommendations(anomalies, method: str) -> list:
        """Generate specific action recommendations based on anomalies"""
        recommendations = []
        
        if not anomalies:
            recommendations.extend([
                "✅ **MAINTAIN MONITORING**: Continue current detection parameters",
                "📊 **PERFORMANCE REVIEW**: Validate detection system effectiveness",
                "🔄 **ROUTINE MAINTENANCE**: Maintain standard operational procedures"
            ])
            return recommendations
        
        anomaly_count = len(anomalies) if isinstance(anomalies, list) else 1
        
        # Immediate actions
        recommendations.append("⚡ **IMMEDIATE ACTIONS**:")
        
        if anomaly_count > 50:
            recommendations.extend([
                "   🚨 Implement comprehensive statistical analysis",
                "   � Conduct multivariate outlier validation",
                "   � Consider isolation of affected data points"
            ])
        elif anomaly_count > 10:
            recommendations.extend([
                "   ⚠️ Perform detailed statistical analysis",
                "   📊 Conduct root cause analysis using statistical methods",
                "   🔍 Investigate correlated variables and patterns"
            ])
        else:
            recommendations.extend([
                "   🔍 Examine individual anomalies statistically",
                "   📋 Document statistical properties of anomalies",
                "   🎯 Validate with appropriate statistical tests"
            ])
        
        # Investigation actions
        recommendations.append("🔍 **INVESTIGATION ACTIONS**:")
        recommendations.extend([
            "   📊 Analyze anomaly distributions and statistical properties",
            "   🕒 Review temporal trends and autocorrelation patterns",
            "   🔗 Check variable relationships and covariance structures",
            "   � Apply multivariate analysis techniques"
        ])
        
        # Prevention actions
        recommendations.append("🛡️ **PREVENTION ACTIONS**:")
        recommendations.extend([
            "   🔧 Optimize statistical detection thresholds",
            "   📈 Enhance monitoring with additional statistical metrics",
            "   📋 Update data validation procedures",
            "   📊 Implement improved outlier detection algorithms"
        ])
        
        # Method-specific recommendations
        if 'statistical' in method.lower():
            recommendations.extend([
                "📊 **STATISTICAL ACTIONS**:",
                "   📈 Validate statistical assumptions and parameters",
                "   🔄 Consider robust statistical methods for outliers"
            ])
        elif 'machine_learning' in method.lower() or 'isolation' in method.lower():
            recommendations.extend([
                "🤖 **ML MODEL ACTIONS**:",
                "   🔄 Retrain model with recent data",
                "   🎯 Tune hyperparameters for better precision"
            ])
        
        # Communication actions
        recommendations.append("📢 **TECHNICAL DOCUMENTATION**:")
        recommendations.extend([
            "   📋 Create statistical summary of anomaly findings",
            "   � Generate visualization of anomaly distributions",
            "   � Document methodological approach to anomaly detection",
            "   📝 Record technical limitations and statistical assumptions"
        ])
        
        # Follow-up actions
        recommendations.append("🔄 **FOLLOW-UP ACTIONS**:")
        recommendations.extend([
            "   📅 Implement periodic statistical quality checks",
            "   📊 Monitor statistical properties of variables over time",
            "   🎯 Measure effectiveness of anomaly detection methods",
            "   📈 Update detection models with new data distributions"
        ])
        
        return recommendations
