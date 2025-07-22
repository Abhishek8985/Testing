"""
Advanced AI Service with Node-Specific Prompts
Completely rewritten for sophisticated multi-node workflow analysis with proper node data handling
Each node type has its own advanced prompt that analyzes the actual node output data
"""

import os
import json
import logging
import time
import threading
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on system environment

# NVIDIA API client
try:
    from openai import OpenAI
    NVIDIA_API_AVAILABLE = True
except ImportError:
    NVIDIA_API_AVAILABLE = False

# Import the prompt router instead of individual generators
from app.api.ai.node_prompt_router import NodePromptRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAIInsightService:
    """
    Advanced AI Service that generates sophisticated insights using node-specific prompts
    for connected workflow analysis. Each node type gets its own advanced prompt based on actual data.
    """
    
    def __init__(self):
        """Initialize the advanced AI service with NVIDIA API and node-specific prompt generators"""
        self.api_key = os.getenv('NVIDIA_API_KEY')
        self.base_url = "https://integrate.api.nvidia.com/v1"
        
        # Initialize client if API is available
        self.client = None
        if NVIDIA_API_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                logger.info("NVIDIA API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize NVIDIA API client: {str(e)}")
        
        # Model priority list (try these in order)
        self.models = [
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "meta/llama-3.1-405b-instruct", 
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-8b-instruct",
            "microsoft/phi-3-medium-4k-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1"
        ]
        
        # Current working model
        self.model = self.models[0]
        
        # Initialize the node prompt router
        self.prompt_router = NodePromptRouter()
    
    def generate_comprehensive_workflow_insights(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive AI insights for connected workflow nodes
        
        Args:
            workflow_data: Dictionary containing all connected node data
            Format: {
                'nodes': {
                    'node_id_1': {
                        'type': 'data_source',
                        'data': {...},
                        'connections': ['node_id_2', 'node_id_3']
                    },
                    'node_id_2': {
                        'type': 'statistical_analysis', 
                        'data': {...},
                        'connections': ['node_id_3']
                    },
                    ...
                },
                'workflow_context': {...}
            }
            
        Returns:
            Dictionary containing comprehensive AI analysis
        """
        try:
            logger.info("Starting comprehensive workflow analysis")
            
            # Extract nodes and context
            nodes = workflow_data.get('nodes', {})
            workflow_context = workflow_data.get('workflow_context', {})
            
            if not nodes:
                return self._generate_error_response("No nodes provided for analysis")
            
            # Analyze workflow structure
            workflow_analysis = self._analyze_workflow_structure(nodes, workflow_context)
            
            # Generate node-specific prompts with actual data
            node_prompts = self._generate_node_prompts(nodes, workflow_context)
            
            # Create comprehensive workflow prompt
            comprehensive_prompt = self._create_comprehensive_workflow_prompt(
                nodes, workflow_context, node_prompts, workflow_analysis
            )
            
            # Generate AI insights using streaming API
            ai_response = self._call_nvidia_api_streaming(comprehensive_prompt)
            
            # Process and structure the response
            structured_insights = self._structure_ai_response(ai_response, nodes, workflow_analysis)
            
            # Add metadata
            structured_insights['success'] = True
            structured_insights['metadata'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'nodes_analyzed': len(nodes),
                'workflow_complexity': workflow_analysis.get('complexity_level', 'moderate'),
                'ai_model_used': self.model,
                'prompt_system': 'advanced_node_specific',
                'analysis_type': 'comprehensive_workflow'
            }
            
            logger.info(f"Successfully generated insights for {len(nodes)} nodes")
            return structured_insights
            
        except Exception as e:
            logger.error(f"Error in comprehensive workflow analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_error_response(str(e))
    
    def generate_single_node_insights(self, node_data: Dict[str, Any], node_id: str, 
                                    node_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate AI insights for a single node
        
        Args:
            node_data: The node's data/output
            node_id: Unique identifier for the node
            node_type: Type of the node (e.g., 'data_source', 'classification')
            context: Optional context about the broader workflow
            
        Returns:
            Dictionary containing AI analysis for the single node
        """
        try:
            logger.info(f"Generating insights for single node: {node_id} (type: {node_type})")
            
            # Generate node-specific prompt
            node_prompt = self.prompt_router.generate_prompt(node_type, node_data, node_id, context)
            
            # Create focused analysis prompt
            focused_prompt = self._create_focused_node_prompt(node_prompt, node_type, context)
            
            # Generate AI insights
            ai_response = self._call_nvidia_api_streaming(focused_prompt)
            
            # Structure the response
            structured_insights = {
                'success': True,
                'node_id': node_id,
                'node_type': node_type,
                'insights': self._parse_ai_response(ai_response),
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'ai_model_used': self.model,
                    'prompt_system': 'advanced_node_specific',
                    'analysis_type': 'single_node'
                }
            }
            
            logger.info(f"Successfully generated insights for node {node_id}")
            return structured_insights
            
        except Exception as e:
            logger.error(f"Error generating insights for node {node_id}: {str(e)}")
            return self._generate_error_response(str(e), node_id)
    
    def _analyze_workflow_structure(self, nodes: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure and complexity of the workflow"""
        analysis = {
            'total_nodes': len(nodes),
            'node_types': {},
            'connections': [],
            'data_flow': [],
            'complexity_level': 'basic',
            'analysis_categories': {
                'data_sources': [],
                'statistical_nodes': [],
                'visualization_nodes': [],
                'ml_nodes': [],
                'processing_nodes': [],
                'eda_nodes': [],
                'cleaning_nodes': [],
                'feature_engineering_nodes': [],
                'anomaly_detection_nodes': []
            }
        }
        
        # Categorize nodes by type, excluding AI nodes and nodes without valid data
        for node_id, node_info in nodes.items():
            node_type = node_info.get('type', 'unknown')
            node_data = node_info.get('data', {})
            
            # Skip AI nodes
            if self._is_ai_node(node_type, node_id):
                logger.info(f"Excluding AI node {node_id} from workflow structure analysis")
                continue
            
            # Skip nodes without valid data
            if not self._has_valid_data(node_data, node_type):
                logger.info(f"Excluding node {node_id} from workflow structure analysis - no valid data")
                continue
            
            # Count node types
            if node_type not in analysis['node_types']:
                analysis['node_types'][node_type] = 0
            analysis['node_types'][node_type] += 1
            
            # Categorize for analysis
            if node_type in ['data_source', 'csv_upload', 'file_upload']:
                analysis['analysis_categories']['data_sources'].append(node_id)
            elif node_type in ['statistical_analysis', 'descriptive_stats', 'processing']:
                analysis['analysis_categories']['statistical_nodes'].append(node_id)
            elif node_type in ['visualization', 'chart', 'plotting', 'basic_plots', 'advanced_plots']:
                analysis['analysis_categories']['visualization_nodes'].append(node_id)
            elif node_type in ['classification', 'regression', 'clustering', 'automl', 'machine_learning']:
                analysis['analysis_categories']['ml_nodes'].append(node_id)
            elif node_type in ['eda', 'exploratory_data_analysis', 'eda_analysis']:
                analysis['analysis_categories']['eda_nodes'].append(node_id)
            elif node_type in ['data_cleaning', 'preprocessing']:
                analysis['analysis_categories']['cleaning_nodes'].append(node_id)
            elif node_type in ['feature_engineering', 'feature_creation']:
                analysis['analysis_categories']['feature_engineering_nodes'].append(node_id)
            elif node_type in ['anomaly_detection', 'outlier_detection', 'univariate_anomaly_detection', 'multivariate_anomaly_detection']:
                analysis['analysis_categories']['anomaly_detection_nodes'].append(node_id)
            else:
                analysis['analysis_categories']['processing_nodes'].append(node_id)
            
            # Track connections
            connections = node_info.get('connections', [])
            for connected_node in connections:
                analysis['connections'].append((node_id, connected_node))
        
        # Update total nodes to reflect only analyzed nodes
        total_analyzed_nodes = sum(len(category_nodes) for category_nodes in analysis['analysis_categories'].values())
        analysis['total_analyzed_nodes'] = total_analyzed_nodes
        
        # Determine complexity level based on analyzed nodes
        total_categories = sum(1 for category_nodes in analysis['analysis_categories'].values() if category_nodes)
        
        if total_categories >= 6 or total_analyzed_nodes >= 10:
            analysis['complexity_level'] = 'advanced'
        elif total_categories >= 4 or total_analyzed_nodes >= 5:
            analysis['complexity_level'] = 'intermediate'
        else:
            analysis['complexity_level'] = 'basic'
        
        # Analyze data flow patterns
        data_sources = analysis['analysis_categories']['data_sources']
        ml_nodes = analysis['analysis_categories']['ml_nodes']
        viz_nodes = analysis['analysis_categories']['visualization_nodes']
        
        if data_sources and ml_nodes:
            analysis['data_flow'].append('data_to_ml_pipeline')
        if data_sources and viz_nodes:
            analysis['data_flow'].append('data_to_visualization_pipeline')
        if len(analysis['analysis_categories']['statistical_nodes']) > 0:
            analysis['data_flow'].append('statistical_analysis_pipeline')
        
        return analysis
    
    def _generate_node_prompts(self, nodes: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Generate sophisticated prompts for each node using actual node data"""
        node_prompts = {}
        
        for node_id, node_info in nodes.items():
            try:
                node_type = node_info.get('type', 'unknown')
                node_data = node_info.get('data', {})
                
                # Skip AI nodes - they should not be included in AI analysis
                if self._is_ai_node(node_type, node_id):
                    logger.info(f"Skipping AI node {node_id} (type: {node_type}) from analysis")
                    continue
                
                # Validate that node has meaningful data
                if not self._has_valid_data(node_data, node_type):
                    logger.info(f"Skipping node {node_id} (type: {node_type}) - no valid data available")
                    continue
                
                logger.info(f"Generating prompt for node {node_id} (type: {node_type}) with data keys: {list(node_data.keys()) if isinstance(node_data, dict) else 'non-dict data'}")
                
                # Use the prompt router to generate the prompt
                try:
                    prompt = self.prompt_router.generate_prompt(
                        node_type=node_type,
                        data=node_data,
                        node_id=node_id,
                        context=context
                    )
                    
                    # Check if prompt indicates an error or no data
                    if "CRITICAL ERROR" in prompt or "No data available" in prompt:
                        logger.warning(f"Prompt generator indicated data issue for node {node_id}: {prompt[:100]}...")
                        continue
                    
                    node_prompts[node_id] = prompt
                    logger.info(f"Generated advanced prompt for {node_id} ({len(prompt)} characters)")
                    
                except Exception as prompt_error:
                    logger.error(f"Error generating prompt for node {node_id}: {str(prompt_error)}")
                    # Use fallback prompt
                    node_prompts[node_id] = self._generate_fallback_node_prompt(node_type, node_data, node_id)
                
            except Exception as e:
                logger.error(f"Error generating prompt for node {node_id}: {str(e)}")
                node_prompts[node_id] = f"Error generating prompt for {node_type} node: {str(e)}"
        
        return node_prompts
    
    def _create_comprehensive_workflow_prompt(self, nodes: Dict[str, Any], context: Dict[str, Any], 
                                            node_prompts: Dict[str, str], 
                                            workflow_analysis: Dict[str, Any]) -> str:
        """Create a comprehensive prompt that combines all node-specific prompts"""
        
        # Workflow overview
        total_nodes = workflow_analysis.get('total_analyzed_nodes', len(nodes))
        complexity = workflow_analysis.get('complexity_level', 'moderate')
        node_categories = workflow_analysis.get('analysis_categories', {})
        
        # Count nodes by category
        category_counts = {k: len(v) for k, v in node_categories.items() if v}
        
        # Create workflow header
        prompt = f"""
🚀 **COMPREHENSIVE WORKFLOW INTELLIGENCE ANALYSIS**
**Analyzing {total_nodes} Connected Nodes | Complexity: {complexity.upper()}**

📊 **WORKFLOW ARCHITECTURE OVERVIEW**:
• Data Sources: {len(node_categories.get('data_sources', []))} nodes
• Statistical Analysis: {len(node_categories.get('statistical_nodes', []))} nodes  
• Machine Learning: {len(node_categories.get('ml_nodes', []))} nodes
• Visualizations: {len(node_categories.get('visualization_nodes', []))} nodes
• Data Processing: {len(node_categories.get('processing_nodes', []))} nodes
• EDA Analysis: {len(node_categories.get('eda_nodes', []))} nodes
• Data Cleaning: {len(node_categories.get('cleaning_nodes', []))} nodes
• Feature Engineering: {len(node_categories.get('feature_engineering_nodes', []))} nodes
• Anomaly Detection: {len(node_categories.get('anomaly_detection_nodes', []))} nodes

🔗 **CONNECTED NODE ANALYSIS**:
The following connected nodes require comprehensive, integrated analysis:

"""
        
        # Add individual node prompts in logical order
        node_order = self._determine_node_analysis_order(nodes, workflow_analysis)
        
        # Filter to only include nodes that actually have prompts
        valid_nodes = [node_id for node_id in node_order if node_id in node_prompts]
        
        logger.info(f"Including {len(valid_nodes)} nodes with valid prompts out of {len(node_order)} total nodes")
        
        for i, node_id in enumerate(valid_nodes, 1):
            node_info = nodes.get(node_id, {})
            node_type = node_info.get('type', 'unknown')
            node_prompt = node_prompts[node_id]  # We know this exists since we filtered
            
            prompt += f"""
═══════════════════════════════════════════════════════════════════

**NODE {i}: {node_id.upper()} | TYPE: {node_type.upper()}**

{node_prompt}

═══════════════════════════════════════════════════════════════════
"""
        
        # Add comprehensive workflow synthesis requirements
        prompt += f"""

🎯 **DATA ANALYSIS SYNTHESIS REQUIREMENTS**:

**REQUIREMENT**: This analysis must synthesize data from ALL {len(valid_nodes)} connected nodes to provide:

1. **DATA INFORMATION SYNTHESIS**: 
   - Combine data information from all nodes
   - Identify cross-node patterns and relationships
   - Summarize key data characteristics

2. **DATA QUALITY ASSESSMENT**:
   - Evaluate data quality progression through the workflow
   - Assess transformation effectiveness across connected nodes
   - Identify data issues and inconsistencies

3. **RESULTS REPORTING**:
   - Synthesize statistical findings with analysis results
   - Combine visualization insights with analytical outcomes
   - Summarize key findings from the data

**RESPONSE STRUCTURE REQUIREMENTS**:

Provide a comprehensive analysis structured as follows:

## DATA SUMMARY
- Key characteristics of the data processed in this workflow
- Data quality assessment
- Main statistical properties

## INTEGRATED DATA ANALYSIS
- Cross-node pattern analysis
- Data transformation results
- Quality progression assessment

## ANALYSIS RESULTS
- Key findings from the data
- Observed patterns and trends
- Notable statistics and metrics

⚡ **CRITICAL REQUIREMENT**: All insights must be based on the ACTUAL data, patterns, and results from the connected nodes. Focus only on data information and results reporting.

**WORKFLOW CONTEXT**: {json.dumps(context, indent=2) if context else 'Standard analytical workflow'}
"""
        
        return prompt.strip()
    
    def _determine_node_analysis_order(self, nodes: Dict[str, Any], workflow_analysis: Dict[str, Any]) -> List[str]:
        """Determine the logical order for analyzing nodes"""
        categories = workflow_analysis.get('analysis_categories', {})
        
        # Define logical analysis order
        order_priority = [
            'data_sources',
            'cleaning_nodes', 
            'eda_nodes',
            'statistical_nodes',
            'feature_engineering_nodes',
            'processing_nodes',
            'ml_nodes',
            'anomaly_detection_nodes',
            'visualization_nodes'
        ]
        
        ordered_nodes = []
        
        # Add nodes in priority order
        for category in order_priority:
            category_nodes = categories.get(category, [])
            ordered_nodes.extend(category_nodes)
        
        # Add any remaining nodes
        all_node_ids = set(nodes.keys())
        ordered_node_ids = set(ordered_nodes)
        remaining_nodes = all_node_ids - ordered_node_ids
        ordered_nodes.extend(list(remaining_nodes))
        
        return ordered_nodes
    
    def _create_focused_node_prompt(self, node_prompt: str, node_type: str, context: Dict[str, Any]) -> str:
        """Create a focused prompt for single node analysis"""
        
        context_str = json.dumps(context, indent=2) if context else "Single node analysis"
        
        focused_prompt = f"""
🎯 **FOCUSED NODE ANALYSIS REQUEST**

**ANALYSIS CONTEXT**: {context_str}

{node_prompt}

**FOCUSED ANALYSIS REQUIREMENTS**:

Please provide a focused analysis of this specific node that includes:

1. **DATA INFORMATION**: Key characteristics of the data in this node
2. **DATA QUALITY**: Data quality assessment and reliability
3. **RESULTS SUMMARY**: Key findings and patterns from this node's data/results
4. **STATISTICAL INSIGHTS**: Important statistical properties observed

**OUTPUT FORMAT**: Provide a well-structured analysis with clear sections and bullet points based on the actual data and results from this node. Focus only on data information and results reporting.
"""
        
        return focused_prompt
    
    def _call_nvidia_api_streaming(self, prompt: str) -> str:
        """Call NVIDIA API with streaming for comprehensive responses"""
        if not self.client:
            logger.warning("NVIDIA API not available, generating fallback response")
            return self._generate_fallback_ai_response(prompt)
        
        try:
            # System prompt for data scientist expertise
            system_prompt = """You are a data scientist with expertise in:
- Data analysis and statistical modeling
- Data quality assessment
- Pattern recognition and data insights
- Technical and statistical reporting

Provide clear and concise data information and results reporting. Focus on presenting factual information about the data, statistical properties, and analytical findings without business implications or recommendations."""

            # Try different models until one works
            for model_attempt, model_name in enumerate(self.models):
                try:
                    logger.info(f"Attempting API call with model: {model_name}")
                    
                    completion = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=8192,  # Increased for comprehensive analysis
                        frequency_penalty=0.1,
                        presence_penalty=0.1,
                        stream=True
                    )
                    
                    # Update working model
                    if model_name != self.model:
                        self.model = model_name
                        logger.info(f"Updated working model to: {model_name}")
                    
                    # Collect streaming response
                    full_response = ""
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            full_response += content
                    
                    logger.info(f"Successfully generated AI response ({len(full_response)} characters)")
                    return full_response
                    
                except Exception as model_error:
                    logger.warning(f"Model {model_name} failed: {str(model_error)}")
                    if model_attempt == len(self.models) - 1:
                        raise model_error
                    continue
            
        except Exception as e:
            logger.error(f"NVIDIA API call failed: {str(e)}")
            return self._generate_fallback_ai_response(prompt)
    
    def _structure_ai_response(self, ai_response: str, nodes: Dict[str, Any], 
                             workflow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the AI response into organized sections"""
        
        # Try to parse structured sections from the response
        sections = self._parse_response_sections(ai_response)
        
        structured_response = {
            'insights': {
                'data_summary': sections.get('data_summary', ''),
                'integrated_analysis': sections.get('integrated_analysis', ''),
                'analysis_results': sections.get('analysis_results', ''),
                'key_insights': self._extract_key_insights(ai_response),
                'data_quality': self._extract_data_quality(ai_response),
                'statistical_properties': self._extract_statistical_properties(ai_response)
            },
            'full_response': ai_response,
            'workflow_analysis': workflow_analysis,
            'node_summary': {
                'total_nodes': len(nodes),
                'node_types': list(set(node.get('type', 'unknown') for node in nodes.values())),
                'complexity_level': workflow_analysis.get('complexity_level', 'moderate')
            }
        }
        
        return structured_response
    
    def _parse_response_sections(self, response: str) -> Dict[str, str]:
        """Parse AI response into structured sections"""
        sections = {}
        
        # Define section markers
        section_markers = {
            'data_summary': ['## DATA SUMMARY', '# DATA SUMMARY', 'DATA SUMMARY'],
            'integrated_analysis': ['## INTEGRATED DATA ANALYSIS', '# INTEGRATED DATA ANALYSIS', 'INTEGRATED DATA ANALYSIS'],
            'analysis_results': ['## ANALYSIS RESULTS', '# ANALYSIS RESULTS', 'ANALYSIS RESULTS']
        }
        
        # Split response into lines
        lines = response.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_upper = line.upper().strip()
            
            # Check if this line starts a new section
            section_found = False
            for section_key, markers in section_markers.items():
                if any(marker.upper() in line_upper for marker in markers):
                    # Save previous section
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    # Start new section
                    current_section = section_key
                    current_content = []
                    section_found = True
                    break
            
            # Add line to current section
            if not section_found and current_section:
                current_content.append(line)
            elif not section_found and not current_section:
                # Before any section markers, treat as data summary
                if 'data_summary' not in sections:
                    sections['data_summary'] = sections.get('data_summary', '') + line + '\n'
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _parse_ai_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI response for single node analysis"""
        return {
            'analysis': ai_response,
            'key_insights': self._extract_key_insights(ai_response),
            'data_quality': self._extract_data_quality(ai_response),
            'statistical_properties': self._extract_statistical_properties(ai_response)
        }
    
    def _extract_key_insights(self, response: str) -> List[str]:
        """Extract key insights from AI response"""
        insights = []
        lines = response.split('\n')
        
        for line in lines:
            # Look for bullet points or numbered insights
            if any(marker in line for marker in ['•', '◦', '▸', '1.', '2.', '3.', '-']):
                if any(keyword in line.lower() for keyword in ['insight', 'finding', 'pattern', 'trend', 'result']):
                    insights.append(line.strip())
        
        return insights[:5]  # Return top 5 insights
    
    def _extract_data_quality(self, response: str) -> List[str]:
        """Extract data quality information from AI response"""
        quality_info = []
        lines = response.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['quality', 'missing', 'valid', 'complete', 'reliable']):
                if any(marker in line for marker in ['•', '◦', '▸', '1.', '2.', '3.', '-']):
                    quality_info.append(line.strip())
        
        return quality_info[:5]  # Return top 5 quality points
    
    def _extract_statistical_properties(self, response: str) -> str:
        """Extract statistical properties from AI response"""
        lines = response.split('\n')
        stat_content = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['statistic', 'mean', 'median', 'mode', 'std', 'variance', 'distribution']):
                stat_content.append(line.strip())
        
        return ' '.join(stat_content[:5])  # Return first 5 statistics-related lines
    
    def _generate_fallback_ai_response(self, prompt: str) -> str:
        """Generate fallback response when AI is not available"""
        return f"""
## DATA SUMMARY
AI API temporarily unavailable. Based on the workflow structure and data patterns, this analysis indicates a data processing pipeline with multiple connected nodes requiring integrated assessment.

## INTEGRATED DATA ANALYSIS  
The connected nodes represent an analytical workflow combining data processing, analysis, and visualization. Each component contributes to the overall data processing and analysis pipeline.

## ANALYSIS RESULTS
• Data processing pipeline established with proper node connections
• Statistical analysis capabilities available for data insights
• Data transformation and processing steps detected in the workflow
• Data quality assessment needed for reliable results
• Pattern detection and trend analysis possible with available data

Note: This is a fallback analysis. For comprehensive data analysis, ensure NVIDIA API access is configured.
"""
    
    def _generate_error_response(self, error_message: str, node_id: str = None) -> Dict[str, Any]:
        """Generate structured error response"""
        return {
            'success': False,
            'error': error_message,
            'node_id': node_id,
            'timestamp': datetime.now().isoformat(),
            'insights': {
                'key_findings': [f'Error occurred during analysis: {error_message}'],
                'data_quality_assessment': {
                    'overall_score': 'Error',
                    'main_issues': [error_message],
                    'recommendations': [
                        "Review input data format and structure",
                        "Check API connectivity and authentication", 
                        "Validate node configuration and types"
                    ]
                },
                'data_analysis': [
                    "Analysis unavailable due to error"
                ],
                'statistical_properties': ['Analysis unavailable due to error'],
                'next_steps': ['Resolve the error and retry analysis']
            }
        }
    
    def test_api_connection(self) -> Dict[str, Any]:
        """Test API connection and available models"""
        results = {
            'api_accessible': False,
            'working_models': [],
            'failed_models': [],
            'recommended_model': None,
            'prompt_system_status': 'active'
        }
        
        if not self.client:
            results['message'] = "NVIDIA API client not initialized"
            return results
        
        test_prompt = "Respond with: AI connection test successful"
        
        for model_name in self.models:
            try:
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": test_prompt}],
                    max_tokens=50,
                    temperature=0.1
                )
                
                response = completion.choices[0].message.content
                if "successful" in response.lower():
                    results['working_models'].append(model_name)
                    if not results['recommended_model']:
                        results['recommended_model'] = model_name
                        
            except Exception as e:
                logger.warning(f"Model {model_name} test failed: {str(e)}")
                results['failed_models'].append(model_name)
        
        results['api_accessible'] = len(results['working_models']) > 0
        
        # Update working model if we found a better one
        if results['recommended_model'] and results['recommended_model'] != self.model:
            self.model = results['recommended_model']
            logger.info(f"Updated working model to: {self.model}")
        
        return results

    def create_background_task(self, comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a background task for streaming analysis"""
        try:
            task_id = f"ai_task_{int(datetime.now().timestamp() * 1000)}"
            
            # Estimate duration based on data complexity
            total_nodes = comprehensive_data.get('workflow_summary', {}).get('total_nodes', 0)
            estimated_duration = min(30 + (total_nodes * 5), 120)  # 30-120 seconds
            
            return {
                'success': True,
                'task_id': task_id,
                'estimated_duration': f"{estimated_duration} seconds",
                'background_task': {
                    'id': task_id,
                    'status': 'created',
                    'created_at': datetime.now().isoformat(),
                    'estimated_completion': (datetime.now().timestamp() + estimated_duration)
                }
            }
        except Exception as e:
            logger.error(f"Error creating background task: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'background_task': None
            }
    
    def execute_streaming_analysis(self, comprehensive_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Execute streaming analysis in background"""
        try:
            logger.info(f"Starting background streaming analysis for task: {task_id}")
            
            # Check if we have the workflow summary format from workflow service
            # or the nodes format for direct API calls
            if 'workflow_summary' in comprehensive_data:
                # Convert workflow summary format to nodes format
                workflow_data = self._convert_workflow_summary_to_nodes_format(comprehensive_data)
            else:
                # Direct nodes format
                workflow_data = comprehensive_data
            
            # Generate comprehensive workflow insights using the existing method
            result = self.generate_comprehensive_workflow_insights(workflow_data)
            
            # Add streaming metadata
            result.update({
                'task_id': task_id,
                'completed_at': datetime.now().isoformat(),
                'processing_mode': 'background_streaming'
            })
            
            logger.info(f"Completed background streaming analysis for task: {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in background streaming analysis for task {task_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'task_id': task_id,
                'completed_at': datetime.now().isoformat()
            }

    def generate_advanced_workflow_analysis(self, comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advanced workflow analysis (compatibility method)"""
        try:
            # This is essentially the same as comprehensive workflow insights
            # but with additional advanced analysis markers
            result = self.generate_comprehensive_workflow_insights(comprehensive_data)
            
            # Add advanced analysis indicators
            result.update({
                'analysis_type': 'advanced_workflow',
                'advanced_features': [
                    'Multi-node synthesis',
                    'Contextual prompting',
                    'Business intelligence focus',
                    'Node-specific insights'
                ]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced workflow analysis: {str(e)}")
            return self._generate_error_response(str(e))
    
    def _convert_workflow_summary_to_nodes_format(self, comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert workflow summary format to nodes format for comprehensive analysis"""
        try:
            # Check if nodes are already in the correct format
            if 'nodes' in comprehensive_data:
                logger.info("Found nodes in direct format, using as-is")
                return {
                    'nodes': comprehensive_data['nodes'],
                    'workflow_context': comprehensive_data.get('workflow_context', {})
                }
            
            # Check for workflow_summary format
            workflow_summary = comprehensive_data.get('workflow_summary', {})
            node_outputs = workflow_summary.get('node_outputs', {})
            
            # Convert node outputs to the expected nodes format
            nodes = {}
            for node_id, node_output in node_outputs.items():
                nodes[node_id] = {
                    'type': node_output.get('type', 'unknown'),
                    'data': node_output.get('data', {}),
                    'connections': []  # Connections aren't preserved in this format
                }
            
            # If no node_outputs but we have workflow_summary, create a synthetic node
            if not nodes and workflow_summary:
                logger.info("Creating synthetic node from workflow summary")
                nodes['workflow_summary_node'] = {
                    'type': 'workflow_summary',
                    'data': workflow_summary,
                    'connections': []
                }
            
            # Create workflow context from summary
            workflow_context = {
                'total_nodes': workflow_summary.get('total_nodes', len(nodes)),
                'node_types': workflow_summary.get('node_types', []),
                'has_data': workflow_summary.get('has_data', False),
                'primary_data_shape': workflow_summary.get('primary_data_shape'),
                'analysis_type': 'comprehensive_workflow_summary'
            }
            
            logger.info(f"Converted workflow summary with {len(nodes)} nodes to comprehensive format")
            
            return {
                'nodes': nodes,
                'workflow_context': workflow_context
            }
            
        except Exception as e:
            logger.error(f"Error converting workflow summary format: {str(e)}")
            # Return minimal format to avoid complete failure
            return {
                'nodes': {},
                'workflow_context': {'error': str(e)}
            }
    
    def _is_ai_node(self, node_type: str, node_id: str) -> bool:
        """Check if a node is an AI node that should be excluded from analysis"""
        ai_node_types = [
            'ai_summary', 'ai_analysis', 'ai_insights', 'ai_data_summary',
            'ai_recommendation', 'ai_report', 'ai_workflow_summary'
        ]
        
        ai_node_patterns = [
            'ai_', 'artificial_intelligence_', 'llm_', 'gpt_', 'chatbot_',
            'assistant_', 'recommendation_engine_'
        ]
        
        # Check if node type is explicitly an AI node
        if node_type.lower() in ai_node_types:
            return True
        
        # Check if node ID contains AI patterns
        node_id_lower = node_id.lower()
        if any(pattern in node_id_lower for pattern in ai_node_patterns):
            return True
        
        # Check if node type contains AI patterns
        node_type_lower = node_type.lower()
        if any(pattern in node_type_lower for pattern in ai_node_patterns):
            return True
        
        return False
    
    def _has_valid_data(self, node_data: Dict[str, Any], node_type: str) -> bool:
        """Check if node has valid, meaningful data for analysis"""
        # Handle DataFrame case first
        if isinstance(node_data, pd.DataFrame):
            return not node_data.empty
        
        # Handle None or empty cases
        if node_data is None:
            return False
        
        # For non-DataFrame data, check if it's truthy (but avoid ambiguous DataFrame evaluation)
        if not isinstance(node_data, (pd.DataFrame, pd.Series)) and not node_data:
            return False
        
        # Check for empty dictionaries or None values
        if isinstance(node_data, dict):
            # Filter out None, empty strings, and empty collections (handle DataFrames separately)
            meaningful_data = {}
            for k, v in node_data.items():
                if v is None:
                    continue
                elif isinstance(v, str) and v == "":
                    continue
                elif isinstance(v, list) and len(v) == 0:
                    continue
                elif isinstance(v, dict) and len(v) == 0:
                    continue
                elif isinstance(v, pd.DataFrame):
                    # Handle DataFrame separately - check if empty
                    if not v.empty:
                        meaningful_data[k] = v
                else:
                    meaningful_data[k] = v
            
            if not meaningful_data:
                return False
            
            # For statistical nodes, ensure we have actual statistical results
            if node_type in ['descriptive_stats', 'statistical_analysis', 'processing']:
                # Look for statistical indicators
                expected_keys = ['statistics', 'dataframe', 'correlations', 'results', 'summary']
                
                # Check if at least one expected key exists with meaningful data
                has_expected_data = False
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            has_expected_data = True
                            break
                        elif isinstance(value, dict) and len(value) > 0:
                            # Check if dict contains statistical data
                            stat_indicators = ['mean', 'median', 'std', 'count', 'min', 'max', 'describe']
                            if any(indicator in str(value).lower() for indicator in stat_indicators):
                                has_expected_data = True
                                break
                        elif isinstance(value, list) and len(value) > 0:
                            has_expected_data = True
                            break
                
                return has_expected_data
            
            # For data source nodes, ensure we have actual dataset
            elif node_type in ['data_source', 'csv_upload', 'file_upload']:
                expected_keys = ['dataframe', 'data', 'source_info', 'file_info', 'shape', 'columns']
                
                has_data_source = False
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            has_data_source = True
                            break
                        elif key in ['shape', 'columns'] and value:
                            has_data_source = True
                            break
                        elif isinstance(value, dict) and len(value) > 0:
                            has_data_source = True
                            break
                
                return has_data_source
            
            # For EDA nodes
            elif node_type in ['eda', 'exploratory_data_analysis', 'eda_analysis']:
                expected_keys = ['eda_results', 'statistics', 'charts', 'dataframe', 'insights']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif value:  # For non-DataFrame values
                            return True
                return False
            
            # For machine learning nodes
            elif node_type in ['classification', 'regression', 'clustering', 'automl', 'machine_learning']:
                expected_keys = ['model', 'predictions', 'performance', 'training_info', 'results']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif value:  # For non-DataFrame values
                            return True
                return False
            
            # For visualization nodes
            elif node_type in ['visualization', 'chart', 'plotting', 'basic_plots', 'advanced_plots']:
                expected_keys = ['charts', 'visualizations', 'chart_metadata', 'insights']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif value:  # For non-DataFrame values
                            return True
                return False
            
            # For data cleaning nodes
            elif node_type in ['data_cleaning', 'preprocessing']:
                expected_keys = ['data', 'dataframe', 'cleaning_summary', 'cleaning_stats', 'before_cleaning', 'after_cleaning']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif value:  # For non-DataFrame values
                            return True
                return False
            
            # For feature engineering nodes
            elif node_type in ['feature_engineering', 'feature_creation']:
                expected_keys = ['features', 'dataframe', 'feature_info', 'transformations']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif value:  # For non-DataFrame values
                            return True
                return False
            
            # For anomaly detection nodes
            elif node_type in ['anomaly_detection', 'outlier_detection']:
                expected_keys = ['anomalies', 'outliers', 'results', 'scores']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif value:  # For non-DataFrame values
                            return True
                return False
            
            # For other nodes, just check if we have meaningful content
            else:
                return True
        
        # Non-dict data
        elif isinstance(node_data, (list, pd.DataFrame)):
            return len(node_data) > 0
        elif isinstance(node_data, (str, int, float)):
            return node_data != "" and node_data != 0
        
        return False

    def _generate_fallback_node_prompt(self, node_type: str, node_data: Dict[str, Any], node_id: str) -> str:
        """Generate a fallback prompt for unknown node types"""
        data_summary = self._summarize_node_data(node_data)
        
        return f"""
🔍 **NODE ANALYSIS: {node_id.upper()} | TYPE: {node_type.upper()}**

**UNKNOWN NODE TYPE DETECTED**
This node type ({node_type}) doesn't have a specialized prompt generator. 
Analyzing based on available data structure and patterns.

**NODE DATA SUMMARY**:
{data_summary}

**ANALYSIS REQUEST**:
Please analyze this node's data and provide insights based on:
1. **Data Structure**: What type of data processing or analysis appears to have occurred?
2. **Key Patterns**: What patterns or trends are evident in the data?
3. **Data Quality Assessment**: How reliable and complete is this data?
4. **Results Summary**: What are the main findings from this data?

**CONTEXT**: This node is part of a larger analytical workflow. Focus only on data information and results reporting.
"""
    
    def _summarize_node_data(self, node_data: Dict[str, Any]) -> str:
        """Create a summary of node data for prompt generation"""
        if not node_data:
            return "• No data available"
        
        summary_parts = []
        
        # Check for common data structures
        for key, value in node_data.items():
            if isinstance(value, pd.DataFrame):
                summary_parts.append(f"• {key}: DataFrame with {value.shape[0]} rows and {value.shape[1]} columns")
            elif isinstance(value, dict):
                if 'accuracy' in value or 'precision' in value or 'recall' in value:
                    summary_parts.append(f"• {key}: ML model metrics")
                elif 'mean' in value or 'std' in value or 'count' in value:
                    summary_parts.append(f"• {key}: Statistical measurements")
                else:
                    summary_parts.append(f"• {key}: Dictionary with {len(value)} keys")
            elif isinstance(value, list):
                summary_parts.append(f"• {key}: List with {len(value)} items")
            elif isinstance(value, (int, float)):
                summary_parts.append(f"• {key}: Numeric value ({value})")
            elif isinstance(value, str):
                summary_parts.append(f"• {key}: Text data")
            else:
                summary_parts.append(f"• {key}: {type(value).__name__}")
        
        return "\n".join(summary_parts) if summary_parts else "• Data structure not recognized"

# Global instance
advanced_ai_service = AdvancedAIInsightService()
