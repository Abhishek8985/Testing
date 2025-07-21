# Superhacker ML Workbench

An advanced machine learning and data analysis platform with AI-powered insights and interactive workflow building capabilities.

## Features

### 🤖 AI-Powered Analysis
- **AI Summary Node**: Comprehensive analysis of your entire workflow using advanced AI
- **Intelligent Insights**: Context-aware recommendations and findings
- **Background Streaming**: Real-time AI analysis with live updates
- **Multi-node Integration**: Analyzes data from all connected workflow nodes

### 📊 Data Processing & Visualization
- **Interactive Workflow Builder**: Drag-and-drop interface for building ML pipelines
- **Multiple Data Sources**: Support for CSV, database, and API data sources
- **Advanced Visualizations**: Statistical plots, correlation matrices, distribution analysis
- **EDA (Exploratory Data Analysis)**: Automated data exploration and insights

### 🧠 Machine Learning Capabilities
- **Classification & Regression**: Multiple algorithms including Random Forest, SVM, Gradient Boosting
- **Clustering Analysis**: K-means, DBSCAN, Hierarchical clustering
- **Anomaly Detection**: Univariate and multivariate outlier detection
- **Feature Engineering**: Automated feature creation and transformation
- **Model Evaluation**: Comprehensive performance metrics and validation

### 🔍 Advanced Analytics
- **Statistical Analysis**: Descriptive statistics, correlation analysis, hypothesis testing
- **Data Quality Assessment**: Missing value analysis, data type validation
- **Pattern Detection**: Automatic identification of data patterns and relationships
- **Business Intelligence**: Actionable insights and recommendations

## Architecture

### Backend (Enhanced Flask API)
- **Python Flask**: RESTful API with advanced workflow processing
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and utilities
- **Matplotlib & Seaborn**: Data visualization and chart generation
- **NVIDIA API Integration**: AI-powered analysis using Llama models

### Frontend (React Application)
- **React + Vite**: Modern frontend with fast development and building
- **Tailwind CSS**: Utility-first styling for responsive design
- **React Flow**: Interactive workflow diagram building
- **Chart.js**: Dynamic data visualization components
- **Real-time Updates**: WebSocket integration for live data streaming

## Installation & Setup

### Prerequisites
- Python 3.8+ with pip
- Node.js 16+ with npm/pnpm
- Git for version control

### Backend Setup
```bash
cd enhanced-backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd superhacker-frontend
npm install
# or
pnpm install

npm run dev
# or
pnpm dev
```

## Usage

1. **Start the Backend**: Run the Flask server on `http://localhost:5000`
2. **Start the Frontend**: Run the React development server on `http://localhost:5173`
3. **Build Workflows**: Use the drag-and-drop interface to create ML pipelines
4. **Analyze Data**: Connect nodes and run workflows to get insights
5. **AI Analysis**: Add AI Summary nodes for comprehensive analysis

## Key Components

### Workflow Nodes
- **Data Sources**: CSV upload, database connections, API integrations
- **Processing**: Data cleaning, feature engineering, validation
- **Analysis**: Statistical analysis, EDA, correlation analysis
- **Machine Learning**: Classification, regression, clustering, anomaly detection
- **Visualization**: Charts, plots, dashboards
- **AI Summary**: Intelligent analysis of entire workflows

### Data Pipeline
1. **Data Ingestion**: Multiple source support with validation
2. **Processing**: Cleaning, transformation, feature engineering
3. **Analysis**: Statistical analysis and pattern detection
4. **Modeling**: ML algorithm training and evaluation
5. **Visualization**: Interactive charts and insights
6. **AI Integration**: Comprehensive workflow analysis

## Recent Enhancements

### AI Summary Integration
- ✅ **Chart Visibility**: Fixed issue where charts weren't visible in AI Summary results
- ✅ **Multi-node Analysis**: AI Summary now analyzes all connected workflow nodes
- ✅ **Chart Collection**: Automatically includes visualizations from connected nodes
- ✅ **Streaming Analysis**: Background AI processing with real-time updates
- ✅ **Enhanced UI**: Improved chart display with proper organization by source node

### Technical Improvements
- ✅ **DataFrame Validation**: Fixed boolean evaluation errors in AI service
- ✅ **Chart Cache Integration**: Proper chart collection from workflow cache
- ✅ **Frontend Chart Display**: Added dedicated chart section for AI Summary nodes
- ✅ **Error Handling**: Improved error reporting and debugging capabilities
- ✅ **Test Coverage**: Comprehensive testing for node outputs and AI integration

## Contributing

This project is part of the Yottaasys Summer Internship 2025 program, focusing on AI-driven and data-centric applications using modern software tools.

### Development Focus Areas
- Pattern Detection & Machine Learning
- Anomaly Detection & Data Visualization
- Web Development (React, Node.js, Flask)
- Python Scripting & Automation
- AI Integration & Advanced Analytics

## License

This project is developed as part of an educational internship program.

---

**Built with ❤️ during Summer Internship 2025**