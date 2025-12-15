"""
Streamlit Web Application for Parkinson's Disease Detection
Production-ready UI with real-time predictions and visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from predict import ParkinsonPredictor
from preprocessing import ParkinsonDataProcessor
from evaluate_model import ParkinsonModelEvaluator
import joblib


# Page configuration
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        font-weight: bold;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #11151d;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor(model_path: str = 'models/svm_model.pkl', 
                   scaler_path: str = 'models/scaler.pkl'):
    """Load model and scaler with caching."""
    try:
        predictor = ParkinsonPredictor(model_path, scaler_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def render_header():
    """Render application header."""
    st.markdown('<h1 class="main-header">🧠 Parkinson\'s Disease Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        <p>AI-Powered Vocal Feature Analysis for Early Detection</p>
        <p><strong>⚕️ For Research and Educational Purposes Only</strong></p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with navigation."""
    st.sidebar.title("🎯 Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "🔬 Single Prediction", "📊 Batch Prediction", 
         "📈 Model Performance", "ℹ️ About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Info")
    st.sidebar.info("""
    This system analyzes vocal features to detect potential Parkinson's Disease indicators.
    
    **Key Features:**
    - Real-time predictions
    - Confidence scoring
    - Batch processing
    - Clinical interpretations
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ Disclaimer")
    st.sidebar.warning("""
    This tool is for research purposes only. 
    Always consult healthcare professionals for medical diagnosis.
    """)
    
    return page


def home_page():
    """Render home page."""
    st.markdown('<h2 class="sub-header">Welcome to Parkinson\'s Disease Detection System</h2>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accurate</h3>
            <p>Advanced ML algorithms for reliable detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Fast</h3>
            <p>Real-time predictions in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔒 Secure</h3>
            <p>Your data is processed locally</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Input Features")
        st.write("Provide vocal feature measurements or upload CSV data")
    
    with col2:
        st.markdown("#### 2️⃣ AI Analysis")
        st.write("Advanced ML model analyzes patterns")
    
    with col3:
        st.markdown("#### 3️⃣ Get Results")
        st.write("Receive diagnosis with confidence scores")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Key Vocal Features Analyzed")
    
    features_df = pd.DataFrame({
        'Feature Category': ['Frequency', 'Jitter', 'Shimmer', 'Harmonics', 'Nonlinear Dynamics'],
        'Description': [
            'Fundamental frequency variations',
            'Pitch variation measures',
            'Amplitude variation measures',
            'Harmonic-to-noise ratio',
            'Complexity measures (RPDE, DFA, PPE)'
        ],
        'Clinical Relevance': [
            'Voice stability',
            'Vocal cord control',
            'Voice quality',
            'Voice clarity',
            'Neural control patterns'
        ]
    })
    
    st.dataframe(features_df, use_container_width=True)


def single_prediction_page(predictor):
    """Render single prediction page."""
    st.markdown('<h2 class="sub-header">🔬 Single Sample Prediction</h2>', 
                unsafe_allow_html=True)
    
    st.info("📝 Enter vocal feature values to get prediction")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Entry", "Upload JSON"]
    )
    
    if input_method == "Manual Entry":
        st.markdown("### Enter Feature Values")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mdvp_fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, value=150.0, 
                                      help="Average vocal fundamental frequency")
            mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, value=200.0)
            mdvp_flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, value=100.0)
            mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, value=0.005)
            mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, value=0.00004)
        
        with col2:
            mdvp_rap = st.number_input("MDVP:RAP", min_value=0.0, value=0.003)
            mdvp_ppq = st.number_input("MDVP:PPQ", min_value=0.0, value=0.003)
            jitter_ddp = st.number_input("Jitter:DDP", min_value=0.0, value=0.009)
            mdvp_shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, value=0.03)
            mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, value=0.3)
        
        with col3:
            shimmer_apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, value=0.015)
            shimmer_apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, value=0.02)
            mdvp_apq = st.number_input("MDVP:APQ", min_value=0.0, value=0.025)
            shimmer_dda = st.number_input("Shimmer:DDA", min_value=0.0, value=0.045)
            nhr = st.number_input("NHR", min_value=0.0, value=0.02)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            hnr = st.number_input("HNR", min_value=0.0, value=20.0)
            rpde = st.number_input("RPDE", min_value=0.0, value=0.5)
            dfa = st.number_input("DFA", min_value=0.0, value=0.7)
        
        with col5:
            spread1 = st.number_input("spread1", value=-5.0)
            spread2 = st.number_input("spread2", value=0.2)
            d2 = st.number_input("D2", min_value=0.0, value=2.5)
        
        with col6:
            ppe = st.number_input("PPE", min_value=0.0, value=0.2)
        
        # Collect features
        features = np.array([[
            mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs,
            mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
            shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr,
            hnr, rpde, dfa, spread1, spread2, d2, ppe
        ]])
        
    else:  # Upload JSON
        uploaded_file = st.file_uploader("Upload JSON file with features", type=['json'])
        
        if uploaded_file is not None:
            data = json.load(uploaded_file)
            features = np.array([list(data.values())])
            st.success("✅ File uploaded successfully!")
            st.json(data)
        else:
            features = None
    
    # Prediction button
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        if features is not None:
            with st.spinner("Analyzing vocal features..."):
                try:
                    # Make prediction
                    result = predictor.predict(features)
                    interpretation = predictor.get_clinical_interpretation(result)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")
                    
                    # Result display
                    if result['prediction'] == 1:
                        st.markdown(f"""
                        <div class="danger-box">
                            <h3>⚠️ Diagnosis: {result['diagnosis']}</h3>
                            <p><strong>Risk Level:</strong> {interpretation['risk_level']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ Diagnosis: {result['diagnosis']}</h3>
                            <p><strong>Risk Level:</strong> {interpretation['risk_level']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    if 'probability' in result:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Healthy Probability",
                                f"{result['probability']['healthy']*100:.2f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Parkinson's Probability",
                                f"{result['probability']['parkinsons']*100:.2f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "Confidence",
                                f"{result['confidence']*100:.2f}%"
                            )
                        
                        # Probability gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result['probability']['parkinsons'] * 100,
                            title={'text': "Parkinson's Risk Score"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkred"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Clinical interpretation
                    st.markdown("### 🏥 Clinical Interpretation")
                    st.info(interpretation['recommendation'])
                    
                    # Download results
                    result_json = json.dumps({**result, 'clinical_interpretation': interpretation}, 
                                           indent=2)
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=result_json,
                        file_name="prediction_result.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        else:
            st.warning("⚠️ Please provide input features")


def batch_prediction_page(predictor):
    """Render batch prediction page."""
    st.markdown('<h2 class="sub-header">📊 Batch Prediction</h2>', 
                unsafe_allow_html=True)
    
    st.info("📤 Upload a CSV file with multiple samples for batch processing")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV should contain feature columns without 'status' or 'name' columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} samples")
            
            # Show preview
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Remove non-feature columns
            if 'name' in df.columns:
                df = df.drop('name', axis=1)
            if 'status' in df.columns:
                actual_labels = df['status'].values
                df = df.drop('status', axis=1)
            else:
                actual_labels = None
            
            # Predict button
            if st.button("🔮 Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner("Processing samples..."):
                    # Make predictions
                    results = predictor.predict_batch(df.values, return_proba=True)
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame([
                        {
                            'Sample': r['sample_id'] + 1,
                            'Diagnosis': r['diagnosis'],
                            'Prediction': r['prediction'],
                            'Confidence': f"{r.get('confidence', 0)*100:.2f}%",
                            'Healthy_Prob': f"{r.get('probability', {}).get('healthy', 0)*100:.2f}%",
                            'Parkinsons_Prob': f"{r.get('probability', {}).get('parkinsons', 0)*100:.2f}%"
                        }
                        for r in results
                    ])
                    
                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Samples", len(results))
                    
                    with col2:
                        healthy_count = sum(1 for r in results if r['prediction'] == 0)
                        st.metric("Healthy", healthy_count)
                    
                    with col3:
                        parkinsons_count = sum(1 for r in results if r['prediction'] == 1)
                        st.metric("Parkinson's", parkinsons_count)
                    
                    with col4:
                        avg_confidence = np.mean([r.get('confidence', 0) for r in results])
                        st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
                    
                    # Visualization
                    fig = px.pie(
                        values=[healthy_count, parkinsons_count],
                        names=['Healthy', "Parkinson's"],
                        title="Diagnosis Distribution",
                        color_discrete_sequence=['#2ecc71', '#e74c3c']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # If actual labels available, show accuracy
                    if actual_labels is not None:
                        predictions = [r['prediction'] for r in results]
                        accuracy = np.mean(np.array(predictions) == actual_labels)
                        st.success(f"🎯 Accuracy on this dataset: {accuracy*100:.2f}%")
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def model_performance_page():
    """Render model performance page."""
    st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', 
                unsafe_allow_html=True)
    
    # Check if metrics file exists
    metrics_path = Path('reports/metrics.json')
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        st.success("✅ Model evaluation metrics loaded")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
        
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
        
        with col4:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.2f}%")
        
        st.markdown("---")
        
        # Clinical metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sensitivity", f"{metrics.get('sensitivity', 0)*100:.2f}%")
        
        with col2:
            st.metric("Specificity", f"{metrics.get('specificity', 0)*100:.2f}%")
        
        with col3:
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0)*100:.2f}%")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Healthy', "Parkinson's"],
                y=['Healthy', "Parkinson's"],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20}
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted Label",
                yaxis_title="True Label"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display images if they exist
        st.markdown("### 📊 Performance Visualizations")
        
        image_files = [
            ('reports/roc_curve.png', 'ROC Curve'),
            ('reports/precision_recall_curve.png', 'Precision-Recall Curve'),
            ('reports/feature_importance.png', 'Feature Importance')
        ]
        
        for img_path, title in image_files:
            if Path(img_path).exists():
                st.markdown(f"#### {title}")
                st.image(img_path, use_container_width=True)
    
    else:
        st.warning("⚠️ Model performance metrics not found. Please run model evaluation first.")
        st.info("Run: `python src/evaluate_model.py` to generate performance reports")


def about_page():
    """Render about page."""
    st.markdown('<h2 class="sub-header">ℹ️ About This System</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Overview
    
    This Parkinson's Disease Detection System uses machine learning to analyze vocal features 
    for early detection of Parkinson's Disease. The system is built with production-grade 
    architecture and follows healthcare AI best practices.
    
    ### 🔬 Technology Stack
    
    - **Machine Learning**: Scikit-learn, XGBoost
    - **Web Framework**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Matplotlib, Seaborn
    
    ### 📊 Dataset
    
    The system is trained on the UCI Parkinson's Disease Dataset, which contains biomedical 
    voice measurements from individuals with and without Parkinson's Disease.
    
    **Features analyzed include:**
    - Vocal fundamental frequency measures
    - Jitter (frequency variation)
    - Shimmer (amplitude variation)
    - Harmonic-to-noise ratios
    - Nonlinear dynamical complexity measures
    
    ### ⚠️ Important Disclaimer
    
    This system is designed for:
    - Research purposes
    - Educational demonstrations
    - Preliminary screening support
    
    **This system should NOT be used as:**
    - A replacement for professional medical diagnosis
    - The sole basis for medical decisions
    - A definitive diagnostic tool
    
    Always consult qualified healthcare professionals for medical diagnosis and treatment.
    
    ### 📚 References
    
    1. UCI Machine Learning Repository: Parkinson's Disease Dataset
    2. Little, M. A., et al. (2007). "Exploiting nonlinear recurrence and fractal scaling 
       properties for voice disorder detection"
    3. Clinical guidelines for Parkinson's Disease diagnosis
    
    ### 👨‍💻 Development
    
    Built with ❤️ using modern ML engineering practices:
    - Clean architecture
    - Modular design
    - Production-quality code
    - Comprehensive testing
    - Clinical validation focus
    """)
    
    st.markdown("---")
    
    st.markdown("### 📞 Contact & Support")
    st.info("""
    For questions, issues, or contributions:
    - Developer Name :Ragul N
    - GitHub: [https://github.com/RAGUL-49]
    - Email: ragul.naa@gmail.com
    - Documentation: See README.md
    """)


def main():
    """Main application entry point."""
    render_header()
    
    # Load predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.error("❌ Failed to load model. Please ensure model files exist in 'models/' directory.")
        st.stop()
    
    # Sidebar navigation
    page = render_sidebar()
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_page()
    elif page == "🔬 Single Prediction":
        single_prediction_page(predictor)
    elif page == "📊 Batch Prediction":
        batch_prediction_page(predictor)
    elif page == "📈 Model Performance":
        model_performance_page()
    elif page == "ℹ️ About":
        about_page()


if __name__ == "__main__":
    main()