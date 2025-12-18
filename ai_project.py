import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import time
warnings.filterwarnings('ignore')

# Page config with custom theme
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Header animations */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    /* Card styling with hover effects */
    .stApp [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .css-1r6slb0:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    /* Button animations */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    /* Metric cards */
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #764ba2;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Success/Warning boxes */
    .stAlert {
        border-radius: 10px;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Section headers */
    h2, h3 {
        color: #667eea;
        font-weight: 700;
        margin-top: 2rem;
        animation: fadeInDown 0.6s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Fade in animation for content */
    .fade-in {
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Animated sidebar
st.sidebar.markdown("# 🎓 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("", ["🏠 Home", "🤖 Train Models", "🎯 Make Prediction", "ℹ️ About"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Source")
data_option = st.sidebar.radio("", ["Sample dataset", "Local CSV"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌟 Quick Stats")

# Dataset functions (keeping your original code)
@st.cache_data
def create_sample_dataset(n_samples=500):
    np.random.seed(42)
    data = {
        'Hours Studied': np.random.randint(0, 10, n_samples),
        'Previous Scores': np.random.randint(0, 100, n_samples),
        'Extracurricular Activities': np.random.choice(['Yes', 'No'], n_samples),
        'Sleep Hours': np.random.randint(4, 9, n_samples),
        'Sample Question Papers Practiced': np.random.randint(0, 5, n_samples),
        'Parent Education': np.random.choice(['High School','Bachelor','Master','PhD'], n_samples),
        'Tutoring': np.random.choice(['Yes','No'], n_samples),
        'Internet Access': np.random.choice(['Yes','No'], n_samples)
    }
    df = pd.DataFrame(data)
    df['final_score'] = (
        df['Hours Studied'] * 6 +
        df['Previous Scores'] * 0.5 +
        df['Sleep Hours'] * 2 +
        df['Sample Question Papers Practiced'] * 4 +
        (df['Extracurricular Activities'] == 'Yes') * 5 +
        (df['Tutoring'] == 'Yes') * 7 +
        (df['Internet Access'] == 'Yes') * 3 +
        (df['Parent Education'] == 'PhD') * 6 +
        (df['Parent Education'] == 'Master') * 4 +
        (df['Parent Education'] == 'Bachelor') * 2 +
        np.random.normal(0, 5, n_samples)
    )
    df['final_score'] = np.clip(df['final_score'], 0, 100)
    return df

@st.cache_data
def load_local_dataset():
    df = pd.read_csv("StudentPerformance.csv")
    if 'Hours Studied' in df.columns and 'Previous Scores' in df.columns:
        df['final_score'] = (
            df['Hours Studied'] * 6 +
            df['Previous Scores'] * 0.5 +
            df['Sleep Hours'] * 2 +
            df['Sample Question Papers Practiced'] * 4 +
            (df['Extracurricular Activities'] == 'Yes') * 5 +
            np.random.normal(0, 5, len(df))
        )
        df['final_score'] = np.clip(df['final_score'], 0, 100)
    else:
        score_cols = [c for c in ['math score','reading score','writing score'] if c in df.columns]
        if score_cols:
            df['final_score'] = df[score_cols].mean(axis=1)
        else:
            df['final_score'] = np.random.uniform(40,95,len(df))
    return df

def preprocess_data(df):
    df_processed = df.copy()
    label_encoders = {}
    categorical_cols = ['Extracurricular Activities','Parent Education','Tutoring','Internet Access']
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    return df_processed, label_encoders

def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
    return results


# Load dataset
df = create_sample_dataset(1000) if data_option == "Sample dataset" else load_local_dataset()

# Update sidebar stats
st.sidebar.metric("📚 Total Students", len(df))
st.sidebar.metric("⭐ Avg Score", f"{df['final_score'].mean():.1f}")

# HOME PAGE
if page == "🏠 Home":
    # Animated title
    st.markdown('<h1 class="main-title">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
    
    # Welcome message with animation
    st.markdown("""
        <div class="fade-in">
            <h3 style='text-align: center; color: #764ba2;'>
                Predict Student Success with AI-Powered Machine Learning
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics with icons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Students", len(df))
    with col2:
        st.metric("📊 Average Score", f"{df['final_score'].mean():.2f}")
    with col3:
        st.metric("🎯 Features", len(df.columns) - 1)
    with col4:
        st.metric("📈 Max Score", f"{df['final_score'].max():.2f}")
    
    st.markdown("---")
    
    # Dataset preview with animation
    st.markdown("### 📋 Dataset Preview")
    with st.spinner("Loading dataset..."):
        time.sleep(0.5)
        st.dataframe(df.head(10), use_container_width=True, height=400)
    
    # Feature distribution
    st.markdown("### 📊 Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['final_score'], bins=20, color='#667eea', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Final Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Student Scores', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# TRAIN MODELS PAGE
elif page == "🤖 Train Models":
    st.markdown('<h1 class="main-title">🤖 Model Training & Evaluation</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: rgba(102, 126, 234, 0.1); border-radius: 10px; margin-bottom: 30px;'>
            <h4 style='color: #764ba2;'>Train multiple ML algorithms and compare their performance</h4>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("⚙️ Preprocessing data...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        df_processed, label_encoders = preprocess_data(df)
        X = df_processed.drop(['final_score','Performance Index'], axis=1, errors='ignore')
        y = df_processed['final_score']
        
        status_text.text("🔄 Splitting dataset...")
        progress_bar.progress(40)
        time.sleep(0.3)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        status_text.text("🤖 Training models...")
        progress_bar.progress(60)
        time.sleep(0.5)
        
        results = train_models(X_train, y_train, X_test, y_test)
        
        status_text.text("✅ Evaluating models...")
        progress_bar.progress(80)
        time.sleep(0.3)
        
        st.session_state.results = results
        st.session_state.label_encoders = label_encoders
        st.session_state.feature_names = X.columns.tolist()
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        st.balloons()
        st.success("✅ Models trained successfully!")
    
    if 'results' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Model Performance Comparison")
        
        comparison_data = []
        for name, result in st.session_state.results.items():
            comparison_data.append({
                'Model': name,
                'RMSE': f"{result['rmse']:.2f}",
                'MAE': f"{result['mae']:.2f}",
                'R² Score': f"{result['r2']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, height=150)
        
        # Best model highlight
        best_model = max(st.session_state.results.items(), key=lambda x: x[1]['r2'])
        st.markdown(f"""
            <div style='text-align: center; padding: 15px; background: linear-gradient(90deg, #667eea, #764ba2); 
                        border-radius: 10px; color: white; margin: 20px 0;'>
                <h3>🏆 Best Model: {best_model[0]} (R² = {best_model[1]['r2']:.4f})</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 R² Score Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            models = list(st.session_state.results.keys())
            r2_scores = [st.session_state.results[m]['r2'] for m in models]
            colors = ['#667eea', '#764ba2', '#f093fb']
            bars = ax.bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
            ax.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=15)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 🎯 Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(8, 5))
            y_pred = best_model[1]['predictions']
            ax.scatter(st.session_state.y_test, y_pred, alpha=0.6, c='#667eea', edgecolors='black', linewidth=0.5)
            ax.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Score', fontsize=11, fontweight='bold')
            ax.set_ylabel('Predicted Score', fontsize=11, fontweight='bold')
            ax.set_title(f'{best_model[0]} Predictions', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

# MAKE PREDICTION PAGE
elif page == "🎯 Make Prediction":
    st.markdown('<h1 class="main-title">🎯 Predict Student Performance</h1>', unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Please train the models first from the 'Train Models' page!")
    else:
        st.markdown("""
            <div style='text-align: center; padding: 15px; background: rgba(102, 126, 234, 0.1); 
                        border-radius: 10px; margin-bottom: 30px;'>
                <h4 style='color: #764ba2;'>Enter student details to predict their final score</h4>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📚 Academic Information")
            hours_studied = st.slider("📖 Hours Studied", 0, 10, 5, help="Daily study hours")
            prev_scores = st.slider("📊 Previous Scores", 0, 100, 70, help="Previous exam scores")
            papers_practiced = st.slider("📄 Sample Papers Practiced", 0, 5, 2, help="Number of practice papers")
            sleep_hours = st.slider("😴 Sleep Hours", 4, 9, 7, help="Daily sleep hours")
        
        with col2:
            st.markdown("#### 🏠 Background Information")
            extracurricular = st.selectbox("🎨 Extracurricular Activities", ['Yes','No'])
            parent_education = st.selectbox("🎓 Parent Education", ['High School','Bachelor','Master','PhD'])
            tutoring = st.selectbox("👨‍🏫 Tutoring", ['Yes','No'])
            internet_access = st.selectbox("🌐 Internet Access", ['Yes','No'])
        
        st.markdown("---")
        
        if st.button("🔮 Predict Score", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing student profile..."):
                time.sleep(1)
                
                input_data = {
                    'Hours Studied': hours_studied,
                    'Previous Scores': prev_scores,
                    'Extracurricular Activities': extracurricular,
                    'Parent Education': parent_education,
                    'Tutoring': tutoring,
                    'Internet Access': internet_access,
                    'Sleep Hours': sleep_hours,
                    'Sample Question Papers Practiced': papers_practiced
                }
                df_input = pd.DataFrame([input_data])
                
                for col, le in st.session_state.label_encoders.items():
                    if col in df_input.columns:
                        df_input[col] = le.transform(df_input[col])
                
                df_input = df_input[st.session_state.feature_names]
                best_model = max(st.session_state.results.items(), key=lambda x: x[1]['r2'])
                prediction = best_model[1]['model'].predict(df_input)[0]
                prediction = max(0, min(prediction, 100))
                
                # Animated results
                st.markdown("---")
                st.markdown("### 🎉 Prediction Results")
                
                # Score display with animation
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea, #764ba2); 
                                    border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.2);'>
                            <h2 style='color: white; margin: 0;'>📊 Predicted Score</h2>
                            <h1 style='color: white; font-size: 3rem; margin: 10px 0;'>{prediction:.1f}</h1>
                            <p style='color: white; margin: 0;'>out of 100</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if prediction >= 90:
                        grade, emoji, color = "A", "🌟", "#4CAF50"
                    elif prediction >= 80:
                        grade, emoji, color = "B", "⭐", "#2196F3"
                    elif prediction >= 70:
                        grade, emoji, color = "C", "✨", "#FF9800"
                    elif prediction >= 60:
                        grade, emoji, color = "D", "💫", "#FF5722"
                    else:
                        grade, emoji, color = "F", "📚", "#F44336"
                    
                    st.markdown(f"""
                        <div style='text-align: center; padding: 30px; background: {color}; 
                                    border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.2);'>
                            <h2 style='color: white; margin: 0;'>🎖️ Grade</h2>
                            <h1 style='color: white; font-size: 3rem; margin: 10px 0;'>{emoji} {grade}</h1>
                            <p style='color: white; margin: 0;'>Performance Level</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #f093fb, #f5576c); 
                                    border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.2);'>
                            <h2 style='color: white; margin: 0;'>🤖 Model</h2>
                            <h3 style='color: white; margin: 10px 0;'>{best_model[0]}</h3>
                            <p style='color: white; margin: 0;'>Accuracy: {best_model[1]['r2']:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Progress bar
                st.markdown("### 📈 Score Visualization")
                st.progress(prediction/100)
                
                # Recommendations
                st.markdown("---")
                st.markdown("### 💡 Personalized Recommendations")
                
                if prediction < 70:
                    st.error("⚠️ **Student needs additional support:**")
                    recommendations = []
                    if hours_studied < 5:
                        recommendations.append("⏰ Increase daily study hours to at least 5 hours")
                    if sleep_hours < 6:
                        recommendations.append("😴 Ensure 7-8 hours of quality sleep")
                    if papers_practiced < 2:
                        recommendations.append("📄 Practice at least 3-4 sample papers")
                    if tutoring == 'No':
                        recommendations.append("👨‍🏫 Consider enrolling in tutoring programs")
                    if extracurricular == 'No':
                        recommendations.append("🎨 Join extracurricular activities for holistic development")
                    
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.success("✅ **Excellent Performance! Keep it up!**")
                    st.markdown("""
                        - 🌟 Continue with current study routine
                        - 📚 Focus on advanced topics
                        - 🎯 Set higher academic goals
                        - 🤝 Help peers who need support
                    """)

# ABOUT PAGE
elif page == "ℹ️ About":
    st.markdown('<h1 class="main-title">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 30px; border-radius: 15px; margin: 20px 0;'>
            <h2 style='color: #764ba2; text-align: center;'>🎓 Student Performance Prediction System</h2>
            <p style='text-align: center; font-size: 1.2rem; color: #555;'>
                AI-Powered Machine Learning for Educational Excellence
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Project Overview")
        st.write("""
        This system uses advanced machine learning algorithms to predict student 
        performance based on academic and personal factors. It helps identify 
        students who need support and provides actionable insights.
        """)
        
        st.markdown("### 🤖 Algorithms Used")
        st.write("""
        - **Linear Regression**: Baseline model
        - **Decision Tree**: Non-linear patterns
        - **Random Forest**: Ensemble method (Best accuracy)
        """)
        
        st.markdown("### 📊 Key Features")
        st.write("""
        - Study hours & Previous scores
        - Sleep patterns & Practice papers
        - Extracurricular activities
        - Parent education & Tutoring
        - Internet access
        """)
    
    with col2:
        st.markdown("### 🛠️ Technologies")
        st.write("""
        - **Python 3.x**
        - **Streamlit** (Web Interface)
        - **Scikit-learn** (ML Models)
        - **Pandas & NumPy** (Data Processing)
        - **Matplotlib & Seaborn** (Visualization)
        """)
        
        # st.markdown("### 👥 Team Members")
        # st.info("""
        # 1. **Nabeel Sikandar** - Dataset & Preprocessing
        # 2. **Pymra**  - Model Training & Evaluation  
        # 3. **Noor** - UI Design & Documentation
        # """)
        
        st.markdown("### 🎯 Project Goals")
        st.write("""
        ✅ 80%+ prediction accuracy  
        ✅ Early identification of at-risk students  
        ✅ Data-driven educational insights  
        ✅ Timely intervention strategies  
        """)
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <p style='color: #764ba2; font-size: 1.1rem;'>
                <b></b>
            </p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 10px; color: #888;'>
        <p>🎓 Student Performance Prediction System | Powered by AI & Machine Learning</p>
    </div>
""", unsafe_allow_html=True)