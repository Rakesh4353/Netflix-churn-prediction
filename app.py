import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib

# Page configuration
st.set_page_config(
    page_title="Netflix Churn Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #E50914;
        color: white;
    }
    .stButton>button:hover {
        background-color: #B2070E;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #221F1F;
    }
    h1, h2, h3 {
        color: #E50914;
    }
    .metric-card {
        background-color: #221F1F;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Create model directory if it doesn't exist
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "netflix_churn_model.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_data():
    file_path = "Netflix Userbase.csv"
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data['Join Date'] = pd.to_datetime(data['Join Date'], format="%d-%m-%y")
    data['Last Payment Date'] = pd.to_datetime(data['Last Payment Date'], format="%d-%m-%y")
    data['Plan Duration'] = data['Plan Duration'].str.extract('(\\d+)').astype(int)
    data['Churn'] = (data['Last Payment Date'] < pd.Timestamp("2023-07-01")).astype(int)
    
    # Save column names before encoding
    categorical_columns = ['Subscription Type', 'Country', 'Gender', 'Device']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    return data

def get_model(X_train=None, y_train=None, force_train=False):
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_NAMES_PATH) and not force_train:
        return joblib.load(MODEL_PATH), joblib.load(FEATURE_NAMES_PATH)
    
    if X_train is None or y_train is None:
        raise ValueError("Training data required for initial model training")
    
    model = RandomForestClassifier(random_state=42, n_estimators=200)
    model.fit(X_train, y_train)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X_train.columns), FEATURE_NAMES_PATH)
    return model, list(X_train.columns)

def create_metrics_cards(data):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Total Users</h3>
                <h2>{len(data):,}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        churn_rate = (data['Churn'].mean() * 100)
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Churn Rate</h3>
                <h2>{churn_rate:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        avg_revenue = data['Monthly Revenue'].mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Avg. Revenue</h3>
                <h2>${avg_revenue:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        avg_duration = data['Plan Duration'].mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Avg. Duration</h3>
                <h2>{avg_duration:.1f} months</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

def create_visualizations(data):
    # Create tabs for different visualization categories
    tab1, tab2, tab3 = st.tabs(["📊 User Demographics", "💰 Revenue Analysis", "🔄 Churn Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age Distribution
            fig_age = px.histogram(
                data, x='Age', 
                title='Age Distribution',
                color_discrete_sequence=['#E50914'],
                marginal='box'
            )
            fig_age.update_layout(template='plotly_dark')
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Subscription Type Distribution
            fig_sub = px.pie(
                data, 
                names='Subscription Type',
                title='Subscription Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_sub.update_layout(template='plotly_dark')
            st.plotly_chart(fig_sub, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by Subscription Type
            fig_rev = px.box(
                data, 
                x='Subscription Type', 
                y='Monthly Revenue',
                title='Revenue by Subscription Type',
                color='Subscription Type',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_rev.update_layout(template='plotly_dark')
            st.plotly_chart(fig_rev, use_container_width=True)
        
        with col2:
            # Monthly Revenue Trend
            monthly_revenue = data.groupby('Join Date')['Monthly Revenue'].mean().reset_index()
            fig_trend = px.line(
                monthly_revenue, 
                x='Join Date', 
                y='Monthly Revenue',
                title='Average Monthly Revenue Trend',
                line_shape='spline'
            )
            fig_trend.update_layout(template='plotly_dark')
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn by Subscription Type
            churn_by_sub = data.groupby('Subscription Type')['Churn'].mean().reset_index()
            fig_churn = px.bar(
                churn_by_sub,
                x='Subscription Type',
                y='Churn',
                title='Churn Rate by Subscription Type',
                color='Subscription Type',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_churn.update_layout(template='plotly_dark')
            st.plotly_chart(fig_churn, use_container_width=True)
        
        with col2:
            # Churn by Age Group
            data['Age_Group'] = pd.cut(data['Age'], bins=[0, 25, 35, 45, 100], labels=['18-25', '26-35', '36-45', '45+'])
            churn_by_age = data.groupby('Age_Group')['Churn'].mean().reset_index()
            fig_age_churn = px.bar(
                churn_by_age,
                x='Age_Group',
                y='Churn',
                title='Churn Rate by Age Group',
                color='Age_Group',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_age_churn.update_layout(template='plotly_dark')
            st.plotly_chart(fig_age_churn, use_container_width=True)

def prepare_user_input(user_data, feature_names):
    new_user = pd.DataFrame(0, index=[0], columns=feature_names)
    new_user['Monthly Revenue'] = user_data['monthly_revenue']
    new_user['Age'] = user_data['age']
    new_user['Plan Duration'] = user_data['plan_duration']
    
    if 'Subscription Type_Standard' in feature_names:
        new_user['Subscription Type_Standard'] = 1 if user_data['subscription_type'] == "Standard" else 0
    if 'Subscription Type_Premium' in feature_names:
        new_user['Subscription Type_Premium'] = 1 if user_data['subscription_type'] == "Premium" else 0
    
    device_columns = [col for col in feature_names if col.startswith('Device_')]
    for col in device_columns:
        device_name = col.replace('Device_', '')
        new_user[col] = 1 if user_data['device'] == device_name else 0
    
    return new_user

def main():
    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", use_container_width=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Churn Prediction", "Model Performance"])
    
    # Load and preprocess data
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Prepare features and target
    X = processed_data.drop(['User ID', 'Churn', 'Join Date', 'Last Payment Date'], axis=1)
    y = processed_data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get or train model
    model, feature_names = get_model(X_train, y_train)
    
    if page == "Dashboard":
        st.title("📊 Netflix User Analysis Dashboard")
        create_metrics_cards(data)
        create_visualizations(data)
    
    elif page == "Churn Prediction":
        st.title("🔮 Churn Prediction Tool")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
                monthly_revenue = st.number_input("Monthly Revenue ($)", min_value=5, max_value=50, value=15)
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
            
            with col2:
                plan_duration = st.number_input("Plan Duration (months)", min_value=1, max_value=24, value=1)
                device = st.selectbox("Primary Device", ["Smartphone", "Tablet", "Smart TV", "Laptop"])
            
            submitted = st.form_submit_button("Predict Churn")
            
            if submitted:
                user_input = {
                    'subscription_type': subscription_type,
                    'monthly_revenue': monthly_revenue,
                    'age': age,
                    'plan_duration': plan_duration,
                    'device': device
                }
                
                new_user_df = prepare_user_input(user_input, feature_names)
                churn_prediction = model.predict(new_user_df)[0]
                churn_probability = model.predict_proba(new_user_df)[0][1]
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if churn_probability*100 >60:
                        st.error("⚠️ High Risk of Churn")
                    elif churn_probability*100 > 40:
                        st.warning("Moderate Risk Of Churn")
                    else:
                        st.success("✅ Low Risk of Churn")
                
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_probability * 100,
                        title={'text': "Churn Probability"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "#E50914"},
                               'steps': [
                                   {'range': [0, 30], 'color': "lightgray"},
                                   {'range': [30, 70], 'color': "gray"},
                                   {'range': [70, 100], 'color': "darkgray"}]}))
                    fig.update_layout(template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
    
    else:  # Model Performance
        st.title("📈 Model Performance Analysis")
        
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=['No Churn', 'Churn'],
                y=['No Churn', 'Churn'],
                colorscale='RdBu',
                zmin=0, zmax=conf_matrix.max(),
                colorbar=dict(title='Count'),
                text=conf_matrix,  # Add text for annotations
                hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'  # Custom hover info
            ))
            
            fig_cm.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                template='plotly_dark'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.subheader("Feature Importance")
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig_imp = px.bar(
                feature_imp,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance Plot'
            )
            fig_imp.update_layout(template='plotly_dark')
            st.plotly_chart(fig_imp, use_container_width=True)
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report, use_container_width=True)

if __name__ == "__main__":
    main()
