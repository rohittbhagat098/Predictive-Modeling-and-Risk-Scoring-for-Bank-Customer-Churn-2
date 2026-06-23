# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and artifacts
@st.cache_resource
def load_artifacts():
    models = joblib.load('models/trained_models.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    results = joblib.load('models/results.pkl')
    feature_importance = pd.read_csv('models/feature_importance.csv')
    return models, scaler, feature_names, results, feature_importance

try:
    models, scaler, feature_names, results, feature_importance = load_artifacts()
    MODEL_LOADED = True
except:
    MODEL_LOADED = False
    st.error("⚠️ Models not found. Please run the training script first.")

# Sidebar navigation
st.sidebar.title("🏦 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "🎯 Risk Calculator", "📈 Feature Importance", 
     "🔮 What-If Simulation", "📋 Model Performance"]
)

# Helper function for risk score
def get_risk_score(probability):
    if probability < 0.3:
        return "Low", "risk-low", "🟢"
    elif probability < 0.6:
        return "Medium", "risk-medium", "🟡"
    else:
        return "High", "risk-high", "🔴"

# ============ DASHBOARD PAGE ============
if page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">Bank Customer Churn Dashboard</h1>', unsafe_allow_html=True)
    
    if MODEL_LOADED:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Best Model", "XGBoost" if results['XGBoost']['ROC-AUC'] == max(r['ROC-AUC'] for r in results.values()) else "Random Forest")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            best_auc = max(r['ROC-AUC'] for r in results.values())
            st.metric("Best ROC-AUC", f"{best_auc:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            best_f1 = max(r['F1-Score'] for r in results.values())
            st.metric("Best F1-Score", f"{best_f1:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Features", len(feature_names))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model performance comparison
        st.subheader("📊 Model Performance Comparison")
        
        df_results = pd.DataFrame(results).T
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=df_results.index,
                y=df_results[metric],
                text=[f'{v:.3f}' for v in df_results[metric]],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance summary
        st.subheader("🔑 Top 10 Churn Drivers")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_imp = px.bar(
                feature_importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance (Random Forest)',
                color='Importance',
                color_continuous_scale='Blues',
                height=400
            )
            fig_imp.update_layout(
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                template='plotly_white'
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with col2:
            st.markdown("#### Key Insights")
            st.markdown("""
            - **Age** is the strongest predictor of churn
            - **Tenure** and **Balance** are significant factors
            - **Activity** status strongly influences churn
            - Product count has complex relationship
            - Geography (Germany) shows higher churn risk
            """)
        
        # Distribution of predictions
        st.subheader("📈 Churn Probability Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Simulate predictions for visualization
            np.random.seed(42)
            sample_probs = np.random.beta(2, 5, 1000)
            
            fig_hist = px.histogram(
                sample_probs,
                nbins=30,
                title="Churn Probability Distribution",
                labels={'value': 'Churn Probability'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_hist.add_vline(x=0.3, line_dash="dash", line_color="green", annotation_text="Low Risk")
            fig_hist.add_vline(x=0.6, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
            fig_hist.update_layout(
                xaxis_range=[0, 1],
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("#### Risk Categories")
            low_count = np.sum(sample_probs < 0.3)
            medium_count = np.sum((sample_probs >= 0.3) & (sample_probs < 0.6))
            high_count = np.sum(sample_probs >= 0.6)
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Low Risk', 'Medium Risk', 'High Risk'],
                values=[low_count, medium_count, high_count],
                marker_colors=['#28a745', '#ffc107', '#dc3545'],
                hole=0.4
            )])
            fig_pie.update_layout(
                title="Risk Distribution",
                height=350,
                template='plotly_white'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

# ============ RISK CALCULATOR PAGE ============
elif page == "🎯 Risk Calculator":
    st.markdown('<h1 class="main-header">🎯 Customer Churn Risk Calculator</h1>', unsafe_allow_html=True)
    
    if MODEL_LOADED:
        st.markdown("Enter customer details to calculate churn risk score.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.slider("Credit Score", 350, 850, 650, help="Customer's credit score")
            age = st.slider("Age", 18, 90, 40, help="Customer's age")
            tenure = st.slider("Tenure (Years with Bank)", 0, 10, 5, help="Number of years with the bank")
            balance = st.number_input("Account Balance (€)", min_value=0.0, max_value=300000.0, value=80000.0, step=1000.0)
        
        with col2:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], help="Number of bank products used")
            has_credit_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            is_active = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            estimated_salary = st.number_input("Estimated Salary (€)", min_value=0.0, max_value=300000.0, value=100000.0, step=5000.0)
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        # Prepare features for prediction
        def prepare_features(credit_score, age, tenure, balance, num_products, has_credit_card, is_active, estimated_salary, geography, gender):
            # Base features
            data = {
                'CreditScore': credit_score,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': num_products,
                'HasCrCard': has_credit_card,
                'IsActiveMember': is_active,
                'EstimatedSalary': estimated_salary,
                'Geography_Germany': 1 if geography == "Germany" else 0,
                'Geography_Spain': 1 if geography == "Spain" else 0,
                'Gender_Male': 1 if gender == "Male" else 0,
            }
            
            # Derived features
            data['Balance_Salary_Ratio'] = balance / (estimated_salary + 1)
            data['Product_Density'] = num_products / (tenure + 1)
            data['Age_Tenure_Ratio'] = age / (tenure + 1)
            data['Engagement_Product'] = is_active * num_products
            data['Credit_Score_Good'] = 1 if credit_score >= 700 else 0
            data['Credit_Score_Fair'] = 1 if 600 <= credit_score < 700 else 0
            data['Credit_Score_Poor'] = 1 if credit_score < 600 else 0
            data['High_Balance'] = 1 if balance > 80000 else 0
            data['Active_With_Card'] = 1 if is_active == 1 and has_credit_card == 1 else 0
            data['Age_Young'] = 1 if age < 30 else 0
            data['Age_Middle'] = 1 if 30 <= age < 50 else 0
            data['Age_Senior'] = 1 if age >= 50 else 0
            data['Tenure_Short'] = 1 if tenure < 3 else 0
            data['Tenure_Medium'] = 1 if 3 <= tenure < 7 else 0
            data['Tenure_Long'] = 1 if tenure >= 7 else 0
            
            # Ensure all features are present
            df = pd.DataFrame([data])
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            return df[feature_names]
        
        # Calculate risk
        if st.button("🚀 Calculate Churn Risk", use_container_width=True):
            with st.spinner("Calculating risk score..."):
                try:
                    # Prepare features
                    features_df = prepare_features(
                        credit_score, age, tenure, balance, num_products, 
                        has_credit_card, is_active, estimated_salary, geography, gender
                    )
                    
                    # Scale features
                    features_scaled = scaler.transform(features_df)
                    
                    # Get predictions from all models
                    predictions = {}
                    for name, model in models.items():
                        if name == 'XGBoost':
                            prob = model.predict_proba(features_df)[0, 1]
                        else:
                            prob = model.predict_proba(features_scaled)[0, 1]
                        predictions[name] = prob
                    
                    # Average probability
                    avg_prob = np.mean(list(predictions.values()))
                    
                    # Display results
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        risk_level, risk_class, risk_icon = get_risk_score(avg_prob)
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;">
                            <h2>Churn Risk Score</h2>
                            <h1 style="font-size: 4rem; margin: 1rem 0;" class="{risk_class}">
                                {risk_icon} {avg_prob*100:.1f}%
                            </h1>
                            <h3 class="{risk_class}">Risk Level: {risk_level}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Model predictions
                    st.subheader("🤖 Model Predictions")
                    
                    pred_df = pd.DataFrame({
                        'Model': list(predictions.keys()),
                        'Probability': [p*100 for p in predictions.values()]
                    }).sort_values('Probability', ascending=False)
                    
                    fig_pred = px.bar(
                        pred_df,
                        x='Model',
                        y='Probability',
                        title='Churn Probability by Model',
                        color='Probability',
                        color_continuous_scale='RdYlGn_r',
                        text=[f'{p:.1f}%' for p in pred_df['Probability']],
                        height=350
                    )
                    fig_pred.update_layout(
                        yaxis_range=[0, 100],
                        template='plotly_white',
                        xaxis_title="Model",
                        yaxis_title="Churn Probability (%)"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Risk factors
                    st.subheader("🔍 Key Risk Factors")
                    
                    risk_factors = []
                    
                    if age > 50:
                        risk_factors.append(("Age", "High age (>50) increases churn risk", "Senior customers are more likely to churn"))
                    if tenure < 3:
                        risk_factors.append(("Tenure", "Short tenure (<3 years)", "New customers have higher churn rates"))
                    if is_active == 0:
                        risk_factors.append(("Activity", "Inactive member", "Inactive members are more likely to churn"))
                    if num_products == 1:
                        risk_factors.append(("Products", "Only 1 product", "Customers with fewer products are more likely to churn"))
                    if balance > 150000:
                        risk_factors.append(("Balance", "High balance", "High-balance customers sometimes churn to competitors"))
                    if credit_score < 600:
                        risk_factors.append(("Credit Score", "Low credit score", "Lower credit scores correlate with higher churn"))
                    if geography == "Germany":
                        risk_factors.append(("Geography", "Germany", "German customers show higher churn rates"))
                    
                    if risk_factors:
                        for factor, label, desc in risk_factors:
                            with st.expander(f"⚠️ {factor}: {label}"):
                                st.write(desc)
                    else:
                        st.info("✅ No major risk factors identified for this customer.")
                
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

# ============ FEATURE IMPORTANCE PAGE ============
elif page == "📈 Feature Importance":
    st.markdown('<h1 class="main-header">📈 Feature Importance Analysis</h1>', unsafe_allow_html=True)
    
    if MODEL_LOADED:
        st.markdown("""
        Understanding which features most strongly influence churn predictions helps 
        banks design targeted retention strategies.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            n_features = st.slider("Number of features to display", 5, 30, 15)
            
            fig_imp = px.bar(
                feature_importance.head(n_features),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Top {n_features} Features by Importance',
                color='Importance',
                color_continuous_scale='Viridis',
                height=max(400, n_features * 25)
            )
            fig_imp.update_layout(
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                template='plotly_white'
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with col2:
            st.markdown("#### Feature Categories")
            st.markdown("""
            **📊 Demographic**
            - Age
            - Geography
            - Gender
            
            **💰 Financial**
            - Balance
            - Credit Score
            - Estimated Salary
            - Balance-to-Salary Ratio
            
            **🏦 Banking Behavior**
            - Tenure
            - Number of Products
            - Active Member
            - Has Credit Card
            
            **📈 Derived Features**
            - Engagement-Product Interaction
            - Age-Tenure Ratio
            - Product Density
            """)
        
        # Detailed feature analysis
        st.subheader("🔍 Detailed Feature Analysis")
        
        # Feature type categorization
        feature_categories = {
            'Demographic': ['Age', 'Geography_Germany', 'Geography_Spain', 'Gender_Male', 
                           'Age_Young', 'Age_Middle', 'Age_Senior'],
            'Financial': ['CreditScore', 'Balance', 'EstimatedSalary', 'Balance_Salary_Ratio',
                         'Credit_Score_Good', 'Credit_Score_Fair', 'Credit_Score_Poor', 'High_Balance'],
            'Banking': ['Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                       'Tenure_Short', 'Tenure_Medium', 'Tenure_Long', 'Active_With_Card'],
            'Derived': ['Product_Density', 'Age_Tenure_Ratio', 'Engagement_Product']
        }
        
        cat_importance = {}
        for cat, features in feature_categories.items():
            cat_importance[cat] = feature_importance[feature_importance['Feature'].isin(features)]['Importance'].sum()
        
        fig_cat = px.pie(
            values=list(cat_importance.values()),
            names=list(cat_importance.keys()),
            title='Feature Importance by Category',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.3
        )
        fig_cat.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig_cat, use_container_width=True)

# ============ WHAT-IF SIMULATION PAGE ============
elif page == "🔮 What-If Simulation":
    st.markdown('<h1 class="main-header">🔮 What-If Scenario Simulator</h1>', unsafe_allow_html=True)
    
    if MODEL_LOADED:
        st.markdown("""
        Adjust customer attributes to see how changes affect churn probability.
        This helps banks understand which interventions are most effective.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Customer Profile")
            
            # Base customer
            base_age = st.slider("Age", 18, 90, 40, key="sim_age")
            base_tenure = st.slider("Tenure (Years)", 0, 10, 5, key="sim_tenure")
            base_balance = st.number_input("Balance (€)", 0, 300000, 80000, key="sim_balance")
            base_products = st.selectbox("Number of Products", [1, 2, 3, 4], key="sim_products")
            base_active = st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key="sim_active")
            base_credit = st.slider("Credit Score", 350, 850, 650, key="sim_credit")
            base_salary = st.number_input("Salary (€)", 0, 300000, 100000, key="sim_salary")
            base_geography = st.selectbox("Geography", ["France", "Spain", "Germany"], key="sim_geo")
        
        with col2:
            st.subheader("🔄 Intervention Scenario")
            
            st.markdown("**Scenario Variables**")
            
            new_active = st.selectbox(
                "Active Status (Intervention)",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="sim_new_active",
                index=0
            )
            
            new_products = st.selectbox(
                "Number of Products (Upsell)",
                [1, 2, 3, 4],
                key="sim_new_products",
                index=1
            )
            
            new_tenure = st.slider(
                "Additional Years (Retention)",
                0, 5, 2,
                key="sim_new_tenure"
            )
            
            # Prepare features for both scenarios
            def prepare_sim_features(age, tenure, balance, products, active, credit, salary, geography):
                data = {
                    'CreditScore': credit,
                    'Age': age,
                    'Tenure': tenure,
                    'Balance': balance,
                    'NumOfProducts': products,
                    'HasCrCard': 1,
                    'IsActiveMember': active,
                    'EstimatedSalary': salary,
                    'Geography_Germany': 1 if geography == "Germany" else 0,
                    'Geography_Spain': 1 if geography == "Spain" else 0,
                    'Gender_Male': 1,
                }
                
                data['Balance_Salary_Ratio'] = balance / (salary + 1)
                data['Product_Density'] = products / (tenure + 1)
                data['Age_Tenure_Ratio'] = age / (tenure + 1)
                data['Engagement_Product'] = active * products
                data['Credit_Score_Good'] = 1 if credit >= 700 else 0
                data['Credit_Score_Fair'] = 1 if 600 <= credit < 700 else 0
                data['Credit_Score_Poor'] = 1 if credit < 600 else 0
                data['High_Balance'] = 1 if balance > 80000 else 0
                data['Active_With_Card'] = active
                data['Age_Young'] = 1 if age < 30 else 0
                data['Age_Middle'] = 1 if 30 <= age < 50 else 0
                data['Age_Senior'] = 1 if age >= 50 else 0
                data['Tenure_Short'] = 1 if tenure < 3 else 0
                data['Tenure_Medium'] = 1 if 3 <= tenure < 7 else 0
                data['Tenure_Long'] = 1 if tenure >= 7 else 0
                
                df = pd.DataFrame([data])
                for feature in feature_names:
                    if feature not in df.columns:
                        df[feature] = 0
                return df[feature_names]
            
            if st.button("🔄 Run Simulation", use_container_width=True):
                with st.spinner("Running simulation..."):
                    # Base scenario
                    base_features = prepare_sim_features(
                        base_age, base_tenure, base_balance, base_products, 
                        base_active, base_credit, base_salary, base_geography
                    )
                    base_scaled = scaler.transform(base_features)
                    base_prob = models['Random Forest'].predict_proba(base_scaled)[0, 1]
                    
                    # New scenario
                    new_features = prepare_sim_features(
                        base_age, base_tenure + new_tenure, base_balance, new_products,
                        new_active, base_credit, base_salary, base_geography
                    )
                    new_scaled = scaler.transform(new_features)
                    new_prob = models['Random Forest'].predict_proba(new_scaled)[0, 1]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Simulation Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risk_level, risk_class, risk_icon = get_risk_score(base_prob)
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;">
                            <h4>Current Risk</h4>
                            <h2 style="font-size: 2.5rem;" class="{risk_class}">{risk_icon} {base_prob*100:.1f}%</h2>
                            <p class="{risk_class}">{risk_level} Risk</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        risk_level, risk_class, risk_icon = get_risk_score(new_prob)
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;">
                            <h4>After Intervention</h4>
                            <h2 style="font-size: 2.5rem;" class="{risk_class}">{risk_icon} {new_prob*100:.1f}%</h2>
                            <p class="{risk_class}">{risk_level} Risk</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        improvement = (base_prob - new_prob) * 100
                        color = "#28a745" if improvement > 0 else "#dc3545"
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;">
                            <h4>Risk Reduction</h4>
                            <h2 style="font-size: 2.5rem; color: {color};">{improvement:.1f}%</h2>
                            <p>{'✅' if improvement > 0 else '⚠️'} {'Decrease' if improvement > 0 else 'Increase'} in churn risk</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Scenario comparison chart
                    fig_compare = go.Figure()
                    
                    fig_compare.add_trace(go.Bar(
                        name='Current',
                        x=['Current', 'After Intervention'],
                        y=[base_prob*100, new_prob*100],
                        text=[f'{base_prob*100:.1f}%', f'{new_prob*100:.1f}%'],
                        textposition='auto',
                        marker_color=['#1f77b4', '#2ca02c']
                    ))
                    
                    fig_compare.update_layout(
                        title='Churn Probability Comparison',
                        yaxis_title="Churn Probability (%)",
                        yaxis_range=[0, 100],
                        template='plotly_white',
                        height=300
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    # Intervention impact breakdown
                    st.subheader("💡 Intervention Impact")
                    
                    changes = []
                    if new_active != base_active:
                        changes.append(f"✅ Changed Active Status from {'Inactive' if base_active==0 else 'Active'} to {'Active' if new_active==1 else 'Inactive'}")
                    if new_products != base_products:
                        changes.append(f"✅ Changed Products from {base_products} to {new_products}")
                    if new_tenure > 0:
                        changes.append(f"✅ Added {new_tenure} more years of tenure")
                    
                    if changes:
                        for change in changes:
                            st.success(change)
                    else:
                        st.info("No interventions applied. Try adjusting the scenario variables.")

# ============ MODEL PERFORMANCE PAGE ============
elif page == "📋 Model Performance":
    st.markdown('<h1 class="main-header">📋 Model Performance Dashboard</h1>', unsafe_allow_html=True)
    
    if MODEL_LOADED:
        st.markdown("""
        Comprehensive evaluation of all trained models with key performance metrics.
        This helps in selecting the best model for deployment.
        """)
        
        # Performance metrics table
        df_results = pd.DataFrame(results).T
        df_results.index.name = 'Model'
        df_results = df_results.reset_index()
        
        # Color formatting
        def color_metrics(val):
            if isinstance(val, (int, float)):
                if val > 0.8:
                    return 'color: #28a745'
                elif val > 0.6:
                    return 'color: #ffc107'
                else:
                    return 'color: #dc3545'
            return ''
        
        styled_df = df_results.style.map(color_metrics, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Model comparison chart
        st.subheader("📊 Model Comparison")
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=df_results[metrics].values,
            x=metrics,
            y=df_results['Model'],
            text=df_results[metrics].values,
            texttemplate='%{text:.3f}',
            textfont={"size": 12},
            colorscale='RdYlGn',
            zmid=0.5,
            zmin=0.5,
            zmax=1.0
        ))
        fig_heat.update_layout(
            title='Model Performance Heatmap',
            xaxis_title="Metric",
            yaxis_title="Model",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Best model details
        st.subheader("🏆 Best Model Analysis")
        
        best_model = max(results, key=lambda x: results[x]['ROC-AUC'])
        best_metrics = results[best_model]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", best_model)
        with col2:
            st.metric("ROC-AUC", f"{best_metrics['ROC-AUC']:.4f}")
        with col3:
            st.metric("F1-Score", f"{best_metrics['F1-Score']:.4f}")
        with col4:
            st.metric("CV Score", f"{best_metrics['CV_Mean']:.4f} ± {best_metrics['CV_Std']:.4f}")
        
        st.markdown(f"""
        **Recommendation**: The **{best_model}** model achieves the best performance with a ROC-AUC of **{best_metrics['ROC-AUC']:.4f}** 
        and F1-Score of **{best_metrics['F1-Score']:.4f}**. This model should be considered for deployment.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>🏦 Bank Customer Churn Prediction System | Built with ❤️ using Streamlit</p>
    <p style="font-size: 0.8rem;">Data source: European Bank Customer Dataset | Unified Mentor Project</p>
</div>
""", unsafe_allow_html=True)