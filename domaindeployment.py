import streamlit as st
import pandas as pd
import joblib
import numpy as np


# Custom CSS to inject into the Streamlit app
st.markdown("""
<style>
    /* This affects the Streamlit app's background color */
    .stApp {
        background-color: #262730;
    }
    
    /* This affects the title color */
    h1 {
        color: #FFCD00;
        background-color: #262730;
    }
    
    /* Modify button color */
    .stButton>button {
        color: #ffffff;
        background-color: #FFCD00;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
    }
    
    /* Modify button hover effect */
    .stButton>button:hover {
        background-color: #e6b800;
    }
</style>
""", unsafe_allow_html=True)

# Your main title using markdown
st.markdown('<h1 style="color: #FFCD00;">Flight Delay Prediction System</h1>', unsafe_allow_html=True)

# Load your trained model
model = joblib.load('domain_rf_model.joblib')

# Define the structure of your web app
st.title('Domain Name Values Prediction')

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded data
    data = pd.read_csv(uploaded_file)
    st.success('Data Successfully Uploaded!')

    # Check if the 'Name' column is in the CSV
    if 'Name' in data.columns:
        # Let the user select a domain to predict
        selected_domain = st.selectbox('Select a Domain to Predict', data['Name'].unique())

        # Get the features of the selected domain
        domain_features = data[data['Name'] == selected_domain][['Age of Domain', 'TLD Score', 'Search Queries Occurrences', 'Length Score', 'Word Composition']].iloc[0]

        # Display the features in separate text input fields
        if not domain_features.empty:
            st.write('Features of the selected domain:')
            st.text_input('Age of Domain', domain_features['Age of Domain'], disabled=True)
            st.text_input('Length Score', domain_features['Length Score'], disabled=True)
            st.text_input('Word Composition', domain_features['Word Composition'], disabled=True)
            st.text_input('TLD Score', domain_features['TLD Score'], disabled=True)
            st.text_input('Search Queries Occurrences', domain_features['Search Queries Occurrences'], disabled=True)
       

            # Button to make prediction
            if st.button('Predict Domain Price'):
                # Prepare the features for prediction
                prediction_features = np.array([[
                    domain_features['Age of Domain'],
                    domain_features['TLD Score'],
                    domain_features['Search Queries Occurrences'],
                    domain_features['Length Score'],
                    domain_features['Word Composition']
                ]])
                # Predict the price
                log_price_prediction = model.predict(prediction_features)
                # Convert the log price prediction back to the original price scale
                original_price_prediction = np.exp(log_price_prediction)  # Replace with appropriate transformation

                # Display the prediction
                st.write(f'The predicted price for "{selected_domain}" is: ${original_price_prediction[0]:,.2f}')
        else:
            st.error('No features found for the selected domain.')
    else:
        st.error('CSV must have a "Name" column.')
