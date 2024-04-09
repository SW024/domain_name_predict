import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your trained model
model = joblib.load('domain_rf_model.joblib')

# Define the structure of your web app
st.title('Random Forest Model Deployment')

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
        domain_features = data[data['Name'] == selected_domain][['Age of Domain', 'TLD Score', 'Search Queries Occurrences', 'Length Score', 'Word Composition']]
        
        # Display the features in separate columns
        if not domain_features.empty:
            st.write('Features of the selected domain:')
            st.table(domain_features.reset_index(drop=True))  # Display features in a table

            # Button to make prediction
            if st.button('Predict Domain Price'):
                # Reshape the data for prediction
                prediction_features = domain_features.values.reshape(1, -1)
                # Predict the price
                log_price_prediction = model.predict(prediction_features)
                
                # Convert the log price prediction back to original scale
                # Assuming you used np.log() to get the log_price during training, you'd use np.exp() to reverse it
                original_price_prediction = np.exp(log_price_prediction)
                
                # Display the prediction
                st.write(f'The predicted price for "{selected_domain}" is: ${original_price_prediction[0]:,.2f}')
        else:
            st.error('No features found for the selected domain.')
    else:
        st.error('CSV must have a Domain column.')
