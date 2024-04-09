import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('domain_rf_model.joblib')

# Define the structure of your web app
st.title('Domain Name Scoring System')

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded data
    data = pd.read_csv(uploaded_file)
    st.write('Data Successfully Uploaded!')
    
    # Assuming there's a column in the CSV called 'Domain' which contains the domain names
    if 'Domain' in data.columns:
        # Let the user select a domain to predict
        selected_domain = st.selectbox('Select a Domain to Predict', data['Domain'].unique())
        
        # Get the features of the selected domain
        domain_features = data[data['Domain'] == selected_domain][['Age of Domain', 'TLD Score', 'Search Queries Occurrences', 'Length Score', 'Word Composition']]
        
        # Display the features
        if not domain_features.empty:
            st.write('Features of the selected domain:')
            st.json(domain_features.iloc[0].to_json())  # Convert to JSON for pretty printing

            # Button to make prediction
            if st.button('Predict Domain Price'):
                # Reshape the data for prediction
                prediction_features = domain_features.values.reshape(1, -1)
                # Predict the price
                prediction = model.predict(prediction_features)
                
                # Display the prediction
                st.write(f'The predicted log price for "{selected_domain}" is: {prediction[0]}')
        else:
            st.error('No features found for the selected domain.')
    else:
        st.error('CSV must have a "Domain" column.')
