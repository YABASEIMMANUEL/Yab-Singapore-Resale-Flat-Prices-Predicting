import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import os

st.set_page_config(page_title="Singapore Resale Flat Price Prediction - Made by: yabase", layout="wide", initial_sidebar_state="expanded")

# Helper function to load pickle files with error handling
def load_pickle(file_name):
    try:
        with open(file_name, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"File not found: {file_name}")
        return None
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        return None

# Load encoders and scaler
town_encoder = load_pickle('town_encoder.pkl')
flat_model_encoder = load_pickle('flat_model_encoder.pkl')
flat_type_encoder = load_pickle('flat_type_encoder.pkl')
address_encoder = load_pickle('address_encoder.pkl')
scaler = load_pickle('scaler.pkl')

if None in [town_encoder, flat_model_encoder, flat_type_encoder, address_encoder, scaler]:
    st.stop()  # Stop the app if any of the files failed to load

storey = ['01 TO 03', '01 TO 05', '04 TO 06', '06 TO 10', '07 TO 09', '10 TO 12', '11 TO 15', '13 TO 15', 
          '16 TO 18', '16 TO 20', '19 TO 21', '21 TO 25', '22 TO 24', '25 TO 27', '26 TO 30', '28 TO 30', 
          '31 TO 33', '31 TO 35', '34 TO 36', '36 TO 40', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', 
          '49 TO 51']

all_addresses = address_encoder.classes_.tolist()
all_towns = town_encoder.classes_.tolist()
all_flat_model = flat_model_encoder.categories_[0].tolist()
flat_type_mapping = flat_type_encoder.categories_[0].tolist()

# Function to get user input
def get_user_input():
    st.subheader(":violet[Fill all the fields and press the button below to view the **Predicted price** of Resale Flat Price : ]")
    cc1, cc2 = st.columns([2, 2])
    
    with cc1:
        year = st.number_input("Year (YYYY) : ", min_value=1990, max_value=2034)
        town = st.selectbox("Town : ", all_towns)
        flat_model = st.selectbox("Flat Model : ", all_flat_model)
        flat_type = st.selectbox("Flat Type : ", list(flat_type_mapping))
        storey_range = st.selectbox("Storey Range : ", storey)
    with cc2:
        floor_area_sqm = st.number_input("Floor Area (sqm) : ")
        price_per_sqm = st.number_input("Price Area (sqm) : ")
        lease_commence_date = st.number_input("Lease Commencement Year (YYYY) : ", min_value=1966, max_value=2023)
        address = st.selectbox('Enter the Address with Block Number with Street name :', all_addresses)

    user_input_data = {
        'year': year,
        'town': town,
        'flat_model': flat_model,
        'flat_type': flat_type,
        'storey_range': storey_range,
        'floor_area_sqm': floor_area_sqm,
        'price_per_sqm': price_per_sqm,
        'lease_commence_date': lease_commence_date,
        'address': address
    }
    
    return pd.DataFrame(user_input_data, index=[0])

# Function to load model
def load_model():
    try:
        with open('ET_regression_model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found: ET_regression_model.pkl")
        return None
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

# Data transformation function
def data_transformation_for_the_model(df):
    def to_upper(df):
        string_cols = df.select_dtypes(include='object').columns
        df[string_cols] = df[string_cols].apply(lambda x: x.str.upper())
        return df
    
    df = (df
          .pipe(to_upper)
          .assign(
              town_ENCODED=lambda x: town_encoder.transform(x[["town"]]),
              flat_model_ENCODED=lambda x: flat_model_encoder.transform(x[["flat_model"]]),
              flat_type_ENCODED=lambda x: flat_type_encoder.transform(x[["flat_type"]]),
              address_ENCODED=lambda x: address_encoder.transform(x[["address"]]),
              median_storey_range=lambda x: x['storey_range'].apply(lambda storey_range: (int(storey_range.split(' TO ')[0]) + int(storey_range.split(' TO ')[1])) / 2).astype(int)
          )
          [['year', 'town_ENCODED', 'flat_model_ENCODED', 'flat_type_ENCODED', 'median_storey_range', 'floor_area_sqm', 'price_per_sqm', 'lease_commence_date', 'address_ENCODED']]
          )

    df_scaled = scaler.transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns)
    
    return df

# Main function to run the app
def main():
    # Sidebar layout with image and navigation
    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("Navigation Menu")
        choice = st.radio("Select a page", ["Home", "Price Prediction", "Multiple Predictions"])
        st.info("This project predicts the resale flat prices and status.")

    # Right side content based on navigation selection
    if choice == "Home":
        st.markdown("<h1 style='text-align: center; font-size: 38px; color: #55ACEE; font-weight: 700; font-family: inherit;'>Welcome to the Singapore Resale Flat Pricing Analysis App!</h1>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 2px solid beige;'>", unsafe_allow_html=True)
        st.subheader("App Insights")
        st.write("- This interactive tool is designed to estimate resale flat prices in Singapore. Whether you are a buyer, seller, or simply curious about housing trends, our app provides reliable price projections using sophisticated machine learning methods.")
        
        st.subheader("Essential Features")
        st.markdown("- **Advanced Price Estimation**: Our app utilizes cutting-edge machine learning algorithms to deliver accurate resale price estimates by analyzing both past data and current market conditions.")
        st.markdown("- **User-Friendly Experience**: The app is designed with simplicity in mind, allowing you to effortlessly input essential information and obtain price estimates instantly, making your decision-making process more efficient.")
        st.markdown("- **Customizable Parameters**: Personalize your estimates by adjusting factors such as floor space, lease year, neighborhood, and flat type to receive customized results")
        st.markdown("- **Insightful Data Visuals**: Visualize market trends and price fluctuations through dynamic charts and graphs, offering a clearer understanding of pricing trends.")
        
        st.subheader("How It Runs")
        st.markdown("1. **Provide Your Details**: Enter relevant information like floor space, lease year, location, and type of flat.")
        st.markdown("2. **Receive Estimates**: Our machine learning models process your data and deliver accurate resale price estimates.")
        st.markdown("3. **Gain Insights**:  Understand the key factors that impact resale prices, enabling you to make data-driven decisions.")
        
        st.subheader("Start Exploring")
        st.write("Interested in forecasting resale flat prices in Singapore? Click the sidebar to begin or learn how our app can benefit you.")
        
        st.subheader("Our Unique Advantage")
        st.markdown("- **Specialized Expertise**: Created by experienced data analysts with a focus on real estate trends to ensure dependable forecasts.")
        st.markdown("- **Precision & Accuracy**:  Our predictions are grounded in thoroughly validated models, ensuring highly accurate price estimates.")
        
        st.subheader("Testimonials")
        st.markdown('"Using the app helped me understand the market better and make smarter investment decisions." - Real Estate Investor')
        st.markdown('"Highly recommend for anyone looking to buy or sell property in Singapore!" - Homeowner')
        
    elif choice == "Price Prediction":
        st.title(":violet[Singapore Resale Flat Price Prediction]")
        user_input_data = get_user_input()

        if st.button("Predict"):
            df = data_transformation_for_the_model(user_input_data)
            model = load_model()
            if model is not None:
                predicted_price = model.predict(df)
                st.success(f'Predicted price :green[$] :green {predicted_price[0]:.2f}')

    elif choice == "Multiple Predictions":
        st.title(":violet[Multiple Resale Flat Price Prediction]")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

            st.write("Uploaded Data")
            st.write(data)

            data_transformed = data_transformation_for_the_model(data)
            model = load_model()
            if model is not None:
                predictions = model.predict(data_transformed)
                data['Predicted_Price_Range'] = predictions

                st.write("Data with Predictions")
                st.write(data)
                
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download data with predictions as CSV",
                    data=csv,
                    file_name='predicted_resale_prices.csv',
                    mime='text/csv',
                )

if __name__ == "__main__":
    main()
