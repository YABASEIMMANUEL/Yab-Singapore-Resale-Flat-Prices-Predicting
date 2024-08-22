# Singapore Resale Flat Price Prediction

## Project Overview
This project aims to develop a machine learning model that predicts the resale prices of flats in Singapore. The model is embedded in a user-friendly online application, which offers accurate price predictions based on historical transaction data. By analyzing factors such as location, flat type, floor area, and lease duration, this tool assists buyers and sellers in making informed real estate decisions.

## Domain
Real Estate

## Project Links


## Prerequisites
To run this project, ensure you have the following installed:
- **Python**: The main programming language.
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computing.
- **streamlit**: For building and deploying the web application.
- **scikit-learn**: For developing the machine learning model.

## Data Source
The dataset used is publicly available and can be accessed [here](https://beta.data.gov.sg/collections/189/view).

## Project Workflow
1. **Data Integration**:
   - The dataset consists of five CSV files covering different periods: 1990-1999, 2000-2012, 2012-2014, 2015-2016, and 2017 onwards.
   - These files are combined into a single dataset for analysis.

2. **Data Preprocessing**:
   - The data is cleaned and formatted for analysis.
   - Key features such as town, flat type, storey range, floor area, flat model, and lease commence date are extracted.
   - Additional features are created to enhance prediction accuracy.

3. **Model Building**:
   - A regression model using a decision tree regressor is developed to predict resale prices.

4. **Web Application Development**:
   - The model is integrated into a Streamlit web application, enabling users to input data and receive resale price predictions.

## How to Use the Application
1. **Access the Predictions Page**:
   - Go to the "Predictions" section of the application.

2. **Enter the Required Details**:
   - Provide information such as Street Name, Block Number, Floor Area (in square meters), Lease Commence Date, and Storey Range.

3. **Get the Prediction**:
   - Click the "PREDICT RESALE PRICE" button to see the estimated resale price based on your inputs.

## Conclusion
This project offers a valuable tool for predicting the resale value of Singapore flats by combining historical data with machine learning techniques. The Streamlit application provides an intuitive interface, making it simple for users to obtain price predictions and navigate the real estate market confidently.
