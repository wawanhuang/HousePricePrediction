# **Housing Price Prediction Model**

## **Project Overview**

This project aims to develop a predictive model for estimating housing prices. The model leverages multiple regression techniques to predict house prices based on various features such as square footage, the number of bedrooms, the number of bathrooms, the neighborhood, and the year the house was built. The model's effectiveness is evaluated using the **Root Mean Squared Error (RMSE)** metric, a standard for measuring the differences between predicted and actual values in regression models.

## **Project Background**

Accurate estimation of property prices is crucial for homeowners, real estate developers, and investors. By predicting house prices based on historical data and various features of a property, stakeholders can make informed decisions regarding property investments, sales, and purchases. This project focuses on using regression models to predict house prices, providing a valuable tool for the real estate industry.

## **Dataset**

### **Source**

The dataset used for this project is sourced from Kaggle, available at the following link: [Housing Price Prediction Data](https://www.kaggle.com/datasets/muhammadbinimran/housing-price-prediction-data/data).

### **Features**

The dataset includes the following features:

- **SquareFeet**: The total area of the house in square feet.
- **Bedrooms**: The number of bedrooms in the house.
- **Bathrooms**: The number of bathrooms/restrooms/washrooms in the house.
- **Neighborhood**: The area or neighborhood where the house is located.
- **YearBuilt**: The year in which the house was built.
- **Price**: The selling price of the house (target variable).

## **Model Development**

### **Libraries Used**

The following Python libraries are utilized in the project:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **plotly.express**: For data visualization.
- **xgboost**: For implementing the XGBoost regression model.
- **scikit-learn**: For machine learning models, preprocessing, and evaluation metrics.
- **feature_engine**: For handling outliers and feature engineering.

### **Data Preprocessing**

Before training the models, the dataset undergoes various preprocessing steps:

- **Outlier Treatment**: Winsorization is used to limit extreme values to reduce the effect of possible outliers.
- **Scaling**: MinMaxScaler is applied to normalize the feature values.
- **One-Hot Encoding**: Categorical variables, such as `Neighborhood`, are encoded using one-hot encoding.

### **Models Used**

Several regression models are employed to predict house prices:

1. **Support Vector Regressor (SVR)**
2. **K-Neighbors Regressor**
3. **Decision Tree Regressor**
4. **Random Forest Regressor**
5. **Gradient Boosting Regressor**
6. **XGBoost Regressor**

### **Model Evaluation**

The models are evaluated using the following metrics:

- **RMSE (Root Mean Squared Error)**: Measures the average magnitude of the error between predicted and actual values.
- **R² Score**: Indicates the proportion of variance in the dependent variable that is predictable from the independent variables.

## **Results and Conclusion**

The model that performs the best in predicting house prices is identified based on the RMSE and R² scores. The results indicate that the **XGBoost Regressor** provides the highest accuracy with an optimal balance between bias and variance.

### **Key Insights**

- **Square Footage**: Larger homes generally have higher prices.
- **Bedrooms and Bathrooms**: These features contribute significantly to the house price, but their impact varies based on the neighborhood and year built.
- **Neighborhood**: The location plays a crucial role in determining the house price, with some neighborhoods being more desirable and thus more expensive.

## **Recommendations**

- **Model Deployment**: The best-performing model can be deployed as a web service or integrated into a real estate application to provide real-time price estimates.
- **Feature Engineering**: Further feature engineering, such as the inclusion of additional variables (e.g., proximity to schools, parks, or commercial areas), could improve the model's accuracy.
- **Regular Updates**: The model should be regularly updated with new data to maintain its predictive accuracy, especially in rapidly changing real estate markets.

## **Try It Out**

You can try the Housing Price Prediction model live [here](https://huggingface.co/spaces/kurniawanew/HousePricePrediction) on Hugging Face Spaces.

## **How to Run the Project**

1. **Clone the Repository**: Download the notebook and associated files.
2. **Install Dependencies**: Install the required Python libraries using `pip install -r requirements.txt`.
3. **Run the Notebook**: Open the Jupyter notebook and execute the cells to preprocess the data, train the models, and evaluate the results.

This project was developed as part of the Data Science program at Hacktiv8. Special thanks to mentors and colleagues for their guidance and support.
