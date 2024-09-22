# Singapore-Flat-Resale-Price-Prediction
### **Introduction**
  The main objective of this project is to predict the resale selling price using regression models. Streamlit is used to enhance accessibility and usability, allowing the users to obtain predictions for resale selling price. 
#### Technologies Used:
* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Pickle
#### Installation
* pip install pandas
* pip install numpy
* pip install scikit-learn
* pip install matplotlib
* pip install seaborn
* pip install streamlit
* pip install pickle
#### Data Understanding
1. Identify Variable Types: Continuous or Discrete
2. Handle Invalid Values
#### Data Preprocessing
1. Handle missing values using Mean, Median, Mode.
2. Detect outliers using IQR or Isolation Forest.
3. Determine skewnes using Log, Square Root or Box-Cox transaformations to treat skewness.
4. Encode categorical variables with One-Hot Encoding, Label Encoding, or Ordinal Encoding.
#### Exploratory Data Analysis
1. Visualize outliers and Skewness using Boxplot, Distplot or Violinplot. 
2. Analyze and treat Skewness
#### Feature Engineering
1. Create new features through aggregation or transformation.
2. Drop highly correlated features using a heatmap.
#### Model Building and Evaluation
1. Split the Data
2. Train and evaluate regression models using metrics like Mean Squared Error, Mean Absolute Error, and R² Score.
4. Optimize with hyperparameter tuning using Cross-Validation and Grid Search.
#### Model GUI
1. Develop a Streamlit App for interactive predictions.
2. Allow the users to input feature values and display predictions.
#### Usage
Steps to be followed for effectively using the application:
1. Access the Streamlit App: Open the application in your browser.
2. Input Data: Enter the values for the given field.
3. Perform Prediction: By clicking the button will able to get results based on the input data.
4. Results: Prediction output will be displayed on the page.
#### Features
- **Data Preprocessing:** Handles missing values,outliers, and skewness.
- **Model Training:** Trains Machine Learning Models and hyperparameter optimization.
- **Interactive GUI:** Provides a user-friendly web interface for viewing predictions based on user input.
- **Interactive Visualizations:** Utilizes EDA techniques to understand data distributions and model performance.
- **Pickle Integration:** Saves and loads models and transformers for seamless use.
