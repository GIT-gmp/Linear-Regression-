# Linear-Regression-
Implement and understand simple &amp; multiple linear regression.

The provided Python code performs a comprehensive linear regression analysis on the Housing.csv dataset. It follows a standard machine learning workflow, encompassing data loading, preprocessing, model training, evaluation, and interpretation.

1. Data Loading and Initial Inspection
Purpose: The first step involves loading the Housing.csv file into a pandas DataFrame. This is the foundation for all subsequent operations.
Error Handling: A try-except block is implemented to gracefully handle the scenario where the Housing.csv file might not be found, preventing the program from crashing and providing a user-friendly error message.
Initial Data Overview:
df.head(): Displays the initial rows of the dataset, offering a quick glimpse into its structure and content.
df.info(): Provides a summary of the DataFrame, including the number of entries, column names, non-null counts, and data types. This is crucial for identifying missing values and understanding the nature of each feature.
df.describe(): Generates descriptive statistics (e.g., mean, standard deviation, min, max, quartiles) for numerical columns, giving insights into the distribution and range of the data.
2. Data Preprocessing
Purpose: Raw datasets often contain non-numerical data or require transformation before they can be used effectively by machine learning algorithms like Linear Regression. This section prepares the data for modeling.
Handling Categorical Features:
Binary Categorical Variables: Columns such as mainroad, guestroom, basement, hotwaterheating, airconditioning, and prefarea contain binary categorical values ('yes'/'no'). These are converted into numerical representations (1 for 'yes', 0 for 'no') using a map function. This is a common practice as linear regression models require numerical input.
Multi-Class Categorical Variables: The furnishingstatus column has more than two categories ('furnished', 'semi-furnished', 'unfurnished'). To handle this, one-hot encoding is applied using pd.get_dummies(). This process creates new binary columns for each category. The drop_first=True argument is used to prevent multicollinearity (the "dummy variable trap"), where one of the new dummy columns is dropped, as its information can be inferred from the others.
Feature and Target Separation: The dataset is explicitly divided into:
Features (X): All columns except 'price' are designated as independent variables, which will be used to predict the target.
Target (y): The 'price' column is identified as the dependent variable, which the model aims to predict.
3. Data Splitting
Purpose: To evaluate the model's performance on unseen data, the dataset is split into training and testing sets. This simulates how the model would perform in a real-world scenario.
train_test_split: The sklearn.model_selection.train_test_split function is used for this purpose:
test_size=0.2: Specifies that 20% of the data will be allocated to the testing set, and the remaining 80% will be used for training.
random_state=42: Ensures that the data split is consistent and reproducible across different runs. This means that if you run the code multiple times, you will always get the same training and testing subsets.
4. Linear Regression Model Training
Model Instantiation: An object of the LinearRegression class from sklearn.linear_model is created. This object represents the linear regression model.
Model Fitting: The model.fit(X_train, y_train) method is called to train the model. During this process, the algorithm learns the optimal coefficients (weights) for each feature that minimize the sum of squared differences between the predicted and actual prices in the training data.
5. Model Evaluation
Purpose: After training, it's crucial to assess how well the model performs. This is done by comparing its predictions on the test set against the actual values.
Predictions: The trained model generates predictions (y_pred) for the X_test (unseen) data.
Performance Metrics: Several standard regression evaluation metrics are calculated:
Mean Absolute Error (MAE): Measures the average magnitude of the errors, without considering their direction. It's robust to outliers. The formula is:
 
 
where 
 is the actual value, 
 is the predicted value, and 
 is the number of samples.
Mean Squared Error (MSE): Calculates the average of the squared differences between predicted and actual values. It penalizes larger errors more heavily than MAE. The formula is:
 
 
Root Mean Squared Error (RMSE): The square root of MSE. It is often preferred over MSE because it is in the same units as the target variable, making it more interpretable. The formula is:
R-squared (
): Also known as the coefficient of determination, it indicates the proportion of the variance in the dependent variable that can be predicted from the independent variables. A value of 1 indicates a perfect fit, while 0 indicates that the model explains none of the variance. The formula is: $$ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}i)^2}{\sum{i=1}^{n} (y_i - \bar{y})^2} $$ where 
 is the mean of the actual values.
6. Model Interpretation and Visualization
Coefficient Analysis:
The model.coef_ attribute provides the learned coefficients for each feature. These coefficients represent the change in the predicted price for a one-unit increase in the corresponding feature, assuming all other features are held constant. This is a key aspect of interpreting linear regression models.
The coefficients are presented in a pandas DataFrame for clear readability.
Regression Line Plot (Actual vs. Predicted):
A scatter plot is generated to visually compare the actual prices (y_test) against the prices predicted by the model (y_pred).
A red dashed line representing a "perfect fit" (where actual values equal predicted values) is added to the plot. Points that lie close to this line indicate accurate predictions. This plot helps to quickly gauge the overall fit of the model.
Residual Plot:
This plot displays the predicted values on the x-axis and the residuals (the difference between actual and predicted values: 
) on the y-axis.
A horizontal red dashed line at 
 is included.
Interpretation: For a good linear regression model, the residuals should be randomly scattered around the zero line, with no discernible patterns (e.g., no U-shape, no fanning out). Patterns in the residuals can indicate issues such as non-linearity in the data, heteroscedasticity (non-constant variance of errors), or missing important variables.
In summary, this code provides a structured and well-commented approach to building, evaluating, and understanding a linear regression model, making it suitable for educational purposes and initial data analysis.
