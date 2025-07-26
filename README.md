# Adult_income-Logistic-Regression
Adult Income Prediction Project
Project Overview
This project leverages the Adult Census Income Dataset to predict whether an individual's annual income exceeds $50,000 based on demographic and personal attributes. The goal is to build a robust machine learning model to classify income levels using features such as age, education, occupation, and marital status, while applying data preprocessing and exploratory data analysis (EDA) techniques.
Key Features

Dataset: Adult Census Income dataset with 48,842 records and 15 features, including numerical (e.g., age, hours-per-week) and categorical (e.g., workclass, education) variables.
Objective: Binary classification to predict income as either <=50K or >50K.
Tools & Libraries: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly.

Methodology

Data Preprocessing:
Removed 52 duplicate records and handled missing values represented as "?" in workclass, occupation, and native-country.
Dropped the fnlwgt column due to its irrelevance to the prediction task.
Categorized features into numerical and categorical for tailored processing.


Exploratory Data Analysis (EDA):
Analyzed the distribution of categorical variables using value_counts.
Visualized the distribution of true and predicted income labels using interactive pie charts with Plotly.


Machine Learning:
Applied preprocessing techniques such as encoding categorical variables and scaling numerical features.
Implemented a Logistic Regression model for income classification (additional models can be explored).
Evaluated model performance using metrics like accuracy and confusion matrix.



Visualizations

Created interactive pie charts to visualize the distribution of true and predicted income labels, providing insights into model performance and data balance.

Key Skills Demonstrated

Data cleaning and preprocessing
Exploratory data analysis
Data visualization with Plotly
Machine learning model development with Scikit-learn
Handling categorical and numerical data

Repository Contents

Jupyter Notebook (Adult_income.ipynb) containing the full code for data loading, preprocessing, EDA, visualization, and model training.
Documentation of the dataset's features and project workflow.

Future Improvements

Experiment with advanced models like Random Forest or XGBoost to improve prediction accuracy.
Incorporate feature engineering to enhance model performance.
Add cross-validation to ensure robust model evaluation.

How to Run

Clone the repository.
Install required libraries: pip install pandas numpy scikit-learn matplotlib seaborn plotly.
Run the Jupyter Notebook to explore the analysis and model.

Feel free to explore the code and provide feedback! Connect with me on LinkedIn for discussions or collaboration.
