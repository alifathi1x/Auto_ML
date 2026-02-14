# Auto_ML


Here is the revised GitHub description with the part about displaying all metrics at the end removed:



Auto Model Selector Dashboard (AutoML System)

Overview

This project is an automated machine learning (AutoML) dashboard built with Python that allows users to upload their own dataset and automatically select the best-performing model based on real evaluation metrics.

Instead of manually testing multiple algorithms, the system trains several predefined models, evaluates their performance, compares them, and automatically selects the best one.

The goal of this project is to simplify the model selection process and combine machine learning with a clean, user-friendly analytical dashboard.



How It Works
	1.	The user uploads a CSV dataset
	2.	The user selects the target column
	3.	The user chooses the problem type (Classification or Regression)
	4.	The system:
	•	Splits the dataset into training and testing sets
	•	Trains multiple algorithms
	•	Applies GridSearch with Cross Validation for hyperparameter tuning
	•	Evaluates each model
	•	Compares results
	•	Automatically selects the best-performing model

All models are trained first, then the best model is selected based on actual performance — not assumptions.



Implemented Algorithms

Classification
	•	Logistic Regression
	•	Random Forest Classifier
	•	Support Vector Machine (SVM)

Regression
	•	Linear Regression
	•	Random Forest Regressor
	•	Support Vector Regressor (SVR)



Evaluation Strategy

For Classification:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1 Score

For Regression:
	•	R² Score



Features
	•	Automatic model comparison
	•	Hyperparameter tuning using GridSearchCV
	•	Cross-validation support
	•	Interactive and modern UI built with Streamlit
	•	Clean and structured visualization using Matplotlib
	•	Model performance comparison charts
	•	Learning Curve visualization
	•	Training time comparison
	•	Confusion Matrix (for classification problems)



Technologies Used
	•	Python
	•	Scikit-learn
	•	Streamlit
	•	Matplotlib
	•	Pandas



Project Structure
	•	app.py – Main Streamlit application
	•	Model selection and training logic implemented inside a reusable class
	•	Visualization components integrated within the dashboard



Use Cases
	•	Educational purposes
	•	Machine learning experimentation
	•	Rapid model benchmarking
	•	Demonstrating automated model selection concepts
	•	Portfolio project for showcasing ML + product mindset



Future Improvements
	•	Add more advanced algorithms (e.g., Gradient Boosting)
	•	Add feature importance visualization
	•	Add model export functionality
	•	Add automatic preprocessing for categorical features
	•	Deploy as a cloud web application



Goal

The main objective of this project is to demonstrate how automated model selection can reduce development time while maintaining transparency through structured evaluation and visual analytics.

This project combines machine learning engineering, model evaluation, and product-oriented dashboard design in a single system.
