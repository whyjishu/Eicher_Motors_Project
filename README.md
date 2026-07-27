Eicher Motors — Financial Performance Analysis & Dashboard

A comprehensive data analysis and machine learning project tracking Eicher Motors' quarterly financial performance from March 2023 to December 2025. This repository includes a complete workflow, from data cleaning and statistical exploration to predictive modeling and an interactive web dashboard.

📋 Project Overview
This project analyzes key financial metrics for Eicher Motors Ltd (NSE: EICHERMOT), including Sales, Operating Profit, OPM %, Net Profit, and EPS. By examining twelve quarters of publicly reported figures, the analysis highlights the company's growth trajectory, powered largely by strong Royal Enfield motorcycle volumes and the VE Commercial Vehicles joint venture.
The project features a comparison of two machine learning models (Logistic Regression and a Decision Tree) to predict whether quarterly profit will grow or decline based on underlying financial drivers.

✨ Key Features
Data Processing: Automated handling of raw financial data, including gap interpolation and chronological sorting.
Exploratory Data Analysis: Calculation of summary statistics (mean, median, standard deviation) and quarter-on-quarter growth rates.
Interactive Visualizations: Dynamic charts built with Plotly, Matplotlib, and Seaborn to visualize Sales vs. Net Profit trends, Operating Margins, and profit distributions.
Predictive Modeling: A live testing environment comparing Logistic Regression and Decision Tree classifiers for trend prediction, including feature importance evaluation.
Streamlit Dashboard: A multi-page, publicly accessible web application styled with Eicher's brand colors, featuring KPI metrics, side-by-side quarter comparisons, and downloadable datasets.

📂 Repository Structure
eicher_financials.csv — The raw, uncleaned financial data exported from Screener.in.
eicher_financials_clean.csv — The processed dataset used to power the dashboard.
Eicher_Motors_Analysis.ipynb — The core Jupyter Notebook containing the data cleaning steps, statistical analysis, interactive charts, and model training workflow.

app.py — The Streamlit application script containing the dashboard's UI and logic.

requirements.txt — The list of required Python dependencies.

Eicher_Motors_Project_Report.docx — The final CA2 project report detailing the findings and methodology.
