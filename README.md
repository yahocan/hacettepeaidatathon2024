# Customer Churn Prediction Datathon

## Project Overview

Customer churn prediction model developed for a datathon competition. Analyzes multiple customer data sources to predict churn probability and identify key retention factors.

## 📊 Data Sources

### Customer Data (`archive/` directory):

- **`CSAT_Survey_Data.csv`** - Customer satisfaction surveys, NPS ratings, usage frequency
- **`Customer_Age_Data.csv`** - Customer demographics
- **`Customer_MRR_Data.csv`** - Monthly recurring revenue
- **`Customer_Revenue_Data.csv`** - Revenue metrics
- **`Help_Ticket_Data.csv`** - Support interactions
- **`Newsletter_Interaction_Data.csv`** - Email engagement
- **`Product_Bug_Task_Data.csv`** - Product issues
- **`RegionAndVertical_Data.csv`** - Geographic and industry data
- **`StatusAndLevel_Data.csv`** - Customer status levels

## 🔧 Setup

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook datathon.ipynb
```

## 📈 Methodology

1. **Data Integration** - Multi-source data cleaning and feature engineering
2. **Model Development** - Machine learning algorithms comparison and optimization
3. **Prediction** - Binary churn classification (0: No Churn, 1: Churn)

## 📋 Files

```
├── archive/                    # Raw data files
├── datathon.ipynb             # Main analysis notebook
├── final_predictions.csv      # Model predictions
└── README.md                  # Documentation
```

## 🎯 Results

Binary churn predictions where:

- **0**: Customer retention (No Churn)
- **1**: Customer churn risk (High Risk)

Key predictors: Customer satisfaction, usage patterns, support interactions, revenue trends.
