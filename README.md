# 🚗 AI-Powered Car Price Predictor

<div align="center">

![Car Price Predictor](https://img.shields.io/badge/ML-Car%20Price%20Prediction-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**🌐 [Live Demo](https://carpricepredictor-5dagmnysz9gnffctzxkncd.streamlit.app/) | 📊 Data Science Project**

*Predict used car prices with 95%+ accuracy using advanced machine learning algorithms*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The **AI-Powered Car Price Predictor** is a comprehensive machine learning web application that accurately predicts the resale value of used cars. Built with Streamlit and powered by multiple ML algorithms, this tool helps buyers and sellers make informed decisions about car pricing.

### 🎨 Key Highlights

- ⚡ **Real-time Predictions** - Get instant price estimates in under 1 second
- 📊 **Interactive Dashboard** - Explore data with beautiful visualizations
- 📦 **Bulk Processing** - Predict prices for multiple cars at once
- 🎯 **High Accuracy** - Achieved 95%+ R² score on test data
- 🌐 **Web-Based** - No installation required, access from anywhere

---

## ✨ Features

### 🎯 Single Car Prediction
- User-friendly form interface
- Input car specifications (Year, Price, Mileage, Fuel Type, etc.)
- Instant price prediction with depreciation analysis
- Price range estimation

### 📦 Bulk Prediction
- CSV file upload for batch processing
- Handles multiple cars simultaneously
- Downloadable results with predictions
- Visual insights and statistics

### 📊 EDA Dashboard
- **Overview Tab**: Dataset statistics and summary
- **Distributions Tab**: Feature distributions and category breakdowns
- **Correlations Tab**: Feature relationships and heatmaps
- **Advanced Tab**: 3D visualizations and key insights

### 🎨 Modern UI/UX
- Responsive design with gradient themes
- Interactive Plotly charts
- Smooth animations and transitions
- Mobile-friendly interface

---

## 🎥 Demo

### 🌐 Live Application
**👉 [Try it now](https://carpricepredictor-5dagmnysz9gnffctzxkncd.streamlit.app/)**

### 🖼️ Quick Preview
```
🏠 Home → Beautiful landing page with metrics
🎯 Single Prediction → Predict individual car prices
📦 Bulk Prediction → Upload CSV for batch processing
📊 EDA Dashboard → Explore data insights
ℹ️ About → Project documentation
```

---

## 🛠️ Tech Stack

### Machine Learning
- **Scikit-learn** - ML algorithms (Linear Regression, Lasso, Random Forest)
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Visualization
- **Plotly** - Interactive charts and graphs
- **Seaborn** - Statistical visualizations
- **Matplotlib** - Static plots

### Web Framework
- **Streamlit** - Web application framework
- **Custom CSS** - Enhanced UI/UX styling

### Model Persistence
- **Joblib** - Model serialization and loading

---

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/car-price-predictor.git
cd car-price-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure model file exists**
```
Make sure car_price_model_final.pkl is in the project directory
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**
```
The app will automatically open at http://localhost:8501
```

---

## 🚀 Usage

### Single Prediction

1. Navigate to **🎯 Single Prediction** from the sidebar
2. Fill in the car details:
   - Year of Manufacture
   - Current Market Price (Lakhs)
   - Kilometers Driven
   - Number of Previous Owners
   - Fuel Type (Petrol/Diesel/CNG)
   - Seller Type (Individual/Dealer)
   - Transmission (Manual/Automatic)
3. Click **Predict Price**
4. View predicted selling price with insights

### Bulk Prediction

1. Navigate to **📦 Bulk Prediction**
2. Download the sample CSV template (optional)
3. Prepare your CSV file with columns:
   ```
   Present_Price, Kms_Driven, Owner, Fuel_Type, Seller_Type, Transmission, Year
   ```
4. Upload your CSV file
5. Click **Generate Predictions**
6. Download results with predicted prices

### CSV Format Example
```csv
Present_Price,Kms_Driven,Owner,Fuel_Type,Seller_Type,Transmission,Year
5.59,27000,0,Petrol,Dealer,Manual,2014
9.54,43000,0,Diesel,Dealer,Manual,2013
```

---

## 📊 Model Performance

### Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | 0.95+ | High accuracy on test data |
| **MAE** | <1.5 Lakhs | Low mean absolute error |
| **Training Data** | 300+ cars | Diverse dataset |
| **Features** | 7 | Input variables |

### Models Compared

1. **Linear Regression** - Fast, interpretable baseline
2. **Lasso Regression** - Regularized model with feature selection
3. **Random Forest** - Best performing ensemble model

The final deployed model is automatically selected based on the highest R² score during training.

---

## 📁 Project Structure

```
car-price-predictor/
├── app.py                          # Main Streamlit application
├── car_price_model_final.pkl      # Trained ML model
├── requirements.txt                # Python dependencies
├── car data.csv                    # Training dataset
├── model_training.ipynb            # Jupyter notebook for training
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

---



## 🔬 Model Training Details

### Data Preprocessing
- Encoded categorical variables (Fuel Type, Seller Type, Transmission)
- Created Car_Age feature from Year
- Handled missing values
- Feature scaling and normalization

### Feature Engineering
- **Present_Price**: Current market price in lakhs
- **Kms_Driven**: Total kilometers driven
- **Fuel_Type**: Petrol (0), Diesel (1), CNG (2)
- **Seller_Type**: Individual (0), Dealer (1)
- **Transmission**: Manual (0), Automatic (1)
- **Owner**: Number of previous owners
- **Car_Age**: 2025 - Year of manufacture

### Training Process
1. Data collection and cleaning
2. Exploratory data analysis
3. Feature encoding and engineering
4. Train-test split (90-10)
5. Model training and comparison
6. Hyperparameter tuning
7. Model evaluation and selection
8. Model serialization

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Arnav Thapliyal**

- GitHub: [@arnavt866](https://github.com/arnavt866)
- LinkedIn: [arnav-thapliyal-756b53316](https://www.linkedin.com/in/arnav-thapliyal-756b53316/)
- Email: arnav.t0806@gmail.com

---

## 🙏 Acknowledgments

- Dataset sourced from Kaggle
- Streamlit for the amazing framework
- Scikit-learn for ML algorithms
- Plotly for beautiful visualizations

