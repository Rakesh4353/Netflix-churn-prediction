# Netflix-churn-prediction
# Netflix Churn Analysis

A Streamlit-based application for analyzing Netflix user churn, predicting churn probability, and visualizing key insights.

## Features
- **Interactive Dashboard**: Visualizes user demographics, revenue trends, and churn analysis.
- **Churn Prediction Tool**: Predicts churn probability based on user input.
- **Machine Learning Model**: Uses a Random Forest classifier to identify churn patterns.
- **Performance Metrics**: Displays confusion matrix, classification report, and feature importance.

## Installation

### Prerequisites
Ensure you have Python installed and install the required libraries.

### Steps:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/netflix-churn-analysis.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd netflix-churn-analysis
   ```
3. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv env
   source env/bin/activate  # On macOS/Linux
   env\Scripts\activate  # On Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit app in your browser after running `streamlit run app.py`.
2. Navigate between **Dashboard**, **Churn Prediction**, and **Model Performance**.
3. Enter user details in the **Churn Prediction** section to estimate churn risk.
4. Explore visualizations for insights on user behavior.

## Technologies Used
- **Python**
- **Streamlit**
- **Scikit-learn** (Machine Learning)
- **Plotly & Seaborn** (Data Visualization)
- **Pandas** (Data Processing)
- **Joblib** (Model Persistence)

## Contributing
Pull requests are welcome. Follow these steps to contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Added feature X"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request.

## License
This project is licensed
