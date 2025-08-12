# InnovateMart Sales Forecasting Project

## 🎯 Project Overview

This project implements an end-to-end time series forecasting solution for InnovateMart, a fictional retail chain. The system predicts daily sales for multiple stores using a Temporal Fusion Transformer (TFT) model and presents results through an interactive Streamlit dashboard.

## 🏗️ Architecture

### Data Simulation Approach
The data simulation creates a realistic retail environment with the following characteristics:

- **Multi-store setup**: 4 stores with different characteristics (small, medium, large)
- **Temporal patterns**: 
  - Weekly seasonality (higher weekend sales)
  - Annual seasonality (holiday boosts, seasonal variations)
  - Long-term growth trends (4-7% annually per store)
- **Business events**:
  - Pre-planned promotional campaigns (Black Friday, holidays, summer sales)
  - External shocks (competitor impact on store_002 starting March 1, 2022)
- **Store characteristics**: Size categories and city population affecting baseline sales
- **Realistic noise**: Random variations to simulate real-world uncertainty

### Model Architecture
- **Model**: Temporal Fusion Transformer (TFT)
- **Framework**: PyTorch Forecasting
- **Features**:
  - Static categorical: store_id, store_size
  - Static real: city_population
  - Time-varying known: promotion_active, is_weekend, day_of_week, month, quarter
  - Time-varying unknown: daily_sales (target)
- **Prediction horizon**: 30 days
- **Encoder length**: 60 days

## 📁 Project Structure

```
├── data_simulation.py          # Data generation script
├── model_training.py           # Model training pipeline
├── streamlit_app.py           # Interactive dashboard
├── run_pipeline.py            # Run complete pipeline 
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── innovatemart_sales_data.csv    # Generated dataset
├── model_predictions.csv         # Model predictions
├── training_dataset.pkl          # Serialized training dataset
└── tft_model_state.pth          # Trained model weights
```

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/asma2020/InnovateMart_Sales_Forecasting_Project_v1.0.0.git
cd InnovateMart_Sales_Forecasting_Project_v1.0.0
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate Data and Train Model
```bash
# Generate synthetic data
python data_simulation.py

# Train the forecasting model
python model_training.py
```

### Step 4: Launch the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

The dashboard will be available at `http://localhost:8501`

## 🎮 Using the Dashboard

### Features Available:
1. **Store Selection**: Choose from 4 different stores via dropdown menu
2. **Historical Sales Visualization**: Interactive time series plot showing:
   - Daily sales trends
   - Promotion periods highlighted in red
   - Seasonal patterns and growth trends

3. **Model Predictions**: Comparison charts showing:
   - Actual vs predicted sales for validation period
   - Model performance metrics (MAE, MAPE, RMSE)

4. **Advanced Analytics**:
   - Feature importance analysis
   - Promotion effectiveness analysis  
   - Store performance comparison

### Navigation:
- Use the sidebar to select different stores
- Explore three analysis tabs: Feature Importance, Promotion Analysis, and Overall Performance
- Hover over charts for detailed data points

## 📊 Key Business Insights

### Promotional Impact
- Promotions provide approximately **80% lift** in daily sales
- Black Friday and holiday periods show highest effectiveness
- Weekend promotions generally outperform weekday promotions

### Seasonality Patterns
- **Weekly**: 20-50% higher sales on weekends depending on store size
- **Annual**: December shows peak sales, January-February are lowest
- **Growth**: Consistent 4-7% annual growth across all stores

### Store Performance
- Large stores in high-population cities show highest absolute sales
- Store_002 experienced permanent 25% sales reduction after competitor entry
- Medium stores show highest growth rates (7% annually)

## 🔧 Technical Implementation Details

### Data Generation Logic
The simulation implements realistic business scenarios:
- Base sales levels determined by store size and city population
- Multiplicative seasonal effects for weekly and annual patterns
- Promotional boosts applied during planned campaign periods
- External shock modeling for competitive impacts
- Gaussian noise for realistic variance

### Model Configuration
- **Architecture**: Temporal Fusion Transformer
- **Optimization**: AdamW optimizer with learning rate 0.03
- **Loss Function**: Quantile Loss (7 quantiles)
- **Regularization**: Dropout (0.1), gradient clipping
- **Training**: Early stopping with patience=10 epochs

### Performance Metrics
The model achieves the following performance on validation data:
- **SMAPE**: ~8-12% across different stores
- **MAPE**: ~10-15% average percentage error
- **MAE**: $500-1500 depending on store sales volume

## 🛠️ Customization Options

### Extending the Dataset
To add more complexity to the simulation:
- Modify store characteristics in `simulate_innovatemart_data()`
- Add new promotional events or external shocks
- Include additional features like weather, economic indicators
- Extend time range or add more stores

### Model Tuning
Key hyperparameters in `model_training.py`:
- `hidden_size`: Increase for more model capacity
- `attention_head_size`: Adjust attention mechanism complexity
- `max_encoder_length`: Change historical window size
- `max_prediction_length`: Modify forecast horizon

### Dashboard Enhancement
The Streamlit app can be extended with:
- Real-time prediction updates
- Interactive parameter tuning
- Export functionality for predictions
- Integration with external data sources

## 🎓 Learning Outcomes

This project demonstrates:
- **End-to-end ML pipeline**: From data generation to deployment
- **Time series modeling**: Handling complex temporal patterns
- **Business context integration**: Realistic retail scenarios
- **Interactive visualization**: User-friendly dashboard creation
- **Model interpretation**: Understanding feature importance and business impact

## 📋 Future Enhancements

Potential improvements include:
- **Multi-step ahead forecasting**: Extend prediction horizon
- **Uncertainty quantification**: Add prediction intervals
- **Online learning**: Update model with new data
- **A/B testing framework**: Compare different promotional strategies
- **Integration APIs**: Connect with real retail systems

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is created for educational and evaluation purposes. The synthetic data and scenarios are fictional and designed to demonstrate technical capabilities in time series forecasting.
