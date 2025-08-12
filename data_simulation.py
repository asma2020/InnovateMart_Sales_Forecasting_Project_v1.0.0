import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def simulate_innovatemart_data():
    """
    Simulate realistic daily sales data for InnovateMart stores
    """
    
    # Define store characteristics
    stores = {
        'store_001': {
            'store_size': 'large',
            'city_population': 500000,
            'base_sales': 15000,
            'growth_rate': 0.05,  # 5% annual growth
            'weekend_boost': 1.4,
            'seasonal_amplitude': 0.3
        },
        'store_002': {
            'store_size': 'medium',
            'city_population': 200000,
            'base_sales': 8000,
            'growth_rate': 0.07,  # 7% annual growth
            'weekend_boost': 1.3,
            'seasonal_amplitude': 0.25
        },
        'store_003': {
            'store_size': 'small',
            'city_population': 80000,
            'base_sales': 4000,
            'growth_rate': 0.04,  # 4% annual growth
            'weekend_boost': 1.2,
            'seasonal_amplitude': 0.2
        },
        'store_004': {
            'store_size': 'large',
            'city_population': 750000,
            'base_sales': 18000,
            'growth_rate': 0.06,  # 6% annual growth
            'weekend_boost': 1.5,
            'seasonal_amplitude': 0.35
        }
    }
    
    # Time range: 2.5 years
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 6, 30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Pre-planned promotional events (known in advance)
    promotion_dates = [
        # Black Friday weeks
        ('2021-11-22', '2021-11-28'),
        ('2022-11-21', '2022-11-27'),
        # Holiday seasons
        ('2021-12-15', '2021-12-31'),
        ('2022-12-15', '2022-12-31'),
        # Summer sales
        ('2021-07-01', '2021-07-15'),
        ('2022-07-01', '2022-07-15'),
        ('2023-07-01', '2023-07-15'),
        # Back to school
        ('2021-08-15', '2021-08-31'),
        ('2022-08-15', '2022-08-31'),
        # Spring promotions
        ('2021-03-15', '2021-03-31'),
        ('2022-03-15', '2022-03-31'),
        ('2023-03-15', '2023-03-31'),
    ]
    
    # Convert promotion dates to date objects
    promotion_periods = []
    for start, end in promotion_dates:
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')
        promotion_periods.append((start_dt, end_dt))
    
    # External shock: competitor opened near store_002 on 2022-03-01
    competitor_shock_date = datetime(2022, 3, 1)
    
    data = []
    
    for store_id, store_info in stores.items():
        for date in dates:
            # Days since start (for trend calculation)
            days_since_start = (date - start_date).days
            
            # Base sales with growth trend
            trend_factor = 1 + (store_info['growth_rate'] * days_since_start / 365.25)
            base_sales = store_info['base_sales'] * trend_factor
            
            # Weekly seasonality (higher sales on weekends)
            day_of_week = date.weekday()  # 0=Monday, 6=Sunday
            if day_of_week >= 5:  # Weekend (Friday, Saturday, Sunday)
                weekly_factor = store_info['weekend_boost']
            else:
                weekly_factor = 1.0
            
            # Annual seasonality (higher in December, lower in January-February)
            day_of_year = date.timetuple().tm_yday
            annual_seasonal = 1 + store_info['seasonal_amplitude'] * np.sin(2 * np.pi * (day_of_year - 60) / 365.25)
            
            # Check if promotion is active
            promotion_active = 0
            promotion_boost = 1.0
            for promo_start, promo_end in promotion_periods:
                if promo_start <= date <= promo_end:
                    promotion_active = 1
                    promotion_boost = 1.8 + np.random.normal(0, 0.1)  # 80% boost with some variation
                    break
            
            # External shock effect (competitor impact on store_002)
            shock_effect = 1.0
            if store_id == 'store_002' and date >= competitor_shock_date:
                shock_effect = 0.75  # 25% permanent reduction in sales
            
            # Calculate daily sales
            daily_sales = (base_sales * 
                         weekly_factor * 
                         annual_seasonal * 
                         promotion_boost * 
                         shock_effect * 
                         np.random.normal(1, 0.1))  # Add some random noise
            
            # Ensure sales are positive
            daily_sales = max(daily_sales, 0)
            
            # Additional features
            is_weekend = 1 if day_of_week >= 5 else 0
            month = date.month
            quarter = (month - 1) // 3 + 1
            
            data.append({
                'date': date,
                'store_id': store_id,
                'daily_sales': round(daily_sales, 2),
                'promotion_active': promotion_active,
                'store_size': store_info['store_size'],
                'city_population': store_info['city_population'],
                'is_weekend': is_weekend,
                'day_of_week': day_of_week,
                'month': month,
                'quarter': quarter,
                'days_since_start': days_since_start
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['store_id', 'date']).reset_index(drop=True)
    
    return df

# Generate the data
if __name__ == "__main__":
    df = simulate_innovatemart_data()
    
    # Save to CSV
    df.to_csv('innovatemart_sales_data.csv', index=False)
    
    # Display basic statistics
    print("Dataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head(10))
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nSales statistics by store:")
    print(df.groupby('store_id')['daily_sales'].describe())
    
    print("\nPromotion statistics:")
    print(df.groupby('promotion_active')['daily_sales'].mean())