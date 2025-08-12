#!/usr/bin/env python3
"""
Complete pipeline runner for InnovateMart Sales Forecasting Project
Run this script to execute the entire pipeline from data generation to model training
"""

import os
import subprocess
import sys
from datetime import datetime

def print_step(step_name, description):
    """Print formatted step information"""
    print("\n" + "="*60)
    print(f"STEP: {step_name}")
    print(f"Description: {description}")
    print("="*60)

def check_dependencies():
    """Check if required packages are installed"""
    print_step("DEPENDENCY CHECK", "Verifying required packages are installed")
    
    required_packages = [
        'pandas', 'numpy', 'torch', 'pytorch_lightning', 
        'pytorch_forecasting', 'streamlit', 'matplotlib', 
        'seaborn', 'plotly', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies satisfied!")
    return True

def run_data_simulation():
    """Run data simulation script"""
    print_step("DATA SIMULATION", "Generating synthetic sales data for InnovateMart")
    
    # Import and run data simulation
    try:
        from data_simulation import simulate_innovatemart_data
        
        print("Generating synthetic data...")
        df = simulate_innovatemart_data()
        
        # Save to CSV
        df.to_csv('innovatemart_sales_data.csv', index=False)
        
        print(f"✅ Data generated successfully!")
        print(f"   - Dataset shape: {df.shape}")
        print(f"   - Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   - Stores: {df['store_id'].nunique()}")
        print(f"   - Total records: {len(df):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data simulation: {e}")
        return False

def run_model_training():
    """Run model training script"""
    print_step("MODEL TRAINING", "Training Temporal Fusion Transformer model")
    
    try:
        # Import training modules
        from model_training import main as train_main
        
        print("Starting model training...")
        print("This may take several minutes depending on your hardware...")
        
        # Run training
        model, training_dataset, pred_df = train_main()
        
        print("✅ Model training completed successfully!")
        print(f"   - Model saved to: tft_model_state.pth")
        print(f"   - Training dataset saved to: training_dataset.pkl")
        print(f"   - Predictions saved to: model_predictions.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file"""
    config_content = '''# InnovateMart Forecasting Configuration
# Modify these parameters to customize the model and data generation

[data_simulation]
n_stores = 4
start_date = 2021-01-01
end_date = 2023-06-30
base_noise_std = 0.1

[model_training]
max_encoder_length = 60
max_prediction_length = 30
hidden_size = 32
attention_head_size = 4
dropout = 0.1
learning_rate = 0.03
max_epochs = 50
batch_size = 64

[streamlit]
page_title = "InnovateMart Sales Forecasting"
theme = "light"
'''
    
    with open('config.ini', 'w') as f:
        f.write(config_content)
    
    print("✅ Sample configuration file created: config.ini")

def verify_outputs():
    """Verify that all expected output files were created"""
    print_step("OUTPUT VERIFICATION", "Checking that all files were generated correctly")
    
    expected_files = [
        'innovatemart_sales_data.csv',
        'model_predictions.csv',
        'training_dataset.pkl',
        'tft_model_state.pth'
    ]
    
    all_files_exist = True
    
    for file_name in expected_files:
        if os.path.exists(file_name):
            file_size = os.path.getsize(file_name)
            print(f"✅ {file_name} - {file_size:,} bytes")
        else:
            print(f"❌ {file_name} - NOT FOUND")
            all_files_exist = False
    
    return all_files_exist

def launch_streamlit():
    """Launch the Streamlit dashboard"""
    print_step("DASHBOARD LAUNCH", "Starting Streamlit application")
    
    print("🚀 Launching Streamlit dashboard...")
    print("   Dashboard will be available at: http://localhost:8501")
    print("   Press Ctrl+C to stop the server")
    print("\nStarting in 3 seconds...")
    
    import time
    time.sleep(3)
    
    try:
        subprocess.run(['streamlit', 'run', 'streamlit_app.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("You can manually run: streamlit run streamlit_app.py")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it using: pip install streamlit")
        print("Then run manually: streamlit run streamlit_app.py")

def main():
    """Main pipeline execution"""
    print("🛒 InnovateMart Sales Forecasting - Complete Pipeline")
    print("=" * 60)
    print(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        sys.exit(1)
    
    # Step 2: Generate data
    if not run_data_simulation():
        print("\n❌ Data simulation failed. Please check the errors above.")
        sys.exit(1)
    
    # Step 3: Train model
    if not run_model_training():
        print("\n❌ Model training failed. Please check the errors above.")
        sys.exit(1)
    
    # Step 4: Verify outputs
    if not verify_outputs():
        print("\n⚠️  Some output files are missing. The pipeline may not have completed successfully.")
    
    # Step 5: Create sample config
    create_sample_config()
    
    # Pipeline completion summary
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Generated files:")
    print("  📊 innovatemart_sales_data.csv - Synthetic sales data")
    print("  🤖 tft_model_state.pth - Trained model weights")
    print("  📈 model_predictions.csv - Model predictions")
    print("  💾 training_dataset.pkl - Training dataset")
    print("  ⚙️  config.ini - Configuration file")
    
    print("\nNext steps:")
    print("  1. Run 'streamlit run streamlit_app.py' to launch the dashboard")
    print("  2. Open http://localhost:8501 in your browser")
    print("  3. Explore the interactive sales forecasting dashboard!")
    
    # Ask if user wants to launch Streamlit
    launch_choice = input("\nWould you like to launch the Streamlit dashboard now? (y/n): ").lower()
    if launch_choice in ['y', 'yes']:
        launch_streamlit()
    else:
        print("\n✅ Pipeline complete! Run 'streamlit run streamlit_app.py' when ready.")

if __name__ == "__main__":
    main()