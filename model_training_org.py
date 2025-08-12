import pandas as pd
import numpy as np
import torch
import pickle
import warnings
from datetime import datetime, timedelta

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

def prepare_data_for_modeling(df):
    """
    Prepare the simulated data for PyTorch Forecasting
    """
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create time index (days since start)
    df['time_idx'] = df['days_since_start']
    
    # Create validation cutoff (last 30 days)
    max_prediction_length = 30
    max_encoder_length = 60
    
    # Split data
    cutoff_date = df['date'].max() - timedelta(days=max_prediction_length)
    
    training = df[df['date'] <= cutoff_date].copy()
    validation = df[df['date'] > cutoff_date].copy()
    
    print(f"Training data: {training.shape[0]} records")
    print(f"Validation data: {validation.shape[0]} records")
    print(f"Training date range: {training['date'].min()} to {training['date'].max()}")
    print(f"Validation date range: {validation['date'].min()} to {validation['date'].max()}")
    
    return training, validation, max_encoder_length, max_prediction_length

def create_datasets(training, validation, max_encoder_length, max_prediction_length):
    """
    Create PyTorch Forecasting datasets
    """
    
    # Create training dataset
    training_dataset = TimeSeriesDataSet(
        training,
        time_idx="time_idx",
        target="daily_sales",
        group_ids=["store_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        
        # Static categoricals (don't change over time)
        static_categoricals=["store_id", "store_size"],
        
        # Static reals (don't change over time)
        static_reals=["city_population"],
        
        # Time-varying known reals (known in advance)
        time_varying_known_reals=[
            "time_idx", "promotion_active", "is_weekend", 
            "day_of_week", "month", "quarter"
        ],
        
        # Time-varying unknown reals (not known in advance, only target)
        time_varying_unknown_reals=["daily_sales"],
        
        # Normalization
        target_normalizer=GroupNormalizer(
            groups=["store_id"], transformation="softplus"
        ),
        
        # Add lags
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        
        # Allow missing values
        allow_missing_timesteps=True,
    )
    
    # Create validation dataset
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, validation, predict=True, stop_randomization=True
    )
    
    # Create data loaders
    batch_size = 64
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )
    
    return training_dataset, validation_dataset, train_dataloader, val_dataloader

def train_model(training_dataset, train_dataloader, val_dataloader):
    """
    Train the Temporal Fusion Transformer model
    """
    # Configure trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    
    logger = TensorBoardLogger("lightning_logs")
    
    trainer = pl.Trainer(
        max_epochs=50,
        gpus=0,  # Use CPU
        gradient_clip_val=0.1,
        limit_train_batches=50,  # Limit for faster training
        callbacks=[early_stop_callback],
        logger=logger,
        enable_progress_bar=True,
    )
    
    # Create model
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=32,  # Small for CPU training
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    
    # Fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # Load best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    return best_tft, trainer

def evaluate_model(model, val_dataloader, training_dataset):
    """
    Evaluate the model and create predictions
    """
    # Calculate metrics
    predictions, x = model.predict(val_dataloader, return_x=True)
    
    # Get actuals
    actuals = torch.cat([y for x, y in iter(val_dataloader)])
    
    # Calculate SMAPE
    smape = SMAPE()
    smape_score = smape(predictions, actuals)
    
    print(f"SMAPE Score: {smape_score:.4f}")
    
    # Create prediction vs actual dataframe
    predictions_vs_actuals = []
    
    for idx, prediction in enumerate(predictions):
        # Get corresponding x data
        x_data = {k: v[idx] for k, v in x.items()}
        
        # Decode the data
        decoder_data = training_dataset.x_to_index(x_data)
        
        # Get store_id and time information
        store_id = decoder_data['groups'][0]  # First group is store_id
        
        # Get prediction time indices
        decoder_time_idx = decoder_data['decoder_time_idx']
        
        for t, (pred, actual) in enumerate(zip(prediction, actuals[idx])):
            predictions_vs_actuals.append({
                'store_id': store_id,
                'time_idx': decoder_time_idx[t].item(),
                'prediction': pred.item(),
                'actual': actual.item()
            })
    
    pred_df = pd.DataFrame(predictions_vs_actuals)
    
    return pred_df, smape_score

def save_model_and_data(model, training_dataset, pred_df):
    """
    Save model and necessary data for the Streamlit app
    """
    # Save model
    model_path = "tft_model.pkl"
    torch.save(model.state_dict(), "tft_model_state.pth")
    
    # Save training dataset (needed for predictions)
    with open("training_dataset.pkl", "wb") as f:
        pickle.dump(training_dataset, f)
    
    # Save predictions
    pred_df.to_csv("model_predictions.csv", index=False)
    
    print(f"Model saved to: tft_model_state.pth")
    print(f"Training dataset saved to: training_dataset.pkl")
    print(f"Predictions saved to: model_predictions.csv")

def main():
    """
    Main function to train the model
    """
    print("Starting model training pipeline...")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('innovatemart_sales_data.csv')
    
    # Prepare data
    print("Preparing data...")
    training, validation, max_encoder_length, max_prediction_length = prepare_data_for_modeling(df)
    
    # Create datasets
    print("Creating datasets...")
    training_dataset, validation_dataset, train_dataloader, val_dataloader = create_datasets(
        training, validation, max_encoder_length, max_prediction_length
    )
    
    # Train model
    print("Training model...")
    model, trainer = train_model(training_dataset, train_dataloader, val_dataloader)
    
    # Evaluate model
    print("Evaluating model...")
    pred_df, smape_score = evaluate_model(model, val_dataloader, training_dataset)
    
    # Save everything
    print("Saving model and data...")
    save_model_and_data(model, training_dataset, pred_df)
    
    print(f"\nTraining completed successfully!")
    print(f"Final SMAPE Score: {smape_score:.4f}")
    
    return model, training_dataset, pred_df

if __name__ == "__main__":
    # Generate data first if it doesn't exist
    try:
        df = pd.read_csv('innovatemart_sales_data.csv')
        print("Data file found, proceeding with training...")
    except FileNotFoundError:
        print("Data file not found. Please run data simulation first.")
        from data_simulation import simulate_innovatemart_data
        print("Generating data...")
        df = simulate_innovatemart_data()
        df.to_csv('innovatemart_sales_data.csv', index=False)
        print("Data generated and saved.")
    
    # Train the model
    model, training_dataset, pred_df = main()