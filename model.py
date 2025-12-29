"""
Cryptocurrency Price Prediction using LSTM
Time series forecasting with Long Short-Term Memory neural networks

Author: Tatsu (@code-craftsman369)
Date: December 30, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data_fetcher import fetch_crypto_data, prepare_data

class CryptoPricePredictor:
    """LSTM-based cryptocurrency price predictor"""
    
    def __init__(self, lookback=60):
        """
        Initialize predictor
        
        Args:
            lookback: Number of previous days to use for prediction
        """
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def create_sequences(self, data):
        """
        Create sequences for LSTM training
        
        Args:
            data: Scaled price data
            
        Returns:
            X, y: Training sequences and targets
        """
        X, y = [], []
        
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
            
        return np.array(X), np.array(y)
    
    def prepare_training_data(self, df, train_split=0.8):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with Close prices
            train_split: Ratio of training data
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Scale data
        scaled_data = self.scaler.fit_transform(df.values)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split train/test
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ Training data: {X_train.shape}")
        print(f"✅ Test data: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """
        Build LSTM model
        
        Args:
            input_shape: Shape of input data (time steps, features)
        """
        self.model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ Model built successfully")
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model
        """
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        # Inverse transform to original scale
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance
        
        Returns:
            Dictionary with metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        return metrics
    
    def plot_results(self, y_true, y_pred, title='Price Prediction'):
        """Plot prediction results"""
        plt.figure(figsize=(14, 5))
        plt.plot(y_true, label='Actual Price', linewidth=2)
        plt.plot(y_pred, label='Predicted Price', linewidth=2)
        plt.title(title, fontsize=14)
        plt.xlabel('Days')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt

def main():
    """Main execution"""
    print("Cryptocurrency Price Prediction with LSTM")
    print("=" * 50)
    
    # Fetch Bitcoin data
    print("\n📊 Fetching Bitcoin data...")
    btc_data = fetch_crypto_data('BTC-USD', period='2y')
    btc_df = prepare_data(btc_data)
    
    if btc_df is None:
        print("❌ Failed to fetch data")
        return
    
    # Initialize predictor
    predictor = CryptoPricePredictor(lookback=60)
    
    # Prepare data
    print("\n🔧 Preparing training data...")
    X_train, X_test, y_train, y_test = predictor.prepare_training_data(btc_df)
    
    # Build model
    print("\n🏗️ Building LSTM model...")
    predictor.build_model(input_shape=(X_train.shape[1], 1))
    
    # Train model
    print("\n🚀 Training model...")
    history = predictor.train(X_train, y_train, X_test, y_test, epochs=50)
    
    # Make predictions
    print("\n📈 Making predictions...")
    predictions = predictor.predict(X_test)
    y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluate
    metrics = predictor.evaluate(y_test_actual, predictions)
    print("\n📊 Model Performance:")
    print(f"  RMSE: ${metrics['RMSE']:.2f}")
    print(f"  MAE: ${metrics['MAE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # Plot results
    print("\n📉 Generating plots...")
    plt = predictor.plot_results(y_test_actual, predictions, 
                                   title='Bitcoin Price Prediction (LSTM)')
    plt.savefig('images/prediction_results.png', dpi=300, bbox_inches='tight')
    print("✅ Plot saved to images/prediction_results.png")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/training_history.png', dpi=300, bbox_inches='tight')
    print("✅ Training history saved to images/training_history.png")
    
    print("\n✅ Prediction completed successfully!")

if __name__ == "__main__":
    main()