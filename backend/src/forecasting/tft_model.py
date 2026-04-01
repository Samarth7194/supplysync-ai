# src/forecasting/tft_model.py

import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pandas as pd
from typing import Dict, List, Optional, Tuple

class TFTForecaster:
    """
    Lead Architect Implementation: Advanced Temporal Fusion Transformer (TFT) 
    for risk-aware, probabilistic supply chain orchestration.
    
    Features:
    - Multi-horizon forecasting
    - Quantile loss for predictive intervals (P10, P50, P90)
    - Variable Selection Networks (VSN) for interpretability
    """
    
    def __init__(self, hidden_size: int = 16, lstm_layers: int = 2, dropout: float = 0.1):
        self.model = None
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.quantiles = [0.1, 0.5, 0.9] # Task 2.2: Quantile Loss Optimization

    def create_model(self, training_data: TimeSeriesDataSet):
        """Constructs the TFT architecture linked to the specific dataset schema."""
        self.model = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=0.03,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            attention_head_size=4,
            dropout=self.dropout,
            hidden_continuous_size=8,
            output_size=len(self.quantiles), # Number of quantiles
            loss=QuantileLoss(quantiles=self.quantiles),
            reduce_on_plateau_patience=4,
        )
        return self.model

    def predict_probabilistic(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Generates calibrated quantiles for risk-aware safety stock decisions.
        
        Returns:
            Dictionary containing 'p10', 'p50', 'p90' forecast tensors.
        """
        if self.model is None:
            raise ValueError("Model has not been initialized or loaded.")
            
        predictions = self.model.predict(data, mode="raw", return_x=False)
        # Predictions shape: [batch, horizon, n_quantiles]
        
        return {
            "p10": predictions.output[..., 0],
            "p50": predictions.output[..., 1],
            "p90": predictions.output[..., 2]
        }

    def get_feature_importance(self, x: Dict) -> Dict[str, float]:
        """
        Task 2.3: Interpretability Layer.
        Extracts variable importance from the TFT Variable Selection Network.
        """
        if self.model is None:
            return {}
            
        interpretation = self.model.interpret_output(self.model.predict(x, mode="raw"), reduction="sum")
        
        # Merge static and dynamic variable importance
        importance = {
            **{k: float(v) for k, v in interpretation["static_variables"].items()},
            **{k: float(v) for k, v in interpretation["encoder_variables"].items()},
            **{k: float(v) for k, v in interpretation["decoder_variables"].items()}
        }
        
        return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

# Design Pattern: Risk-Aware Decision Logic integration
def calculate_risk_adjusted_reorder(p90_forecast: float, current_stock: int, lead_time: int) -> float:
    """
    Implements Task 2.2: Use the 90th percentile to drive safety stock, 
    ensuring 90% service level protection.
    """
    lead_time_p90 = p90_forecast * lead_time
    shortfall = max(0, lead_time_p90 - current_stock)
    return shortfall
