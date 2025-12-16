"""
Base Model - Abstract interface for all trading ML models
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime

from config import config


logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all trading models"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        self.metrics: Dict[str, float] = {}
        self.feature_names: List[str] = []
        self.training_history: List[Dict] = []
        self.created_at = datetime.now()
        self.last_trained: Optional[datetime] = None

    @abstractmethod
    def build(self, input_shape: Tuple, **kwargs) -> None:
        """Build the model architecture"""
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (for classification)"""
        pass

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Returns:
            Dict with evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score
        )

        predictions = self.predict(X_test)
        metrics = {}

        # Check if classification or regression
        if len(np.unique(y_test)) <= 10:  # Classification
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            metrics['precision'] = precision_score(y_test, predictions, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, predictions, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_test, predictions, average='weighted', zero_division=0)
        else:  # Regression
            metrics['mse'] = mean_squared_error(y_test, predictions)
            metrics['mae'] = mean_absolute_error(y_test, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test, predictions)

        self.metrics = metrics
        return metrics

    def save(self, path: str = None) -> str:
        """Save model to disk"""
        if path is None:
            path = Path(config.models_dir) / 'saved' / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save metadata
        metadata = {
            'name': self.name,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'feature_names': self.feature_names,
            'created_at': self.created_at.isoformat(),
            'last_trained': self.last_trained.isoformat() if self.last_trained else None
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {path}")
        return str(path)

    def load(self, path: str) -> None:
        """Load model from disk"""
        path = Path(path)

        # Load model
        with open(path / 'model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.name = metadata['name']
        self.is_trained = metadata['is_trained']
        self.metrics = metadata['metrics']
        self.feature_names = metadata['feature_names']
        self.created_at = datetime.fromisoformat(metadata['created_at'])
        if metadata['last_trained']:
            self.last_trained = datetime.fromisoformat(metadata['last_trained'])

        logger.info(f"Model loaded from {path}")

    def get_signal(self, X: np.ndarray, threshold: float = 0.6) -> int:
        """
        Get trading signal from prediction

        Args:
            X: Input features
            threshold: Probability threshold for signal

        Returns:
            1 (buy), -1 (sell), or 0 (hold)
        """
        try:
            proba = self.predict_proba(X)
            if len(proba.shape) > 1:
                proba = proba[-1]  # Get last prediction

            # Assuming 3 classes: [down, neutral, up]
            if len(proba) == 3:
                if proba[2] > threshold:  # Strong up signal
                    return 1
                elif proba[0] > threshold:  # Strong down signal
                    return -1
            elif len(proba) == 2:
                if proba[1] > threshold:
                    return 1
                elif proba[0] > threshold:
                    return -1

            return 0  # Hold
        except Exception as e:
            logger.error(f"Error getting signal: {e}")
            return 0


class ModelRegistry:
    """Registry for managing multiple models"""

    def __init__(self):
        self.models: Dict[str, BaseModel] = {}

    def register(self, model: BaseModel) -> None:
        """Register a model"""
        self.models[model.name] = model
        logger.info(f"Registered model: {model.name}")

    def get(self, name: str) -> Optional[BaseModel]:
        """Get a model by name"""
        return self.models.get(name)

    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())

    def remove(self, name: str) -> None:
        """Remove a model"""
        if name in self.models:
            del self.models[name]
            logger.info(f"Removed model: {name}")


# Global model registry
model_registry = ModelRegistry()
