"""
Ensemble Model - Combines multiple models for robust predictions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from .base_model import BaseModel
from .lstm_model import LSTMModel, TransformerModel
from .xgboost_model import XGBoostModel, LightGBMModel, RandomForestModel
from config import config


logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines predictions from multiple models.
    Supports weighted averaging, stacking, and voting.
    """

    def __init__(
        self,
        name: str = "ensemble",
        ensemble_method: str = "weighted_average"
    ):
        super().__init__(name)
        self.models: Dict[str, BaseModel] = {}
        self.weights: Dict[str, float] = {}
        self.ensemble_method = ensemble_method  # weighted_average, voting, stacking
        self.meta_model = None  # For stacking

    def add_model(
        self,
        model: BaseModel,
        weight: float = 1.0
    ) -> None:
        """Add a model to the ensemble"""
        self.models[model.name] = model
        self.weights[model.name] = weight
        logger.info(f"Added {model.name} to ensemble with weight {weight}")

    def build(
        self,
        input_shape: Tuple,
        sequence_input_shape: Tuple = None,
        include_lstm: bool = True,
        include_transformer: bool = True,
        include_xgboost: bool = True,
        include_lightgbm: bool = True,
        include_rf: bool = True,
        num_classes: int = 3,
        **kwargs
    ) -> None:
        """
        Build ensemble with default model configurations

        Args:
            input_shape: Shape for tabular models (num_features,)
            sequence_input_shape: Shape for sequence models (seq_len, num_features)
            include_*: Whether to include specific models
            num_classes: Number of output classes
        """
        weights = config.model.ensemble_weights

        # LSTM model
        if include_lstm and sequence_input_shape:
            lstm = LSTMModel("lstm")
            lstm.build(sequence_input_shape, num_classes=num_classes)
            self.add_model(lstm, weights.get('lstm', 0.3))

        # Transformer model
        if include_transformer and sequence_input_shape:
            transformer = TransformerModel("transformer")
            transformer.build(sequence_input_shape, num_classes=num_classes)
            self.add_model(transformer, weights.get('transformer', 0.2))

        # XGBoost model
        if include_xgboost:
            xgb = XGBoostModel("xgboost")
            xgb.build(input_shape, num_classes=num_classes)
            self.add_model(xgb, weights.get('xgboost', 0.3))

        # LightGBM model
        if include_lightgbm:
            lgb = LightGBMModel("lightgbm")
            lgb.build(input_shape, num_classes=num_classes)
            self.add_model(lgb, weights.get('lightgbm', 0.1))

        # Random Forest model
        if include_rf:
            rf = RandomForestModel("random_forest")
            rf.build(input_shape, num_classes=num_classes)
            self.add_model(rf, weights.get('random_forest', 0.1))

        self.num_classes = num_classes
        logger.info(f"Built ensemble with {len(self.models)} models")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_train_seq: Optional[np.ndarray] = None,
        X_val_seq: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train all models in the ensemble

        Args:
            X_train: Training features (tabular)
            y_train: Training labels
            X_val: Validation features (tabular)
            y_val: Validation labels
            X_train_seq: Training sequences (for LSTM/Transformer)
            X_val_seq: Validation sequences
        """
        results = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            try:
                # Use sequence data for LSTM/Transformer
                if name in ['lstm', 'transformer'] and X_train_seq is not None:
                    result = model.train(X_train_seq, y_train, X_val_seq, y_val, **kwargs)
                else:
                    result = model.train(X_train, y_train, X_val, y_val, **kwargs)

                results[name] = result
                logger.info(f"Trained {name} successfully")

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = {'error': str(e)}

        self.is_trained = any(m.is_trained for m in self.models.values())
        self.last_trained = datetime.now()

        # Train meta-model for stacking
        if self.ensemble_method == "stacking" and X_val is not None:
            self._train_meta_model(X_val, y_val, X_val_seq)

        return results

    def _train_meta_model(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_val_seq: Optional[np.ndarray] = None
    ) -> None:
        """Train meta-model for stacking ensemble"""
        from sklearn.linear_model import LogisticRegression

        # Get predictions from base models
        meta_features = self._get_meta_features(X_val, X_val_seq)

        # Train meta-model
        self.meta_model = LogisticRegression(max_iter=1000)
        self.meta_model.fit(meta_features, y_val)

        logger.info("Trained meta-model for stacking")

    def _get_meta_features(
        self,
        X: np.ndarray,
        X_seq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get predictions from all models as meta-features"""
        meta_features = []

        for name, model in self.models.items():
            if not model.is_trained:
                continue

            try:
                if name in ['lstm', 'transformer'] and X_seq is not None:
                    proba = model.predict_proba(X_seq)
                else:
                    proba = model.predict_proba(X)

                # Flatten if needed
                if len(proba.shape) > 1:
                    meta_features.append(proba)
                else:
                    meta_features.append(proba.reshape(-1, 1))

            except Exception as e:
                logger.warning(f"Could not get predictions from {name}: {e}")

        return np.hstack(meta_features)

    def predict(
        self,
        X: np.ndarray,
        X_seq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")

        if self.ensemble_method == "stacking" and self.meta_model is not None:
            meta_features = self._get_meta_features(X, X_seq)
            return self.meta_model.predict(meta_features)

        proba = self.predict_proba(X, X_seq)
        return np.argmax(proba, axis=1)

    def predict_proba(
        self,
        X: np.ndarray,
        X_seq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get ensemble probability predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")

        if self.ensemble_method == "stacking" and self.meta_model is not None:
            meta_features = self._get_meta_features(X, X_seq)
            return self.meta_model.predict_proba(meta_features)

        # Weighted average of predictions
        predictions = []
        total_weight = 0

        for name, model in self.models.items():
            if not model.is_trained:
                continue

            try:
                weight = self.weights.get(name, 1.0)

                if name in ['lstm', 'transformer'] and X_seq is not None:
                    proba = model.predict_proba(X_seq)
                else:
                    proba = model.predict_proba(X)

                predictions.append(proba * weight)
                total_weight += weight

            except Exception as e:
                logger.warning(f"Could not get predictions from {name}: {e}")

        if not predictions:
            raise ValueError("No valid predictions from ensemble models")

        # Normalize by total weight
        ensemble_proba = np.sum(predictions, axis=0) / total_weight

        return ensemble_proba

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_test_seq: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all models and ensemble"""
        results = {}

        # Evaluate individual models
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    if name in ['lstm', 'transformer'] and X_test_seq is not None:
                        metrics = model.evaluate(X_test_seq, y_test)
                    else:
                        metrics = model.evaluate(X_test, y_test)
                    results[name] = metrics
                except Exception as e:
                    logger.error(f"Failed to evaluate {name}: {e}")

        # Evaluate ensemble
        predictions = self.predict(X_test, X_test_seq)
        from sklearn.metrics import accuracy_score, f1_score
        results['ensemble'] = {
            'accuracy': accuracy_score(y_test, predictions),
            'f1': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }

        self.metrics = results['ensemble']
        return results

    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get a specific model from the ensemble"""
        return self.models.get(name)

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update model weights"""
        for name, weight in weights.items():
            if name in self.models:
                self.weights[name] = weight
        logger.info(f"Updated ensemble weights: {self.weights}")

    def get_signal(
        self,
        X: np.ndarray,
        X_seq: Optional[np.ndarray] = None,
        threshold: float = 0.6
    ) -> int:
        """Get trading signal from ensemble"""
        try:
            proba = self.predict_proba(X, X_seq)
            if len(proba.shape) > 1:
                proba = proba[-1]

            if len(proba) == 3:  # [down, neutral, up]
                if proba[2] > threshold:
                    return 1  # Buy
                elif proba[0] > threshold:
                    return -1  # Sell
            elif len(proba) == 2:
                if proba[1] > threshold:
                    return 1
                elif proba[0] > threshold:
                    return -1

            return 0  # Hold

        except Exception as e:
            logger.error(f"Error getting ensemble signal: {e}")
            return 0
