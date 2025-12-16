"""
LSTM Model - Long Short-Term Memory neural network for time series prediction
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .base_model import BaseModel
from config import config


logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """LSTM-based model for sequence prediction"""

    def __init__(self, name: str = "lstm"):
        super().__init__(name)
        self.history = None

    def build(
        self,
        input_shape: Tuple,
        lstm_units: List[int] = None,
        dropout: float = 0.2,
        dense_units: List[int] = None,
        num_classes: int = 3,
        **kwargs
    ) -> None:
        """
        Build LSTM model architecture

        Args:
            input_shape: (sequence_length, num_features)
            lstm_units: List of units for each LSTM layer
            dropout: Dropout rate
            dense_units: List of units for dense layers
            num_classes: Number of output classes
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                LSTM, Dense, Dropout, BatchNormalization,
                Bidirectional, Input
            )
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            lstm_units = lstm_units or [128, 64, 32]
            dense_units = dense_units or [64, 32]

            model = Sequential()

            # Input layer
            model.add(Input(shape=input_shape))

            # LSTM layers
            for i, units in enumerate(lstm_units):
                return_sequences = i < len(lstm_units) - 1
                model.add(Bidirectional(LSTM(
                    units,
                    return_sequences=return_sequences,
                    recurrent_dropout=dropout * 0.5
                )))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))

            # Dense layers
            for units in dense_units:
                model.add(Dense(units, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(dropout * 0.5))

            # Output layer
            if num_classes > 2:
                model.add(Dense(num_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
            else:
                model.add(Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'

            model.compile(
                optimizer=Adam(learning_rate=config.model.learning_rate),
                loss=loss,
                metrics=['accuracy']
            )

            self.model = model
            self.num_classes = num_classes
            logger.info(f"Built LSTM model with input shape {input_shape}")

        except ImportError:
            logger.error("TensorFlow not installed. Run: pip install tensorflow")
            raise

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = None,
        batch_size: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the LSTM model"""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

        epochs = epochs or config.model.epochs
        batch_size = batch_size or config.model.batch_size

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.model.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True
        self.last_trained = __import__('datetime').datetime.now()

        return {
            'history': self.history.history,
            'epochs_trained': len(self.history.history['loss'])
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        proba = self.model.predict(X, verbose=0)

        if self.num_classes > 2:
            return np.argmax(proba, axis=1)
        else:
            return (proba > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X, verbose=0)

    def save(self, path: str = None) -> str:
        """Save model with Keras format"""
        from pathlib import Path
        from datetime import datetime

        if path is None:
            path = Path(config.models_dir) / 'saved' / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        self.model.save(path / 'model.keras')

        # Save metadata
        import json
        metadata = {
            'name': self.name,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'num_classes': self.num_classes,
            'feature_names': self.feature_names,
            'created_at': self.created_at.isoformat(),
            'last_trained': self.last_trained.isoformat() if self.last_trained else None
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"LSTM model saved to {path}")
        return str(path)

    def load(self, path: str) -> None:
        """Load Keras model"""
        from pathlib import Path
        from datetime import datetime
        import json
        import tensorflow as tf

        path = Path(path)

        # Load Keras model
        self.model = tf.keras.models.load_model(path / 'model.keras')

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.name = metadata['name']
        self.is_trained = metadata['is_trained']
        self.metrics = metadata['metrics']
        self.num_classes = metadata['num_classes']
        self.feature_names = metadata['feature_names']
        self.created_at = datetime.fromisoformat(metadata['created_at'])
        if metadata['last_trained']:
            self.last_trained = datetime.fromisoformat(metadata['last_trained'])

        logger.info(f"LSTM model loaded from {path}")


class TransformerModel(BaseModel):
    """Transformer-based model for sequence prediction"""

    def __init__(self, name: str = "transformer"):
        super().__init__(name)
        self.history = None

    def build(
        self,
        input_shape: Tuple,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_transformer_blocks: int = 2,
        mlp_units: List[int] = None,
        dropout: float = 0.2,
        num_classes: int = 3,
        **kwargs
    ) -> None:
        """Build Transformer model"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model

            mlp_units = mlp_units or [128, 64]

            def transformer_block(x, head_size, num_heads, ff_dim, dropout):
                # Multi-head attention
                attn_output = layers.MultiHeadAttention(
                    key_dim=head_size, num_heads=num_heads, dropout=dropout
                )(x, x)
                attn_output = layers.Dropout(dropout)(attn_output)
                out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

                # Feed-forward network
                ffn_output = layers.Dense(ff_dim, activation='relu')(out1)
                ffn_output = layers.Dropout(dropout)(ffn_output)
                ffn_output = layers.Dense(x.shape[-1])(ffn_output)
                return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

            # Build model
            inputs = layers.Input(shape=input_shape)
            x = inputs

            # Position embedding
            positions = tf.range(start=0, limit=input_shape[0], delta=1)
            position_embedding = layers.Embedding(
                input_dim=input_shape[0], output_dim=input_shape[1]
            )(positions)
            x = x + position_embedding

            # Transformer blocks
            head_size = input_shape[1] // num_heads
            for _ in range(num_transformer_blocks):
                x = transformer_block(x, head_size, num_heads, ff_dim, dropout)

            # Global pooling
            x = layers.GlobalAveragePooling1D()(x)

            # MLP layers
            for units in mlp_units:
                x = layers.Dense(units, activation='relu')(x)
                x = layers.Dropout(dropout)(x)

            # Output
            if num_classes > 2:
                outputs = layers.Dense(num_classes, activation='softmax')(x)
                loss = 'sparse_categorical_crossentropy'
            else:
                outputs = layers.Dense(1, activation='sigmoid')(x)
                loss = 'binary_crossentropy'

            self.model = Model(inputs, outputs)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config.model.learning_rate),
                loss=loss,
                metrics=['accuracy']
            )

            self.num_classes = num_classes
            logger.info(f"Built Transformer model with input shape {input_shape}")

        except ImportError:
            logger.error("TensorFlow not installed. Run: pip install tensorflow")
            raise

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = None,
        batch_size: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the Transformer model"""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        epochs = epochs or config.model.epochs
        batch_size = batch_size or config.model.batch_size

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.model.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True
        self.last_trained = __import__('datetime').datetime.now()

        return {
            'history': self.history.history,
            'epochs_trained': len(self.history.history['loss'])
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        proba = self.model.predict(X, verbose=0)
        if self.num_classes > 2:
            return np.argmax(proba, axis=1)
        return (proba > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X, verbose=0)
