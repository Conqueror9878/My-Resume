"""
XGBoost Model - Gradient boosting for tabular data
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .base_model import BaseModel
from config import config


logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost-based model for classification/regression"""

    def __init__(self, name: str = "xgboost", task: str = "classification"):
        super().__init__(name)
        self.task = task
        self.num_classes = None

    def build(
        self,
        input_shape: Tuple = None,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        num_classes: int = 3,
        **kwargs
    ) -> None:
        """
        Build XGBoost model

        Args:
            input_shape: Not used, kept for API consistency
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            num_classes: Number of classes for classification
        """
        try:
            import xgboost as xgb

            self.num_classes = num_classes

            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
                **kwargs
            }

            if self.task == "classification":
                if num_classes > 2:
                    params['objective'] = 'multi:softprob'
                    params['num_class'] = num_classes
                else:
                    params['objective'] = 'binary:logistic'

                self.model = xgb.XGBClassifier(**params)
            else:
                params['objective'] = 'reg:squarederror'
                self.model = xgb.XGBRegressor(**params)

            logger.info(f"Built XGBoost model for {self.task}")

        except ImportError:
            logger.error("XGBoost not installed. Run: pip install xgboost")
            raise

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the XGBoost model"""

        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        self.is_trained = True
        self.last_trained = __import__('datetime').datetime.now()

        # Get feature importances
        importance = self.model.feature_importances_

        return {
            'feature_importance': importance,
            'best_iteration': getattr(self.model, 'best_iteration', None)
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        if self.task == "classification":
            return self.model.predict_proba(X)
        else:
            # For regression, return predictions as-is
            return self.model.predict(X)


class LightGBMModel(BaseModel):
    """LightGBM-based model for classification/regression"""

    def __init__(self, name: str = "lightgbm", task: str = "classification"):
        super().__init__(name)
        self.task = task
        self.num_classes = None

    def build(
        self,
        input_shape: Tuple = None,
        n_estimators: int = 500,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        num_classes: int = 3,
        **kwargs
    ) -> None:
        """Build LightGBM model"""
        try:
            import lightgbm as lgb

            self.num_classes = num_classes

            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'num_leaves': num_leaves,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                **kwargs
            }

            if self.task == "classification":
                if num_classes > 2:
                    params['objective'] = 'multiclass'
                    params['num_class'] = num_classes
                else:
                    params['objective'] = 'binary'

                self.model = lgb.LGBMClassifier(**params)
            else:
                params['objective'] = 'regression'
                self.model = lgb.LGBMRegressor(**params)

            logger.info(f"Built LightGBM model for {self.task}")

        except ImportError:
            logger.error("LightGBM not installed. Run: pip install lightgbm")
            raise

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the LightGBM model"""

        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set
        )

        self.is_trained = True
        self.last_trained = __import__('datetime').datetime.now()

        return {
            'feature_importance': self.model.feature_importances_,
            'best_iteration': getattr(self.model, 'best_iteration_', None)
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        if self.task == "classification":
            return self.model.predict_proba(X)
        return self.model.predict(X)


class RandomForestModel(BaseModel):
    """Random Forest model"""

    def __init__(self, name: str = "random_forest", task: str = "classification"):
        super().__init__(name)
        self.task = task

    def build(
        self,
        input_shape: Tuple = None,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_split: int = 5,
        num_classes: int = 3,
        **kwargs
    ) -> None:
        """Build Random Forest model"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        self.num_classes = num_classes

        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': 42,
            'n_jobs': -1,
            **kwargs
        }

        if self.task == "classification":
            self.model = RandomForestClassifier(**params)
        else:
            self.model = RandomForestRegressor(**params)

        logger.info(f"Built Random Forest model for {self.task}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train Random Forest model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.last_trained = __import__('datetime').datetime.now()

        return {'feature_importance': self.model.feature_importances_}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        if self.task == "classification":
            return self.model.predict_proba(X)
        return self.model.predict(X)
