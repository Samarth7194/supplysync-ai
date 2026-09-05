"""Model artifact loading, metadata, and integrity checks."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_schema_checksum


class ModelArtifactValidationError(Exception):
    """Raised when an artifact exists but should not be used for inference."""

class ModelService:
    """
    Centralized model loading, caching, and management service.
    """
    
    def __init__(self, model_dir: str = "backend/saved_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self._model_cache = {}
        self._feature_cache = {}
        self._model_metadata = {}
        
        # Setup structured logging
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger for model operations."""
        
        logger = logging.getLogger("model_service")
        logger.setLevel(logging.INFO)
        
        # Create handler if it doesn't exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    @staticmethod
    def checksum_file(path: Path | str) -> str:
        """Compute a SHA-256 checksum for a model artifact."""
        digest = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def version_from_checksum(model_name: str, saved_at: str, checksum: str) -> str:
        """Return a deterministic model version from timestamp + checksum."""
        stamp = saved_at.replace("+00:00", "Z").replace("-", "").replace(":", "")
        return f"{model_name}-{stamp}-{checksum[:12]}"

    def metadata_path(self, model_name: str) -> Path:
        return self.model_dir / f"{model_name}_metadata.json"

    def artifact_path(self, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> Path:
        artifact_file = (metadata or {}).get("artifact_file") or f"{model_name}.pkl"
        return self.model_dir / str(artifact_file)
    
    def save_model(self, model: Any, model_name: str, metadata: Optional[Dict] = None) -> str:
        """
        Save a trained model with metadata.
        
        Parameters:
        - model: Trained model object
        - model_name: Name for the model file
        - metadata: Model metadata (training date, features, etc.)
        
        Returns:
        - Path to saved model
        """
        
        model_path = self.artifact_path(model_name, metadata)

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save metadata
        if metadata is None:
            metadata = {}

        # Store only the filename in the portable metadata so the committed
        # JSON doesn't leak the original developer's absolute filesystem
        # path. The loader resolves the file via ``model_dir``.
        saved_at = metadata.get("saved_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
        checksum = self.checksum_file(model_path)
        feature_columns = list(metadata.get("features") or FEATURE_COLUMNS)
        metadata.update(
            {
                "saved_at": saved_at,
                "model_name": model_name,
                "model_family": "lightgbm" if "lightgbm" in model_name else model_name,
                "model_type": "ml",
                "artifact_file": model_path.name,
                "artifact_checksum": checksum,
                "checksum_algorithm": "sha256",
                "version": metadata.get("version") or self.version_from_checksum(model_name, saved_at, checksum),
                "feature_schema_version": metadata.get("feature_schema_version") or FEATURE_SCHEMA_VERSION,
                "feature_schema_checksum": metadata.get("feature_schema_checksum")
                or feature_schema_checksum(feature_columns),
                "lifecycle_status": metadata.get("lifecycle_status") or "candidate",
            }
        )
        
        metadata_file = metadata.get("metadata_file") or f"{model_name}_metadata.json"
        metadata["metadata_file"] = str(metadata_file)
        metadata_path = self.model_dir / str(metadata_file)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update cache
        self._model_cache[model_name] = model
        self._model_metadata[model_name] = metadata
        
        self.logger.info(f"Model saved: {model_name} at {model_path}")
        
        return str(model_path)
    
    def load_model(self, model_name: str, use_cache: bool = True) -> Any:
        """
        Load a trained model with caching.
        
        Parameters:
        - model_name: Name of the model to load
        - use_cache: Whether to use cached model if available
        
        Returns:
        - Loaded model object
        """
        
        # Check cache first
        if use_cache and model_name in self._model_cache:
            self.logger.info(f"Model loaded from cache: {model_name}")
            return self._model_cache[model_name]
        
        # Load from disk only after validating metadata and checksum.
        metadata = self.validate_model_artifact(model_name)
        model_path = self.artifact_path(model_name, metadata)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Update cache
        if use_cache:
            self._model_cache[model_name] = model
        
        self._model_metadata[model_name] = metadata
        
        self.logger.info(f"Model loaded from disk: {model_name}")
        
        return model
    
    def get_model_metadata(self, model_name: str) -> Dict:
        """
        Get model metadata.
        
        Parameters:
        - model_name: Name of the model
        
        Returns:
        - Model metadata dictionary
        """
        
        if model_name in self._model_metadata:
            return self._model_metadata[model_name]
        
        # Load from disk
        metadata_path = self.metadata_path(model_name)
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self._model_metadata[model_name] = metadata
            return metadata
        
        return {}

    def validate_model_artifact(self, model_name: str) -> Dict[str, Any]:
        """Validate artifact presence, checksum, and feature compatibility."""
        metadata = self.get_model_metadata(model_name)
        if not metadata:
            raise ModelArtifactValidationError(f"Model metadata not found for {model_name}")

        model_type = metadata.get("model_type", "ml")
        if model_type != "ml":
            raise ModelArtifactValidationError(f"Unsupported model_type for artifact loading: {model_type}")

        artifact_path = self.artifact_path(model_name, metadata)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model not found: {artifact_path}")

        expected_checksum = metadata.get("artifact_checksum")
        if expected_checksum:
            actual_checksum = self.checksum_file(artifact_path)
            if actual_checksum != expected_checksum:
                raise ModelArtifactValidationError(
                    f"Checksum mismatch for {model_name}: expected {expected_checksum}, got {actual_checksum}"
                )

        feature_version = metadata.get("feature_schema_version")
        if feature_version and feature_version != FEATURE_SCHEMA_VERSION:
            raise ModelArtifactValidationError(
                f"Feature schema mismatch for {model_name}: artifact={feature_version}, runtime={FEATURE_SCHEMA_VERSION}"
            )

        features = metadata.get("features")
        if features and list(features) != FEATURE_COLUMNS:
            raise ModelArtifactValidationError("Artifact feature column order does not match runtime schema")
        expected_feature_checksum = metadata.get("feature_schema_checksum")
        if expected_feature_checksum and expected_feature_checksum != feature_schema_checksum(FEATURE_COLUMNS):
            raise ModelArtifactValidationError("Artifact feature schema checksum does not match runtime schema")

        return metadata

    def artifact_status(self, model_name: str) -> Dict[str, Any]:
        """Return credential/path-safe model artifact status for health/model-info."""
        try:
            metadata = self.validate_model_artifact(model_name)
            return {
                "valid": True,
                "model_name": model_name,
                "version": metadata.get("version"),
                "feature_schema_version": metadata.get("feature_schema_version"),
                "lifecycle_status": metadata.get("lifecycle_status"),
                "checksum": metadata.get("artifact_checksum"),
            }
        except Exception as exc:  # noqa: BLE001 - status should be diagnostic, not fatal
            return {
                "valid": False,
                "model_name": model_name,
                "error": type(exc).__name__,
            }
    
    def cache_features(self, sku: str, features: pd.DataFrame, cache_name: str = "latest") -> None:
        """
        Cache features for a SKU.
        
        Parameters:
        - sku: SKU identifier
        - features: Feature DataFrame
        - cache_name: Name for the cache entry
        """
        
        cache_key = f"{sku}_{cache_name}"
        self._feature_cache[cache_key] = features
        
        self.logger.info(f"Features cached: {cache_key} with shape {features.shape}")
    
    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear model and/or feature cache.
        
        Parameters:
        - model_name: Specific model to clear, or None for all
        """
        
        if model_name:
            if model_name in self._model_cache:
                del self._model_cache[model_name]
            if model_name in self._model_metadata:
                del self._model_metadata[model_name]
                self.logger.info(f"Cleared cache for model: {model_name}")
        else:
            self._model_cache.clear()
            self._feature_cache.clear()
            self._model_metadata.clear()
            self.logger.info("Cleared all caches")
    
    def list_available_models(self) -> list:
        """
        List all available models.
        
        Returns:
        - List of model names
        """
        
        model_files = list(self.model_dir.glob("*.pkl"))
        model_names = [f.stem for f in model_files]
        
        return sorted(model_names)
    
# Global model service instance
_model_service = None

def get_model_service(model_dir: str = "backend/saved_models") -> ModelService:
    """
    Get global model service instance.
    
    Parameters:
    - model_dir: Directory for model storage
    
    Returns:
    - ModelService instance
    """
    
    global _model_service
    
    if _model_service is None or Path(model_dir) != _model_service.model_dir:
        _model_service = ModelService(model_dir)
    
    return _model_service
