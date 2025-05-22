import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor
from sklearn.impute import KNNImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

class NO2DownscalingModel:
    def __init__(self):
        # Use an ensemble of multiple models for better performance
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        self.nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        
        # Add Gaussian Process model for uncertainty estimation
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            random_state=42
        )
        
        # Create ensemble model
        self.model = VotingRegressor([
            ('rf', self.rf_model),
            ('xgb', self.xgb_model),
            ('nn', self.nn_model)
        ])
        
        self.scaler = StandardScaler()
        # Replace SimpleImputer with KNNImputer
        self.imputer = KNNImputer(
            n_neighbors=5,
            weights='uniform',
            metric='nan_euclidean'
        )
        self.metrics = {}
        self.uncertainty = None
        
    def prepare_features(self, data):
        """Prepare features for the model with enhanced feature engineering."""
        rows, cols = data.shape
        X = []
        y = []
        
        # Create spatial features with advanced engineering
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(data[i, j]):
                    # Enhanced feature set with spatial context
                    X.append([
                        i/rows,                      # normalized row position
                        j/cols,                      # normalized column position
                        data[i, j],                  # NO2 value
                        np.sin(i/rows * 2 * np.pi),  # sinusoidal transformation of position
                        np.cos(j/cols * 2 * np.pi),  # cosine transformation of position
                        i*j/(rows*cols),             # interaction term
                        (i/rows) ** 2,               # squared terms for non-linear relationships
                        (j/cols) ** 2,
                        np.exp(-((i/rows)**2 + (j/cols)**2)),  # radial basis function
                        np.sin(i/rows * 4 * np.pi),  # higher frequency components
                        np.cos(j/cols * 4 * np.pi)
                    ])
                    y.append(data[i, j])
        
        return np.array(X), np.array(y)
    
    def train(self, data):
        """Train the downscaling model with cross-validation and uncertainty estimation."""
        X, y = self.prepare_features(data)
        X = self.scaler.fit_transform(X)
        X = self.imputer.fit_transform(X)  # Use KNN imputation
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train main ensemble model
        self.model.fit(X_train, y_train)
        
        # Train Gaussian Process model for uncertainty estimation
        self.gp_model.fit(X_train, y_train)
        
        # Calculate and store model performance metrics
        y_pred = self.model.predict(X_val)
        self.metrics = {
            'mse': mean_squared_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred)
        }
        
        # Calculate uncertainty on validation set
        _, std = self.gp_model.predict(X_val, return_std=True)
        self.uncertainty = {
            'mean_uncertainty': np.mean(std),
            'max_uncertainty': np.max(std),
            'min_uncertainty': np.min(std)
        }
        
        return X_val, y_val
    
    def predict(self, data, scale_factor=3):
        """Generate high-resolution predictions with uncertainty estimates."""
        rows, cols = data.shape
        new_rows = rows * scale_factor
        new_cols = cols * scale_factor
        
        # Create high-resolution grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, new_cols),
            np.linspace(0, 1, new_rows)
        )
        
        # Base features
        base_features = np.column_stack([
            grid_x.ravel(),
            grid_y.ravel(),
            np.repeat(data.ravel(), scale_factor**2)
        ])
        
        # Add enhanced features
        sin_x = np.sin(grid_x.ravel() * 2 * np.pi)
        cos_y = np.cos(grid_y.ravel() * 2 * np.pi)
        interaction = grid_x.ravel() * grid_y.ravel()
        squared_x = grid_x.ravel() ** 2
        squared_y = grid_y.ravel() ** 2
        rbf = np.exp(-(grid_x.ravel()**2 + grid_y.ravel()**2))
        sin_x_high = np.sin(grid_x.ravel() * 4 * np.pi)
        cos_y_high = np.cos(grid_y.ravel() * 4 * np.pi)
        
        X_pred = np.column_stack([
            base_features,
            sin_x,
            cos_y,
            interaction,
            squared_x,
            squared_y,
            rbf,
            sin_x_high,
            cos_y_high
        ])
        
        X_pred = self.scaler.transform(X_pred)
        X_pred = self.imputer.transform(X_pred)  # Use KNN imputation
        
        # Get predictions and uncertainty estimates
        predictions = self.model.predict(X_pred)
        _, uncertainty = self.gp_model.predict(X_pred, return_std=True)
        
        # Reshape predictions and uncertainty
        predictions = predictions.reshape(new_rows, new_cols)
        uncertainty = uncertainty.reshape(new_rows, new_cols)
        
        return predictions, uncertainty
    
    def get_metrics(self):
        """Return model performance metrics and uncertainty estimates."""
        return {
            'performance': self.metrics,
            'uncertainty': self.uncertainty
        }
