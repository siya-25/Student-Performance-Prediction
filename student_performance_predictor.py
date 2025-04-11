import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import logging
import os

class StudentPerformancePredictor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.pca = None
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.model_metrics = {}
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.feature_names = None
        self.numeric_columns = None  # Will store actual numeric column names after preprocessing
        self.categorical_columns = None  # Will store actual categorical column names after preprocessing
        # Define expected column names exactly as in CSV
        self.numeric_features = ['Age', 'StudyHoursPerWeek', 'AttendanceRate']
        self.categorical_features = ['Gender', 'Major']
        self.binary_features = ['PartTimeJob', 'ExtraCurricularActivities']
        self.target_column = 'GPA'
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_and_preprocess_data(self):
        try:
            # Load data
            self.logger.info(f"Loading data from {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            # Log initial data shape and columns
            self.logger.info(f"Initial data shape: {df.shape}")
            self.logger.info(f"Initial columns: {df.columns.tolist()}")
            
            # Drop StudentID if present
            if 'StudentID' in df.columns:
                df = df.drop('StudentID', axis=1)
            
            # Ensure column names are consistent
            df.columns = df.columns.str.strip()
            
            # Separate target variable early
            y = df[self.target_column].copy()
            X = df.drop(self.target_column, axis=1)
            
            # Convert numeric columns to float
            for col in self.numeric_features:
                if col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                else:
                    self.logger.warning(f"Missing expected numeric column: {col}")
            
            # Handle missing values
            X = self.handle_missing_values(X)
            
            # Feature engineering
            X = self.perform_feature_engineering(X)
            
            # Store feature names (excluding target)
            self.feature_names = X.columns.tolist()
            
            # Verify we still have data
            if len(X) == 0:
                raise ValueError("No valid data remaining after preprocessing")
            
            # Log data shapes and features
            self.logger.info(f"Features shape: {X.shape}")
            self.logger.info(f"Target shape: {y.shape}")
            self.logger.info(f"Feature names: {self.feature_names}")
            
            # Encode categorical variables
            X = self.encode_categorical_features(X)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            # Apply PCA
            self.apply_pca()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def handle_missing_values(self, df):
        try:
            # Log initial missing values
            missing_values = df.isna().sum()
            self.logger.info(f"Initial missing values:\n{missing_values}")
            
            # Identify numeric and categorical columns
            self.numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            self.categorical_columns = df.select_dtypes(include=['object']).columns
            
            # Fill missing numeric values with median
            if len(self.numeric_columns) > 0:
                self.numeric_imputer.fit(df[self.numeric_columns])
                df[self.numeric_columns] = self.numeric_imputer.transform(df[self.numeric_columns])
            
            # Fill missing categorical values with mode
            if len(self.categorical_columns) > 0:
                self.categorical_imputer.fit(df[self.categorical_columns])
                df[self.categorical_columns] = self.categorical_imputer.transform(df[self.categorical_columns])
            
            # Log remaining missing values
            remaining_missing = df.isna().sum()
            self.logger.info(f"Remaining missing values after imputation:\n{remaining_missing}")
            
            # Drop rows with NaN values only if absolutely necessary
            if df.isna().any().any():
                self.logger.warning("Some NaN values remain after imputation. Dropping rows with NaN values.")
                initial_rows = len(df)
                df = df.dropna()
                dropped_rows = initial_rows - len(df)
                self.logger.info(f"Dropped {dropped_rows} rows with NaN values")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise

    def perform_feature_engineering(self, df):
        try:
            # Log initial data shape
            self.logger.info(f"Starting feature engineering with shape: {df.shape}")
            
            # Convert binary columns to 0/1
            for col in self.binary_features:
                if col in df.columns:
                    df[col] = df[col].map({'Yes': 1, 'No': 0})
                    # Fill NaN values with 0 (assuming No)
                    df[col] = df[col].fillna(0)
            
            # Create interaction features
            if all(col in df.columns for col in ['StudyHoursPerWeek', 'AttendanceRate']):
                df['study_attendance_ratio'] = df['StudyHoursPerWeek'] * df['AttendanceRate'] / 100.0
            
            # Log final data shape
            self.logger.info(f"Final data shape after feature engineering: {df.shape}")
            self.logger.info(f"Final columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise

    def encode_categorical_features(self, X):
        try:
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            return X
        except Exception as e:
            self.logger.error(f"Error encoding categorical features: {str(e)}")
            raise

    def apply_pca(self):
        try:
            # Verify no NaN values before PCA
            if np.isnan(self.X_train_scaled).any():
                self.logger.error("NaN values found in scaled features. Cannot apply PCA.")
                raise ValueError("NaN values in scaled features")
            
            # Determine optimal number of components
            pca = PCA()
            pca.fit(self.X_train_scaled)
            
            # Find number of components that explain 95% of variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= 0.95) + 1
            
            # Apply PCA
            self.pca = PCA(n_components=n_components)
            self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
            self.X_test_pca = self.pca.transform(self.X_test_scaled)
            
            self.logger.info(f"Applied PCA with {n_components} components")
            
        except Exception as e:
            self.logger.error(f"Error applying PCA: {str(e)}")
            raise

    def train_and_evaluate_models(self):
        try:
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=1.0),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'SVR': SVR(kernel='rbf'),
                'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            }
            
            best_score = float('-inf')
            
            for name, model in models.items():
                try:
                    # Train model
                    model.fit(self.X_train_pca, self.y_train)
                    
                    # Make predictions
                    y_pred = model.predict(self.X_test_pca)
                    
                    # Calculate metrics
                    metrics = {
                        'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                        'MAE': mean_absolute_error(self.y_test, y_pred),
                        'R2': r2_score(self.y_test, y_pred)
                    }
                    
                    # Store metrics
                    self.model_metrics[name] = metrics
                    
                    # Update best model
                    if metrics['R2'] > best_score:
                        best_score = metrics['R2']
                        self.best_model = model
                        self.best_model_name = name
                    
                    self.logger.info(f"{name} - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
            
            # Get feature importance if available
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                self.feature_importance = self.best_model.coef_
            
            self.logger.info(f"Best model: {self.best_model_name} with R2: {best_score:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def predict(self, features):
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            # Log input features
            self.logger.info(f"Input features: {features}")
            
            # Convert numeric features
            for col in self.numeric_features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert binary features
            for col in self.binary_features:
                if col in df.columns:
                    df[col] = df[col].map({'Yes': 1, 'No': 0})
                    df[col] = df[col].fillna(0)
            
            # Create interaction features
            if all(col in df.columns for col in ['StudyHoursPerWeek', 'AttendanceRate']):
                df['study_attendance_ratio'] = df['StudyHoursPerWeek'] * df['AttendanceRate'] / 100.0
            
            # Handle missing values using the fitted imputers
            numeric_cols = [col for col in self.numeric_columns if col in df.columns]
            categorical_cols = [col for col in self.categorical_columns if col in df.columns]
            
            if len(numeric_cols) > 0:
                df[numeric_cols] = self.numeric_imputer.transform(df[numeric_cols])
            if len(categorical_cols) > 0:
                df[categorical_cols] = self.categorical_imputer.transform(df[categorical_cols])
            
            # Encode categorical features
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    df[col] = encoder.transform([str(df[col].iloc[0])])
            
            # Ensure all feature columns are present and in correct order
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match training data
            df = df[self.feature_names]  # Only use feature_names, which excludes target
            
            # Log preprocessed features
            self.logger.info(f"Preprocessed features shape: {df.shape}")
            self.logger.info(f"Preprocessed columns: {df.columns.tolist()}")
            
            # Scale features
            scaled_features = self.scaler.transform(df)
            
            # Apply PCA
            pca_features = self.pca.transform(scaled_features)
            
            # Make prediction
            prediction = self.best_model.predict(pca_features)[0]
            
            # Calculate risk level based on GPA
            if prediction >= 3.5:
                risk_level = "low"
            elif prediction >= 2.5:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            return {
                'prediction': prediction,
                'risk_level': risk_level,
                'model_used': self.best_model_name,
                'confidence': self.calculate_confidence(prediction),
                'metrics': self.model_metrics[self.best_model_name]
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise

    def calculate_confidence(self, prediction):
        # Calculate confidence based on model's performance metrics
        metrics = self.model_metrics[self.best_model_name]
        r2 = metrics['R2']
        
        # Scale confidence based on R2 score
        confidence = min(100, max(0, (r2 + 1) * 50))
        return confidence

    def save_model(self, path='model.joblib'):
        try:
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'pca': self.pca,
                'label_encoders': self.label_encoders,
                'metrics': self.model_metrics,
                'feature_importance': self.feature_importance,
                'numeric_imputer': self.numeric_imputer,
                'categorical_imputer': self.categorical_imputer,
                'feature_names': self.feature_names,
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
                'binary_features': self.binary_features,
                'numeric_columns': self.numeric_columns,
                'categorical_columns': self.categorical_columns
            }
            joblib.dump(model_data, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path='model.joblib'):
        try:
            model_data = joblib.load(path)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.label_encoders = model_data['label_encoders']
            self.model_metrics = model_data['metrics']
            self.feature_importance = model_data['feature_importance']
            self.numeric_imputer = model_data['numeric_imputer']
            self.categorical_imputer = model_data['categorical_imputer']
            self.feature_names = model_data['feature_names']
            self.numeric_features = model_data['numeric_features']
            self.categorical_features = model_data['categorical_features']
            self.binary_features = model_data['binary_features']
            self.numeric_columns = model_data['numeric_columns']
            self.categorical_columns = model_data['categorical_columns']
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

def main():
    try:
        # Example usage
        csv_path = r"C:\Users\siyap\OneDrive\Desktop\FDS\student_performance_data.csv"
        predictor = StudentPerformancePredictor(csv_path)
        
        # Load and preprocess data
        predictor.load_and_preprocess_data()
        
        # Train and evaluate models
        predictor.train_and_evaluate_models()
        
        # Save the best model
        predictor.save_model()
        
        # Example prediction
        sample_features = {
            'Gender': 'Male',
            'Age': 24,
            'StudyHoursPerWeek': 37,
            'AttendanceRate': 90.75,
            'Major': 'Arts',
            'PartTimeJob': 'Yes',
            'ExtraCurricularActivities': 'No'
        }
        
        result = predictor.predict(sample_features)
        print("\nPrediction Results:")
        print(f"Predicted GPA: {result['prediction']:.2f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Model Used: {result['model_used']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nModel Metrics:")
        print(f"R² Score: {result['metrics']['R2']:.4f}")
        print(f"RMSE: {result['metrics']['RMSE']:.4f}")
        print(f"MAE: {result['metrics']['MAE']:.4f}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 