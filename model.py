import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import pickle
warnings.filterwarnings('ignore')


def load_dataset_from_csv(path='final_tool_wear_dataset.csv'):
    print(f"📥 Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded shape: {df.shape}")

    # Detect whether the dataset is already processed (contains encoded/material_removal_rate etc.)
    processed_markers = {'material_encoded', 'coolant_encoded', 'material_removal_rate', 'total_cutting_force'}
    if processed_markers.issubset(set(df.columns)):
        print("Detected processed dataset (contains engineered features and encodings). Skipping preprocessing.")
        return df, None, None

    # Otherwise, expect raw dataset with 'material_type' and 'coolant_type'
    if 'material_type' in df.columns and 'coolant_type' in df.columns:
        print("Detected raw dataset (categoricals present). Running preprocessing...")
        return preprocess_complete_dataset(df)

    raise ValueError("CSV does not appear to be either a processed dataset or raw dataset with 'material_type'/'coolant_type'.")


def preprocess_complete_dataset(df):
    """Comprehensive preprocessing and feature engineering.
    Returns: df_processed, le_material, le_coolant
    """
    print("\n📊 Data Preprocessing & Feature Engineering")
    print("="*50)

    df_processed = df.copy()

    # Encode categorical variables
    le_material = LabelEncoder()
    le_coolant = LabelEncoder()

    df_processed['material_encoded'] = le_material.fit_transform(df_processed['material_type'])
    df_processed['coolant_encoded'] = le_coolant.fit_transform(df_processed['coolant_type'])

    print(f"Material encoding: {dict(zip(le_material.classes_, le_material.transform(le_material.classes_)))}")
    print(f"Coolant encoding: {dict(zip(le_coolant.classes_, le_coolant.transform(le_coolant.classes_)))}")

    # Feature engineering
    df_processed['material_removal_rate'] = (df_processed['cutting_speed'] *
                                             df_processed['feed_rate'] *
                                             df_processed['depth_of_cut'])

    df_processed['total_cutting_force'] = np.sqrt(
        df_processed['cutting_force_x']**2 +
        df_processed['cutting_force_y']**2 +
        df_processed['cutting_force_z']**2
    )

    df_processed['power_to_force_ratio'] = df_processed['spindle_power'] / (df_processed['total_cutting_force'] + 1)
    df_processed['thermal_load'] = (df_processed['cutting_speed'] * df_processed['spindle_power']) / 1000
    df_processed['wear_intensity'] = (df_processed['cutting_speed'] *
                                     df_processed['material_hardness'] *
                                     df_processed['vibration_level']) / 10000

    # Remove original categorical columns if present
    if 'material_type' in df_processed.columns and 'coolant_type' in df_processed.columns:
        df_processed = df_processed.drop(['material_type', 'coolant_type'], axis=1)

    print(f"Features after engineering: {df_processed.shape[1]-1} (excluding target)")  # -1 for target

    # Save encoders for future inverse transforms (optional)
    with open('le_material.pkl', 'wb') as f:
        pickle.dump(le_material, f)
    with open('le_coolant.pkl', 'wb') as f:
        pickle.dump(le_coolant, f)
    print("Saved LabelEncoders: le_material.pkl, le_coolant.pkl")

    return df_processed, le_material, le_coolant


def train_cutting_tool_models(X, y):
    """Train and evaluate multiple models"""
    print("\n🤖 Model Training & Evaluation")
    print("="*50)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(X.columns)}")

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)
    }

    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")

        if name == 'Neural Network':
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        results[name] = {
            'model': model,
            'predictions': pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")

    return results, y_test, scaler


def perform_cross_validation(models_dict, X, y, cv=5):
    """Perform cross-validation"""
    print(f"\n🔄 {cv}-Fold Cross Validation")
    print("="*50)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for name, model_data in models_dict.items():
        model = model_data['model']

        if name == 'Neural Network':
            X_data = X_scaled
        else:
            X_data = X

        cv_scores = cross_val_score(model, X_data, y, cv=cv, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)

        print(f"{name}:")
        print(f"  CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")


def analyze_feature_importance(rf_model, feature_names):
    """Analyze and display feature importance"""
    print("\n🔍 Feature Importance Analysis")
    print("="*50)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 10 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")

    return importance_df


def create_visualizations(results, y_test, importance_df):
    """Create comprehensive visualizations"""
    print("\n📈 Creating Visualizations")
    print("="*50)

    # Set up the plotting environment
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))

    # 1. Model Performance Comparison
    ax1 = plt.subplot(3, 3, 1)
    models = list(results.keys())
    rmse_values = [results[model]['rmse'] for model in models]
    bars = ax1.bar(models, rmse_values)
    ax1.set_title('Model Performance - RMSE', fontweight='bold')
    ax1.set_ylabel('RMSE (mm)')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0005,
                f'{value:.4f}', ha='center', va='bottom')

    # 2. R² Score Comparison
    ax2 = plt.subplot(3, 3, 2)
    r2_values = [results[model]['r2'] for model in models]
    bars2 = ax2.bar(models, r2_values)
    ax2.set_title('Model Performance - R² Score', fontweight='bold')
    ax2.set_ylabel('R² Score')
    ax2.grid(True, alpha=0.3)

    for bar, value in zip(bars2, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom')

    # 3. Feature Importance
    ax3 = plt.subplot(3, 3, 3)
    top_features = importance_df.head(8)
    ax3.barh(range(len(top_features)), top_features['importance'])
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_title('Top 8 Feature Importance', fontweight='bold')
    ax3.set_xlabel('Importance')
    ax3.grid(True, alpha=0.3)

    # 4-6. Actual vs Predicted for each model
    for i, (model_name, model_data) in enumerate(results.items(), 4):
        ax = plt.subplot(3, 3, i)
        predictions = model_data['predictions']

        ax.scatter(y_test, predictions, alpha=0.6, s=30)
        min_val, max_val = y_test.min(), y_test.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        ax.set_xlabel('Actual Tool Wear (mm)')
        ax.set_ylabel('Predicted Tool Wear (mm)')
        ax.set_title(f'{model_name}\nR² = {model_data["r2"]:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 7. Residuals plot for best model
    ax7 = plt.subplot(3, 3, 7)
    best_model = max(results.keys(), key=lambda x: results[x]['r2'])
    residuals = y_test - results[best_model]['predictions']
    ax7.scatter(results[best_model]['predictions'], residuals, alpha=0.6)
    ax7.axhline(y=0, color='r', linestyle='--')
    ax7.set_xlabel('Predicted Tool Wear (mm)')
    ax7.set_ylabel('Residuals (mm)')
    ax7.set_title(f'Residual Plot - {best_model}', fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # 8. Error distribution
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Residuals (mm)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Residual Distribution', fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # 9. Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = f"""
    🎯 PROJECT SUMMARY
    ==================

    📊 Dataset: {len(y_test)*5} samples
    🔧 Features: {len(importance_df)} engineered
    🤖 Models: {len(results)} algorithms

    🏆 Best Model: {best_model}
    📈 Best R²: {results[best_model]['r2']:.4f}
    📉 Best RMSE: {results[best_model]['rmse']:.4f} mm

    💡 Key Insights:
    • {importance_df.iloc[0]['feature']} most important
    • Excellent prediction accuracy achieved
    • Ready for deployment

    ✅ Status: COMPLETED
    """

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.suptitle('Cutting Tool Wear Prediction - Complete Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('tool_wear_complete_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Comprehensive analysis visualization created!")


def predict_tool_wear_example(best_model, scaler, X_features, model_name, le_material=None, le_coolant=None):
    """Demonstrate prediction on sample data. Accepts optional LabelEncoders to inverse map labels."""
    print("\n🔮 Tool Wear Prediction Example")
    print("="*50)

    # Create sample machining parameters
    sample_data = pd.DataFrame({
        'cutting_speed': [200, 350, 150],
        'feed_rate': [0.1, 0.2, 0.08],
        'depth_of_cut': [1.5, 2.5, 1.0],
        'material_hardness': [250, 320, 180],
        'material_encoded': [2, 0, 1],
        'coolant_encoded': [1, 0, 2],
        'tool_nose_radius': [0.8, 1.2, 0.6],
        'cutting_force_x': [200, 400, 150],
        'cutting_force_y': [300, 600, 200],
        'cutting_force_z': [500, 900, 350],
        'spindle_power': [8, 18, 5],
        'vibration_level': [1.2, 2.0, 0.8],
        'machining_time': [30, 90, 20],
        'surface_roughness': [2.5, 4.5, 1.8]
    })

    # Add engineered features
    sample_data['material_removal_rate'] = (sample_data['cutting_speed'] *
                                           sample_data['feed_rate'] *
                                           sample_data['depth_of_cut'])

    sample_data['total_cutting_force'] = np.sqrt(
        sample_data['cutting_force_x']**2 +
        sample_data['cutting_force_y']**2 +
        sample_data['cutting_force_z']**2
    )

    sample_data['power_to_force_ratio'] = sample_data['spindle_power'] / (sample_data['total_cutting_force'] + 1)
    sample_data['thermal_load'] = (sample_data['cutting_speed'] * sample_data['spindle_power']) / 1000
    sample_data['wear_intensity'] = (sample_data['cutting_speed'] *
                                   sample_data['material_hardness'] *
                                   sample_data['vibration_level']) / 10000

    # Ensure same feature order as training and that all features exist
    missing_cols = [c for c in X_features.columns if c not in sample_data.columns]
    if missing_cols:
        raise ValueError(f"Sample data missing these training features: {missing_cols}")

    sample_data = sample_data[X_features.columns]

    # Ensure encoded columns are integers
    if 'material_encoded' in sample_data.columns:
        sample_data['material_encoded'] = sample_data['material_encoded'].astype(int)
    if 'coolant_encoded' in sample_data.columns:
        sample_data['coolant_encoded'] = sample_data['coolant_encoded'].astype(int)

    # Make predictions
    if model_name == 'Neural Network':
        sample_scaled = scaler.transform(sample_data)
        predictions = best_model.predict(sample_scaled)
    else:
        predictions = best_model.predict(sample_data)

    print("Sample Predictions:")

    for i, pred in enumerate(predictions):
        mat_idx = int(sample_data.iloc[i]['material_encoded']) if 'material_encoded' in sample_data.columns else None
        cool_idx = int(sample_data.iloc[i]['coolant_encoded']) if 'coolant_encoded' in sample_data.columns else None

        # Map back to labels when encoders available
        if le_material is not None and mat_idx is not None:
            material_label = le_material.inverse_transform([mat_idx])[0]
        else:
            # fallback order (if you know the mapping, adjust accordingly)
            material_label = mat_idx

        if le_coolant is not None and cool_idx is not None:
            coolant_label = le_coolant.inverse_transform([cool_idx])[0]
        else:
            coolant_label = cool_idx

        print(f"Sample {i+1}:")
        print(f"  Material: {material_label}")
        print(f"  Coolant: {coolant_label}")
        print(f"  Cutting Speed: {sample_data.iloc[i]['cutting_speed']:.0f} m/min")
        print(f"  Feed Rate: {sample_data.iloc[i]['feed_rate']:.2f} mm/rev")
        print(f"  → Predicted Tool Wear: {pred:.4f} mm")

        if pred > 0.3:
            print(f"  ⚠️  HIGH WEAR - Consider tool replacement!")
        elif pred > 0.2:
            print(f"  🟡 MODERATE WEAR - Monitor closely")
        else:
            print(f"  ✅ LOW WEAR - Tool in good condition")
        print()


def main_execution(csv_path='final_tool_wear_dataset.csv'):
    """Complete project execution reading dataset from CSV"""
    print("🔧 CUTTING TOOL WEAR PREDICTION PROJECT")
    print("🎯 Complete ML System for Predictive Maintenance")
    print("="*60)

    try:
        # Step 1: Load dataset from CSV (auto-detect processed/raw)
        df_processed, le_material, le_coolant = load_dataset_from_csv(csv_path)

        # Step 2: Prepare features and target
        if 'tool_wear' not in df_processed.columns:
            raise ValueError("Target column 'tool_wear' not found in dataset.")

        X = df_processed.drop('tool_wear', axis=1)
        y = df_processed['tool_wear']

        # Step 3: Train models
        results, y_test, scaler = train_cutting_tool_models(X, y)

        # Step 4: Cross-validation
        perform_cross_validation(results, X, y)

        # Step 5: Feature importance
        rf_importance = analyze_feature_importance(results['Random Forest']['model'], X.columns)

        # Step 6: Create visualizations
        create_visualizations(results, y_test, rf_importance)

        # Step 7: Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_model = results[best_model_name]['model']

        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   R² Score: {results[best_model_name]['r2']:.4f}")
        print(f"   RMSE: {results[best_model_name]['rmse']:.4f} mm")

        # Step 8: Prediction demonstration
        predict_tool_wear_example(best_model, scaler, X, best_model_name, le_material=le_material, le_coolant=le_coolant)

        # Step 9: (Optional) Save processed dataset back
        df_processed.to_csv('final_tool_wear_dataset_processed.csv', index=False)
        print("\n💾 Processed results saved to 'final_tool_wear_dataset_processed.csv'")

        # Project summary
        print("\n" + "="*60)
        print("✅ PROJECT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("🎯 Achievements:")
        print(f"   • Dataset: {len(df_processed)} samples with realistic machining relationships")
        print(f"   • Features: {len(X.columns)} engineered parameters")
        print(f"   • Models: {len(results)} algorithms tested and evaluated")
        print(f"   • Best Performance: R² = {results[best_model_name]['r2']:.4f}")
        print(f"   • Prediction Accuracy: RMSE = {results[best_model_name]['rmse']:.4f} mm")

        print("\n🚀 Next Steps:")
        print("   • Integrate with real CNC machining data")
        print("   • Deploy model for real-time monitoring")
        print("   • Implement automated alerts for tool replacement")
        print("   • Optimize machining parameters based on predictions")

        return {
            'dataset': df_processed,
            'models': results,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'feature_importance': rf_importance,
            'scaler': scaler,
            'le_material': le_material,
            'le_coolant': le_coolant
        }

    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        return None


if __name__ == "__main__":
    # Configure matplotlib
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100

    # Execute complete project (in Colab set path or upload file first)
    # Example for Colab: run the following before executing this script:
    # from google.colab import files
    # uploaded = files.upload()  # then upload 'final_tool_wear_dataset.csv'

    project_results = main_execution(csv_path='final_tool_wear_dataset.csv')

    if project_results:
        print("\n🔧 Project objects available for further use:")
        print("   • project_results['best_model'] - Best trained model")
        print("   • project_results['dataset'] - Final processed dataset")
        print("   • project_results['feature_importance'] - Feature analysis")
        print("   • project_results['scaler'] - Data scaler for new predictions")
