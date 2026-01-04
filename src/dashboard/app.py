"""
Dashboard Web Application
Simple Flask dashboard to display NHL game predictions
"""
import sys
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, jsonify
from flask_cors import CORS

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config_loader import load_config

app = Flask(__name__)
CORS(app)


def load_predictions():
    """Load predictions from CSV file"""
    predictions_path = Path(__file__).parent.parent.parent / "data" / "predictions.csv"
    
    if not predictions_path.exists():
        return pd.DataFrame()
    
    return pd.read_csv(predictions_path)


@app.route('/')
def index():
    """Main dashboard page"""
    config = load_config()
    # Get metadata for display
    metadata = {
        'prediction_horizon_days': config['features']['prediction_horizon_days'],
        'model_type': config['training']['model_type'].upper(),
        'model_version': config['model_registry'].get('model_version', 'latest'),
        'course_name': 'Scalable Machine Learning',
        'course_code': 'ID2223'
    }
    return render_template('index.html', **metadata)

@app.route('/api/metadata')
def get_metadata():
    config = load_config()
    
        # Try to get feature count from model
    feature_count = 25      
    try:
        import joblib
        from pathlib import Path
        model_path = Path(__file__).parent.parent.parent / "models" / config['model_registry']['model_name'] / "model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            if hasattr(model, 'feature_names_in_'):
                feature_count = len(model.feature_names_in_)
            elif hasattr(model, 'get_booster'):
                booster = model.get_booster()
                if hasattr(booster, 'feature_names') and booster.feature_names:
                    feature_count = len(booster.feature_names)
    except Exception:
        pass
    
    return jsonify({
        'prediction_horizon_days': config['features']['prediction_horizon_days'],
        'model_type': config['training']['model_type'].upper(),
        'model_version': str(config['model_registry'].get('model_version', 'latest')),
        'feature_count': feature_count,
        'course_name': 'Scalable Machine Learning',
        'course_code': 'ID2223',
        'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    })


@app.route('/api/predictions')
def get_predictions():
    """API endpoint to get predictions"""
    predictions_df = load_predictions()
    
    if predictions_df.empty:
        return jsonify({
            'error': 'No predictions available. Please run the inference pipeline first.'
        }), 404
    
    predictions = predictions_df.to_dict('records')
    
    return jsonify({
        'predictions': predictions,
        'count': len(predictions)
    })


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


def main():
    """Run the dashboard server"""
    config = load_config()
    dashboard_config = config['dashboard']
    
    app.run(
        host=dashboard_config['host'],
        port=dashboard_config['port'],
        debug=dashboard_config['debug']
    )


if __name__ == "__main__":
    main()

