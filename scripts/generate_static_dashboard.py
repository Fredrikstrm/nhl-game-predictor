"""
Generate static HTML dashboard for GitHub Pages
Reads predictions CSV and template HTML, generates static version with embedded data
"""
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from config.config_loader import load_config


def generate_static_dashboard():
    """Generate static HTML dashboard with embedded predictions"""
    project_root = Path(__file__).parent.parent
    predictions_path = project_root / "data" / "predictions.csv"
    template_path = project_root / "src" / "dashboard" / "templates" / "index.html"
    output_path = project_root / "docs" / "index.html"
    
    # Load config
    config = load_config()
    
    # Load predictions
    if not predictions_path.exists():
        print(f"Warning: No predictions file found at {predictions_path}")
        predictions_df = pd.DataFrame()
    else:
        predictions_df = pd.read_csv(predictions_path)
        print(f"Loaded {len(predictions_df)} predictions from {predictions_path}")
    
    # Read template HTML
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found at {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Try to get feature count from model if available
    feature_count = 25  # Default
    try:
        import joblib
        model_path = project_root / "models" / config['model_registry']['model_name'] / "model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            if hasattr(model, 'feature_names_in_'):
                feature_count = len(model.feature_names_in_)
            elif hasattr(model, 'get_booster'):
                booster = model.get_booster()
                if hasattr(booster, 'feature_names'):
                    feature_count = len(booster.feature_names) if booster.feature_names else 25
    except Exception as e:
        print(f"Could not load model to get feature count: {e}, using default 25")
    
    # Prepare metadata
    metadata = {
        'prediction_horizon_days': config['features']['prediction_horizon_days'],
        'model_type': config['training']['model_type'].upper(),
        'model_version': str(config['model_registry'].get('model_version', 'latest')),
        'feature_count': feature_count,
        'course_name': 'Scalable Machine Learning',
        'course_code': 'ID2223',
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    }
    
    # Convert predictions to JSON
    if len(predictions_df) > 0:
        predictions_data = {
            'predictions': predictions_df.to_dict('records'),
            'count': len(predictions_df)
        }
    else:
        predictions_data = {
            'predictions': [],
            'count': 0,
            'error': 'No predictions available'
        }
    
    # Embed data in script tags
    embedded_data_script = f"""
    <script>
        // Embedded predictions data (for static GitHub Pages)
        const embeddedPredictions = {json.dumps(predictions_data, indent=8)};
        const embeddedMetadata = {json.dumps(metadata, indent=8)};
        
        // Override loadPredictions to use embedded data
        const originalLoadPredictions = loadPredictions;
        loadPredictions = function() {{
            const loadingEl = document.getElementById('loading');
            const errorEl = document.getElementById('error');
            const gamesContainer = document.getElementById('games-container');
            const statsEl = document.getElementById('stats');
            
            loadingEl.style.display = 'block';
            errorEl.style.display = 'none';
            gamesContainer.innerHTML = '';
            
            // Simulate async for consistency
            setTimeout(() => {{
                loadingEl.style.display = 'none';
                
                if (embeddedPredictions.error) {{
                    errorEl.textContent = `Error: ${{embeddedPredictions.error}}`;
                    errorEl.style.display = 'block';
                    return;
                }}
                
                if (embeddedPredictions.predictions && embeddedPredictions.predictions.length > 0) {{
                    statsEl.textContent = `Showing ${{embeddedPredictions.count}} upcoming game${{embeddedPredictions.count !== 1 ? 's' : ''}}`;
                    gamesContainer.innerHTML = embeddedPredictions.predictions.map(createGameCard).join('');
                }} else {{
                    gamesContainer.innerHTML = '<div class="error">No upcoming games found</div>';
                }}
            }}, 100);
        }};
        
        // Override loadMetadata to use embedded data
        const originalLoadMetadata = loadMetadata;
        loadMetadata = function() {{
            document.getElementById('horizon-days').textContent = embeddedMetadata.prediction_horizon_days;
            document.getElementById('model-info').textContent = `${{embeddedMetadata.model_type}} v${{embeddedMetadata.model_version}}`;
            document.getElementById('feature-count').textContent = embeddedMetadata.feature_count;
            document.getElementById('last-updated').textContent = embeddedMetadata.last_updated;
        }};
    </script>
    """
    
    # Insert embedded data script before closing body tag
    html = html.replace('</body>', embedded_data_script + '\n</body>')
    
    # Update refresh button to work in static mode (reloads embedded data)
    html = html.replace(
        '<button class="refresh-btn" onclick="loadPredictions()">🔄 Refresh Predictions</button>',
        '<button class="refresh-btn" onclick="loadPredictions(); loadMetadata();">🔄 Refresh Predictions</button>'
    )
    
    # Remove auto-refresh interval (not needed for static page)
    html = html.replace(
        "// Auto-refresh every 5 minutes\n        setInterval(loadPredictions, 5 * 60 * 1000);",
        "// Static page - refresh button manually reloads embedded data"
    )
    
    # Create docs directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write static HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Generated static dashboard at {output_path}")
    print(f"   - {len(predictions_df)} predictions embedded")
    print(f"   - Model: {metadata['model_type']} v{metadata['model_version']}")
    print(f"   - Features: {metadata['feature_count']}")
    return output_path


if __name__ == "__main__":
    generate_static_dashboard()

