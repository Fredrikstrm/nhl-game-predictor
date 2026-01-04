# GitHub Pages Setup Guide

This guide explains how to set up automated inference and dashboard deployment to GitHub Pages.

## Overview

The GitHub Actions workflow (`.github/workflows/inference-and-deploy.yml`) will:
1. **Run daily at 8 AM UTC** (or manually via workflow_dispatch)
2. **Execute inference pipeline** to generate new predictions
3. **Generate static HTML dashboard** with embedded predictions
4. **Deploy to GitHub Pages** automatically

## Setup Steps

### 1. Add GitHub Secret

Go to your GitHub repository → Settings → Secrets and variables → Actions → New repository secret

Add this secret:
- `HOPSWORKS_API_KEY` - Your Hopsworks API key

**Note:** The project name and host are read from `config.yaml` (under the `hopsworks` section). Make sure to update `config.yaml` with your Hopsworks project name before pushing.

### 2. Enable GitHub Pages

1. Go to repository Settings → Pages
2. Under "Source", select: **Deploy from a branch**
3. Branch: **gh-pages** (will be created automatically by the workflow)
4. Folder: **/ (root)**
5. Click Save

### 3. Test Locally (Optional)

Before pushing, you can test the static dashboard generation:

```bash
# Make sure you have predictions
python src/inference_pipeline/main.py

# Generate static dashboard
python scripts/generate_static_dashboard.py

# View locally (optional)
open docs/index.html
```

### 4. Push to GitHub

```bash
git add .
git commit -m "Add GitHub Actions workflow for inference and dashboard deployment"
git push origin main
```

### 5. Trigger Workflow

The workflow will run:
- **Automatically** on push to `main` branch (if inference pipeline files changed)
- **Daily at 8 AM UTC** via schedule
- **Manually** via GitHub UI: Actions → "Run Inference and Deploy Dashboard" → Run workflow

## Dashboard URL

Once deployed, your dashboard will be available at:
```
https://<your-username>.github.io/<repository-name>/
```

For example: `https://fredrikstrom.github.io/nhl-predictor/`

## How It Works

1. **Static HTML Generation**: The `scripts/generate_static_dashboard.py` script:
   - Reads your beautiful `index.html` template
   - Embeds predictions as JSON in a `<script>` tag
   - Embeds metadata (model info, course info, etc.)
   - Replaces API calls with direct data access
   - Saves to `docs/index.html`

2. **GitHub Pages**: Serves the static HTML file with all CSS/JS embedded, so it looks exactly like your localhost version!

3. **Auto-updates**: The workflow runs daily, so your dashboard stays current with new predictions.

## Troubleshooting

- **Workflow fails**: Check Actions tab for error logs
- **No predictions**: Make sure Hopsworks secrets are set correctly
- **Dashboard not updating**: Check if workflow ran successfully in Actions tab
- **Styling broken**: The static HTML includes all CSS inline, so it should work

## Manual Deployment

If you want to manually update the dashboard:

```bash
# Run inference
python src/inference_pipeline/main.py

# Generate static dashboard
python scripts/generate_static_dashboard.py

# Commit and push
git add docs/index.html
git commit -m "Update dashboard"
git push origin main
```

The GitHub Actions workflow will then deploy it automatically.

