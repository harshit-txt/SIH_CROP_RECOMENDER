# Fasal Sathi — AI-Based Crop Recommendation System (SIH MVP)

## Run locally
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Backend runs at http://localhost:5000

Then open `frontend/index.html` directly in a browser (double-click it,
or use VS Code "Live Server"). It's a static file — no build step needed.

## API
- GET  /api/soil-defaults?state=Jharkhand
- POST /api/recommend   { season, soil_type, ph, n, p, k, moisture }
- GET  /api/health

## Notes for judges
- Scoring is an explainable rule-based engine (agronomic ranges + market
  economics + sustainability rules) — structured so each scoring function
  can be swapped for a trained Scikit-Learn model later.
- Regional soil defaults and market prices are mock/illustrative data for
  the MVP; production version would pull from Soil Health Card API,
  Agmarknet, and IMD weather data.
- UI supports English, Hindi, and Santali (transliterated for this demo).
