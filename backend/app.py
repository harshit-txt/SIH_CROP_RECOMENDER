"""
AI-Based Crop Recommendation System — Backend (MVP)
====================================================
Smart India Hackathon — Pre-Qualification Prototype

Stack: Python + Flask (lightweight, zero-config, easy to demo/deploy)

This backend exposes two endpoints:
  1. GET  /api/soil-defaults?state=<state>   -> auto-fill regional soil/climate data
  2. POST /api/recommend                     -> ranked crop recommendations

The "AI" here is an explainable, rule-based scoring engine (a common and fully
acceptable approach for an MVP / pre-qualification round). It is structured so
that `calculate_suitability_score()`, `calculate_profitability_score()`, and
`calculate_soil_health_score()` can each be swapped later for a trained
Scikit-Learn model (e.g. RandomForestRegressor on real yield/soil datasets)
without changing the API contract — this is worth mentioning live to judges.

Run locally:
    pip install flask
    python app.py
Server starts at http://localhost:5000
"""

import os
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)


# ---------------------------------------------------------------------------
# CORS — allow the frontend (served from file:// or any localhost port) to
# call this API without needing the flask-cors package as a dependency.
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ---------------------------------------------------------------------------
# 1. REGIONAL SOIL & CLIMATE DEFAULTS DATABASE (mock — replace with a real
#    Soil Health Card API / govt. dataset in production)
# ---------------------------------------------------------------------------
REGIONAL_DEFAULTS = {
    "Jharkhand":     {"soil_type": "Red & Laterite", "ph": 5.8, "n": 40, "p": 22, "k": 35, "moisture": 45},
    "Bihar":         {"soil_type": "Alluvial",        "ph": 6.8, "n": 55, "p": 30, "k": 40, "moisture": 55},
    "West Bengal":   {"soil_type": "Alluvial",        "ph": 6.5, "n": 60, "p": 28, "k": 42, "moisture": 65},
    "Odisha":        {"soil_type": "Laterite",        "ph": 5.9, "n": 42, "p": 20, "k": 32, "moisture": 58},
    "Uttar Pradesh": {"soil_type": "Alluvial",        "ph": 7.2, "n": 65, "p": 35, "k": 45, "moisture": 48},
    "Punjab":        {"soil_type": "Alluvial",        "ph": 7.5, "n": 70, "p": 38, "k": 50, "moisture": 50},
    "Maharashtra":   {"soil_type": "Black",           "ph": 7.0, "n": 50, "p": 25, "k": 55, "moisture": 40},
    "Madhya Pradesh":{"soil_type": "Black",           "ph": 7.1, "n": 48, "p": 24, "k": 52, "moisture": 42},
    "Rajasthan":     {"soil_type": "Sandy",           "ph": 7.8, "n": 25, "p": 15, "k": 20, "moisture": 22},
    "Karnataka":     {"soil_type": "Red",             "ph": 6.3, "n": 45, "p": 22, "k": 38, "moisture": 46},
}


# ---------------------------------------------------------------------------
# 2. CROP KNOWLEDGE BASE (mock agronomic + market dataset)
#    NPK values = kg/ha ideal midpoint range. Prices/yield are illustrative
#    (INR) — in production, wire market_price_per_ton to a live mandi/
#    Agmarknet price-feed API.
# ---------------------------------------------------------------------------
CROP_DB = [
    {
        "id": "rice", "name": "Rice", "name_hi": "धान", "name_sat": "बाबा दा (Baba Da)",
        "season": ["Kharif"], "soil_types": ["Alluvial", "Red & Laterite", "Clay", "Laterite"],
        "ph_range": (5.5, 7.0), "n_range": (40, 80), "p_range": (20, 40), "k_range": (30, 50),
        "moisture_range": (60, 90), "water_requirement": "high",
        "yield_per_acre_ton": 2.2, "market_price_per_ton": 21000, "cost_per_acre": 18000,
        "nitrogen_fixing": False, "rotation_group": "cereal", "soil_impact_base": 45,
    },
    {
        "id": "maize", "name": "Maize", "name_hi": "मक्का", "name_sat": "जनुम (Janum)",
        "season": ["Kharif", "Rabi"], "soil_types": ["Red & Laterite", "Alluvial", "Loamy", "Black"],
        "ph_range": (5.5, 7.5), "n_range": (50, 90), "p_range": (20, 35), "k_range": (25, 40),
        "moisture_range": (40, 65), "water_requirement": "medium",
        "yield_per_acre_ton": 2.6, "market_price_per_ton": 19000, "cost_per_acre": 14000,
        "nitrogen_fixing": False, "rotation_group": "cereal", "soil_impact_base": 55,
    },
    {
        "id": "ragi", "name": "Ragi (Finger Millet)", "name_hi": "रागी / मड़ुआ", "name_sat": "मडुवा (Maduwa)",
        "season": ["Kharif"], "soil_types": ["Red & Laterite", "Red", "Laterite", "Sandy"],
        "ph_range": (5.0, 7.0), "n_range": (20, 40), "p_range": (10, 20), "k_range": (15, 30),
        "moisture_range": (30, 50), "water_requirement": "low",
        "yield_per_acre_ton": 1.1, "market_price_per_ton": 33000, "cost_per_acre": 7000,
        "nitrogen_fixing": False, "rotation_group": "cereal", "soil_impact_base": 70,
    },
    {
        "id": "arhar", "name": "Arhar / Tur (Pigeon Pea)", "name_hi": "अरहर / तूर", "name_sat": "हड़ाम दाल (Haram Dal)",
        "season": ["Kharif"], "soil_types": ["Red & Laterite", "Black", "Red", "Loamy"],
        "ph_range": (5.5, 7.5), "n_range": (15, 30), "p_range": (20, 35), "k_range": (15, 30),
        "moisture_range": (35, 55), "water_requirement": "low",
        "yield_per_acre_ton": 0.9, "market_price_per_ton": 65000, "cost_per_acre": 9000,
        "nitrogen_fixing": True, "rotation_group": "legume", "soil_impact_base": 85,
    },
    {
        "id": "groundnut", "name": "Groundnut", "name_hi": "मूंगफली", "name_sat": "बादाम गाड़ा (Badam Gada)",
        "season": ["Kharif"], "soil_types": ["Red", "Sandy", "Red & Laterite", "Loamy"],
        "ph_range": (5.5, 7.0), "n_range": (15, 30), "p_range": (25, 40), "k_range": (25, 40),
        "moisture_range": (35, 55), "water_requirement": "low",
        "yield_per_acre_ton": 1.3, "market_price_per_ton": 58000, "cost_per_acre": 16000,
        "nitrogen_fixing": True, "rotation_group": "legume", "soil_impact_base": 80,
    },
    {
        "id": "wheat", "name": "Wheat", "name_hi": "गेहूं", "name_sat": "गहूम (Gahum)",
        "season": ["Rabi"], "soil_types": ["Alluvial", "Loamy", "Black"],
        "ph_range": (6.0, 7.5), "n_range": (60, 100), "p_range": (30, 50), "k_range": (25, 40),
        "moisture_range": (35, 55), "water_requirement": "medium",
        "yield_per_acre_ton": 1.8, "market_price_per_ton": 22500, "cost_per_acre": 15000,
        "nitrogen_fixing": False, "rotation_group": "cereal", "soil_impact_base": 50,
    },
    {
        "id": "mustard", "name": "Mustard", "name_hi": "सरसों", "name_sat": "सोरोसोम (Sorosom)",
        "season": ["Rabi"], "soil_types": ["Alluvial", "Loamy", "Red & Laterite", "Sandy"],
        "ph_range": (6.0, 7.5), "n_range": (30, 55), "p_range": (15, 30), "k_range": (15, 30),
        "moisture_range": (25, 45), "water_requirement": "low",
        "yield_per_acre_ton": 0.7, "market_price_per_ton": 55000, "cost_per_acre": 8000,
        "nitrogen_fixing": False, "rotation_group": "oilseed", "soil_impact_base": 60,
    },
    {
        "id": "chickpea", "name": "Chickpea (Gram)", "name_hi": "चना", "name_sat": "चोना दाल (Chona Dal)",
        "season": ["Rabi"], "soil_types": ["Black", "Alluvial", "Loamy", "Red & Laterite"],
        "ph_range": (6.0, 7.5), "n_range": (15, 30), "p_range": (25, 40), "k_range": (15, 30),
        "moisture_range": (25, 45), "water_requirement": "low",
        "yield_per_acre_ton": 0.9, "market_price_per_ton": 52000, "cost_per_acre": 9500,
        "nitrogen_fixing": True, "rotation_group": "legume", "soil_impact_base": 85,
    },
    {
        "id": "potato", "name": "Potato", "name_hi": "आलू", "name_sat": "आलू (Aalu)",
        "season": ["Rabi"], "soil_types": ["Alluvial", "Loamy", "Red & Laterite"],
        "ph_range": (5.0, 6.5), "n_range": (80, 120), "p_range": (40, 60), "k_range": (60, 90),
        "moisture_range": (55, 75), "water_requirement": "high",
        "yield_per_acre_ton": 8.0, "market_price_per_ton": 12000, "cost_per_acre": 35000,
        "nitrogen_fixing": False, "rotation_group": "vegetable", "soil_impact_base": 35,
    },
    {
        "id": "soybean", "name": "Soybean", "name_hi": "सोयाबीन", "name_sat": "सोयाबीन (Soybean)",
        "season": ["Kharif"], "soil_types": ["Black", "Loamy", "Alluvial"],
        "ph_range": (6.0, 7.5), "n_range": (20, 35), "p_range": (25, 45), "k_range": (25, 40),
        "moisture_range": (45, 65), "water_requirement": "medium",
        "yield_per_acre_ton": 1.2, "market_price_per_ton": 45000, "cost_per_acre": 13000,
        "nitrogen_fixing": True, "rotation_group": "legume", "soil_impact_base": 82,
    },
    {
        "id": "sugarcane", "name": "Sugarcane", "name_hi": "गन्ना", "name_sat": "कुसुम गाड़ा (Kusum Gada)",
        "season": ["Kharif", "Zaid"], "soil_types": ["Alluvial", "Black", "Loamy"],
        "ph_range": (6.0, 7.5), "n_range": (100, 150), "p_range": (40, 60), "k_range": (60, 100),
        "moisture_range": (70, 90), "water_requirement": "high",
        "yield_per_acre_ton": 35.0, "market_price_per_ton": 3200, "cost_per_acre": 45000,
        "nitrogen_fixing": False, "rotation_group": "cash", "soil_impact_base": 20,
    },
    {
        "id": "sesame", "name": "Sesame (Til)", "name_hi": "तिल", "name_sat": "तिल (Til)",
        "season": ["Zaid", "Kharif"], "soil_types": ["Sandy", "Red", "Red & Laterite", "Loamy"],
        "ph_range": (5.5, 7.5), "n_range": (15, 30), "p_range": (15, 25), "k_range": (15, 25),
        "moisture_range": (25, 40), "water_requirement": "low",
        "yield_per_acre_ton": 0.35, "market_price_per_ton": 95000, "cost_per_acre": 6000,
        "nitrogen_fixing": False, "rotation_group": "oilseed", "soil_impact_base": 65,
    },
    {
        "id": "tomato", "name": "Tomato", "name_hi": "टमाटर", "name_sat": "टमाटर (Tamatar)",
        "season": ["Zaid", "Rabi"], "soil_types": ["Loamy", "Alluvial", "Red & Laterite"],
        "ph_range": (6.0, 7.0), "n_range": (60, 100), "p_range": (40, 60), "k_range": (50, 80),
        "moisture_range": (50, 70), "water_requirement": "medium",
        "yield_per_acre_ton": 9.0, "market_price_per_ton": 15000, "cost_per_acre": 30000,
        "nitrogen_fixing": False, "rotation_group": "vegetable", "soil_impact_base": 40,
    },
    {
        "id": "cucumber", "name": "Cucumber / Muskmelon", "name_hi": "खीरा / खरबूजा", "name_sat": "खीरा (Khira)",
        "season": ["Zaid"], "soil_types": ["Sandy", "Loamy", "Alluvial"],
        "ph_range": (6.0, 7.0), "n_range": (30, 50), "p_range": (25, 40), "k_range": (25, 40),
        "moisture_range": (45, 65), "water_requirement": "medium",
        "yield_per_acre_ton": 6.0, "market_price_per_ton": 14000, "cost_per_acre": 17000,
        "nitrogen_fixing": False, "rotation_group": "vegetable", "soil_impact_base": 50,
    },
]


# ---------------------------------------------------------------------------
# 3. SCORING ENGINE
# ---------------------------------------------------------------------------

def _range_closeness_score(value, low, high):
    """
    Returns 0-100: 100 if value is within [low, high],
    decaying linearly the further outside the range it falls.
    """
    if low <= value <= high:
        return 100.0
    span = max(high - low, 1e-6)
    if value < low:
        distance = low - value
    else:
        distance = value - high
    penalty = (distance / span) * 100
    return max(0.0, 100.0 - penalty)


def calculate_suitability_score(crop, soil_type, ph, n, p, k, moisture):
    """
    Combines soil-type match, pH, N, P, K and moisture closeness into a
    single 0-100 suitability score. This is the piece you'd replace with a
    trained classifier/regressor once real labelled yield data is available.
    """
    soil_match = 100.0 if soil_type in crop["soil_types"] else 40.0
    ph_score = _range_closeness_score(ph, *crop["ph_range"])
    n_score = _range_closeness_score(n, *crop["n_range"])
    p_score = _range_closeness_score(p, *crop["p_range"])
    k_score = _range_closeness_score(k, *crop["k_range"])
    moisture_score = _range_closeness_score(moisture, *crop["moisture_range"])

    weighted = (
        soil_match * 0.25
        + ph_score * 0.15
        + n_score * 0.15
        + p_score * 0.15
        + k_score * 0.15
        + moisture_score * 0.15
    )
    return round(weighted, 1)


def calculate_profitability_score(crop, suitability_score):
    """
    Estimates expected yield (scaled down by soil suitability, since poor
    conditions reduce real-world yield), then computes gross return, net
    profit, and a normalized 0-100 profitability score.
    """
    yield_efficiency = 0.5 + (suitability_score / 100) * 0.5  # 50%-100% of max yield
    expected_yield_ton = round(crop["yield_per_acre_ton"] * yield_efficiency, 2)
    gross_return = expected_yield_ton * crop["market_price_per_ton"]
    net_profit = round(gross_return - crop["cost_per_acre"], 0)
    roi_percent = round((net_profit / crop["cost_per_acre"]) * 100, 1) if crop["cost_per_acre"] else 0

    return {
        "expected_yield_ton_per_acre": expected_yield_ton,
        "net_profit_per_acre_inr": net_profit,
        "roi_percent": roi_percent,
    }


def calculate_soil_health_score(crop, moisture_available):
    """
    Sustainability score: rewards nitrogen-fixing legumes (natural soil
    enrichment) and low-water-stress crops; penalizes crops whose water
    demand exceeds what the field can realistically provide long-term.
    """
    score = crop["soil_impact_base"]

    if crop["nitrogen_fixing"]:
        score += 15  # bonus: naturally replenishes soil nitrogen

    water_penalty = {
        "low": 0,
        "medium": 5 if moisture_available < crop["moisture_range"][0] else 0,
        "high": 15 if moisture_available < crop["moisture_range"][0] else 5,
    }
    score -= water_penalty.get(crop["water_requirement"], 0)

    return {
        "soil_health_score": round(max(0, min(100, score)), 1),
        "nitrogen_fixing": crop["nitrogen_fixing"],
        "rotation_group": crop["rotation_group"],
        "rotation_tip": (
            "Nitrogen-fixing crop — excellent for restoring soil fertility. "
            "Safe to grow after cereals like rice/maize/wheat."
            if crop["nitrogen_fixing"]
            else f"Rotate with a legume (e.g. Arhar, Chickpea, Groundnut) next season "
                 f"to replenish nitrogen after growing this {crop['rotation_group']} crop."
        ),
    }


def recommend_crops(season, soil_type, ph, n, p, k, moisture):
    """Main pipeline: filter by season -> score -> rank -> attach badges."""
    results = []

    for crop in CROP_DB:
        if season not in crop["season"]:
            continue  # hard filter: only agronomically valid crops for this season

        suitability = calculate_suitability_score(crop, soil_type, ph, n, p, k, moisture)
        profit = calculate_profitability_score(crop, suitability)
        soil_health = calculate_soil_health_score(crop, moisture)

        final_score = round(
            suitability * 0.40
            + min(100, max(0, profit["roi_percent"])) * 0.35
            + soil_health["soil_health_score"] * 0.25,
            1,
        )

        results.append({
            "crop_id": crop["id"],
            "name": crop["name"],
            "name_hi": crop["name_hi"],
            "name_sat": crop["name_sat"],
            "suitability_score": suitability,
            "profitability": profit,
            "soil_health": soil_health,
            "final_score": final_score,
        })

    # Rank by overall final score
    results.sort(key=lambda r: r["final_score"], reverse=True)
    top_results = results[:5]

    # Attach badges
    if top_results:
        highest_return_crop = max(top_results, key=lambda r: r["profitability"]["net_profit_per_acre_inr"])
        for r in top_results:
            r["badges"] = []
            if r["crop_id"] == highest_return_crop["crop_id"]:
                r["badges"].append("highest_return")
            if r["soil_health"]["soil_health_score"] >= 75:
                r["badges"].append("soil_safe")
        top_results[0]["badges"].append("best_match")

    return top_results


# ---------------------------------------------------------------------------
# 4. API ROUTES
# ---------------------------------------------------------------------------

@app.route("/api/soil-defaults", methods=["GET"])
def soil_defaults():
    """Auto-fill regional soil/climate defaults based on selected state."""
    state = request.args.get("state", "")
    data = REGIONAL_DEFAULTS.get(state)
    if not data:
        return jsonify({"error": f"No default data available for '{state}'"}), 404
    return jsonify({"state": state, **data})


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Expected JSON body:
    {
        "season": "Kharif" | "Rabi" | "Zaid",
        "soil_type": "Alluvial" | "Black" | "Red" | "Red & Laterite" | "Laterite" | "Sandy" | "Clay" | "Loamy",
        "ph": 6.2,
        "n": 40, "p": 20, "k": 30,
        "moisture": 50
    }
    """
    data = request.get_json(force=True, silent=True) or {}

    required_fields = ["season", "soil_type", "ph", "n", "p", "k", "moisture"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        recommendations = recommend_crops(
            season=data["season"],
            soil_type=data["soil_type"],
            ph=float(data["ph"]),
            n=float(data["n"]),
            p=float(data["p"]),
            k=float(data["k"]),
            moisture=float(data["moisture"]),
        )
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input values: {e}"}), 400

    return jsonify({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input": data,
        "recommendations": recommendations,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "crops_in_db": len(CROP_DB)})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
