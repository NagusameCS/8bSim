from __future__ import annotations
import json
import os
from typing import Any, Dict


DEFAULTS: Dict[str, Any] = {
    "economy": {
        "alpha": 0.33,                # capital share in Cobb-Douglas
        "delta": 0.05,                # depreciation rate
        "g_A": 0.01,                  # baseline TFP growth
        "sigma_shock": 0.015,         # volatility of TFP shocks (std dev)
        "k_y_ratio_init": 3.0,        # initial capital-output ratio
        "labor_share_working_age": 0.65,  # fraction of population as effective labor
        "neighbor_spillover": 0.05,   # how much neighbor GDP growth spills into A growth
        "investment_rate_base": 0.22, # baseline investment rate
        "investment_rate_rich": 0.26, # used if GDPpc > rich_threshold
        "rich_threshold": 30000.0
    },
    "geopolitics": {
        "enabled": True,
        "relation_scale": 1.0,                 # scales [-1,1] relation impact
        "neighbor_friend_bonus": 0.2,          # base bonus for neighbors
        "sanctions_threshold": -0.4,           # relation below this triggers sanctions drag
        "sanctions_penalty": -0.004,           # TFP growth drag when under threshold
        "migration_geo_bias_floor": 0.1,       # minimum geo bias multiplier
        "min_population_floor": 5,             # do not let migration drain below this in a step
        "emigration_max_fraction": 0.2,        # cap emigration (rate<0) to this fraction per year
        "relations": {},                       # optional override: {"<cid>": {"<other>": value}}
        "rivalries": [                         # optional named rivalries to seed
            {"a": "United States", "b": "China", "w": -0.35},
            {"a": "United States", "b": "Russia", "w": -0.35},
            {"a": "India", "b": "Pakistan", "w": -0.45},
            {"a": "Iran", "b": "Saudi Arabia", "w": -0.35}
        ]
    },
    "language": {
        "w_local": 1.0,
        "w_neighbor": 0.6,
        "w_global": 0.45,
        "aggression": 1.25,
        "campaign_base": 0.02,
        "campaign_global_weight": 0.15,
        "campaign_neighbor_weight": 0.10,
        "max_langs_per_agent": 3
    },
    "population": {
        "prod_to_births_multiplier": 0.5,  # how strongly production scales births
        "baseline_target_growth": 0.003,   # default net growth target if unknown
        "min_target_growth": -0.01,
        "max_target_growth": 0.03
    },
    "migration": {
        "global_wave_probability": 0.1,
        "global_wave_fraction": 0.001,
        "gdp_weight": 1.0,
        "language_bonus": 0.5
    }
    ,
    "ai": {
        "enabled": True,
        "choices_per_year": 2,
        "conquest_aggressiveness": 0.1,
        "conquest_power_ratio_threshold": 1.5,
        "conquest_shared_language_min": 0.3,
        "conquest_neighbor_only": True
    }
}


def settings_path() -> str:
    return os.path.join(os.path.dirname(__file__), "settings.json")


def load_settings() -> Dict[str, Any]:
    path = settings_path()
    if not os.path.exists(path):
        return json.loads(json.dumps(DEFAULTS))
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # On any failure, return defaults
        return json.loads(json.dumps(DEFAULTS))
    # Shallow merge with defaults to add new keys safely
    merged = json.loads(json.dumps(DEFAULTS))
    _deep_update(merged, data)
    return merged


def save_settings(data: Dict[str, Any]) -> None:
    # Merge into defaults then persist (keeps future-proof keys)
    merged = json.loads(json.dumps(DEFAULTS))
    _deep_update(merged, data)
    path = settings_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


def reset_to_defaults() -> Dict[str, Any]:
    save_settings(DEFAULTS)
    return json.loads(json.dumps(DEFAULTS))


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
