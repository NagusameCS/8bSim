import streamlit as st
import pandas as pd
from simulation.settings import load_settings, save_settings, reset_to_defaults
from simulation.main import setup_from_config

st.set_page_config(layout="wide")
st.title("World Geopolitics Settings")
st.caption("Tune geopolitical relations and their influence on migration, language diffusion, and economic spillovers.")

settings = load_settings()
geo = settings.get("geopolitics", {})

colA, colB = st.columns(2)
with colA:
    enabled = st.checkbox("Enable geopolitics", value=geo.get("enabled", True))
    relation_scale = st.slider("Relation scale", 0.0, 2.0, float(geo.get("relation_scale", 1.0)), 0.05)
    neighbor_bonus = st.slider("Neighbor friend bonus", 0.0, 0.5, float(geo.get("neighbor_friend_bonus", 0.2)), 0.01)
    sanctions_threshold = st.slider("Sanctions threshold (relation)", -1.0, 0.0, float(geo.get("sanctions_threshold", -0.4)), 0.05)
    sanctions_penalty = st.slider("Sanctions penalty (TFP growth)", -0.02, 0.0, float(geo.get("sanctions_penalty", -0.004)), 0.001)

with colB:
    mig_floor = st.slider("Migration geo bias floor", 0.0, 1.0, float(geo.get("migration_geo_bias_floor", 0.1)), 0.05)
    min_pop_floor = st.number_input("Min population floor (per-country step)", min_value=0, value=int(geo.get("min_population_floor", 5)))
    emig_cap = st.slider("Emigration max fraction per year", 0.0, 1.0, float(geo.get("emigration_max_fraction", 0.2)), 0.01)

st.markdown("---")

# Show editable relations if desired (static list from current config)
countries, _ = setup_from_config("simulation/config.json")
id_to_name = {c.id: c.name for c in countries}
name_to_id = {v: k for k, v in id_to_name.items()}

relations = geo.get("relations", {}) or {}
df_rows = []
for a_id, row in relations.items():
    for b_id, val in row.items():
        try:
            a_i = int(a_id); b_i = int(b_id)
            df_rows.append({"From": id_to_name.get(a_i, str(a_i)), "To": id_to_name.get(b_i, str(b_i)), "Relation": float(val)})
        except Exception:
            continue
rel_df = pd.DataFrame(df_rows) if df_rows else pd.DataFrame(columns=["From", "To", "Relation"])
st.subheader("Custom relation overrides (optional)")
st.caption("Add rows as overrides. Values in [-1, 1]. If empty, defaults (neighbors + rivalries) are used.")
edited = st.data_editor(rel_df, num_rows="dynamic")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button("Save"):
        # Persist configuration
        settings.setdefault("geopolitics", {})
        settings["geopolitics"].update({
            "enabled": enabled,
            "relation_scale": relation_scale,
            "neighbor_friend_bonus": neighbor_bonus,
            "sanctions_threshold": sanctions_threshold,
            "sanctions_penalty": sanctions_penalty,
            "migration_geo_bias_floor": mig_floor,
            "min_population_floor": int(min_pop_floor),
            "emigration_max_fraction": emig_cap,
        })
        # Rebuild overrides from edited table
        overrides: dict[str, dict[str, float]] = {}
        for _, row in edited.iterrows():
            try:
                a = name_to_id.get(row["From"], None)
                b = name_to_id.get(row["To"], None)
                r = float(row["Relation"])
            except Exception:
                continue
            if a is None or b is None:
                continue
            overrides.setdefault(str(a), {})[str(b)] = max(-1.0, min(1.0, r))
        settings["geopolitics"]["relations"] = overrides
        save_settings(settings)
        st.success("Geopolitics settings saved.")
with col2:
    if st.button("Reset to defaults"):
        reset_to_defaults()
        st.warning("Settings reset to defaults. Reload the app to see changes.")
