# 8bSim

8bSim is a discrete-time world simulation of population, economy, and language dynamics across multiple countries. Agents are born, age, migrate, learn/forget languages, and new policies shape the landscape. Countries pursue dominance via soft power and, rarely, conquest under strict conditions.

## Highlights

- Infection-style language diffusion with logistic adoption, local/neighbor/global influence, and attrition
- Realistic macroeconomy (Cobb–Douglas with capital accumulation, TFP growth, shocks, and spillovers)
- Migration waves and GDP/language-driven destination choice with overwrite effects during large influxes
- Geopolitics that biases language, migration, and economic spillovers (plus sanctions drag)
- Country AI: per-year choices with cautious conquest (shared-language and power thresholds) and soft policy pushes
- World Settings pages (Economy, Population, Language, Migration, Geopolitics) with persistence
- Modern Streamlit UI with a world map showing dominant language prevalence and rich tooltips

## Quickstart

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Run the interactive app

```bash
python -m streamlit run app.py
```

3) (Optional) Run a headless sample

```bash
python -m simulation.main
```

## UI guide

- World Map
    - Colors = dominant language prevalence (%). Hover for:
        - Dominant language and share
        - Top languages list
        - Population (scaled)
        - GDP per capita proxy
- Live progress shows yearly metrics and per-country charts (Population, Languages, Economy)
- World Settings pages let you tune parameters and persist them between runs
    - Geopolitics page: enable/disable, relation weights, sanctions, migration caps, and custom relation overrides

## Structure

- simulation/config.json — world blueprint (countries, neighbors, initial languages, populations)
- simulation/simulation_models.py — Language, Agent, Country models
- simulation/main.py — config loader and headless runner
- simulation/simulation.py — core engine (demographics, economy, language, migration, AI, geopolitics)
- simulation/settings.py — defaults + persistent settings (JSON)
- pages/World Settings.py — settings UI (economy, language, migration, population)
- pages/World Geopolitics.py — relations editor and geopolitics knobs
- app.py — Streamlit app (live run + charts + final map)

## AI and conquest rules (overview)

- Each country can make a small number of choices per year
- Conquest only when:
    - Shared dominant language or a language with significant prevalence in both countries
    - Attacker power ratio >= threshold, and deterrence heuristic passes
    - By default neighbors-only
- After a conquest, the attacker becomes a threat to others (relations reduced), discouraging rapid follow-ups

## Notes

- The simulation loads settings at each step, so UI changes apply without restarts
- For large scale_factor runs, performance depends on your machine; tune the number of years in the sidebar

## License

See LICENSE