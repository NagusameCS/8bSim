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

## How to run

The commands below are for Windows PowerShell.

1) Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Start the interactive app

```powershell
python -m streamlit run app.py
```

- Open the app in your browser at http://localhost:8501
- Stop the app with Ctrl+C in the terminal

4) Run headless in the terminal

```powershell
python -m simulation.main
```

- This sample run executes 20 years with scale_factor=1000 (see `simulation/main.py`) and prints stats to the console.

5) Reset settings to defaults

- Close the app, then delete `simulation/settings.json` (it will be recreated from defaults on next run), or use the UI’s reset if provided.

6) Deactivate the virtual environment (optional)

```powershell
deactivate
```
