import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pycountry
from streamlit_plotly_events import plotly_events
from simulation.main import setup_from_config
from simulation.simulation import Simulation
import json

st.set_page_config(layout="wide")

DISPLAY_POP_SCALE = 100

st.title("8bSim - Language Infection and Population Dynamics")
st.caption("Use World Settings to tune dynamics. The world map shows each country's dominant language prevalence with richer tooltips.")


# --- Helper Functions ---
@st.cache_data
def get_iso_alpha(country_name):
    """Get ISO alpha-3 code for a country name with common alias fallbacks."""
    ALIASES = {
        "United States": "USA",
        "United States of America": "USA",
        "Russia": "RUS",
        "Russian Federation": "RUS",
        "United Kingdom": "GBR",
        "UK": "GBR",
        "South Korea": "KOR",
        "Korea, Republic of": "KOR",
        "North Korea": "PRK",
        "Iran": "IRN",
        "Iran, Islamic Republic of": "IRN",
        "Turkey": "TUR",
        "Vietnam": "VNM",
        "Viet Nam": "VNM",
        "Egypt": "EGY",
        "Saudi Arabia": "SAU",
        "Côte d’Ivoire": "CIV",
        "Cote d'Ivoire": "CIV",
    }
    if country_name in ALIASES:
        return ALIASES[country_name]
    try:
        return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
    except LookupError:
        return None


def format_population(population_value):
    """Format population numbers using the display scale."""
    scaled_value = population_value * DISPLAY_POP_SCALE
    return f"{int(round(scaled_value)):,}"


def prepare_map_dataframe(sim, stats, econ_stats=None):
    """Prepare a dataframe for the choropleth map, always using dominant language prevalence as color.

    econ_stats: optional dict from sim.get_economy_stats() for the same step to show GDPpc proxy in tooltips.
    """
    map_rows = []
    for country_id, country_stats in stats.items():
        country_name = sim.countries[country_id].name
        iso_code = get_iso_alpha(country_name)
        if not iso_code:
            continue
        # Normalize ISO codes to uppercase alpha-3 for Plotly
        iso_code = str(iso_code).upper()

        population = country_stats["population"]
        lang_prev = country_stats.get("language_prevalence", {}) or {}
        sorted_langs = sorted(
            lang_prev.items(), key=lambda item: item[1], reverse=True
        )

        if sorted_langs:
            language_lines = [
                f"{sim.languages[lang_id].name}: {share * 100:.1f}%"
                for lang_id, share in sorted_langs[:5]
                if lang_id in sim.languages
            ]
            languages_tooltip = "Languages: " + "; ".join(language_lines)
        else:
            languages_tooltip = "Languages: Data unavailable"

        # Dominant language and prevalence
        dom_lang = None
        dom_prev = 0.0
        if lang_prev:
            dom_id, dom_prev = max(lang_prev.items(), key=lambda kv: kv[1])
            dom_lang = sim.languages.get(dom_id).name if dom_id in sim.languages else "Unknown"
        # Color value is dominant language prevalence (%)
        value = dom_prev * 100.0
        colorbar = "Dominant Language Prevalence (%)"

        # GDP per Capita (proxy) tooltip
        gdp_pc_proxy = None
        if econ_stats is not None:
            gdp_pc_proxy = econ_stats.get(country_id, {}).get("gdp_pc_proxy", None)
        gdp_pc_fmt = f"${gdp_pc_proxy:,.0f}" if isinstance(gdp_pc_proxy, (int, float)) else "N/A"

        map_rows.append(
            {
                "iso_alpha": iso_code,
                "country": country_name,
                "value": value,
                "languages_tooltip": languages_tooltip,
                "population_display": format_population(population),
                "colorbar": colorbar,
                "dominant_language": dom_lang or "N/A",
                "dominant_prevalence": f"{dom_prev*100:.1f}%",
                "gdp_pc_proxy_fmt": gdp_pc_fmt,
            }
        )

    map_df = pd.DataFrame(map_rows)
    return map_df


def render_map(map_df, title, key):
    """Render a choropleth map using Plotly Express (more reliable)."""
    if map_df.empty:
        st.info("Data will appear once the simulation produces results.")
        return None

    range_color = (0, 100)
    fig = px.choropleth(
        map_df,
        locations="iso_alpha",
        color="value",
        hover_name="country",
        hover_data={
            "value": ":.2f",
            "population_display": True,
            "languages_tooltip": True,
            "dominant_language": True,
            "dominant_prevalence": True,
            "gdp_pc_proxy_fmt": True,
            "iso_alpha": False,
        },
        color_continuous_scale="Turbo",
        range_color=range_color,
        projection="natural earth",
    )
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(
            title=map_df["colorbar"].iloc[0] if not map_df.empty else "",
            ticksuffix="%",
            tickformat=".0f",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    return None


# --- State Management ---
if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False
if "history" not in st.session_state:
    st.session_state.history = None
if "sim" not in st.session_state:
    st.session_state.sim = None
if "econ_history" not in st.session_state:
    st.session_state.econ_history = None
if "selected_country_index" not in st.session_state:
    st.session_state.selected_country_index = 0

st.sidebar.header("Simulation Parameters")
num_years = st.sidebar.slider("Number of Years to Simulate", 1, 100, 20)

if st.sidebar.button("Run Simulation", disabled=st.session_state.get("running", False)):
    st.session_state.running = True
    st.session_state.simulation_run = True

    with st.spinner("Running simulation..."):
        # Initialize simulation from config
        initial_countries, initial_languages = setup_from_config("simulation/config.json")
        sim = Simulation(countries=initial_countries, languages=initial_languages, scale_factor=1000)
        st.session_state.sim = sim

        # --- UI Placeholders for Live Update ---
        st.header("Live Simulation Progress")
        col1, col2, col3 = st.columns(3)
        year_text = col1.empty()
        pop_text = col2.empty()
        migration_text = col3.empty()

        map_placeholder = st.empty()
        charts_placeholder = st.empty()
        progress = st.progress(0, text="Initializing...")

        history = []
        econ_history = []

        country_names = [c.name for c in sim.countries.values()]

        # Base key for the live map; we'll append the year to ensure uniqueness within a single run
        LIVE_MAP_KEY = "live_map"

        for year in range(num_years):
            total_migrations = sim.run_step()
            stats = sim.get_stats()
            econ_stats = sim.get_economy_stats()
            history.append(stats)
            econ_history.append(econ_stats)

            # --- Update Live Metrics ---
            world_pop = sum(c_stats["population"] for c_stats in stats.values())
            year_text.metric("Year", f"{year + 1}/{num_years}")
            pop_text.metric("World Population", format_population(world_pop))
            migration_text.metric("Migrations This Year", f"{total_migrations:,}")
            progress.progress(int(((year + 1) / num_years) * 100), text=f"Year {year + 1} of {num_years}")

            # --- Update Live Map ---
            map_df = prepare_map_dataframe(sim, stats, econ_stats)
            # Clear previous map before rendering a new one in the same run to avoid duplicate keys
            map_placeholder.empty()
            with map_placeholder.container():
                _ = render_map(map_df, "World Map (Live)", f"{LIVE_MAP_KEY}_{year}")
            # Clicks are optional now; we render all countries below regardless

            # --- Update Live Charts for Selected Country ---
            with charts_placeholder.container():
                st.subheader("Country Charts (Live)")
                for country in sim.countries.values():
                    with st.expander(country.name, expanded=False):
                        # Population Chart
                        pop_history_scaled = [
                            year_stats[country.id]["population"] * DISPLAY_POP_SCALE
                            for year_stats in history
                        ]
                        pop_chart = pd.DataFrame(
                            {
                                "Year": range(1, len(pop_history_scaled) + 1),
                                f"Population (×{DISPLAY_POP_SCALE})": pop_history_scaled,
                            }
                        )
                        st.line_chart(pop_chart, x="Year", y=f"Population (×{DISPLAY_POP_SCALE})")

                        # Language Chart
                        lang_history = [
                            year_stats[country.id]["language_prevalence"] for year_stats in history
                        ]

                        all_lang_names = [lang.name for lang in sim.languages.values()]
                        lang_data = {lang_name: [0] * len(history) for lang_name in all_lang_names}

                        for i, year_data in enumerate(lang_history):
                            for lang_id, prevalence in year_data.items():
                                lang_name = sim.languages[lang_id].name
                                if lang_name in lang_data:
                                    lang_data[lang_name][i] = prevalence * 100

                        lang_df = pd.DataFrame(lang_data)
                        lang_df["Year"] = range(1, len(history) + 1)

                        active_lang_cols = lang_df.drop(columns="Year").columns[
                            (lang_df.drop(columns="Year") != 0).any()
                        ]
                        cols_to_show = ["Year"] + [col for col in active_lang_cols if col != "Year"]
                        lang_df = lang_df[cols_to_show]

                        st.line_chart(lang_df, x="Year")

                        # Economy Charts: GDPpc proxy and Production Index
                        econ_series = [year_stats.get(country.id, {}) for year_stats in econ_history]
                        gdp_pc = [row.get("gdp_pc_proxy", 0.0) for row in econ_series]
                        prod_idx = [row.get("production_index", 1.0) for row in econ_series]
                        econ_df = pd.DataFrame({
                            "Year": range(1, len(econ_series) + 1),
                            "GDP per Capita (proxy)": gdp_pc,
                            "Production Index": prod_idx,
                        })
                        st.line_chart(econ_df, x="Year")

        st.session_state.history = history
        st.session_state.econ_history = econ_history
        st.session_state.running = False
        st.success("Simulation Complete!")
        progress.empty()
        # Rerun to clear live view and show final results
        st.rerun()

if (
    st.session_state.simulation_run
    and st.session_state.history
    and not st.session_state.get("running", False)
):
    sim = st.session_state.sim
    history = st.session_state.history
    econ_history = st.session_state.econ_history or []
    num_years_run = len(history)

    # --- History Viewer ---
    st.header("History Viewer")
    st.caption("Browse every year, see births/deaths, migrations, conquests, and dominant-language changes.")
    if hasattr(sim, "event_log") and sim.event_log:
        years = [e.get("year", i + 1) for i, e in enumerate(sim.event_log)]
        sel_year = st.slider("Year to inspect", min_value=years[0], max_value=years[-1], value=years[-1])
        idx = years.index(sel_year)
        entry = sim.event_log[idx]

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Births (world)", f"{sum(entry.get('births', {}).values()):,}")
        with colB:
            st.metric("Deaths (world)", f"{sum(entry.get('deaths', {}).values()):,}")
        with colC:
            st.metric("Migrations (world)", f"{entry.get('migrations_total', 0):,}")

        # Per-country births/deaths table
        with st.expander("Per-country vital stats"):
            rows = []
            for c in sim.countries.values():
                rows.append({
                    "Country": c.name,
                    "Births": entry.get("births", {}).get(c.id, 0),
                    "Deaths": entry.get("deaths", {}).get(c.id, 0),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Country"))

        # Migration flows
        with st.expander("Migration flows"):
            flows = entry.get("migration_flows", [])
            if flows:
                def name_or_world(x):
                    if x is None:
                        return "Outside World"
                    return sim.countries.get(x).name if x in sim.countries else str(x)
                flow_rows = [{
                    "From": name_or_world(f.get("source")),
                    "To": name_or_world(f.get("dest")),
                    "Count": f.get("count", 0)
                } for f in flows]
                st.dataframe(pd.DataFrame(flow_rows).sort_values("Count", ascending=False))
            else:
                st.info("No recorded migration flows this year.")

        # Conquests
        with st.expander("Conquests"):
            cons = entry.get("conquests", [])
            if cons:
                con_rows = [{
                    "Attacker": sim.countries[a].name if a in sim.countries else a,
                    "Target": sim.countries[t].name if t in sim.countries else t,
                } for (a, t) in cons]
                st.table(pd.DataFrame(con_rows))
            else:
                st.info("No conquests this year.")

        # Dominant language changes
        with st.expander("Dominant language changes"):
            changes = entry.get("dominant_language_changes", {})
            if changes:
                rows = []
                for cid, ch in changes.items():
                    c = sim.countries.get(cid)
                    from_name = sim.languages[ch.get("from")].name if ch.get("from") in sim.languages else "N/A"
                    to_name = sim.languages[ch.get("to")].name if ch.get("to") in sim.languages else "N/A"
                    rows.append({"Country": c.name if c else cid, "From": from_name, "To": to_name})
                st.table(pd.DataFrame(rows).set_index("Country"))
            else:
                st.info("No dominant-language changes this year.")

        # Export full history
        with st.expander("Export history"):
            hist_json = {
                "events": sim.event_log,
                "stats": history,
                "economy": econ_history,
            }
            st.download_button("Download JSON", data=json.dumps(hist_json, indent=2), file_name="8bSim_history.json")
    else:
        st.info("No event history recorded. Run a simulation to populate history.")

    # --- Display Final Results ---
    st.header("Final Simulation Results")
    st.caption(
        f"Population figures below are displayed at ×{DISPLAY_POP_SCALE} relative to agent counts."
    )

    country_names = [c.name for c in sim.countries.values()]
    num_countries = len(country_names)
    if num_countries == 0:
        st.warning("No countries available in the simulation.")
    else:
        final_year_stats = history[-1]
        # Use last available economy stats for tooltips if present
        last_econ = econ_history[-1] if econ_history else None
        final_map_df = prepare_map_dataframe(sim, final_year_stats, last_econ)
        with st.container():
            _ = render_map(final_map_df, "Final Year Map", "final_map")
        # Render charts for all countries
        st.subheader("Country Charts (Final)")
        for country in sim.countries.values():
            with st.expander(country.name, expanded=False):
                # Population Chart
                pop_history_scaled = [
                    year_stats[country.id]["population"] * DISPLAY_POP_SCALE
                    for year_stats in history
                ]
                pop_chart = pd.DataFrame(
                    {
                        "Year": range(1, num_years_run + 1),
                        f"Population (×{DISPLAY_POP_SCALE})": pop_history_scaled,
                    }
                )
                st.line_chart(pop_chart, x="Year", y=f"Population (×{DISPLAY_POP_SCALE})")

                # Language Prevalence Chart
                lang_history = [
                    year_stats[country.id]["language_prevalence"] for year_stats in history
                ]

                all_lang_names = [lang.name for lang in sim.languages.values()]
                lang_data = {lang_name: [0] * num_years_run for lang_name in all_lang_names}

                for i, year_data in enumerate(lang_history):
                    for lang_id, prevalence in year_data.items():
                        lang_name = sim.languages[lang_id].name
                        if lang_name in lang_data:
                            lang_data[lang_name][i] = prevalence * 100

                lang_df = pd.DataFrame(lang_data)
                lang_df["Year"] = range(1, num_years_run + 1)

                active_lang_cols = lang_df.drop(columns="Year").columns[
                    (lang_df.drop(columns="Year") != 0).any()
                ]
                cols_to_show = ["Year"] + [col for col in active_lang_cols if col != "Year"]
                lang_df = lang_df[cols_to_show]

                st.line_chart(lang_df, x="Year")

                # Economy Charts
                if econ_history:
                    econ_series = [year_stats.get(country.id, {}) for year_stats in econ_history]
                    gdp_pc = [row.get("gdp_pc_proxy", 0.0) for row in econ_series]
                    prod_idx = [row.get("production_index", 1.0) for row in econ_series]
                    econ_df = pd.DataFrame({
                        "Year": range(1, len(econ_series) + 1),
                        "GDP per Capita (proxy)": gdp_pc,
                        "Production Index": prod_idx,
                    })
                    st.line_chart(econ_df, x="Year")

        # --- Final Stats Table ---
        st.subheader("Final Statistics by Country")
        final_stats_data = []
        for country in sim.countries.values():
            final_pop = history[-1][country.id]["population"]
            lang_prev = history[-1][country.id]["language_prevalence"]

            prevalence_str = ", ".join(
                [
                    f"{sim.languages[l_id].name}: {p:.2%}"
                    for l_id, p in sorted(lang_prev.items(), key=lambda item: item[1], reverse=True)
                ]
            )

            final_stats_data.append(
                {
                    "Country": country.name,
                    f"Final Population (×{DISPLAY_POP_SCALE})": format_population(final_pop),
                    "Language Prevalence": prevalence_str,
                }
            )

        final_stats_df = pd.DataFrame(final_stats_data).set_index("Country")
        st.table(final_stats_df)

elif not st.session_state.get("running", False):
    st.info("Adjust the simulation parameters on the left and click 'Run Simulation' to start.")

