import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pycountry
from streamlit_plotly_events import plotly_events
from simulation.main import setup_from_config
from simulation.simulation import Simulation

st.set_page_config(layout="wide")

DISPLAY_POP_SCALE = 100

st.title("8bSim - Population and Language Dynamics Simulation")
st.caption(
    f"Population figures are displayed at ×{DISPLAY_POP_SCALE} relative to the underlying agent counts."
)


# --- Helper Functions ---
@st.cache_data
def get_iso_alpha(country_name):
    """Get ISO alpha-3 code for a country name."""
    try:
        return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
    except LookupError:
        return None


def format_population(population_value):
    """Format population numbers using the display scale."""
    scaled_value = population_value * DISPLAY_POP_SCALE
    return f"{int(round(scaled_value)):,}"


def prepare_map_dataframe(sim, stats):
    """Prepare a dataframe for the choropleth map."""
    map_rows = []
    for country_id, country_stats in stats.items():
        country_name = sim.countries[country_id].name
        iso_code = get_iso_alpha(country_name)
        if not iso_code:
            continue

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

        map_rows.append(
            {
                "iso_alpha": iso_code,
                "country": country_name,
                "population": population,
                "population_display": format_population(population),
                "languages_tooltip": languages_tooltip,
            }
        )

    map_df = pd.DataFrame(map_rows)
    if not map_df.empty:
        map_df["display_population"] = map_df["population"] * DISPLAY_POP_SCALE
    return map_df


def render_population_map(map_df, title, key):
    """Render an interactive population map and return the clicked country name."""
    if map_df.empty:
        st.info("Population data will appear once the simulation produces results.")
        return None

    fig = go.Figure(
        data=go.Choropleth(
            locations=map_df["iso_alpha"],
            z=map_df["display_population"],
            text=map_df["country"],
            colorscale="Blues",
            autocolorscale=False,
            reversescale=False,
            marker_line_color="darkgray",
            marker_line_width=0.5,
            colorbar_title=f"Population (×{DISPLAY_POP_SCALE})",
            customdata=map_df[["population_display", "languages_tooltip"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Population: %{customdata[0]}<br>"
                "%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title_text=title,
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="equirectangular",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    clicked_points = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        key=key,
    )
    if clicked_points:
        location = clicked_points[0].get("location")
        if location:
            match = map_df.loc[map_df["iso_alpha"] == location]
            if not match.empty:
                return match.iloc[0]["country"]
    return None


# --- State Management ---
if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False
if "history" not in st.session_state:
    st.session_state.history = None
if "sim" not in st.session_state:
    st.session_state.sim = None
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

        history = []

        country_names = [c.name for c in sim.countries.values()]
        selected_country_name_live = st.sidebar.selectbox(
            "Select a country to view live",
            country_names,
            key="live_country_select",
        )

        for year in range(num_years):
            total_migrations = sim.run_step()
            stats = sim.get_stats()
            history.append(stats)

            # --- Update Live Metrics ---
            world_pop = sum(c_stats["population"] for c_stats in stats.values())
            year_text.metric("Year", f"{year + 1}/{num_years}")
            pop_text.metric("World Population", format_population(world_pop))
            migration_text.metric("Migrations This Year", f"{total_migrations:,}")

            # --- Update Live Map ---
            map_df = prepare_map_dataframe(sim, stats)
            with map_placeholder.container():
                clicked_country_live = render_population_map(
                    map_df, "World Population Map (Live)", "live_map"
                )
            if clicked_country_live and clicked_country_live in country_names:
                st.session_state.selected_country_index = country_names.index(clicked_country_live)
                st.session_state.live_country_select = clicked_country_live
                selected_country_name_live = clicked_country_live

            # --- Update Live Charts for Selected Country ---
            with charts_placeholder.container():
                if selected_country_name_live:
                    selected_country = next(
                        c for c in sim.countries.values() if c.name == selected_country_name_live
                    )

                    # Population Chart
                    st.subheader(f"Population: {selected_country.name}")
                    pop_history_scaled = [
                        year_stats[selected_country.id]["population"] * DISPLAY_POP_SCALE
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
                    st.subheader(f"Languages: {selected_country.name}")
                    lang_history = [
                        year_stats[selected_country.id]["language_prevalence"] for year_stats in history
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

                    active_lang_cols = lang_df.columns[
                        (lang_df.drop(columns="Year") != 0).any()
                    ]
                    cols_to_show = ["Year"] + [col for col in active_lang_cols if col != "Year"]
                    lang_df = lang_df[cols_to_show]

                    st.line_chart(lang_df, x="Year")

        st.session_state.history = history
        st.session_state.running = False
        st.success("Simulation Complete!")
        # Rerun to clear live view and show final results
        st.rerun()

if (
    st.session_state.simulation_run
    and st.session_state.history
    and not st.session_state.get("running", False)
):
    sim = st.session_state.sim
    history = st.session_state.history
    num_years_run = len(history)

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
        final_map_df = prepare_map_dataframe(sim, final_year_stats)
        with st.container():
            clicked_country_final = render_population_map(
                final_map_df, "Final Year Population Map", "final_map"
            )
        if clicked_country_final and clicked_country_final in country_names:
            st.session_state.selected_country_index = country_names.index(clicked_country_final)

        # --- Country Selector with Arrows ---
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("⬅️ Previous"):
                st.session_state.selected_country_index = (
                    st.session_state.selected_country_index - 1 + num_countries
                ) % num_countries
                st.rerun()
        with col3:
            if st.button("Next ➡️"):
                st.session_state.selected_country_index = (
                    st.session_state.selected_country_index + 1
                ) % num_countries
                st.rerun()

        selected_index = st.session_state.selected_country_index % num_countries
        selected_country_name = country_names[selected_index]
        with col2:
            st.markdown(
                f"<h3 style='text-align: center;'>{selected_country_name}</h3>",
                unsafe_allow_html=True,
            )

        # Get selected country object
        selected_country = next(
            (c for c in sim.countries.values() if c.name == selected_country_name), None
        )

        if selected_country:
            # --- Population Chart ---
            st.subheader(f"Population Over Time: {selected_country.name}")
            pop_history_scaled = [
                year_stats[selected_country.id]["population"] * DISPLAY_POP_SCALE
                for year_stats in history
            ]
            pop_chart = pd.DataFrame(
                {
                    "Year": range(1, num_years_run + 1),
                    f"Population (×{DISPLAY_POP_SCALE})": pop_history_scaled,
                }
            )
            st.line_chart(pop_chart, x="Year", y=f"Population (×{DISPLAY_POP_SCALE})")

            # --- Language Prevalence Chart ---
            st.subheader(f"Language Prevalence Over Time: {selected_country.name}")
            lang_history = [
                year_stats[selected_country.id]["language_prevalence"] for year_stats in history
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

            active_lang_cols = lang_df.columns[
                (lang_df.loc[:, lang_df.columns != "Year"] != 0).any()
            ]
            cols_to_show = ["Year"] + [col for col in active_lang_cols if col != "Year"]
            lang_df = lang_df[cols_to_show]

            st.line_chart(lang_df, x="Year")

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

