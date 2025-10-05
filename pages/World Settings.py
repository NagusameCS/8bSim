import streamlit as st
from simulation.settings import load_settings, save_settings, reset_to_defaults

st.set_page_config(page_title="World Settings", layout="wide")
st.title("World Settings")
st.caption("Tune the world economy, language diffusion, migration, and population parameters. Settings are saved to disk.")

settings = load_settings()

with st.sidebar:
    st.subheader("Presets")
    if st.button("Reset to Defaults"):
        settings = reset_to_defaults()
        st.success("Settings reset to defaults.")

with st.form("settings_form"):
    tabs = st.tabs(["Economy", "Population", "Language", "Migration"]) 

    with tabs[0]:
        st.markdown("### Economy")
        econ = settings["economy"]
        c1, c2, c3 = st.columns(3)
        econ["alpha"] = c1.number_input("Capital Share (α)", 0.1, 0.9, float(econ["alpha"]), 0.01, help="Share of output attributed to capital in Cobb-Douglas production.")
        econ["delta"] = c2.number_input("Depreciation (δ)", 0.0, 0.5, float(econ["delta"]), 0.005, help="Fraction of capital stock that depreciates each year.")
        econ["g_A"] = c3.number_input("Baseline TFP Growth (gA)", 0.0, 0.1, float(econ["g_A"]), 0.001, help="Baseline annual technology/productivity growth.")
        c4, c5, c6 = st.columns(3)
        econ["sigma_shock"] = c4.number_input("TFP Shock Volatility (σ)", 0.0, 0.2, float(econ["sigma_shock"]), 0.001, help="Std dev of random productivity shocks each year.")
        econ["k_y_ratio_init"] = c5.number_input("Initial K/Y Ratio", 0.5, 10.0, float(econ["k_y_ratio_init"]), 0.1, help="Initial capital to output ratio used to seed the economy.")
        econ["labor_share_working_age"] = c6.number_input("Working-age Labor Share", 0.0, 1.0, float(econ["labor_share_working_age"]), 0.01, help="Fraction of population treated as effective labor.")
        c7, c8, c9 = st.columns(3)
        econ["neighbor_spillover"] = c7.number_input("Neighbor Growth Spillover", 0.0, 0.5, float(econ["neighbor_spillover"]), 0.01, help="How much neighbor GDPpc growth feeds into local TFP growth.")
        econ["investment_rate_base"] = c8.number_input("Investment Rate (Base)", 0.0, 1.0, float(econ["investment_rate_base"]), 0.01, help="Share of output invested in capital formation.")
        econ["investment_rate_rich"] = c9.number_input("Investment Rate (Rich)", 0.0, 1.0, float(econ["investment_rate_rich"]), 0.01, help="Investment share used for countries above the rich threshold.")
        econ["rich_threshold"] = st.number_input("Rich GDP per Capita Threshold", 1000.0, 100000.0, float(econ["rich_threshold"]), 100.0, help="Threshold for using the 'rich' investment rate.")

    with tabs[1]:
        st.markdown("### Population")
        pop = settings["population"]
        c1, c2, c3 = st.columns(3)
        pop["prod_to_births_multiplier"] = c1.number_input("Production→Births Multiplier", 0.0, 2.0, float(pop["prod_to_births_multiplier"]), 0.05, help="How strongly production growth scales births in a year.")
        pop["baseline_target_growth"] = c2.number_input("Baseline Target Growth", -0.05, 0.1, float(pop["baseline_target_growth"]), 0.001, help="Default net population growth rate if fertility inference is missing.")
        c3a, c3b = st.columns(2)
        pop["min_target_growth"] = c3a.number_input("Min Target Growth", -0.2, 0.0, float(pop["min_target_growth"]), 0.001, help="Lower bound on per-country net target growth.")
        pop["max_target_growth"] = c3b.number_input("Max Target Growth", 0.0, 0.5, float(pop["max_target_growth"]), 0.001, help="Upper bound on per-country net target growth.")

    with tabs[2]:
        st.markdown("### Language")
        lang = settings["language"]
        c1, c2, c3 = st.columns(3)
        lang["w_local"] = c1.number_input("Local Weight", 0.0, 5.0, float(lang["w_local"]), 0.05, help="Weight of in-country prevalence in adoption probability.")
        lang["w_neighbor"] = c2.number_input("Neighbor Weight", 0.0, 5.0, float(lang["w_neighbor"]), 0.05, help="Weight of neighbors' prevalence.")
        lang["w_global"] = c3.number_input("Global Weight", 0.0, 5.0, float(lang["w_global"]), 0.05, help="Weight of global language power.")
        c4, c5, c6 = st.columns(3)
        lang["aggression"] = c4.number_input("Aggression Multiplier", 0.0, 5.0, float(lang["aggression"]), 0.05, help="How hard globally powerful languages push adoption.")
        lang["max_langs_per_agent"] = c5.number_input("Max Languages per Agent", 1, 10, int(lang["max_langs_per_agent"]), help="Upper cap on languages known per agent.")
        c7, c8, c9 = st.columns(3)
        lang["campaign_base"] = c7.number_input("Campaign Base Rate", 0.0, 0.5, float(lang["campaign_base"]), 0.005, help="Base chance per year for campaign adoption.")
        lang["campaign_global_weight"] = c8.number_input("Campaign Global Weight", 0.0, 1.0, float(lang["campaign_global_weight"]), 0.01, help="Sensitivity of campaigns to global power.")
        lang["campaign_neighbor_weight"] = c9.number_input("Campaign Neighbor Weight", 0.0, 1.0, float(lang["campaign_neighbor_weight"]), 0.01, help="Sensitivity of campaigns to neighbor prevalence.")

    with tabs[3]:
        st.markdown("### Migration")
        mig = settings["migration"]
        c1, c2, c3 = st.columns(3)
        mig["global_wave_probability"] = c1.number_input("Global Wave Probability", 0.0, 1.0, float(mig["global_wave_probability"]), 0.01, help="Chance of a global migration wave in any year.")
        mig["global_wave_fraction"] = c2.number_input("Global Wave Fraction", 0.0, 0.05, float(mig["global_wave_fraction"]), 0.0005, help="Fraction of world population that moves during a global wave.")
        mig["gdp_weight"] = c3.number_input("GDP Weight", 0.0, 5.0, float(mig["gdp_weight"]), 0.1, help="How strongly GDP per capita affects migration destination choice.")
        mig["language_bonus"] = st.number_input("Language Bonus", 0.0, 5.0, float(mig["language_bonus"]), 0.1, help="Bonus factor if an agent's language is prevalent at destination.")

    submitted = st.form_submit_button("Save Settings")
    if submitted:
        save_settings(settings)
        st.success("Settings saved.")

st.markdown("""
Tips:
- Use a moderate aggression and campaign rate to avoid unrealistic overnight dominance.
- Increase investment and TFP growth for faster catch-up in developing economies.
- Constrain max languages per agent to maintain realism for multilingual adoption.
""")
