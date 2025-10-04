from simulation_models import Country, Language, Agent
from simulation_logic import Simulation
import random

def setup_initial_state():
    """Creates a simple initial state for the simulation."""
    
    # 1. Create Languages
    lang1 = Language(lang_id=1, name="English")
    lang2 = Language(lang_id=2, name="Spanish")
    languages = [lang1, lang2]

    # 2. Create Countries
    # Simplified life expectancy and fertility for demonstration
    life_expectancy_A = {0: 0.99, 10: 0.99, 20: 0.98, 30: 0.98, 40: 0.97, 50: 0.96, 60: 0.9, 70: 0.8, 80: 0.6, 90: 0.4}
    fertility_A = {20: 0.05, 30: 0.04}

    country_A = Country(country_id=101, name="CountryA", gdp_per_capita=50000, 
                        life_expectancy=life_expectancy_A, fertility_rates=fertility_A,
                        income_distribution={'low': 0.2, 'middle': 0.5, 'high': 0.3},
                        migration_rate=0.01)

    life_expectancy_B = {0: 0.98, 10: 0.98, 20: 0.97, 30: 0.97, 40: 0.96, 50: 0.95, 60: 0.85, 70: 0.75, 80: 0.5, 90: 0.3}
    fertility_B = {20: 0.06, 30: 0.05}

    country_B = Country(country_id=102, name="CountryB", gdp_per_capita=30000,
                        life_expectancy=life_expectancy_B, fertility_rates=fertility_B,
                        income_distribution={'low': 0.6, 'middle': 0.3, 'high': 0.1},
                        migration_rate=0.02)

    # Set up neighbors
    country_A.neighbors.append(country_B)
    country_B.neighbors.append(country_A)
    
    countries = [country_A, country_B]

    # 3. Create Initial Population of Agents
    agent_id_counter = 0
    # Populate Country A
    for _ in range(100): # Create 100 agents for Country A
        age = random.randint(0, 80)
        stratum = random.choice(['low', 'mid', 'high'])
        agent = Agent(agent_id=agent_id_counter, country_id=country_A.id, age=age, economic_stratum=stratum)
        agent.languages.append(lang1) # All start with English
        if random.random() < 0.2: # 20% also speak Spanish
            agent.languages.append(lang2)
        country_A.agents.append(agent)
        agent_id_counter += 1

    # Populate Country B
    for _ in range(80): # Create 80 agents for Country B
        age = random.randint(0, 80)
        stratum = random.choice(['low', 'mid', 'high'])
        agent = Agent(agent_id=agent_id_counter, country_id=country_B.id, age=age, economic_stratum=stratum)
        agent.languages.append(lang2) # All start with Spanish
        if random.random() < 0.1: # 10% also speak English
            agent.languages.append(lang1)
        country_B.agents.append(agent)
        agent_id_counter += 1
        
    return countries, languages

if __name__ == "__main__":
    # Setup the simulation
    initial_countries, initial_languages = setup_initial_state()
    sim = Simulation(countries=initial_countries, languages=initial_languages)

    # Run the simulation for 20 years
    for year in range(20):
        sim.run_step()
        # Optional: Print yearly stats to observe changes
        print(f"\n--- Year {year} Stats ---")
        for country in sim.countries.values():
            print(f"Country: {country.name}, Population: {len(country.agents)}")
            # Format prevalence for readability
            prevalence_str = ", ".join([f"{sim.languages[l_id].name}: {p:.2%}" for l_id, p in country.language_prevalence.items()])
            print(f"  Prevalence: {prevalence_str}")


    # Print some final stats
    print("\n--- Simulation End Report ---")
    for country in sim.countries.values():
        living_agents = [agent for agent in country.agents if agent.alive]
        print(f"\nCountry: {country.name}")
        print(f"  - Final Population: {len(living_agents)}")
        print(f"  - Language Prevalence: {country.language_prevalence}")
