import json
import random
from .simulation_models import Country, Language, Agent
from .simulation import Simulation

def setup_from_config(config_path):
    """Sets up the simulation from a JSON config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    languages = [Language(lang['id'], lang['name']) for lang in config['languages']]
    lang_map = {lang.name: lang for lang in languages}

    countries = []
    for country_data in config['countries']:
        country = Country(
            country_id=country_data['id'],
            name=country_data['name'],
            gdp_per_capita=country_data['gdp_per_capita'],
            life_expectancy=country_data['life_expectancy'],
            fertility_rates=country_data['fertility_rates'],
            income_distribution=country_data['income_distribution'],
            migration_rate=country_data['migration_rate'],
            neighbors=country_data['neighbors'],
            initial_population=country_data.get('initial_population'),
            strategy=country_data.get('strategy', [])
        )
        countries.append(country)

    agent_id_counter = 0
    for country_data in config['countries']:
        country = next(c for c in countries if c.id == country_data['id'])
        N = country_data['initial_population']
        # Enforce 1/3 mono, 1/3 bi, 1/3 tri target buckets
        mono_target = N // 3
        bi_target = N // 3
        tri_target = N - mono_target - bi_target
        counts = {1: 0, 2: 0, 3: 0}
        for _ in range(N):
            age = random.randint(0, 80)
            stratum = random.choices(
                list(country.income_distribution.keys()),
                weights=list(country.income_distribution.values()),
                k=1
            )[0]
            
            agent = Agent(agent_id_counter, country.id, age, stratum)
            agent_id_counter += 1

            # Assign initial languages based on probabilities in config
            for lang_name, probability in country_data['initial_languages'].items():
                if random.random() < probability:
                    agent.languages.append(lang_map[lang_name])
            
            # Ensure agent has at least one language if any are defined
            if not agent.languages and country_data['initial_languages']:
                # Assign the most probable language as a fallback
                primary_lang = max(country_data['initial_languages'], key=country_data['initial_languages'].get)
                agent.languages.append(lang_map[primary_lang])

            # Enforce cap of 3 languages
            if len(agent.languages) > 3:
                agent.languages = agent.languages[:3]

            # Adjust to match mono/bi/tri targets as closely as possible
            desired = 1
            if counts[1] >= mono_target and counts[2] < bi_target:
                desired = 2
            if counts[1] >= mono_target and counts[2] >= bi_target:
                desired = 3
            # Trim or augment with most probable locals to reach desired count
            if len(agent.languages) > desired:
                agent.languages = agent.languages[:desired]
            elif len(agent.languages) < desired:
                sorted_langs = sorted(country_data['initial_languages'].items(), key=lambda kv: kv[1], reverse=True)
                for name, _p in sorted_langs:
                    if len(agent.languages) >= desired:
                        break
                    L = lang_map[name]
                    if L not in agent.languages:
                        agent.languages.append(L)
            counts[desired] += 1

            country.agents.append(agent)

    return countries, languages

if __name__ == "__main__":
    # Setup the simulation from the config file
    initial_countries, initial_languages = setup_from_config('simulation/config.json')
    sim = Simulation(countries=initial_countries, languages=initial_languages, scale_factor=1000)

    # Run the simulation for 20 years
    for year in range(20):
        sim.run_step()
        sim.print_stats()

    # Print a final report
    print("\n--- Simulation End Report ---")
    for country in sim.countries.values():
        print(f"\nCountry: {country.name}")
        print(f"  - Final Population: {len(country.agents)}")
        prevalence_str = ", ".join([f"{sim.languages[l_id].name}: {p:.2%}" for l_id, p in country.language_prevalence.items()])
        print(f"  - Language Prevalence: {prevalence_str if prevalence_str else 'N/A'}")
