import random
from simulation_models import Agent, Country, Language

class Simulation:
    """Manages the overall simulation and its yearly steps."""
    def __init__(self, countries, languages):
        self.countries = {c.id: c for c in countries}
        self.languages = {l.id: l for l in languages}
        self.current_year = 0
        self.dead_agent_archive = {} # {year: [dead_agents]}

        # Find the highest initial agent ID to avoid collisions
        max_id = 0
        for country in self.countries.values():
            if country.agents:
                max_id = max(max_id, max(agent.id for agent in country.agents))
        self.next_agent_id = max_id + 1

    def _initialize_population(self, population_data):
        """Initializes the agent population based on provided data."""
        # This method will be responsible for creating agent objects
        # and assigning them to countries according to census data.
        # For now, it's a placeholder.
        print("Initializing population...")
        pass

    def _age_and_mortality(self):
        """Ages all agents and handles mortality."""
        for country in self.countries.values():
            new_agent_list = []
            for agent in country.agents:
                if not agent.alive:
                    continue

                agent.age += 1
                
                # Mortality check
                # A more detailed implementation would use the country's life expectancy table
                age_group = (agent.age // 10) * 10 
                survival_prob = country.life_expectancy.get(age_group, 0.99)
                if random.random() > survival_prob:
                    agent.alive = False
                    if self.current_year not in self.dead_agent_archive:
                        self.dead_agent_archive[self.current_year] = []
                    self.dead_agent_archive[self.current_year].append(agent)
                else:
                    new_agent_list.append(agent)
            
            # The list of agents in the country is replaced with the list of survivors
            country.agents = new_agent_list


    def _births(self):
        """Handles new births in each country."""
        for country in self.countries.values():
            living_agents = [agent for agent in country.agents if agent.alive]
            potential_parents = [agent for agent in living_agents if 20 <= agent.age < 40]
            
            if not potential_parents:
                continue

            num_births = 0
            for age_group, rate in country.fertility_rates.items():
                num_in_group = len([p for p in potential_parents if p.age // 10 * 10 == age_group])
                num_births += int(num_in_group * rate)

            for _ in range(num_births):
                # Create a new child
                new_agent = Agent(
                    agent_id=self.next_agent_id,
                    country_id=country.id,
                    age=0,
                    economic_stratum=self._get_random_stratum(country.income_distribution)
                )
                self.next_agent_id += 1

                # Inherit languages from two random parents
                parent1 = random.choice(potential_parents)
                parent2 = random.choice(potential_parents)
                
                parent_langs = set(parent1.languages + parent2.languages)
                
                if len(parent_langs) > 3:
                    # Keep the 3 with the highest local prevalence
                    sorted_langs = sorted(parent_langs, 
                                          key=lambda l: country.language_prevalence.get(l.id, 0), 
                                          reverse=True)
                    new_agent.languages = sorted_langs[:3]
                elif not parent_langs:
                    # Edge case: if no languages, assign a dominant one from a parent
                    if parent1.languages:
                        new_agent.languages.append(random.choice(parent1.languages))
                else:
                    new_agent.languages = list(parent_langs)

                country.agents.append(new_agent)

    def _get_random_stratum(self, distribution):
        """Selects an economic stratum based on a probability distribution."""
        rand = random.random()
        cumulative = 0
        for stratum, probability in distribution.items():
            cumulative += probability
            if rand < cumulative:
                return stratum
        return list(distribution.keys())[-1] # Fallback

    def _language_acquisition(self):
        """Handles language acquisition for agents, based on age, stratum, and environment."""
        stratum_multipliers = {"high": 1.5, "middle": 1.0, "low": 0.75}
        base_prob_child = 0.1
        base_prob_adult = 0.05

        for country in self.countries.values():
            if not country.language_prevalence:
                continue # Skip if no languages are prevalent in the country

            for agent in country.agents:
                if not agent.alive or len(agent.languages) >= 3:
                    continue

                # Determine learning probability based on age
                if 0 <= agent.age <= 10:
                    prob = base_prob_child
                elif agent.age > 10:
                    # Probability decreases linearly for adults
                    prob = base_prob_adult * max(0, 1 - (agent.age - 11) / 90)
                else:
                    continue
                
                # Adjust probability by economic stratum
                modified_prob = prob * stratum_multipliers.get(agent.economic_stratum, 1.0)

                if random.random() < modified_prob:
                    # Agent gets to learn a new language this year
                    
                    # Find candidate languages (those the agent doesn't already know)
                    agent_lang_ids = {lang.id for lang in agent.languages}
                    candidate_langs = [
                        self.languages[lang_id] for lang_id in country.language_prevalence 
                        if lang_id not in agent_lang_ids
                    ]

                    if not candidate_langs:
                        continue

                    # Weigh candidates by prevalence
                    weights = [
                        country.language_prevalence[lang.id] for lang in candidate_langs
                    ]
                    
                    if sum(weights) > 0:
                        new_language = random.choices(candidate_langs, weights=weights, k=1)[0]
                        agent.languages.append(new_language)

    def _language_attrition(self):
        """Handles language attrition for agents."""
        base_attrition_prob = 0.02

        for country in self.countries.values():
            if not country.language_prevalence:
                continue

            # Get the top 3 languages by prevalence in the country
            top_3_lang_ids = set(
                sorted(country.language_prevalence, 
                       key=country.language_prevalence.get, 
                       reverse=True)[:3]
            )

            for agent in country.agents:
                if not agent.alive or len(agent.languages) <= 1:
                    continue

                languages_to_remove = []
                for lang in agent.languages:
                    # Attrition only applies to languages NOT in the top 3
                    if lang.id not in top_3_lang_ids:
                        prevalence = country.language_prevalence.get(lang.id, 0)
                        # Probability of attrition increases as prevalence decreases
                        attrition_prob = base_attrition_prob * (1 - prevalence)
                        
                        if random.random() < attrition_prob:
                            languages_to_remove.append(lang)
                
                if languages_to_remove:
                    # Ensure the agent doesn't lose all their languages
                    if len(agent.languages) - len(languages_to_remove) < 1:
                        # If attrition would remove all languages, only remove some
                        # This is a safeguard; the len(agent.languages) <= 1 check should prevent this
                        languages_to_remove.pop() 

                    for lang in languages_to_remove:
                        agent.languages.remove(lang)


    def _migration(self):
        """Handles agent migration between countries."""
        linguistic_bonus = 0.5
        migrants_by_destination = {country_id: [] for country_id in self.countries}
        
        source_countries = list(self.countries.values())
        if len(source_countries) < 2:
            return # Migration is not possible with only one country

        for source_country in source_countries:
            num_migrants = int(len(source_country.agents) * source_country.migration_rate)
            if num_migrants == 0 or not source_country.agents:
                continue

            potential_migrants = random.sample(source_country.agents, k=min(num_migrants, len(source_country.agents)))

            for agent in potential_migrants:
                if not agent.alive:
                    continue

                destination_scores = {}
                for dest_country in self.countries.values():
                    if dest_country.id == source_country.id:
                        continue

                    # Economic incentive
                    gdp_ratio = dest_country.gdp_per_capita / source_country.gdp_per_capita
                    
                    # Linguistic incentive
                    bonus = 0
                    agent_lang_ids = {lang.id for lang in agent.languages}
                    if any(lang_id in dest_country.language_prevalence for lang_id in agent_lang_ids):
                        bonus = linguistic_bonus

                    score = gdp_ratio * (1 + bonus)
                    destination_scores[dest_country.id] = score
                
                if not destination_scores:
                    continue

                # Choose the best destination
                best_dest_id = max(destination_scores, key=destination_scores.get)
                
                # Move the agent
                agent.country_id = best_dest_id
                migrants_by_destination[best_dest_id].append(agent)
                source_country.agents.remove(agent)

        # Add migrants to their new countries
        for country_id, new_arrivals in migrants_by_destination.items():
            self.countries[country_id].agents.extend(new_arrivals)

    
    def _update_country_stats(self):
        """Updates country-level statistics like language prevalence."""
        for country in self.countries.values():
            country.update_language_prevalence()

    def run_step(self):
        """Runs a single year of the simulation."""
        print(f"--- Year {self.current_year} ---")
        
        # F. Update Country Stats (from previous year's state)
        self._update_country_stats()
        
        # A. Aging & Mortality
        self._age_and_mortality()
        
        # B. Births
        self._births()
        
        # C. Language Acquisition
        self._language_acquisition()
        
        # D. Language Attrition
        self._language_attrition()
        
        # E. Migration
        self._migration()
        
        self.current_year += 1

    def run(self, years: int):
        """Runs the simulation for a given number of years."""
        for _ in range(years):
            self.run_step()
        print("Simulation finished.")
