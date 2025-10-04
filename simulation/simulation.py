import math
import random
from .simulation_models import Agent

class Simulation:
    """Manages the overall simulation and its yearly steps."""
    def __init__(self, countries, languages, scale_factor=1):
        self.countries = {c.id: c for c in countries}
        self.languages = {l.id: l for l in languages}
        self.current_year = 0
        self.scale_factor = scale_factor
        
        # Scale up initial population
        for country in self.countries.values():
            # This is a simplified way to scale. A more robust implementation
            # might involve creating new agents with varied characteristics.
            current_pop = len(country.agents)
            if current_pop > 0:
                num_to_add = (current_pop * self.scale_factor) - current_pop
                new_agents = random.choices(country.agents, k=num_to_add)
                country.agents.extend(new_agents)

        self.next_agent_id = sum(len(c.agents) for c in self.countries.values()) + 1

    def _age_and_mortality(self):
        """Ages all agents and handles mortality."""
        for country in self.countries.values():
            survivors = []
            for agent in country.agents:
                agent.age += 1
                age_group = (agent.age // 10) * 10
                survival_prob = country.life_expectancy.get(age_group, 0.99)
                if random.random() < survival_prob:
                    survivors.append(agent)
            country.agents = survivors

    def _births(self):
        """Handles new births in each country."""
        for country in self.countries.values():
            living_agents = country.agents
            potential_parents = [a for a in living_agents if 20 <= a.age < 40]
            if not potential_parents:
                continue

            num_births = 0
            for age_group, rate in country.fertility_rates.items():
                num_in_group = len([p for p in potential_parents if p.age // 10 * 10 == age_group])
                num_births += int(num_in_group * rate)

            for _ in range(num_births):
                stratum = self._get_random_stratum(country.income_distribution)
                new_agent = Agent(self.next_agent_id, country.id, 0, stratum)
                self.next_agent_id += 1

                parent1, parent2 = random.choices(potential_parents, k=2)
                parent_langs = set(parent1.languages + parent2.languages)
                
                if len(parent_langs) > 3:
                    sorted_langs = sorted(parent_langs, 
                                          key=lambda l: country.language_prevalence.get(l.id, 0), 
                                          reverse=True)
                    new_agent.languages = sorted_langs[:3]
                else:
                    new_agent.languages = list(parent_langs)
                country.agents.append(new_agent)

    def _get_random_stratum(self, distribution):
        rand = random.random()
        cumulative = 0
        for stratum, probability in distribution.items():
            cumulative += probability
            if rand < cumulative:
                return stratum
        return list(distribution.keys())[-1]

    def _language_acquisition(self):
        """Handles language acquisition for agents with non-linear adoption dynamics."""
        stratum_multipliers = {"high": 1.5, "middle": 1.0, "low": 0.75}
        education_multipliers = {0: 0.8, 1: 1.0, 2: 1.2, 3: 1.5}
        base_prob_child, base_prob_adult = 0.1, 0.05
        network_steepness = 8

        for country in self.countries.values():
            if not country.language_prevalence:
                continue

            prevalence_cache = {
                lang_id: country.language_prevalence.get(lang_id, 0.0)
                for lang_id in country.language_prevalence
            }

            for agent in country.agents:
                if len(agent.languages) >= 3:
                    continue

                age_factor = (
                    base_prob_child
                    if 0 <= agent.age <= 10
                    else base_prob_adult * max(0, 1 - (agent.age - 11) / 90)
                )
                social_factor = stratum_multipliers.get(agent.economic_stratum, 1.0)
                education_factor = education_multipliers.get(agent.education_level, 1.0)

                agent_lang_ids = {lang.id for lang in agent.languages}
                candidate_lang_ids = [
                    lang_id
                    for lang_id in country.language_prevalence
                    if lang_id not in agent_lang_ids
                ]
                if not candidate_lang_ids:
                    continue

                network_pressures = [
                    1
                    / (1 + math.exp(-network_steepness * (prevalence_cache.get(lang_id, 0.0) - 0.35)))
                    for lang_id in candidate_lang_ids
                ]
                diversity_interests = [
                    (1 - prevalence_cache.get(lang_id, 0.0)) ** 1.5
                    for lang_id in candidate_lang_ids
                ]

                network_pressure = max(network_pressures) if network_pressures else 0.0
                diversity_interest = (
                    sum(diversity_interests) / len(diversity_interests)
                    if diversity_interests
                    else 0.0
                )

                pressure_multiplier = 0.55 + 0.35 * network_pressure + 0.1 * diversity_interest
                modified_prob = age_factor * social_factor * education_factor * pressure_multiplier
                modified_prob = max(0.0, min(0.75, modified_prob))

                if random.random() >= modified_prob:
                    continue

                candidate_langs = [self.languages[lang_id] for lang_id in candidate_lang_ids]
                weights = []
                for lang in candidate_langs:
                    prevalence = prevalence_cache.get(lang.id, 0.0)
                    network_weight = 1 / (
                        1 + math.exp(-network_steepness * (prevalence - 0.35))
                    )
                    saturation_penalty = 0.6 + 0.4 * (1 - prevalence)
                    rarity_boost = (1 - prevalence) ** 1.2
                    combined_weight = (network_weight ** 1.5) * saturation_penalty + 0.3 * rarity_boost
                    weights.append(max(1e-3, combined_weight))

                total_weight = sum(weights)
                if total_weight <= 0:
                    continue

                normalized_weights = [w / total_weight for w in weights]
                new_language = random.choices(candidate_langs, weights=normalized_weights, k=1)[0]
                agent.languages.append(new_language)

    def _language_attrition(self):
        """Handles language attrition for agents."""
        base_attrition_prob = 0.02
        for country in self.countries.values():
            if not country.language_prevalence: continue
            top_3_lang_ids = set(sorted(country.language_prevalence, key=country.language_prevalence.get, reverse=True)[:3])

            for agent in country.agents:
                if len(agent.languages) <= 1: continue
                
                langs_to_remove = [lang for lang in agent.languages if lang.id not in top_3_lang_ids and random.random() < base_attrition_prob * (1 - country.language_prevalence.get(lang.id, 0))]
                
                if len(agent.languages) - len(langs_to_remove) < 1 and langs_to_remove:
                    langs_to_remove.pop()
                
                for lang in langs_to_remove:
                    agent.languages.remove(lang)

    def get_stats(self):
        """Returns a dictionary with current simulation statistics."""
        stats = {}
        for country_id, country in self.countries.items():
            stats[country_id] = {
                "population": len(country.agents),
                "language_prevalence": country.language_prevalence
            }
        return stats

    def run_step(self):
        """Runs a single year of the simulation."""
        self.current_year += 1
        self.update_language_prevalence()
        self._age_and_mortality()
        self._births()
        self._language_acquisition()
        self._language_attrition()
        self._migration()

    def update_language_prevalence(self):
        """Updates the language prevalence for all countries."""
        for country in self.countries.values():
            country.update_language_prevalence()

    def _migration(self):
        """Handles agent migration between countries."""
        migrants_by_destination = {country_id: [] for country_id in self.countries}
        migrated_agent_ids = set()
        total_migrations = 0

        # Global migration event (small chance)
        if random.random() < 0.1:  # 10% chance of a global migration wave each year
            all_agents = [agent for country in self.countries.values() for agent in country.agents]
            num_global_migrants = int(len(all_agents) * 0.001)  # 0.1% of total population
            
            # Ensure we don't sample more than available agents
            k = min(num_global_migrants, len(all_agents))
            if k > 0:
                global_migrants = random.sample(all_agents, k=k)

                for agent in global_migrants:
                    if agent.id in migrated_agent_ids:
                        continue

                    source_country = self.countries[agent.country_id]

                    # Destination choice is global, not just neighbors
                    destination_scores = {}
                    for dest_id, dest_country in self.countries.items():
                        if dest_id == source_country.id: continue
                        gdp_ratio = dest_country.gdp_per_capita / source_country.gdp_per_capita
                        bonus = 0.5 if any(lang.id in dest_country.language_prevalence for lang in agent.languages) else 0
                        destination_scores[dest_id] = gdp_ratio * (1 + bonus)

                    if not destination_scores: continue
                    best_dest_id = max(destination_scores, key=destination_scores.get)

                    agent.country_id = best_dest_id
                    migrants_by_destination[best_dest_id].append(agent)
                    source_country.agents.remove(agent)
                    migrated_agent_ids.add(agent.id)
                    total_migrations += 1

        for source_country in self.countries.values():
            # Handle both positive (immigration) and negative (emigration) rates
            migration_rate = source_country.migration_rate
            if migration_rate == 0:
                continue

            num_migrants = int(len(source_country.agents) * abs(migration_rate))
            if num_migrants == 0 or not source_country.agents:
                continue
            
            # Ensure k is not negative for random.sample
            k = min(num_migrants, len(source_country.agents))
            if k <= 0:
                continue

            # Filter out agents who have already migrated globally
            potential_migrants_pool = [agent for agent in source_country.agents if agent.id not in migrated_agent_ids]
            if not potential_migrants_pool:
                continue
            
            k = min(k, len(potential_migrants_pool))
            potential_migrants = random.sample(potential_migrants_pool, k=k)

            # If the rate is negative, it's emigration, and we don't need to calculate a destination
            if migration_rate < 0:
                for agent in potential_migrants:
                    source_country.agents.remove(agent)
                    migrated_agent_ids.add(agent.id) # Also track emigrants
                    total_migrations += 1
                continue  # Move to the next country

            for agent in potential_migrants:
                destination_scores = {}
                for dest_id in source_country.neighbor_ids:
                    dest_country = self.countries[dest_id]
                    gdp_ratio = dest_country.gdp_per_capita / source_country.gdp_per_capita
                    bonus = 0.5 if any(lang.id in dest_country.language_prevalence for lang in agent.languages) else 0
                    destination_scores[dest_id] = gdp_ratio * (1 + bonus)

                if not destination_scores: continue
                best_dest_id = max(destination_scores, key=destination_scores.get)

                agent.country_id = best_dest_id
                migrants_by_destination[best_dest_id].append(agent)
                source_country.agents.remove(agent)
                migrated_agent_ids.add(agent.id)
                total_migrations += 1

        for country_id, new_arrivals in migrants_by_destination.items():
            self.countries[country_id].agents.extend(new_arrivals)
        
        return total_migrations

    def _update_country_stats(self):
        """Updates country-level statistics like language prevalence."""
        for country in self.countries.values():
            country.update_language_prevalence()

    def run_step(self):
        """Runs a single year of the simulation."""
        self._update_country_stats()
        self._age_and_mortality()
        self._births()
        self._education()
        self._update_economic_stratum()
        self._language_acquisition()
        self._language_attrition()
        total_migrations = self._migration()
        self.current_year += 1
        return total_migrations

    def _education(self):
        """Handles education progression for agents."""
        for country in self.countries.values():
            for agent in country.agents:
                # Agents can only improve education between ages 5 and 30
                if 5 <= agent.age <= 30 and agent.education_level < 3:
                    # Higher chance to educate in higher economic stratum
                    stratum_bonus = {"high": 0.1, "middle": 0.05, "low": 0.02}
                    chance = 0.1 + stratum_bonus.get(agent.economic_stratum, 0)
                    if random.random() < chance:
                        agent.education_level += 1

    def _update_economic_stratum(self):
        """Updates agents' economic stratum based on age and education."""
        for country in self.countries.values():
            for agent in country.agents:
                # Stratum change is more likely between 25 and 60
                if 25 <= agent.age <= 60:
                    # Education is a major factor
                    education_bonus = agent.education_level * 0.05
                    # Small chance of random upward or downward mobility
                    mobility_chance = 0.05 + education_bonus

                    if random.random() < mobility_chance:
                        current_stratum_index = list(country.income_distribution.keys()).index(agent.economic_stratum)
                        if current_stratum_index > 0 and random.random() < 0.4: # 40% chance of downward
                            agent.economic_stratum = list(country.income_distribution.keys())[current_stratum_index - 1]
                        elif current_stratum_index < len(country.income_distribution.keys()) - 1: # 60% chance of upward
                             agent.economic_stratum = list(country.income_distribution.keys())[current_stratum_index + 1]

    def print_stats(self):
        """Prints statistics for the current year."""
        print(f"\n--- Year {self.current_year} Stats ---")
        for country in self.countries.values():
            print(f"Country: {country.name}, Population: {len(country.agents)}")
            prevalence_str = ", ".join([f"{self.languages[l_id].name}: {p:.2%}" for l_id, p in country.language_prevalence.items()])
            print(f"  Prevalence: {prevalence_str if prevalence_str else 'N/A'}")