import math
import random
from .simulation_models import Agent
from .settings import load_settings

class Simulation:
    """Manages the overall simulation and its yearly steps."""
    def __init__(self, countries, languages, scale_factor=1):
        self.countries = {c.id: c for c in countries}
        self.languages = {l.id: l for l in languages}
        self.current_year = 0
        self.scale_factor = scale_factor
        self.settings = load_settings()
        # Macro dynamics
        self.production_index = {c.id: 1.0 for c in countries}  # multiplicative index (starts at 1.0)
        self.production_growth = {}  # per-country annual growth rate for production index
        self.population_growth = {}  # per-country target net population growth rate (exogenous target)
        # Language influence context (computed each step)
        self._lang_neighbor_prevalence = {}
        self._lang_global_power = {}
        # Economy state per country: capital K and productivity A
        self._capital = {}
        self._tfp = {}
        # Event/history capture
        self.event_log = []  # list of dicts per year
        self._last_migration_flows = []  # list of {source, dest, count}
        self._last_ai_conquests = []  # list of (attacker_id, target_id)
        
        # Establish next agent id before any scaling
        cur_max_id = -1
        for c in self.countries.values():
            for a in c.agents:
                if a.id > cur_max_id:
                    cur_max_id = a.id
        self.next_agent_id = int(cur_max_id + 1)

        # Scale up initial population by cloning, not by duplicating references
        for country in self.countries.values():
            current_pop = len(country.agents)
            if current_pop > 0 and self.scale_factor > 1:
                num_to_add = (current_pop * self.scale_factor) - current_pop
                if num_to_add > 0:
                    bases = random.choices(country.agents, k=num_to_add)
                    for base in bases:
                        clone = self._clone_agent_from(base, country.id)
                        country.agents.append(clone)

        # Initialize production and population growth matrices with reasonable defaults
        for c in countries:
            # Production growth: mildly higher for emerging economies, taper for rich
            # Normalize GDP per capita roughly into [0, 1]
            gdp = max(500.0, float(c.gdp_per_capita))
            gdp_norm = min(1.0, max(0.0, (gdp - 500.0) / 70000.0))
            # Base between 0.8% and 2.5%
            prod_rate = 0.025 - 0.017 * gdp_norm
            self.production_growth[c.id] = max(0.005, min(0.03, prod_rate))

            # Population target net growth: derive from fertility mix, bounded
            avg_fert = 0.0
            if getattr(c, 'fertility_rates', None):
                try:
                    avg_fert = sum(c.fertility_rates.values()) / max(1, len(c.fertility_rates))
                except Exception:
                    avg_fert = 0.06
            # Start at slight positive, adjust by fertility relative to ~0.06 baseline, small penalty for very high GDP (aging)
            pop_rate = 0.003 + (avg_fert - 0.06) * 0.5 - gdp_norm * 0.002
            self.population_growth[c.id] = max(-0.01, min(0.03, pop_rate))

            # Initialize economy state (K and A) using settings
            econ = self.settings.get("economy", {})
            k_y = float(econ.get("k_y_ratio_init", 3.0))
            # Proxy Y0 ~ GDPpc * population
            Y0 = gdp * max(1, len(c.agents))
            K0 = max(1.0, k_y * Y0)
            self._capital[c.id] = K0
            self._tfp[c.id] = 1.0  # normalized starting TFP

        # Initialize basic geopolitics relations matrix [-1,1]
        self._relations = self._init_geopolitics_relations()

    def _clone_agent_from(self, base_agent: Agent, country_id: int) -> Agent:
        """Clone an agent with a new unique id (no shared references)."""
        a = Agent(self.next_agent_id, country_id, base_agent.age, base_agent.economic_stratum)
        self.next_agent_id += 1
        a.alive = True
        a.education_level = base_agent.education_level
        a.languages = list(base_agent.languages)
        return a

    def _init_geopolitics_relations(self):
        """Simple static geopolitics map using neighbor friendliness and a few rivalries."""
        relations = {cid: {oid: 0.0 for oid in self.countries.keys()} for cid in self.countries.keys()}
        # Friendly baseline for neighbors
        geo_cfg = self.settings.get("geopolitics", {})
        neigh_bonus = float(geo_cfg.get("neighbor_friend_bonus", 0.2))
        for cid, c in self.countries.items():
            for nid in c.neighbor_ids:
                if nid in self.countries:
                    relations[cid][nid] = min(1.0, relations[cid][nid] + neigh_bonus)
                    relations[nid][cid] = min(1.0, relations[nid][cid] + neigh_bonus)
        # A few rivalries by name
        name_to_id = {c.name: c.id for c in self.countries.values()}
        rival_entries = geo_cfg.get("rivalries", [])
        for entry in rival_entries:
            a, b, w = entry.get("a"), entry.get("b"), float(entry.get("w", -0.3))
            if a in name_to_id and b in name_to_id:
                ia, ib = name_to_id[a], name_to_id[b]
                relations[ia][ib] = w
                relations[ib][ia] = w
        # Explicit overrides by ids if provided
        overrides = geo_cfg.get("relations", {}) or {}
        try:
            for sa, row in overrides.items():
                ca = int(sa)
                for sb, val in row.items():
                    cb = int(sb)
                    if ca in relations and cb in relations[ca]:
                        relations[ca][cb] = float(val)
        except Exception:
            pass
        return relations

    def _rel(self, a: int, b: int) -> float:
        if not self.settings.get("geopolitics", {}).get("enabled", True):
            return 0.0
        scale = float(self.settings.get("geopolitics", {}).get("relation_scale", 1.0))
        return scale * float(self._relations.get(a, {}).get(b, 0.0))

    def _geo_weight(self, a: int, b: int) -> float:
        """Convert relation [-1,1] to a non-negative weight for averaging/biasing."""
        cfg = self.settings.get("geopolitics", {})
        floor = float(cfg.get("migration_geo_bias_floor", 0.1))
        return max(floor, 1.0 + self._rel(a, b))

    def _age_and_mortality(self):
        """Ages all agents, handles mortality, and returns death counts per country."""
        deaths = {}
        for country in self.countries.values():
            survivors = []
            dead = 0
            for agent in country.agents:
                agent.age += 1
                age_group = (agent.age // 10) * 10
                survival_prob = country.life_expectancy.get(age_group, 0.99)
                # Production buffers mortality slightly
                prod_buff = 1.0 + 0.01 * (self.production_index.get(country.id, 1.0) - 1.0)
                survival_prob = max(0.0, min(1.0, survival_prob * prod_buff))
                if random.random() < survival_prob:
                    survivors.append(agent)
                else:
                    dead += 1
            country.agents = survivors
            deaths[country.id] = dead
        return deaths

    def _births(self):
        """Handles new births in each country and returns birth counts per country."""
        births = {}
        for country in self.countries.values():
            living_agents = country.agents
            potential_parents = [a for a in living_agents if 20 <= a.age < 40]
            if not potential_parents:
                births[country.id] = 0
                continue

            num_births = 0
            for age_group, rate in country.fertility_rates.items():
                num_in_group = len([p for p in potential_parents if p.age // 10 * 10 == age_group])
                num_births += int(num_in_group * rate)

            # Adjust births by production and population target growth (configurable)
            prod_factor = float(self.settings.get("population", {}).get("prod_to_births_multiplier", 0.5))
            prod_mult = 1.0 + prod_factor * (self.production_index.get(country.id, 1.0) - 1.0)
            growth_mult = 1.0 + max(0.0, self.population_growth.get(country.id, 0.0))
            num_births = int(num_births * prod_mult * growth_mult)

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
            births[country.id] = num_births
        return births

    def _get_random_stratum(self, distribution):
        rand = random.random()
        cumulative = 0
        for stratum, probability in distribution.items():
            cumulative += probability
            if rand < cumulative:
                return stratum
        return list(distribution.keys())[-1]

    def _language_acquisition(self):
        """Language acquisition modeled as disease-like spread with cap of 3 languages per agent.

        For each agent and each non-spoken language in their country, compute an adoption
        probability similar to infection: p = 1 - exp(-beta * prevalence * contacts) * susceptibility,
        where susceptibility depends on age, education, and stratum. Each agent adopts at most
        one new language per year, and total known languages are capped at 3.
        Cross-country diffusion is incorporated via neighbor prevalence and global language power.
        """
        # Base transmission/contact parameters
        base_beta_child = 0.25  # higher for children
        base_beta_adult = 0.12
        contacts_per_year = 10  # average meaningful exposures

        # Susceptibility multipliers
        stratum_mult = {"high": 1.15, "middle": 1.0, "low": 0.9}
        education_mult = {0: 0.9, 1: 1.0, 2: 1.1, 3: 1.25}

        # Diffusion weights (configurable)
        lang_cfg = self.settings.get("language", {})
        w_local = float(lang_cfg.get("w_local", 1.0))
        w_neighbor = float(lang_cfg.get("w_neighbor", 0.6))
        w_global = float(lang_cfg.get("w_global", 0.45))  # "aggressive" global push
        aggression = float(lang_cfg.get("aggression", 1.25))  # scales infection probability for powerful languages

        for country in self.countries.values():
            if not country.language_prevalence:
                continue

            prevalence = country.language_prevalence
            neighbor_prev = self._lang_neighbor_prevalence.get(country.id, {})

            for agent in country.agents:
                # Cap at configurable max languages per agent
                if len(agent.languages) >= int(lang_cfg.get("max_langs_per_agent", 3)):
                    continue

                # Susceptibility based on age, education, stratum
                beta = base_beta_child if agent.age <= 12 else base_beta_adult
                age_tail_off = 1.0 if agent.age <= 30 else max(0.3, 1.0 - (agent.age - 30) / 80)
                susceptibility = (
                    beta
                    * stratum_mult.get(agent.economic_stratum, 1.0)
                    * education_mult.get(agent.education_level, 1.0)
                    * age_tail_off
                )

                spoken = {l.id for l in agent.languages}
                # Consider all known languages in system to allow foreign adoption
                candidate_ids = [lid for lid in self.languages.keys() if lid not in spoken]
                if not candidate_ids:
                    continue

                # Compute per-language infection probability (curved response)
                probs = []
                langs = []
                for lid in candidate_ids:
                    p_local = prevalence.get(lid, 0.0)
                    p_neighbor = neighbor_prev.get(lid, 0.0)
                    p_global = self._lang_global_power.get(lid, 0.0)
                    # Effective prevalence combining sources
                    p_eff = w_local * p_local + w_neighbor * p_neighbor + w_global * p_global
                    if p_eff <= 0:
                        continue
                    # Aggression boost for globally strong languages
                    power_boost = 1.0 + aggression * p_global
                    # Avoid runaway: reduce effect if already knows many languages
                    saturation = 1.0 - (len(agent.languages) / max(1.0, float(self.settings.get("language", {}).get("max_langs_per_agent", 3))))
                    # Curved response: logistic-like curve to avoid linearity
                    # base hazard h = susceptibility * p_eff * contacts
                    h = susceptibility * p_eff * contacts_per_year
                    # Curve: p = (1 / (1 + exp(-h))) - 0.5 adjusted to [0,1], or use softplus approximation
                    p_infect = 1.0 / (1.0 + math.exp(-h))
                    p_infect *= power_boost
                    p_infect = max(0.0, min(1.0, p_infect * saturation))
                    if p_infect > 1e-4:
                        probs.append(p_infect)
                        langs.append(self.languages[lid])

                if not probs:
                    continue

                # Choose at most one new language, weighted by probability
                total = sum(probs)
                weights = [q / total for q in probs]
                new_lang = random.choices(langs, weights=weights, k=1)[0]
                agent.languages.append(new_lang)
                # Enforce hard cap
                max_langs = int(lang_cfg.get("max_langs_per_agent", 3))
                if len(agent.languages) > max_langs:
                    agent.languages = agent.languages[:max_langs]

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

    def _language_campaigns(self):
        """Aggressive global campaigns: globally dominant languages push to replace weaker ones."""
        if not self._lang_global_power:
            return
        # Identify top global languages
        top_langs = sorted(self._lang_global_power.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_ids = [lid for lid, _ in top_langs]
        # Campaign intensity scales with global power
        for country in self.countries.values():
            if not country.agents:
                continue
            # Slightly stronger pressure if neighbors also speak it a lot
            neighbor_prev = self._lang_neighbor_prevalence.get(country.id, {})
            for agent in country.agents:
                # Skip if already at cap and already speaks any top language
                if any(l.id in top_ids for l in agent.languages) and len(agent.languages) >= 2:
                    continue
                for lid in top_ids:
                    lang_cfg = self.settings.get("language", {})
                    p_global = self._lang_global_power.get(lid, 0.0)
                    p_neighbor = neighbor_prev.get(lid, 0.0)
                    base = float(lang_cfg.get("campaign_base", 0.02))
                    wg = float(lang_cfg.get("campaign_global_weight", 0.15))
                    wn = float(lang_cfg.get("campaign_neighbor_weight", 0.10))
                    pressure = base + wg * p_global + wn * p_neighbor  # annual chance to adopt via campaign
                    if random.random() < pressure:
                        # Adopt or replace weakest language (lowest local prevalence)
                        new_lang = self.languages[lid]
                        if new_lang.id not in [l.id for l in agent.languages]:
                            max_langs = int(lang_cfg.get("max_langs_per_agent", 3))
                            if len(agent.languages) < max_langs:
                                agent.languages.append(new_lang)
                            else:
                                # Replace the language that is least prevalent locally
                                weakest = min(
                                    agent.languages,
                                    key=lambda L: country.language_prevalence.get(L.id, 0.0)
                                )
                                agent.languages.remove(weakest)
                                agent.languages.append(new_lang)
                        break  # Only one campaign adoption per year per agent

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
        """Runs a single year of the simulation (legacy entry point - kept for backward compatibility)."""
        return self._run_step_impl()

    def update_language_prevalence(self):
        """Updates the language prevalence for all countries."""
        for country in self.countries.values():
            country.update_language_prevalence()

    def _migration(self):
        """Handles agent migration between countries. Also records aggregated flows for the year."""
        migrants_by_destination = {country_id: [] for country_id in self.countries}
        migrated_agent_ids = set()
        total_migrations = 0
        # Aggregate flows as list of dicts {source, dest, count}
        flow_counter = {}
        def inc_flow(src, dst):
            key = (src, dst)
            flow_counter[key] = flow_counter.get(key, 0) + 1

        # Global migration event (configurable chance)
        mig_cfg = self.settings.get("migration", {})
        wave_p = float(mig_cfg.get("global_wave_probability", 0.1))
        wave_frac = float(mig_cfg.get("global_wave_fraction", 0.001))
        gdp_w = float(mig_cfg.get("gdp_weight", 1.0))
        lang_bonus = float(mig_cfg.get("language_bonus", 0.5))

        if random.random() < wave_p:
            all_agents = [agent for country in self.countries.values() for agent in country.agents]
            num_global_migrants = int(len(all_agents) * wave_frac)
            
            # Ensure we don't sample more than available agents
            k = min(num_global_migrants, len(all_agents))
            if k > 0:
                global_migrants = random.sample(all_agents, k=k)
                # Deduplicate by agent id to avoid processing the same underlying object multiple times
                seen_global = set()
                unique_global_migrants = []
                for a in global_migrants:
                    if a.id not in seen_global:
                        seen_global.add(a.id)
                        unique_global_migrants.append(a)

                for agent in unique_global_migrants:
                    if agent.id in migrated_agent_ids:
                        continue

                    source_country = self.countries[agent.country_id]

                    # Destination choice is global, not just neighbors
                    destination_scores = {}
                    for dest_id, dest_country in self.countries.items():
                        if dest_id == source_country.id: continue
                        gdp_ratio = (dest_country.gdp_per_capita / max(1e-6, source_country.gdp_per_capita)) ** gdp_w
                        bonus = lang_bonus if any(lang.id in dest_country.language_prevalence for lang in agent.languages) else 0
                        # Geopolitics factor: tilt towards friends, away from rivals
                        geo = self._geo_weight(source_country.id, dest_id)
                        destination_scores[dest_id] = gdp_ratio * (1 + bonus) * geo

                    if not destination_scores: continue
                    best_dest_id = max(destination_scores, key=destination_scores.get)

                    agent.country_id = best_dest_id
                    migrants_by_destination[best_dest_id].append(agent)
                    # Guard against cases where the agent was already removed
                    if agent in source_country.agents:
                        source_country.agents.remove(agent)
                    migrated_agent_ids.add(agent.id)
                    total_migrations += 1
                    inc_flow(source_country.id, best_dest_id)

        for source_country in self.countries.values():
            # Handle both positive (immigration) and negative (emigration) rates
            migration_rate = source_country.migration_rate
            if migration_rate == 0:
                continue

            num_migrants = int(len(source_country.agents) * abs(migration_rate))
            # Cap emigration fraction per year to avoid mass exodus
            if migration_rate < 0:
                geo_cfg = self.settings.get("geopolitics", {})
                cap_frac = float(geo_cfg.get("emigration_max_fraction", 0.2))
                num_migrants = min(num_migrants, int(len(source_country.agents) * cap_frac))
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
            # Deduplicate by agent id within this source country selection
            unique_potential = []
            seen_ids = set()
            for a in potential_migrants:
                if a.id not in seen_ids:
                    seen_ids.add(a.id)
                    unique_potential.append(a)

            # If the rate is negative, it's emigration, and we don't need to calculate a destination
            if migration_rate < 0:
                for agent in unique_potential:
                    if agent in source_country.agents:
                        # Keep at least a small base population from this path
                        min_floor = int(self.settings.get("geopolitics", {}).get("min_population_floor", 5))
                        if len(source_country.agents) <= min_floor:
                            break
                        source_country.agents.remove(agent)
                    migrated_agent_ids.add(agent.id) # Also track emigrants
                    total_migrations += 1
                    # Emigration to outside the modeled world
                    inc_flow(source_country.id, None)
                continue  # Move to the next country

            for agent in unique_potential:
                destination_scores = {}
                for dest_id in source_country.neighbor_ids:
                    # Skip invalid neighbor ids or self
                    if dest_id == source_country.id or dest_id not in self.countries:
                        continue
                    dest_country = self.countries[dest_id]
                    gdp_ratio = (dest_country.gdp_per_capita / max(1e-6, source_country.gdp_per_capita)) ** gdp_w
                    bonus = lang_bonus if any(lang.id in dest_country.language_prevalence for lang in agent.languages) else 0
                    geo = self._geo_weight(source_country.id, dest_id)
                    destination_scores[dest_id] = gdp_ratio * (1 + bonus) * geo

                if not destination_scores: continue
                best_dest_id = max(destination_scores, key=destination_scores.get)

                agent.country_id = best_dest_id
                migrants_by_destination[best_dest_id].append(agent)
                if agent in source_country.agents:
                    min_floor = int(self.settings.get("geopolitics", {}).get("min_population_floor", 5))
                    if len(source_country.agents) <= min_floor:
                        continue
                    source_country.agents.remove(agent)
                migrated_agent_ids.add(agent.id)
                total_migrations += 1
                inc_flow(source_country.id, best_dest_id)

        # Apply overwrite effects at destination based on influx and density
        for country_id, new_arrivals in migrants_by_destination.items():
            dest = self.countries[country_id]
            if new_arrivals:
                dest.agents.extend(new_arrivals)
                # If a large simultaneous influx relative to population density, boost the prevalence of migrants' top languages
                pop = max(1, len(dest.agents))
                influx_ratio = len(new_arrivals) / pop
                if influx_ratio > 0.05:  # 5% sudden influx threshold
                    # Count languages among arrivals
                    lang_counts = {}
                    for a in new_arrivals:
                        for L in a.languages:
                            lang_counts[L.id] = lang_counts.get(L.id, 0) + 1
                    # Compute density factor (smaller countries more sensitive)
                    density_factor = min(2.0, 0.5 + 0.5 * (1000000.0 / (1000.0 + pop)))
                    # Increase production-linked acquisition next step by adjusting country prevalence immediately
                    if lang_counts:
                        total = sum(lang_counts.values())
                        for lid, cnt in lang_counts.items():
                            add_share = influx_ratio * density_factor * (cnt / total) * 0.5
                            current = dest.language_prevalence.get(lid, 0.0)
                            # Blend, keep within [0,1]
                            dest.language_prevalence[lid] = max(0.0, min(1.0, current + add_share))
        
        # Save flows for this year as list of dicts
        self._last_migration_flows = [
            {"source": s, "dest": d, "count": c} for (s, d), c in flow_counter.items()
        ]
        return total_migrations

    def _update_country_stats(self):
        """Updates country-level statistics like language prevalence."""
        for country in self.countries.values():
            country.update_language_prevalence()

    def _apply_strategies(self):
        """Allow countries to strategize and execute step plans affecting language and migration.

        Strategies can include actions like:
        - language_campaign: {lang_id, intensity}
        - migration_policy: {openness, language_bonus}
        - education_push: {lang_id, coverage}
        """
        for country in self.countries.values():
            if not country.strategy:
                continue
            for action in country.strategy:
                typ = action.get("type")
                if typ == "language_campaign":
                    lid = action.get("lang_id")
                    intensity = float(action.get("intensity", 0.1))
                    if lid in self.languages:
                        # bias prevalence upward slightly to seed adoption
                        base = country.language_prevalence.get(lid, 0.0)
                        country.language_prevalence[lid] = max(0.0, min(1.0, base + intensity * 0.01))
                elif typ == "migration_policy":
                    # tweak country-specific migration rate temporarily
                    openness = float(action.get("openness", 0.0))
                    country.migration_rate = max(-0.02, min(0.02, country.migration_rate + openness))
                elif typ == "education_push":
                    lid = action.get("lang_id")
                    coverage = float(action.get("coverage", 0.0))
                    if lid in self.languages and coverage > 0:
                        # Encourage agents under 25 to learn this language with small probability bonus
                        for agent in country.agents:
                            if agent.age <= 25 and self.languages[lid] not in agent.languages:
                                if random.random() < min(0.25, 0.02 + 0.1 * coverage):
                                    agent.languages.append(self.languages[lid])
                                    if len(agent.languages) > 3:
                                        agent.languages = agent.languages[:3]

    def _run_step_impl(self):
        """Runs a single year of the simulation with production/pop growth and cross-country language dynamics."""
        # Reload settings at the start of each step so UI changes take effect without restart
        try:
            self.settings = load_settings()
        except Exception:
            # If reload fails, keep prior settings
            pass
        # Record start-of-year populations for growth targeting
        start_pop = {cid: len(c.agents) for cid, c in self.countries.items()}
        # Snapshot dominant language at start for change detection
        start_dom = {}
        for cid, c in self.countries.items():
            if c.language_prevalence:
                lid = max(c.language_prevalence, key=c.language_prevalence.get)
                start_dom[cid] = lid
            else:
                start_dom[cid] = None
        # Update production index
        self._update_production()
        # Update economy (capital and TFP) and incorporate into production index
        self._update_economy()
        # Refresh prevalence and compute influence context before acquisition
        self._update_country_stats()
        self._compute_language_influence_context()
        # AI decisions (may include conquest) before demographics
        self._ai_decisions()
        # Demographics
        deaths = self._age_and_mortality()
        births = self._births()
        self._education()
        self._update_economic_stratum()
        # Language dynamics
        self._language_acquisition()
        self._language_attrition()
        self._language_campaigns()
        # Execute strategic actions which may have post-dynamics adjustments
        self._apply_strategies()
        # Migration
        total_migrations = self._migration()
        # Apply exogenous population growth targets to counter systemic drift
        self._apply_population_growth_targets(start_pop)
        # Enforce ongoing 1/3 mono/bi/tri rule per country
        self._enforce_language_bucket_rule()
        # Refresh prevalence after all changes
        self._update_country_stats()
        # Build per-year event log entry
        econ_stats = self.get_economy_stats()
        year_entry = {
            "year": self.current_year + 1,
            "births": births,
            "deaths": deaths,
            "migrations_total": total_migrations,
            "migration_flows": list(self._last_migration_flows),
            "conquests": list(self._last_ai_conquests),
            "dominant_language_changes": {},
            "economy": {cid: {"production_index": econ_stats[cid]["production_index"],
                                "gdp_pc_proxy": econ_stats[cid]["gdp_pc_proxy"]}
                         for cid in self.countries.keys()},
        }
        for cid, c in self.countries.items():
            new_dom = None
            if c.language_prevalence:
                new_dom = max(c.language_prevalence, key=c.language_prevalence.get)
            old_dom = start_dom.get(cid)
            if new_dom != old_dom:
                year_entry["dominant_language_changes"][cid] = {"from": old_dom, "to": new_dom}
        self.event_log.append(year_entry)
        self.current_year += 1
        return total_migrations

    def get_economy_stats(self):
        """Return a dict of economy stats per country: production index, capital, TFP, GDP per capita proxy."""
        econ = self.settings.get("economy", {})
        alpha = float(econ.get("alpha", 0.33))
        labor_share = float(econ.get("labor_share_working_age", 0.65))
        out = {}
        for cid, c in self.countries.items():
            pop = max(1, len(c.agents))
            L = labor_share * pop
            K = max(1.0, float(self._capital.get(cid, 1.0)))
            A = max(1e-6, float(self._tfp.get(cid, 1.0)))
            Y = max(1.0, A * (K ** alpha) * (L ** (1.0 - alpha)))
            gdp_pc_proxy = Y / pop
            out[cid] = {
                "production_index": float(self.production_index.get(cid, 1.0)),
                "capital": K,
                "tfp": A,
                "gdp_pc_proxy": gdp_pc_proxy,
            }
        return out

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

    # --- New helper methods ---
    def _update_production(self):
        for cid in list(self.countries.keys()):
            g = self.production_growth.get(cid, 0.01)
            self.production_index[cid] = self.production_index.get(cid, 1.0) * (1.0 + g)

    def _update_economy(self):
        """Cobb-Douglas with TFP shocks and investment-driven capital accumulation per country.
        Y = A * K^α * L^(1-α). We do not store Y explicitly; we adjust production_index to reflect relative growth.
        """
        econ = self.settings.get("economy", {})
        alpha = float(econ.get("alpha", 0.33))
        delta = float(econ.get("delta", 0.05))
        gA = float(econ.get("g_A", 0.01))
        sigma = float(econ.get("sigma_shock", 0.015))
        spill = float(econ.get("neighbor_spillover", 0.05))
        inv_base = float(econ.get("investment_rate_base", 0.22))
        inv_rich = float(econ.get("investment_rate_rich", 0.26))
        rich_thr = float(econ.get("rich_threshold", 30000.0))
        labor_share = float(econ.get("labor_share_working_age", 0.65))

        # Precompute neighbor growth weighted by geopolitics
        neighbor_growth = {}
        for cid, c in self.countries.items():
            weighted_sum = 0.0
            weight_total = 0.0
            for nid in c.neighbor_ids:
                if nid in self.countries:
                    w = self._geo_weight(cid, nid)
                    weighted_sum += w * self.production_growth.get(nid, 0.01)
                    weight_total += w
            neighbor_growth[cid] = (weighted_sum / weight_total) if weight_total > 0 else 0.0

        for cid, c in self.countries.items():
            pop = max(1, len(c.agents))
            L = labor_share * pop
            K = self._capital.get(cid, 1.0)
            A = self._tfp.get(cid, 1.0)

            # TFP evolves with baseline growth, neighbor spillover (geopolitics-weighted), sanctions drag, and a random shock
            shock = random.gauss(0.0, sigma)
            sanctions_penalty = 0.0
            geo_cfg = self.settings.get("geopolitics", {})
            thr = float(geo_cfg.get("sanctions_threshold", -0.4))
            pen = float(geo_cfg.get("sanctions_penalty", -0.004))
            for nid in c.neighbor_ids:
                rel = self._rel(cid, nid)
                if rel < thr:
                    sanctions_penalty += pen
            A_new = A * math.exp(gA + spill * neighbor_growth.get(cid, 0.0) + sanctions_penalty + shock)

            # Output proxy for investment rule of thumb: y ~ A*K^α*L^(1-α)
            Y = max(1.0, A_new * (K ** alpha) * (L ** (1.0 - alpha)))
            invest_rate = inv_rich if c.gdp_per_capita >= rich_thr else inv_base
            I = invest_rate * Y
            K_new = max(1.0, (1.0 - delta) * K + I)

            # Update state
            self._tfp[cid] = A_new
            self._capital[cid] = K_new

            # Feed back into production_index as relative growth driver
            # Effective growth ~ (Y_new / Y_old). We don't keep Y_old; approximate via capital growth and A growth
            # Derive an effective growth multiplier m ≈ (K_new/K)^α * (A_new/A)
            cap_mult = (K_new / K) ** max(0.0, min(1.0, alpha)) if K > 0 else 1.0
            tfp_mult = (A_new / A) if A > 0 else 1.0
            m = cap_mult * tfp_mult
            # Blend into production_index to avoid double counting with fixed production_growth
            self.production_index[cid] *= max(0.95, min(1.10, m))

    def _compute_language_influence_context(self):
        """Compute neighbor prevalence and global language power used for cross-country diffusion."""
        # Neighbor prevalence weighted by geopolitics (friends influence more)
        neighbor_prev = {}
        for cid, country in self.countries.items():
            agg = {}
            wsum = 0.0
            for nid in country.neighbor_ids:
                if nid in self.countries:
                    nprev = self.countries[nid].language_prevalence
                    if nprev:
                        w = self._geo_weight(cid, nid)
                        for lid, p in nprev.items():
                            agg[lid] = agg.get(lid, 0.0) + w * p
                        wsum += w
            neighbor_prev[cid] = ({lid: val / wsum for lid, val in agg.items()} if wsum > 0 else {})
        self._lang_neighbor_prevalence = neighbor_prev

        # Global power: based on total speakers (approx prevalence*pop) weighted by economic heft
        total_pop = sum(len(c.agents) for c in self.countries.values()) or 1
        # Compute per-language global speaker counts and gdp-weighted counts
        speaker_counts = {}
        gdp_weighted = {}
        for c in self.countries.values():
            pop = len(c.agents)
            if pop == 0 or not c.language_prevalence:
                continue
            for lid, p in c.language_prevalence.items():
                speakers = p * pop
                speaker_counts[lid] = speaker_counts.get(lid, 0.0) + speakers
                gdp_weighted[lid] = gdp_weighted.get(lid, 0.0) + speakers * max(1.0, float(c.gdp_per_capita))
        if not speaker_counts:
            self._lang_global_power = {}
            return
        # Normalize to [0,1] style power
        max_speakers = max(speaker_counts.values())
        max_gdpw = max(gdp_weighted.values()) if gdp_weighted else 1.0
        global_power = {}
        for lid in self.languages.keys():
            share = speaker_counts.get(lid, 0.0) / max(1.0, max_speakers)
            econ = gdp_weighted.get(lid, 0.0) / max(1.0, max_gdpw)
            # Blend with small floor to allow tiny languages to still move locally
            power = 0.6 * share + 0.4 * econ
            global_power[lid] = max(0.0, min(1.0, power))
        self._lang_global_power = global_power

    def _country_power(self, cid: int) -> float:
        """Compute a simple country power proxy based on economy and population."""
        econ = self.settings.get("economy", {})
        alpha = float(econ.get("alpha", 0.33))
        labor_share = float(econ.get("labor_share_working_age", 0.65))
        c = self.countries[cid]
        pop = max(1, len(c.agents))
        L = labor_share * pop
        K = max(1.0, float(self._capital.get(cid, 1.0)))
        A = max(1e-6, float(self._tfp.get(cid, 1.0)))
        Y = max(1.0, A * (K ** alpha) * (L ** (1.0 - alpha)))
        pi = max(0.1, float(self.production_index.get(cid, 1.0)))
        return Y * pi

    def _ai_decisions(self):
        ai_cfg = self.settings.get("ai", {})
        if not ai_cfg.get("enabled", True):
            return
        # Reset conquest log for this year
        self._last_ai_conquests = []
        choices_per_year = int(ai_cfg.get("choices_per_year", 2))
        ratio_thr = float(ai_cfg.get("conquest_power_ratio_threshold", 1.5))
        neighbor_only = bool(ai_cfg.get("conquest_neighbor_only", True))
        shared_lang_min = float(ai_cfg.get("conquest_shared_language_min", 0.3))

        powers = {cid: self._country_power(cid) for cid in self.countries.keys()}
        actions = []
        for cid, c in list(self.countries.items()):
            if len(c.agents) == 0:
                continue
            remaining = choices_per_year
            if remaining > 0:
                candidates = c.neighbor_ids if neighbor_only else list(self.countries.keys())
                best_target, best_gain = None, 0.0
                clp = c.language_prevalence or {}
                for tid in candidates:
                    if tid == cid or tid not in self.countries:
                        continue
                    t = self.countries[tid]
                    if len(t.agents) == 0:
                        continue
                    tlp = t.language_prevalence or {}
                    if not clp or not tlp:
                        continue
                    dom_c = max(clp, key=clp.get)
                    dom_t = max(tlp, key=tlp.get)
                    shared = dom_c == dom_t
                    if not shared:
                        for lid, pv in clp.items():
                            if pv >= shared_lang_min and tlp.get(lid, 0.0) >= shared_lang_min:
                                shared = True
                                break
                    if not shared:
                        continue
                    p_att = powers.get(cid, 1.0)
                    p_def = powers.get(tid, 1.0)
                    if p_att / max(1.0, p_def) < ratio_thr:
                        continue
                    hostile_sum = 0.0
                    for oid in self.countries.keys():
                        if oid in (cid, tid):
                            continue
                        if self._rel(cid, oid) <= 0:
                            hostile_sum += powers.get(oid, 0.0)
                    if p_att < 0.2 * max(1.0, hostile_sum):
                        continue
                    gain = p_def * (1.1 if neighbor_only else 1.0)
                    if gain > best_gain:
                        best_gain, best_target = gain, tid
                if best_target is not None:
                    actions.append(("conquer", cid, best_target))
                    remaining -= 1
            if remaining > 0 and (c.language_prevalence or {}):
                dom = max(c.language_prevalence, key=c.language_prevalence.get)
                for agent in c.agents:
                    if agent.age <= 25 and self.languages[dom] not in agent.languages:
                        if random.random() < 0.02:
                            agent.languages.append(self.languages[dom])
                            if len(agent.languages) > 3:
                                agent.languages = agent.languages[:3]

        applied_targets = set()
        for act, a_id, t_id in actions:
            if act == "conquer" and t_id not in applied_targets and a_id in self.countries and t_id in self.countries:
                self._conquer(a_id, t_id)
                applied_targets.add(t_id)
                self._last_ai_conquests.append((a_id, t_id))

    def _conquer(self, attacker_id: int, target_id: int):
        attacker = self.countries[attacker_id]
        target = self.countries[target_id]
        if len(target.agents) == 0:
            return
        for a in list(target.agents):
            a.country_id = attacker_id
            attacker.agents.append(a)
        target.agents = []
        for oid in self.countries.keys():
            if oid == attacker_id:
                continue
            cur = self._relations.get(oid, {}).get(attacker_id, 0.0)
            new_val = min(cur, -0.5)
            self._relations.setdefault(oid, {})[attacker_id] = new_val
            self._relations.setdefault(attacker_id, {})[oid] = new_val
        attacker.update_language_prevalence()
        target.update_language_prevalence()

    def _apply_population_growth_targets(self, start_pop):
        """Adjust populations to reach exogenous per-country growth targets, by adding/removing agents at the margin.

        This helps prevent systemic declines and represents policy/structural effects not captured by micro rules.
        """
        pop_cfg = self.settings.get("population", {})
        min_g = float(pop_cfg.get("min_target_growth", -0.01))
        max_g = float(pop_cfg.get("max_target_growth", 0.03))
        for cid, country in self.countries.items():
            target_rate = max(min_g, min(max_g, self.population_growth.get(cid, 0.0)))
            # Target relative to start-of-year population
            base = max(0, start_pop.get(cid, len(country.agents)))
            target = int(round(base * (1.0 + target_rate)))
            diff = target - len(country.agents)
            if diff > 0:
                # Add newborns anchored to random current parents where possible
                potential_parents = [a for a in country.agents if 20 <= a.age < 40] or country.agents
                for _ in range(diff):
                    if not country.agents:
                        break
                    stratum = self._get_random_stratum(country.income_distribution)
                    new_agent = Agent(self.next_agent_id, country.id, 0, stratum)
                    self.next_agent_id += 1
                    if potential_parents:
                        p1 = random.choice(potential_parents)
                        # Inherit the most prevalent local/parent language
                        langs = list(p1.languages)
                        if not langs and country.language_prevalence:
                            best_lid = max(country.language_prevalence, key=country.language_prevalence.get)
                            langs = [self.languages[best_lid]]
                        max_langs = int(self.settings.get("language", {}).get("max_langs_per_agent", 3))
                        new_agent.languages = langs[:max_langs]
                    country.agents.append(new_agent)
            elif diff < 0:
                # Remove random agents (simulate unobserved outflows/declines)
                k = min(-diff, len(country.agents))
                if k > 0:
                    to_remove = set(random.sample(country.agents, k=k))
                    country.agents = [a for a in country.agents if a not in to_remove]

    def _enforce_language_bucket_rule(self):
        """Keep each country's population split ~1/3 mono, 1/3 bi, 1/3 tri annually.

        - Never allow 0 languages; assign one based on local prevalence or global power.
        - Cap at 3 languages and trim weakest if needed.
        - When adding languages, prefer locally prevalent and globally powerful ones.
        """
        # Optional toggle; default enabled
        if not self.settings.get("language", {}).get("enforce_bucket_rule", True):
            return

        # Precompute global desirability from cached power (available after _compute_language_influence_context)
        global_power = dict(self._lang_global_power or {})

        for cid, country in self.countries.items():
            agents = country.agents
            n = len(agents)
            if n == 0:
                continue

            # Build local counts from current agent languages (reflects post-migration/births)
            local_counts = {}
            for a in agents:
                for L in a.languages:
                    local_counts[L.id] = local_counts.get(L.id, 0) + 1
            # Convert to local shares
            total_lang_mentions = sum(local_counts.values()) or 1
            local_share = {lid: cnt / total_lang_mentions for lid, cnt in local_counts.items()}

            # Desirability score: blend local share and global power
            def desirability(lid: int) -> float:
                loc = local_share.get(lid, 0.0)
                glo = global_power.get(lid, 0.0)
                return 0.7 * loc + 0.3 * glo + 1e-6  # small epsilon to break ties

            # Helper: best new language for an agent
            def add_best_language(agent):
                spoken = {L.id for L in agent.languages}
                candidates = [lid for lid in self.languages.keys() if lid not in spoken]
                if not candidates:
                    return False
                best = max(candidates, key=desirability)
                agent.languages.append(self.languages[best])
                return True

            # Helper: remove weakest language from an agent (by desirability)
            def remove_weakest_language(agent):
                if not agent.languages:
                    return False
                weakest = min(agent.languages, key=lambda L: desirability(L.id))
                agent.languages.remove(weakest)
                return True

            max_langs = int(self.settings.get("language", {}).get("max_langs_per_agent", 3))

            # Normalize: ensure 1..3 languages per agent
            for a in agents:
                # Cap
                if len(a.languages) > max_langs:
                    # Trim least desirable until within cap
                    while len(a.languages) > max_langs:
                        remove_weakest_language(a)
                # Floor: ensure at least 1 language
                if len(a.languages) == 0:
                    # Pick best according to local/global preferences, otherwise arbitrary
                    candidates = list(self.languages.keys())
                    if candidates:
                        best = max(candidates, key=desirability)
                        a.languages = [self.languages[best]]

            # Compute targets (rounded split with residual to tri)
            target_mono = int(round(n / 3))
            target_bi = int(round(n / 3))
            target_tri = n - target_mono - target_bi

            # Bucket agents by current language count (after normalization)
            mono = [a for a in agents if len(a.languages) == 1]
            bi = [a for a in agents if len(a.languages) == 2]
            tri = [a for a in agents if len(a.languages) >= 3]

            # 1) Fix trilingual count first
            if len(tri) > target_tri:
                need = len(tri) - target_tri
                # Demote by removing weakest language
                # Prefer demoting those whose weakest language is very weak
                tri_sorted = sorted(tri, key=lambda a: desirability(min(a.languages, key=lambda L: desirability(L.id)).id))
                for a in tri_sorted[:need]:
                    remove_weakest_language(a)  # now 2 languages
            elif len(tri) < target_tri:
                need = target_tri - len(tri)
                # Promote bi first
                bi_candidates = [a for a in bi]
                random.shuffle(bi_candidates)
                while need > 0 and bi_candidates:
                    a = bi_candidates.pop()
                    if len(a.languages) == 2 and add_best_language(a):
                        need -= 1
                # If still need, promote mono by adding two
                mono_candidates = [a for a in mono]
                random.shuffle(mono_candidates)
                while need > 0 and mono_candidates:
                    a = mono_candidates.pop()
                    if len(a.languages) == 1 and add_best_language(a):
                        add_best_language(a)
                        need -= 1

            # Recompute buckets after tri adjustments
            mono = [a for a in agents if len(a.languages) == 1]
            bi = [a for a in agents if len(a.languages) == 2]
            tri = [a for a in agents if len(a.languages) >= 3]

            # 2) Fix bilingual count using mono <-> bi only (keep tri fixed)
            if len(bi) > target_bi:
                need = len(bi) - target_bi
                # Demote bi -> mono by removing weakest
                bi_sorted = sorted(bi, key=lambda a: desirability(min(a.languages, key=lambda L: desirability(L.id)).id))
                for a in bi_sorted[:need]:
                    remove_weakest_language(a)
            elif len(bi) < target_bi:
                need = target_bi - len(bi)
                mono_candidates = [a for a in mono]
                random.shuffle(mono_candidates)
                for a in mono_candidates:
                    if need <= 0:
                        break
                    if len(a.languages) == 1 and add_best_language(a):
                        need -= 1

            # Final sanity: cap at 3, floor at 1
            for a in agents:
                while len(a.languages) > max_langs:
                    remove_weakest_language(a)
                if len(a.languages) == 0:
                    candidates = list(self.languages.keys())
                    if candidates:
                        best = max(candidates, key=desirability)
                        a.languages = [self.languages[best]]