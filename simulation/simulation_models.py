class Language:
    """Represents a language in the simulation."""
    def __init__(self, lang_id: int, name: str):
        self.id = lang_id
        self.name = name

class Agent:
    """Represents an individual agent in the simulation."""
    def __init__(self, agent_id: int, country_id: int, age: int, economic_stratum: str):
        self.id = agent_id
        self.alive = True
        self.age = age
        self.country_id = country_id
        self.economic_stratum = economic_stratum
        self.education_level = 0  # 0: None, 1: Primary, 2: Secondary, 3: Tertiary
        self.languages = []

class Country:
    """Represents a country in the simulation."""
    def __init__(self, country_id: int, name: str, gdp_per_capita: float, life_expectancy: dict, 
                 fertility_rates: dict, income_distribution: dict, migration_rate: float, neighbors: list):
        self.id = country_id
        self.name = name
        self.gdp_per_capita = gdp_per_capita
        self.life_expectancy = {int(k): v for k, v in life_expectancy.items()}
        self.fertility_rates = {int(k): v for k, v in fertility_rates.items()}
        self.income_distribution = income_distribution
        self.migration_rate = migration_rate
        self.neighbor_ids = neighbors
        self.agents = []
        self.language_prevalence = {}

    def update_language_prevalence(self):
        """Recalculates the language prevalence map based on the current population."""
        living_agents = [agent for agent in self.agents if agent.alive]
        population_count = len(living_agents)

        if population_count == 0:
            self.language_prevalence = {}
            return

        lang_counts = {}
        for agent in living_agents:
            for lang in agent.languages:
                lang_counts[lang.id] = lang_counts.get(lang.id, 0) + 1
        
        self.language_prevalence = {
            lang_id: count / population_count for lang_id, count in lang_counts.items()
        }
