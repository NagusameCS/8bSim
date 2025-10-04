import random

class Language:
    """Represents a language in the simulation."""
    def __init__(self, lang_id: int, name: str):
        self.id = lang_id
        self.name = name
        self.global_speaker_count = 0

class Agent:
    """Represents an individual agent in the simulation."""
    def __init__(self, agent_id: int, country_id: int, age: int, economic_stratum: str):
        self.id = agent_id
        self.alive = True
        self.age = age
        self.country_id = country_id
        self.economic_stratum = economic_stratum  # 'low', 'mid', 'high'
        
        # 40% of agents have a monolingual predisposition
        self.is_monolingual_predisposed = (random.random() < 0.4)
        
        self.languages = []  # List of Language objects
        self.exposure_counters = {}  # {lang_id: years}
        self.attrition_counters = {} # {lang_id: years}
        
        # Tracks the year the agent is next allowed to move
        self.next_migration_year = 0

class Country:
    """Represents a country in the simulation."""
    def __init__(self, country_id: int, name: str, gdp_per_capita: float, life_expectancy: dict, fertility_rates: dict, income_distribution: dict, migration_rate: float):
        self.id = country_id
        self.name = name
        self.gdp_per_capita = gdp_per_capita
        self.life_expectancy = life_expectancy # {age_group: probability}
        self.fertility_rates = fertility_rates # {age_group: rate}
        self.income_distribution = income_distribution # {'low': 0.3, 'mid': 0.6, 'high': 0.1}
        self.migration_rate = migration_rate # Annual percentage of population that will migrate
        
        self.agents = []  # List of Agent objects residing in the country
        self.neighbors = [] # List of Country objects
        self.language_prevalence = {} # {lang_id: fraction}
        
    def update_language_prevalence(self):
        """Recalculates the language prevalence map based on the current population."""
        if not self.agents:
            self.language_prevalence = {}
            return

        lang_counts = {}
        total_speakers = 0
        
        # Filter for living agents
        living_agents = [agent for agent in self.agents if agent.alive]
        population_count = len(living_agents)

        if population_count == 0:
            self.language_prevalence = {}
            return

        for agent in living_agents:
            for lang in agent.languages:
                lang_counts[lang.id] = lang_counts.get(lang.id, 0) + 1
        
        for lang_id, count in lang_counts.items():
            self.language_prevalence[lang_id] = count / population_count
