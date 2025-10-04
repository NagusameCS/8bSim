# 8bSim

8bSim is a discrete-time simulation of population and language dynamics within a set of countries. It models a world of people who are born, age, and die. As they live, they can learn and forget languages, and their children inherit languages from them. This creates a dynamic system where the linguistic landscape of each country evolves over time.

## How It Works

The program is composed of four main files that work together to create the simulation.

### 1. `simulation/config.json` - The Simulation's Blueprint

This file defines the initial state of the world. It contains:
*   **Languages**: A list of all possible languages in the simulation.
*   **Countries**: A list of countries, each with properties like:
    *   `id` and `name`.
    *   `gdp_per_capita`, `life_expectancy` (by age group), and `fertility_rates` (by age group).
    *   `income_distribution`: The proportion of the population in 'high', 'middle', and 'low' economic strata.
    *   `migration_rate` and a list of `neighbors`.
    *   `initial_population`: The number of agents to create at the start.
    *   `initial_languages`: The probability for an initial agent to know a specific language.

### 2. `simulation/simulation_models.py` - The Building Blocks

This file defines the core data structures of the simulation:

*   **`Language`**: A simple class to represent a language with an `id` and a `name`.
*   **`Agent`**: Represents an individual person. Each agent has:
    *   An `id`, `age`, and the `country_id` they live in.
    *   An `economic_stratum` ('high', 'middle', 'low').
    *   A list of `languages` they speak.
*   **`Country`**: Represents a nation. It holds:
    *   The demographic and economic data from `config.json`.
    *   A list of `agents` that live in the country.
    *   A `language_prevalence` dictionary, which tracks the popularity of each language within the country. This is updated each year.

### 3. `simulation/main.py` - The Orchestrator

This is the main entry point of the application. When you run `python simulation/main.py`:

1.  **`setup_from_config()`**: This function reads `simulation/config.json`.
    *   It creates `Language` and `Country` objects.
    *   It then creates the initial population of `Agent` objects for each country, assigning them an age, economic stratum, and initial languages based on the probabilities in the config.
2.  **Simulation Initialization**: It creates a `Simulation` object, passing in the countries and languages.
3.  **The Main Loop**: It runs the simulation for a fixed number of years (in this case, 20). In each year, it calls `sim.run_step()` and `sim.print_stats()`.
4.  **Final Report**: After the loop, it prints a summary of the final population and language distribution for each country.

### 4. `simulation/simulation.py` - The Engine Room

This file contains the `Simulation` class, which drives the core logic of the world's evolution year by year. The `run_step()` method executes the following steps in order:

1.  **Update Language Prevalence**: Before anything else, it recalculates how common each language is in each country based on the current population.
2.  **Aging and Mortality (`_age_and_mortality`)**: Every agent's age is increased by one. Then, based on their age group and their country's life expectancy table, they have a chance to die.
3.  **Births (`_births`)**: New agents are born. The number of births is determined by the country's fertility rates and the number of agents in the child-bearing age range. A newborn's languages are determined by the languages spoken by two randomly chosen parents.
4.  **Language Acquisition (`_language_acquisition`)**: Agents have a chance to learn a new language.
    *   Children are more likely to learn than adults.
    *   Agents in a higher economic stratum are more likely to learn.
    *   The language they learn is chosen based on the current prevalence of languages in their country (they are more likely to learn a more common language).
    *   Agents can know a maximum of 3 languages.
5.  **Language Attrition (`_language_attrition`)**: If an agent speaks more than one language, they have a small chance to forget a language, especially if it's not one of the top 3 most prevalent languages in their country.
6.  **Migration (`_migration`)**: Agents have a chance to move to a neighboring country. The likelihood is based on the country's `migration_rate` and the GDP difference between the home country and the potential destination.

## How to Run

The project has two modes: a command-line simulation and a web-based user interface.

### Command-Line Simulation

To run the original simulation and see the output in your terminal, execute the main script from the root directory:

```bash
python -m simulation.main
```

### Web-based User Interface

The interactive UI provides visualizations of the simulation results.

**1. Install Dependencies:**

First, you need to install the required Python packages. Run the following command in your terminal:
```bash
pip install -r requirements.txt
```

**2. Launch the UI:**

Once the dependencies are installed, start the Streamlit application:
```bash
python -m streamlit run app.py
```

Your web browser should open with the user interface. You can modify the `config.json` file to change the initial conditions of the simulation.