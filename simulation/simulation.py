#!/usr/bin/env python3
"""
lang_disease_sim.py
Prototype: agent-based language-as-disease simulation using memmap + swap-with-last deletions.
Designed for correctness & testability. Parameterized for scaling.

Key rules implemented:
- Agents have up to 3 languages (lang0..lang2).
- Exposure slots E=4: track years >=20% and years <10% for candidate languages.
- Acquire language when years_ge20 >= 4 (non-continuous count).
- Lose language when years_below10 >= 4 (cumulative).
- Births: child inherits union of parents' languages (cap 3; pick by prevalence if >3).
- Deletion: swap-with-last to keep dense array (O(1)).
- Migration: decide moves only to neighboring countries; write moved agents to per-destination buffers.
- Chunked processing to keep RAM low.
"""

import os
import math
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random
from collections import Counter

# -----------------------------
# Config / Parameters
# -----------------------------
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# Testing sizes (change when scaling)
N_INIT = 100_000           # initial agents for test runs
MAX_AGENTS_CAP = 200_000   # capacity of memmap (grow by reallocation if necessary)
CHUNK_AGENTS = 50_000      # chunk size to process at once (tune to fit RAM)
E_SLOTS = 4                # number of exposure slots per agent
MAX_LANGUAGES = 3          # per-agent limit

# Language universe (for prototype)
LANGS = ["English", "Mandarin", "Hindi", "Spanish", "Arabic", "Bengali", "Portuguese"]
LANG_ID = {lang:i+1 for i,lang in enumerate(LANGS)}  # 0 means empty
NUM_LANGS = len(LANGS) + 1  # including 0

# Countries in prototype
COUNTRIES = {
    1: {"name":"CountryA", "neighbors":[2], "gdp":10000, "pop_init_pct":0.6, "lang_dist": {"English":0.7, "Spanish":0.2, "Hindi":0.1}},
    2: {"name":"CountryB", "neighbors":[1], "gdp":20000, "pop_init_pct":0.4, "lang_dist": {"English":0.1, "Mandarin":0.6, "Arabic":0.3}}
}
NUM_COUNTRIES = max(COUNTRIES.keys())

# Demographic & movement rules
PREDISPOSED_MONO_PCT = 0.40
ACQUIRE_YEARS_REQ = 4
ATTRITION_YEARS_REQ = 4
ACQUIRE_THRESHOLD = 0.20  # >=20% local prevalence
ATTRITION_THRESHOLD = 0.10  # <10% local prevalence
MOVE_RULES = {
    "under20_min_years": 7,
    "20to39_min_years": 10,
    "40to59_max_moves": 2,
    "60_plus_allowed": False
}

# File names
MEMMAP_PREFIX = DATA_DIR / "agents"
LOGICAL_LENGTH_FILE = DATA_DIR / "logical_length.npy"
COUNTRY_AGG_SNAP = DATA_DIR / "country_agg_year_{:d}.npz"
MIGRATION_BUFF_DIR = DATA_DIR / "mig_buffers"
MIGRATION_BUFF_DIR.mkdir(exist_ok=True)

# -----------------------------
# Memmap schema & helpers
# -----------------------------
# We'll use row-major contiguous record for ease of swap-with-last single-write.
# Define a dtype for one record.
agent_dtype = np.dtype([
    ("age", np.uint8),               # 1 byte
    ("country", np.uint16),          # 2
    ("income_stratum", np.uint8),    # 1
    ("predisposed_mono", np.uint8),  # 1
    ("lang_count", np.uint8),        # 1
    ("lang0", np.uint16),            # 2
    ("lang1", np.uint16),            # 2
    ("lang2", np.uint16),            # 2
    # exposure slots: E_SLOTS*(lang:uint16, years_ge20:uint8, years_below10:uint8)
    ("exp_langs", np.uint16, (E_SLOTS,)),
    ("exp_ge20", np.uint8, (E_SLOTS,)),
    ("exp_below10", np.uint8, (E_SLOTS,)),
    ("last_move_year", np.uint16),    # 2
    ("move_count_since_40", np.uint8) # 1
], align=False)

RECORD_SIZE = agent_dtype.itemsize

def ensure_memmap(path_prefix: Path, capacity: int) -> np.memmap:
    """Create or open a memmap file with given capacity (#records)."""
    fname = f"{path_prefix}.dat"
    if not Path(fname).exists():
        # create file with zeros
        fp = np.memmap(fname, dtype=agent_dtype, mode='w+', shape=(capacity,))
        fp.flush()
        del fp
    mm = np.memmap(fname, dtype=agent_dtype, mode='r+', shape=(capacity,))
    return mm

def load_logical_length() -> int:
    if Path(LOGICAL_LENGTH_FILE).exists():
        return int(np.load(LOGICAL_LENGTH_FILE))
    return 0

def save_logical_length(n: int):
    np.save(LOGICAL_LENGTH_FILE, np.array([n], dtype=np.uint64))

# -----------------------------
# Utility functions
# -----------------------------
def country_prevalence_from_chunk(chunk: np.ndarray) -> Dict[int, Dict[int,int]]:
    """Count languages by country in chunk. Returns nested dict: country -> lang_id -> count."""
    res = {}
    # iterate unique countries in chunk
    countries_in_chunk, idxs = np.unique(chunk["country"], return_inverse=True)
    for i, c in enumerate(countries_in_chunk):
        if c == 0:
            continue
        mask = (chunk["country"] == c)
        langs = np.concatenate([chunk["lang0"][mask].astype(np.int64),
                                chunk["lang1"][mask].astype(np.int64),
                                chunk["lang2"][mask].astype(np.int64)])
        langs = langs[langs != 0]
        if langs.size == 0:
            res[int(c)] = {}
        else:
            unique, counts = np.unique(langs, return_counts=True)
            res[int(c)] = {int(l): int(cnt) for l, cnt in zip(unique, counts)}
    return res

def merge_country_counts(a: Dict[int, Dict[int,int]], b: Dict[int, Dict[int,int]]):
    for c, mapping in b.items():
        if c not in a:
            a[c] = mapping.copy()
        else:
            for l, cnt in mapping.items():
                a[c][l] = a[c].get(l, 0) + cnt

# -----------------------------
# Initialize memmap and seed agents (small prototype)
# -----------------------------
def init_memmap_and_seed(capacity: int, n_init: int) -> Tuple[np.memmap, int]:
    mm = ensure_memmap(MEMMAP_PREFIX, capacity)
    logical_len = load_logical_length()
    if logical_len == 0:
        # seed n_init agents
        rng = np.random.default_rng(12345)
        ages = rng.integers(0, 80, size=n_init, dtype=np.uint8)
        # assign countries proportional to prototypical pop_init_pct
        country_ids = np.array([1]*int(n_init*COUNTRIES[1]["pop_init_pct"]) + [2]*(n_init - int(n_init*COUNTRIES[1]["pop_init_pct"])), dtype=np.uint16)
        rng.shuffle(country_ids)
        for i in range(n_init):
            rec = mm[i]
            rec["age"] = int(ages[i])
            rec["country"] = int(country_ids[i])
            rec["income_stratum"] = 1
            rec["predisposed_mono"] = 1 if rng.random() < PREDISPOSED_MONO_PCT else 0
            # sample 1-3 languages according to country distribution
            cinfo = COUNTRIES[int(country_ids[i])]
            langs = list(cinfo["lang_dist"].items())
            # sample at least 1
            # choose up to 3 without replacement weighted by probabilities
            names, probs = zip(*langs)
            probs = np.array(probs)
            probs = probs / probs.sum()
            k = rng.integers(1, MAX_LANGUAGES+1)
            selected = rng.choice(len(names), size=k, replace=False, p=probs)
            lang_ids = [LANG_ID[names[j]] for j in selected]
            rec["lang_count"] = len(lang_ids)
            rec["lang0"] = lang_ids[0] if len(lang_ids) > 0 else 0
            rec["lang1"] = lang_ids[1] if len(lang_ids) > 1 else 0
            rec["lang2"] = lang_ids[2] if len(lang_ids) > 2 else 0
            rec["exp_langs"][:] = 0
            rec["exp_ge20"][:] = 0
            rec["exp_below10"][:] = 0
            rec["last_move_year"] = 0
            rec["move_count_since_40"] = 0
            mm[i] = rec
        logical_len = n_init
        save_logical_length(logical_len)
        mm.flush()
        print(f"Seeded {n_init} agents into memmap (capacity {capacity}).")
    else:
        print(f"Memmap already initialized with logical length {logical_len}.")
    return mm, logical_len

# -----------------------------
# Swap-with-last deletion and append
# -----------------------------
def remove_agent_at_pos(mm: np.memmap, pos: int, logical_len: int) -> int:
    """Remove agent at `pos` by swapping last record into pos and shrinking logical length.
    Returns new logical_length.
    """
    last = logical_len - 1
    if pos < 0 or pos >= logical_len:
        raise IndexError("pos out of range")
    if pos == last:
        # just shrink
        logical_len -= 1
        save_logical_length(logical_len)
        return logical_len
    # move last into pos
    mm[pos] = mm[last]
    logical_len -= 1
    save_logical_length(logical_len)
    return logical_len

def append_agent(mm: np.memmap, logical_len: int, record: np.void) -> int:
    """Append a single agent record at end; grows memmap capacity error if out of space."""
    if logical_len >= mm.shape[0]:
        raise MemoryError("Memmap capacity exceeded. Need to reallocate/grow file before append.")
    mm[logical_len] = record
    logical_len += 1
    save_logical_length(logical_len)
    return logical_len

# -----------------------------
# Country prevalence aggregation (yearly)
# -----------------------------
def compute_country_prevalence(mm: np.memmap, logical_len: int, chunk_agents: int) -> Dict[int, Dict[int,float]]:
    """Compute per-country language prevalence (fractions) using chunked reductions."""
    country_counts = {}   # country -> lang_id -> count
    country_pops = {}     # country -> total people (for fraction)
    start = 0
    while start < logical_len:
        size = min(chunk_agents, logical_len - start)
        chunk = mm[start:start+size]
        # country populations
        unique_c, counts_c = np.unique(chunk["country"], return_counts=True)
        for c, cnt in zip(unique_c, counts_c):
            if c == 0: continue
            country_pops[int(c)] = country_pops.get(int(c), 0) + int(cnt)
        # language counts per country
        per_chunk = country_prevalence_from_chunk(chunk)
        merge_country_counts(country_counts, per_chunk)
        start += size
    # convert counts -> fractions
    prevalences = {}
    for c, mapping in country_counts.items():
        pop = max(1, country_pops.get(int(c), 1))
        prevalences[c] = {l: cnt / (pop*1.0) for l, cnt in mapping.items()}
    return prevalences

# -----------------------------
# Exposure update, acquisition, attrition (chunked)
# -----------------------------
def process_exposure_and_language_changes(mm: np.memmap, logical_len: int,
                                          country_prevalences: Dict[int, Dict[int,float]],
                                          current_year: int, chunk_agents: int) -> int:
    """Chunked processing: update exposure slots, apply acquisitions & attrition.
    Returns logical_len (unchanged except for deletions triggered by any attrition rule that removes last language?).
    """
    start = 0
    rng = np.random.default_rng(42 + current_year)
    while start < logical_len:
        size = min(chunk_agents, logical_len - start)
        buf = mm[start:start+size]  # this is a view; writebacks persist to memmap
        # For each country in this chunk, get its prevalence dict once
        countries_in_chunk = np.unique(buf["country"])
        for c in countries_in_chunk:
            if c == 0: continue
            c = int(c)
            cprev = country_prevalences.get(c, {})
            # Precompute thresholds
            # For all agents in chunk with country==c:
            mask = (buf["country"] == c)
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue
            for local_idx in idxs:
                rec = buf[local_idx]
                # update exposure slots for any top languages in cprev
                # We'll consider languages present in cprev with any fraction
                for lid, frac in cprev.items():
                    # only track languages with at least 0.01 prevalence to avoid noise
                    if frac < 0.01:
                        continue
                    # find this language in exposure slots if present
                    exp_langs = rec["exp_langs"]
                    if lid in exp_langs:
                        slot = int(np.where(exp_langs == lid)[0][0])
                        if frac >= ACQUIRE_THRESHOLD:
                            rec["exp_ge20"][slot] = min(255, int(rec["exp_ge20"][slot]) + 1)
                        if frac < ATTRITION_THRESHOLD:
                            rec["exp_below10"][slot] = min(255, int(rec["exp_below10"][slot]) + 1)
                    else:
                        # try to insert in a free slot or replace the least useful slot
                        zero_slots = np.where(exp_langs == 0)[0]
                        if zero_slots.size > 0:
                            s = int(zero_slots[0])
                            rec["exp_langs"][s] = lid
                            rec["exp_ge20"][s] = 1 if frac >= ACQUIRE_THRESHOLD else 0
                            rec["exp_below10"][s] = 1 if frac < ATTRITION_THRESHOLD else 0
                        else:
                            # find slot with lowest exp_ge20 as replacement heuristic
                            ge20 = rec["exp_ge20"].astype(np.int64)
                            s = int(np.argmin(ge20))
                            # replace only if new prevalence is better than whatever's reflected by ge20 (heuristic)
                            # (this heuristic is simple; you can make more complex)
                            rec["exp_langs"][s] = lid
                            rec["exp_ge20"][s] = 1 if frac >= ACQUIRE_THRESHOLD else 0
                            rec["exp_below10"][s] = 1 if frac < ATTRITION_THRESHOLD else 0
                # Acquisition: if any slot has ge20 >= ACQUIRE_YEARS_REQ, and lang not already present -> acquire
                for s in range(E_SLOTS):
                    lid = int(rec["exp_langs"][s])
                    if lid == 0: continue
                    if int(rec["exp_ge20"][s]) >= ACQUIRE_YEARS_REQ:
                        # acquire if not present
                        present = (rec["lang0"] == lid) or (rec["lang1"] == lid) or (rec["lang2"] == lid)
                        if not present:
                            if rec["lang_count"] < MAX_LANGUAGES:
                                # append into first empty slot
                                if rec["lang0"] == 0:
                                    rec["lang0"] = lid
                                elif rec["lang1"] == 0:
                                    rec["lang1"] = lid
                                elif rec["lang2"] == 0:
                                    rec["lang2"] = lid
                                rec["lang_count"] = rec["lang_count"] + 1
                            else:
                                # replacement policy: drop smallest local prevalence language among its current ones
                                # compute local prevalence for each held language
                                held = [int(rec["lang0"]), int(rec["lang1"]), int(rec["lang2"])]
                                # handle zeros defensively
                                held = [h for h in held if h != 0]
                                if len(held) == 0:
                                    continue
                                best_drop = None
                                best_drop_val = 1.0
                                for idx_h, h in enumerate(held):
                                    h_frac = country_prevalences.get(c, {}).get(h, 0.0)
                                    if h_frac < best_drop_val:
                                        best_drop_val = h_frac
                                        best_drop = h
                                # replace best_drop with lid
                                if best_drop is not None and best_drop_val < country_prevalences.get(c, {}).get(lid, 0.999):
                                    # find slot and replace
                                    if rec["lang0"] == best_drop:
                                        rec["lang0"] = lid
                                    elif rec["lang1"] == best_drop:
                                        rec["lang1"] = lid
                                    elif rec["lang2"] == best_drop:
                                        rec["lang2"] = lid
                                    # lang_count unchanged
                # Attrition: if any of agent's current languages has exp_below10 >= ATTRITION_YEARS_REQ, remove it
                for lang_slot in ["lang0", "lang1", "lang2"]:
                    lid = int(rec[lang_slot])
                    if lid == 0:
                        continue
                    # find matching exposure slot
                    exp_langs = rec["exp_langs"]
                    found_idx = np.where(exp_langs == lid)[0]
                    years_below = 0
                    if found_idx.size > 0:
                        years_below = int(rec["exp_below10"][int(found_idx[0])])
                    else:
                        # if not tracked in exposure slots, assume vulnerable (incremental)
                        years_below = 0
                    if years_below >= ATTRITION_YEARS_REQ:
                        # remove language
                        rec[lang_slot] = 0
                        rec["lang_count"] = max(0, int(rec["lang_count"]) - 1)
                # write back modified record
                buf[local_idx] = rec
        # small flush
        mm[start:start+size] = buf
        start += size
    mm.flush()
    return logical_len

# -----------------------------
# Birth routine (vectorized per country)
# -----------------------------
def births_by_country(mm: np.memmap, logical_len: int, year: int) -> int:
    """Simple births: expected births = round(population * fertility_rate).
    For prototype, we use a tiny per-country crude fertility function.
    """
    # crude fertility rate mapping by country
    fertility_by_country = {1: 0.01, 2: 0.012}  # births per person per year (toy)
    # compute county populations quickly
    pops = {}
    start = 0
    while start < logical_len:
        size = min(CHUNK_AGENTS, logical_len - start)
        chunk = mm[start:start+size]
        unique_c, counts = np.unique(chunk["country"], return_counts=True)
        for c, cnt in zip(unique_c, counts):
            if c == 0: continue
            pops[int(c)] = pops.get(int(c), 0) + int(cnt)
        start += size
    # perform births
    rng = np.random.default_rng(1000 + year)
    for c, pop in pops.items():
        fert = fertility_by_country.get(int(c), 0.01)
        births = int(round(pop * fert))
        for _ in range(births):
            # create empty record
            rec = np.zeros(1, dtype=agent_dtype)[0]
            rec["age"] = 0
            rec["country"] = int(c)
            rec["income_stratum"] = 1
            rec["predisposed_mono"] = 1 if rng.random() < PREDISPOSED_MONO_PCT else 0
            # pick parents randomly from the country (simple)
            # we sample two parents uniformly by scanning chunked until find some in country
            # (inefficient for huge N; for prototype ok)
            # For prototype, child inherits languages: we union a random two parents' languages
            # find up to two parents
            parent_langs = []
            # sample 2 random positions globally and hope they land in country (simple)
            attempts = 0
            while len(parent_langs) == 0 and attempts < 20:
                idx = rng.integers(0, max(1,logical_len))
                p = mm[idx]
                if p["country"] == c:
                    parent_langs = [int(p["lang0"]), int(p["lang1"]), int(p["lang2"])]
                    parent_langs = [x for x in parent_langs if x != 0]
                attempts += 1
            if len(parent_langs) == 0:
                # fallback: sample country language dist
                cinfo = COUNTRIES[int(c)]
                names, probs = zip(*cinfo["lang_dist"].items())
                probs = np.array(probs)
                probs = probs / probs.sum()
                k = rng.integers(1, MAX_LANGUAGES+1)
                selected = rng.choice(len(names), size=k, replace=False, p=probs)
                parent_langs = [LANG_ID[names[j]] for j in selected]
            # child inherits union of parent_langs (prototype uses just the one parent set)
            inherited = list(dict.fromkeys(parent_langs))  # preserve order unique
            if len(inherited) > MAX_LANGUAGES:
                inherited = inherited[:MAX_LANGUAGES]
            rec["lang_count"] = len(inherited)
            rec["lang0"] = inherited[0] if len(inherited) > 0 else 0
            rec["lang1"] = inherited[1] if len(inherited) > 1 else 0
            rec["lang2"] = inherited[2] if len(inherited) > 2 else 0
            rec["exp_langs"][:] = 0
            rec["exp_ge20"][:] = 0
            rec["exp_below10"][:] = 0
            rec["last_move_year"] = 0
            rec["move_count_since_40"] = 0
            logical_len = append_agent(mm, logical_len, rec)
    return logical_len

# -----------------------------
# Simplified migration: decide and buffer moves (per destination append-only files)
# -----------------------------
def decide_and_buffer_migration(mm: np.memmap, logical_len: int, current_year: int, chunk_agents: int) -> int:
    """
    Simple migration: randomly move a tiny fraction of eligible agents to neighbor countries.
    Writes moved-records to append-only per-destination files and removes moved agents via swap-with-last.
    """
    rng = np.random.default_rng(2000 + current_year)
    start = 0
    moved = 0
    while start < logical_len:
        size = min(chunk_agents, logical_len - start)
        buf = mm[start:start+size]
        for local_idx in range(size):
            rec = buf[local_idx]
            age = int(rec["age"])
            if age >= 60:
                continue
            # choose minimal movement probabilities by age buckets and predisposition
            allowed = True
            if age < 20:
                min_years = MOVE_RULES["under20_min_years"]
                if current_year - int(rec["last_move_year"]) < min_years:
                    allowed = False
            elif 20 <= age <= 39:
                min_years = MOVE_RULES["20to39_min_years"]
                if current_year - int(rec["last_move_year"]) < min_years:
                    allowed = False
            # 40-59 handled via move_count_since_40; we'll allow small probability if under limit
            if not allowed:
                continue
            # tiny probability to move this year
            if rng.random() < 0.0005:
                src_country = int(rec["country"])
                neigh = COUNTRIES[src_country]["neighbors"]
                if len(neigh) == 0:
                    continue
                dest = int(rng.choice(neigh))
                # preference: only move if dest sustains at least one of their languages above threshold
                holds = False
                for langslot in ("lang0","lang1","lang2"):
                    lid = int(rec[langslot])
                    if lid == 0: continue
                    # quick check: compute dest prevalence by scanning mm (expensive) - for prototype, use country initial dist
                    dest_info = COUNTRIES[dest]
                    # check if any of the agent's language is among dest_info's language names
                    # crude mapping: invert LANG_ID
                    inv_lang = {v:k for k,v in LANG_ID.items()}
                    name = inv_lang.get(lid, None)
                    if name and name in dest_info["lang_dist"] and dest_info["lang_dist"][name] > 0.01:
                        holds = True
                        break
                if not holds:
                    continue
                # write to buffer (append file)
                buf_fname = MIGRATION_BUFF_DIR / f"migrate_to_{dest}.bin"
                # write row bytes for appended record
                with open(buf_fname, "ab") as f:
                    f.write(rec.tobytes())
                # remove agent by swapping with last
                global_pos = start + local_idx
                logical_len = remove_agent_at_pos(mm, global_pos, logical_len)
                moved += 1
                # after swap, the record that was swapped into current position is new; we should re-evaluate this same index,
                # so adjust buf and local_idx accordingly by reloading buf or re-reading mm; for simplicity, we reload entire chunk
                buf = mm[start:start+size]
        start += size
    if moved:
        print(f"Year {current_year}: moved {moved} agents and buffered to per-destination files.")
    return logical_len

def apply_migration_buffers(mm: np.memmap, logical_len: int) -> int:
    """Read per-destination buffer files and append those agents into mm (they are already full records)."""
    for fpath in MIGRATION_BUFF_DIR.glob("migrate_to_*.bin"):
        dest_id = int(fpath.stem.split("_")[-1])
        with open(fpath, "rb") as f:
            data = f.read()
            rec_size = agent_dtype.itemsize
            n_rec = len(data) // rec_size
            if n_rec == 0:
                f.close()
                fpath.unlink()
                continue
            arr = np.frombuffer(data, dtype=agent_dtype)
            # arr is length n_rec; we need to set their country id to dest_id to be consistent
            for i in range(n_rec):
                rec = arr[i]
                rec = rec.copy()
                rec["country"] = dest_id
                logical_len = append_agent(mm, logical_len, rec)
        # delete buffer after consuming
        fpath.unlink()
    return logical_len

# -----------------------------
# Year loop & driver
# -----------------------------
def run_simulation(year_start: int, year_end: int, capacity: int, n_init: int, chunk_agents: int):
    mm, logical_len = init_memmap_and_seed(capacity, n_init)
    for year in range(year_start, year_end+1):
        t0 = time.time()
        print(f"=== Year {year}: starting with {logical_len} agents ===")
        # 1) compute country prevalence
        prevalences = compute_country_prevalence(mm, logical_len, chunk_agents)
        # 2) process exposure & language changes
        logical_len = process_exposure_and_language_changes(mm, logical_len, prevalences, year, chunk_agents)
        # 3) migration decision & buffer
        logical_len = decide_and_buffer_migration(mm, logical_len, year, chunk_agents)
        # 4) apply migration buffers (append moved agents now in destination countries)
        logical_len = apply_migration_buffers(mm, logical_len)
        # 5) births
        logical_len = births_by_country(mm, logical_len, year)
        # 6) optional: recompute prevalences and write summary
        prevalences_after = compute_country_prevalence(mm, logical_len, chunk_agents)
        # snapshot small aggregated data
        np.savez(COUNTRY_AGG_SNAP.format(year), prevalences=prevalences_after, total_agents=np.array([logical_len]))
        mm.flush()
        t1 = time.time()
        print(f"Year {year} done. Agents: {logical_len}. Time: {t1-t0:.1f}s")
    print("Simulation finished.")

# -----------------------------
# Main (run small test)
# -----------------------------
if __name__ == "__main__":
    # 1. Define Languages
    LANGUAGES = {"EN", "ES", "FR", "DE", "ZH"}

    # 2. Create Countries
    country1 = Country(country_id=1, name="CountryA", neighbors=[2], gdp_per_capita=50000, life_expectancy=80, fertility_rate=0.015)
    country2 = Country(country_id=2, name="CountryB", neighbors=[1], gdp_per_capita=30000, life_expectancy=75, fertility_rate=0.020)
    
    countries = [country1, country2]

    # 3. Initialize Agents
    initial_agents = []
    agent_id_counter = 0
    for i in range(200): # 100 for each country
        country = country1 if i < 100 else country2
        
        # Simplified initial language assignment
        langs = set()
        if country.id == 1:
            langs = random.sample(list(LANGUAGES), k=random.randint(1,2)) if "EN" not in langs else {"EN"}
        else:
            langs = random.sample(list(LANGUAGES), k=random.randint(1,2)) if "ES" not in langs else {"ES"}


        agent = Agent(
            agent_id=agent_id_counter,
            country_id=country.id,
            age=random.randint(0, 80),
            languages=langs,
            economic_stratum=random.choice(['low', 'mid', 'high']),
            monolingual_predisposition=random.random() < 0.4
        )
        initial_agents.append(agent)
        country.add_agent(agent)
        agent_id_counter += 1

    # 4. Create and Run Simulation
    sim = Simulation(countries, initial_agents, LANGUAGES)
    
    # Initial setup
    for country in sim.countries.values():
        country.update_language_prevalence()

    # Run for 20 years
    for _ in range(20):
        sim.run_step()

    # 7. Output final state (simplified)
    print("\n--- Final State ---")
    for country in sim.countries.values():
        print(f"Country: {country.name}")
        print(f"  Population: {len([a for a in country.agents if a.is_alive])}")
        
        lang_dist = Counter()
        for agent in country.agents:
            if agent.is_alive:
                lang_dist.update(agent.languages)
        
        total_living = len([a for a in country.agents if a.is_alive])
        if total_living > 0:
            prevalence = {lang: count/total_living for lang, count in lang_dist.items()}
            print(f"  Language Prevalence: { {k: round(v, 2) for k,v in prevalence.items()} }")

        mono, multi = 0, 0
        for agent in country.agents:
            if agent.is_alive:
                if len(agent.languages) == 1:
                    mono += 1
                else:
                    multi += 1
        print(f"  Monolingual: {mono}, Multilingual: {multi}")