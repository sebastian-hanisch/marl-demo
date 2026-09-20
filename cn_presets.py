"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus
dem OR-Demo-Portfolio, siehe constraint-programming-demo/csp_presets.py). Zusätzlich zu den
fünf Szenario-Reglern: Trainingsumgebung, Episoden, Rauschen σ und Trainings-Seed."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import cn_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_jobs_slider": SettingSpec("jobs", int, C.DEFAULT_N_JOBS, C.N_JOBS_MIN, C.N_JOBS_MAX),
    "n_agents_slider": SettingSpec("agents", int, C.DEFAULT_N_AGENTS, C.N_AGENTS_MIN, C.N_AGENTS_MAX),
    "duration_variability_slider": SettingSpec(
        "var", float, C.DEFAULT_DURATION_VARIABILITY, C.DURATION_VARIABILITY_MIN, C.DURATION_VARIABILITY_MAX
    ),
    "travel_time_per_unit_slider": SettingSpec(
        "travel", float, C.DEFAULT_TRAVEL_TIME_PER_UNIT, C.TRAVEL_TIME_PER_UNIT_MIN, C.TRAVEL_TIME_PER_UNIT_MAX
    ),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "env_mode_radio": SettingSpec("env", str, C.DEFAULT_ENV_MODE),
    "episodes_slider": SettingSpec(
        "eps", int, C.DEFAULT_EPISODES, min(C.EPISODES_CHOICES), max(C.EPISODES_CHOICES)
    ),
    "sigma_slider": SettingSpec("sigma", float, C.DEFAULT_SIGMA, C.SIGMA_MIN, C.SIGMA_MAX),
    "train_seed_input": SettingSpec("tseed", int, C.DEFAULT_TRAIN_SEED, C.TRAIN_SEED_MIN, C.TRAIN_SEED_MAX),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default
    if "force_regen" not in st.session_state:
        st.session_state["force_regen"] = False


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    # Sonderfälle: die Trainingsumgebung muss eine bekannte sein, die Episodenzahl eine der
    # wählbaren Stufen (select_slider akzeptiert nur Werte aus seiner Optionsliste).
    if st.session_state.get("env_mode_radio") not in C.ENV_LABELS:
        st.session_state["env_mode_radio"] = C.DEFAULT_ENV_MODE
    st.session_state["episodes_slider"] = snap_episodes(st.session_state.get("episodes_slider", C.DEFAULT_EPISODES))
    st.session_state["permalink_loaded"] = True


def snap_episodes(value):
    """Nächstliegende wählbare Episodenzahl (Permalinks können beliebige Zahlen enthalten)."""
    return min(C.EPISODES_CHOICES, key=lambda choice: abs(choice - value))


def training_key(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed, env_mode, sigma, episodes, train_seed):
    """Alles, was ein Trainingsergebnis bestimmt (σ zählt nur im Modus "wiederkehrend"). Dient als
    Cache-/Owner-Schlüssel in der App und für die Preset-Automatik der Seed-Lotterie."""
    sigma_effective = sigma if env_mode == C.ENV_RECURRING else 0.0
    return (
        int(n_jobs), int(n_agents), duration_variability, travel_time_per_unit, int(seed),
        env_mode, sigma_effective, int(episodes), int(train_seed),
    )


def sync_query_params(
    n_jobs, n_agents, duration_variability, travel_time_per_unit, seed, env_mode, episodes, sigma, train_seed,
):
    try:
        st.query_params["jobs"] = str(int(n_jobs))
        st.query_params["agents"] = str(int(n_agents))
        st.query_params["var"] = str(duration_variability)
        st.query_params["travel"] = str(travel_time_per_unit)
        st.query_params["seed"] = str(int(seed))
        st.query_params["env"] = str(env_mode)
        st.query_params["eps"] = str(int(episodes))
        st.query_params["sigma"] = str(sigma)
        st.query_params["tseed"] = str(int(train_seed))
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["n_jobs_slider"] = p["n_jobs"]
    st.session_state["n_agents_slider"] = p["n_agents"]
    st.session_state["duration_variability_slider"] = p["duration_variability"]
    st.session_state["travel_time_per_unit_slider"] = p["travel_time_per_unit"]
    st.session_state["seed_input"] = p["seed"]
    st.session_state["env_mode_radio"] = p["env_mode"]
    st.session_state["episodes_slider"] = p["episodes"]
    st.session_state["sigma_slider"] = p["sigma"]
    st.session_state["train_seed_input"] = p["train_seed"]
    st.session_state["lottery_preset_key"] = (
        training_key(
            p["n_jobs"], p["n_agents"], p["duration_variability"], p["travel_time_per_unit"], p["seed"],
            p["env_mode"], p["sigma"], p["episodes"], p["train_seed"],
        )
        if p.get("auto_lottery") else None
    )
    st.session_state["force_regen"] = True


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
    st.session_state["force_regen"] = True
