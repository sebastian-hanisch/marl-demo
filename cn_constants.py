"""Defaults, Slider-Grenzen und Presets für die MARL-Demo (unabhängiges Q-Learning).
Szenario-Konstanten (bis ORTOOLS_TIME_LIMIT_SECONDS) sind größtenteils wortgleich aus
contract-net-demo übernommen - dasselbe Vehikel -, bis auf die Regler-Grenzen für
Aufträge/Agenten, die hier auf das Training zugeschnitten sind."""

DEFAULT_N_JOBS = 8
DEFAULT_N_AGENTS = 3
DEFAULT_DURATION_VARIABILITY = 0.3
DEFAULT_TRAVEL_TIME_PER_UNIT = 1.0
DEFAULT_SEED = 5

# Die Zustandstabelle wächst mit n_jobs (S = n_jobs * 12), die Trainingszeit mit
# n_jobs * n_agents - bei 10 Aufträgen/4 Agenten und 100k Episoden bleibt sie unter ~10 s.
N_JOBS_MIN, N_JOBS_MAX = 4, 10
N_AGENTS_MIN, N_AGENTS_MAX = 2, 4
DURATION_VARIABILITY_MIN, DURATION_VARIABILITY_MAX = 0.0, 1.0
TRAVEL_TIME_PER_UNIT_MIN, TRAVEL_TIME_PER_UNIT_MAX = 0.2, 2.0

POSITION_RANGE_MAX = 20.0
DURATION_BASE_RANGE = (5, 15)
SPIKE_PROBABILITY_SCALE = 0.4
SPIKE_MULTIPLIER = 4.0

# OR-Tools-Referenzlauf: harte Zeitgrenze, damit ein Preset niemals hängt.
ORTOOLS_TIME_LIMIT_SECONDS = 10.0
# Für die vielen Held-out-Lösungen: pro Instanz kürzer (typisch 0.05-0.5 s).
ORTOOLS_HELDOUT_TIME_LIMIT_SECONDS = 5.0

# --- Lernverfahren (IQL) -------------------------------------------------------

# Aktionen jedes Agenten pro angekündigtem Auftrag: ein Aufschlag auf sein eigenes
# Contract-Net-Gebot. Aktion 0 = "normal" - ein untrainierter Agent (alle Q gleich)
# wählt per Tie-Break Aktion 0 und verhält sich exakt wie Contract Net.
BIAS_DELTA_MIN = 25.0
BIAS_ACTIONS = (0.0, BIAS_DELTA_MIN, -BIAS_DELTA_MIN)
ACTION_NAMES = ("normal bieten", "hoch bieten (ablehnen)", "niedrig bieten (greifen)")

ALPHA = 0.1
EPSILON_END = 0.05
EPSILON_DECAY_FRACTION = 0.7  # ε fällt linear über die ersten 70 % der Episoden
REWARD_SCALE = 10.0  # Belohnung = -Makespan / REWARD_SCALE

# Beobachtung: Bucket-Kanten für die eigene Freizeit (als Anteile von n_jobs*MEAN_DURATION/n_agents,
# also der "fairen Last") und für die Anfahrt zum angekündigten Auftrag (min).
MEAN_DURATION = 10.0
FREE_TIME_EDGE_FRACTIONS = (0.3, 0.7, 1.1)
TRAVEL_EDGES = (3.0, 8.0)

ENV_RECURRING = "recurring"
ENV_RANDOM = "random"
ENV_LABELS = {
    ENV_RECURRING: "Wiederkehrendes Szenario",
    ENV_RANDOM: "Zufällige Instanzen",
}
DEFAULT_ENV_MODE = ENV_RECURRING

EPISODES_CHOICES = (100, 300, 1000, 3000, 10000, 40000, 100000)
DEFAULT_EPISODES = 40000
SIGMA_MIN, SIGMA_MAX = 0.0, 0.6
DEFAULT_SIGMA = 0.3
TRAIN_SEED_MIN, TRAIN_SEED_MAX = 0, 999
DEFAULT_TRAIN_SEED = 0

# --- Auswertung ---------------------------------------------------------------

N_HELDOUT = 30              # Held-out-Instanzen (Rauschen-Ziehungen bzw. frische Instanzen)
N_FRESH_SCENARIOS = 40      # frische Szenarien für den Generalisierungs-Check
N_RANDOM_BIAS_DRAWS = 30    # Zufalls-Bias-Baseline
RANDOM_POOL_SIZE = 1000     # Trainingspool im Modus "zufällige Instanzen"
N_LOTTERY_SEEDS = 6
LOTTERY_EPISODE_CAP = 40000
N_LEARNING_CURVE_POINTS_MAX = len(EPISODES_CHOICES)

# Schwellwerte der Verdict-Kaskade (gegen gemessene Spannen kalibriert)
WORSE_THAN_CNP_THRESHOLD_PCT = 2.0      # IQL gilt als schlechter, wenn nominal ODER Held-out > +2 %
CLEARLY_BETTER_THRESHOLD_PCT = 3.0      # Erfolg nur bei klar besser (> 3 %) nominal UND Held-out
LOTTERY_SPREAD_WARNING_PCT = 5.0        # Seed-Std der Held-out-Ergebnisse in % von CNP
CROSSPLAY_PENALTY_WARNING_PCT = 3.0     # Cross-Play-Strafe in % von CNP

# Seeds/Preset-Werte empirisch kalibriert (Wegwerf-Sweeps, seither gelöscht) - nicht der erste
# Versuch übernommen. Gemessene Werte (n=8, k=3, Streuung 0.3, Anfahrt 1.0; % vs. Contract Net):
#   Lernen schlägt CNP:  nominal -21.6, Held-out -14.7, 6 Trainings-Seeds alle besser
#   Kaum Vorteil:        nominal +1.1,  Held-out -2.0
#   Seed entscheidet:    nominal -6.2, Lotterie -12.7..+39.3, Held-out-Std 11.3 %, Cross-Play +5.4 %
#   Schlechter als CNP:  nominal +39.4, Held-out +14.4, alle 6 Seeds schlechter (σ=0.6)
#   Zu wenig Training:   nominal +38.7, Held-out +35.3 (gleiches Szenario wie Preset 1)
#   Zufällige Instanzen: nominal +21.5, Held-out +10.3, alle 6 Seeds schlechter
PRESETS = {
    "Lernen schlägt Contract Net": {
        "n_jobs": 8, "n_agents": 3, "duration_variability": 0.3, "travel_time_per_unit": 1.0,
        "seed": 8, "env_mode": ENV_RECURRING, "sigma": 0.3, "episodes": 40000, "train_seed": 3,
    },
    "Kaum Vorteil": {
        "n_jobs": 8, "n_agents": 3, "duration_variability": 0.3, "travel_time_per_unit": 1.0,
        "seed": 9, "env_mode": ENV_RECURRING, "sigma": 0.3, "episodes": 40000, "train_seed": 0,
    },
    "Trainings-Seed entscheidet": {
        "n_jobs": 8, "n_agents": 3, "duration_variability": 0.3, "travel_time_per_unit": 1.0,
        "seed": 4, "env_mode": ENV_RECURRING, "sigma": 0.3, "episodes": 40000, "train_seed": 2,
        "auto_lottery": True,  # startet die Seed-Lotterie sofort - sie IST die Aussage dieses Presets
    },
    "Schlechter als Contract Net": {
        "n_jobs": 8, "n_agents": 3, "duration_variability": 0.3, "travel_time_per_unit": 1.0,
        "seed": 3, "env_mode": ENV_RECURRING, "sigma": 0.6, "episodes": 40000, "train_seed": 0,
    },
    "Zu wenig Training": {
        "n_jobs": 8, "n_agents": 3, "duration_variability": 0.3, "travel_time_per_unit": 1.0,
        "seed": 8, "env_mode": ENV_RECURRING, "sigma": 0.3, "episodes": 300, "train_seed": 3,
    },
    "Zufällige Instanzen: kein Vorteil": {
        "n_jobs": 8, "n_agents": 3, "duration_variability": 0.3, "travel_time_per_unit": 1.0,
        "seed": 6, "env_mode": ENV_RANDOM, "sigma": 0.3, "episodes": 40000, "train_seed": 0,
    },
}

PRESET_HELP = {
    "Lernen schlägt Contract Net": "Auf dem wiederkehrenden Szenario lernen die Agenten, "
        "Aufträge gezielt abzulehnen bzw. zu greifen - der Makespan liegt deutlich unter dem "
        "des rohen Contract Net.",
    "Kaum Vorteil": "Hier hat Contract Net von Haus aus wenig Spielraum - das Lernen bringt "
        "praktisch nichts.",
    "Trainings-Seed entscheidet": "Derselbe Aufbau, nur ein anderer Trainings-Seed: das Ergebnis "
        "schwankt spürbar. Startet die Seed-Lotterie und zeigt die ganze Spanne.",
    "Schlechter als Contract Net": "Starkes Rauschen (σ=0.6): die gelernte Policy landet über "
        "alle Trainings-Seeds schlechter als das rohe Contract Net.",
    "Zu wenig Training": "Nur 300 Episoden: die halb gelernte Policy ist deutlich schlechter "
        "als Contract Net - dasselbe Szenario mit 40000 Episoden schlägt es.",
    "Zufällige Instanzen: kein Vorteil": "Statt eines wiederkehrenden Szenarios sieht das Training "
        "immer neue Instanzen - die gelernte Policy schlägt Contract Net nicht.",
}

# Regressions-Bänder für tests/test_marl_evaluation.py (Prozent gegenüber Contract Net, negativ =
# besser als CNP): nominal = die gezeigte Instanz, heldout = Mittel über die Held-out-Instanzen.
PRESET_EXPECTED_BANDS = {
    "Lernen schlägt Contract Net": {"nominal_pct": (-30.0, -12.0), "heldout_pct": (-24.0, -8.0)},
    "Kaum Vorteil": {"nominal_pct": (-4.0, 6.0), "heldout_pct": (-8.0, 3.0)},
    "Trainings-Seed entscheidet": {"nominal_pct": (-12.0, 0.0), "heldout_pct": (-12.0, 2.0)},
    "Schlechter als Contract Net": {"nominal_pct": (22.0, 60.0), "heldout_pct": (6.0, 24.0)},
    "Zu wenig Training": {"nominal_pct": (20.0, 60.0), "heldout_pct": (20.0, 50.0)},
    "Zufällige Instanzen: kein Vorteil": {"nominal_pct": (10.0, 35.0), "heldout_pct": (2.0, 20.0)},
}
