"""
Multi-Agenten-Reinforcement-Learning (unabhängiges Q-Learning) an der Kran-Auftragsvergabe –
interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der "Konzepte"-Reihe, Multi-Agenten-Koordinations-Linie - ein UNABHÄNGIGER Zweig
direkt vom Contract-Net-Protocol-Root (contract-net-demo). Statt eines festen Gebotsprotokolls
lernen die Agenten selbst, wie sie ihre Gebote anpassen. MARL ist das Feld (mehrere lernende
Agenten in einer gemeinsamen Umgebung), unabhängiges Q-Learning (IQL) das einfachste Lernverfahren
darin. Eigene Schwächen, als gemessene Zahlen gezeigt: Trainingsaufwand, keine Garantie,
Nicht-Stationarität.
"""

import time

import streamlit as st

import cn_constants as C
from cn_evaluation import stats_up_to_step
from cn_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
    training_key,
)
from cn_protocol import run_protocol
from cn_scenario import generate_instance
from cn_visualization import build_bid_chart, build_schedule_figure
from marl_env import heldout_instances, run_dispatch
from marl_evaluation import (
    cnp_makespans,
    comparison,
    generalisation_check,
    heldout_optima,
    learning_curve,
    seed_lottery,
)
from marl_exact import enumerate_joint_policies, fixed_order_ceiling, joint_policy_makespan, miniature_instance
from marl_iql import epsilon_at, greedy_policy, train_iql
from marl_obs import describe_state, make_observer
from marl_visualization import (
    build_crossplay_heatmap,
    build_learning_curve,
    build_lottery_chart,
    build_policy_step_chart,
    build_ratio_histogram,
)

st.set_page_config(page_title="MARL (IQL) – Sebastian Hanisch", layout="wide")


def _instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed):
    return generate_instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed)


@st.cache_data(show_spinner=False)
def _compute_cnp(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed):
    instance = _instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed)
    return instance, run_protocol(instance)


@st.cache_data(show_spinner=False)
def _compute_heldout(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed, env_mode, sigma):
    """Held-out-Instanzen samt CNP- und CP-SAT-Werten - hängt nur vom Szenario ab, nicht vom Training."""
    instance = _instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed)
    held = heldout_instances(instance, env_mode, sigma, duration_variability, seed)
    return held, cnp_makespans(held), heldout_optima(held)


@st.cache_data(show_spinner=False)
def _compute_training(tkey):
    n_jobs, n_agents, var, travel, seed, env_mode, sigma, episodes, train_seed = tkey
    instance = _instance(n_jobs, n_agents, var, travel, seed)
    checkpoints = [c for c in C.EPISODES_CHOICES if c < episodes]
    return train_iql(instance, env_mode, sigma, episodes, train_seed, var, checkpoints=checkpoints)


@st.cache_data(show_spinner=False)
def _compute_evaluation(tkey):
    n_jobs, n_agents, var, travel, seed, env_mode, sigma, episodes, train_seed = tkey
    instance = _instance(n_jobs, n_agents, var, travel, seed)
    train = _compute_training(tkey)
    held, held_cnp, held_opt = _compute_heldout(n_jobs, n_agents, var, travel, seed, env_mode, sigma)
    return {
        "cmp": comparison(instance, train.q, held, held_cnp, held_opt, C.ORTOOLS_TIME_LIMIT_SECONDS),
        "curve": learning_curve(train, instance, held),
        "gen": generalisation_check(train.q, instance, var, seed),
        "ceiling": fixed_order_ceiling(instance),
    }


@st.cache_data(show_spinner=False)
def _compute_lottery(tkey):
    n_jobs, n_agents, var, travel, seed, env_mode, sigma, episodes, train_seed = tkey
    instance = _instance(n_jobs, n_agents, var, travel, seed)
    held, held_cnp, _ = _compute_heldout(n_jobs, n_agents, var, travel, seed, env_mode, sigma)
    return seed_lottery(instance, env_mode, sigma, episodes, train_seed, var, held, held_cnp)


@st.cache_data(show_spinner=False)
def _compute_miniature():
    instance = miniature_instance()
    results = enumerate_joint_policies(instance)
    normal, up, down = 0, 1, 2
    policy_a = ((normal, normal), (down, normal))
    policy_b = ((up, normal), (normal, normal))
    return {
        "n_total": len(results),
        "n_optimal": sum(1 for _, ms in results if ms == min(r[1] for r in results)),
        "optimum": min(r[1] for r in results),
        "cnp": joint_policy_makespan(instance, ((normal, normal), (normal, normal))),
        "a": joint_policy_makespan(instance, policy_a),
        "b": joint_policy_makespan(instance, policy_b),
        "mixed": joint_policy_makespan(instance, (policy_a[0], policy_b[1])),
    }


def _fmt_int(n):
    """Tausendertrennung deutsch (Punkt)."""
    return f"{n:,}".replace(",", ".")


def _fmt_seconds(seconds):
    return f"{seconds:.1f} s" if seconds >= 1 else f"{seconds:.2f} s"


def _start_lottery(key):
    st.session_state["lottery_owner"] = key


def _delta_metric(column, label, value, reference, reference_label, help_text=None):
    """Delta-Regel des Portfolios: Wert DIESER Karte minus Referenz, niedriger ist besser."""
    delta = value - reference
    if abs(delta) < 1e-6:
        column.metric(label, f"{value:.1f} min", delta="±0.0 min", delta_color="off", help=help_text)
    else:
        column.metric(
            label, f"{value:.1f} min", delta=f"{delta:+.1f} min ggü. {reference_label}",
            delta_color="inverse", help=help_text,
        )


st.title("🤖 Multi-Agenten-Reinforcement-Learning (unabhängiges Q-Learning) an der Kran-Auftragsvergabe")
st.markdown(
    """
Ein **unabhängiger Zweig** direkt vom Contract-Net-Protocol-Root (**contract-net-demo**). Statt eines
festen Gebotsprotokolls **lernen** die Agenten selbst, wie sie ihre Gebote anpassen: aus wiederholter
Interaktion und einer gemeinsamen Belohnung (kleiner Makespan). **MARL** ist das Feld - mehrere lernende
Agenten in einer gemeinsamen Umgebung -, **unabhängiges Q-Learning (IQL)** das einfachste Lernverfahren
darin: jeder Agent lernt für sich und behandelt die anderen als Teil der Umgebung.
"""
)
st.caption(
    "Contract Net (online, fest verdrahtet, keine Rücksicht auf spätere Aufträge) wird hier nicht durch "
    "ein Protokoll, sondern durch Erfahrung verbessert. Eigene Schwächen, alle als gemessene Zahlen "
    "unten: Trainingsaufwand, keine Optimalitätsgarantie, Nicht-Stationarität. Das zentrale CP-SAT "
    "bleibt der praktische Industriestandard - dezentrale Verfahren tauschen globale Optimalität "
    "gegen Dezentralität."
)

with st.expander("Wie funktioniert diese Demo?", expanded=True):
    st.markdown(
        r"""
**Phase 1 - Contract Net Protocol (Rekapitulation)**: wie in contract-net-demo - Aufträge werden einzeln
angekündigt, das niedrigste Gebot gewinnt, endgültig. Dient als Vergleichs-Basislinie.

**Das Lernproblem**: derselbe Ablauf, nur wählt jeder Agent (Kran) pro angekündigtem Auftrag eine
**Aktion**: `normal bieten`, `hoch bieten` (+25 min Aufschlag - er lehnt den Auftrag eher ab) oder
`niedrig bieten` (−25 min - er greift ihn eher). Den Zuschlag vergibt weiterhin Contract Nets Regel
(niedrigstes abgegebenes Gebot). Ein Agent, der nichts gelernt hat, bietet immer normal - er verhält
sich **exakt wie Contract Net** (getestet). Alles Gelernte ist also eine Abweichung von Contract Net.

**Beobachtung (streng lokal)**: ein Agent sieht nur, welcher Auftrag gerade dran ist (öffentlicher
Zähler), wie lange er selbst noch beschäftigt ist und wie weit er zum Auftrag fahren müsste - grob in
4 x 3 Stufen pro Auftrag gebucketet, also höchstens 120 Zustände. Keine Gebote und keine Zustände der
anderen Agenten. Das Lernverfahren (**Q-Learning**) ist eine Tabelle: pro Zustand und Aktion ein Wert,
der schätzt, wie gut diese Aktion für den Makespan am Ende ist.

**Belohnung**: für alle Agenten dieselbe, erst am Episodenende: `−Makespan / 10`. Welcher Agent mit
welcher Aktion den Makespan verursacht hat, weiß niemand (**Credit-Assignment**). Zwischen den Aufträgen
gibt es keine Belohnung.

**Trainingsumgebung (Umschalter)**: *Wiederkehrendes Szenario* - dieselben Aufträge und Positionen wie
gezeigt, in jeder Episode mit schwankenden Dauern (jede Dauer wird mit `exp(N(0, σ))` multipliziert;
etwa derselbe Tagesplan mit schwankenden Bearbeitungszeiten). *Zufällige Instanzen* - jede Episode eine
neue Instanz gleicher Größe. Genau dieser Kontrast ist eine der Lehren: gelernt wird nur, was wiederkehrt.
(Bei σ = 0 sind Training und Test identisch - die Agenten merken sich den Plan.)

**Nur das RELATIVE Gebot zählt**: hoch bieten für alle Agenten gleichzeitig ändert nichts. Deshalb
entstehen Konventionen wie "Agent 2 greift, Agent 1 hält sich zurück" - und deshalb sind zwei getrennt
trainierte Teams nicht ohne Weiteres kombinierbar (Cross-Play, siehe unten).

**Held-out**: die gelernte Policy wird immer auch auf 30 Instanzen getestet, die das Training nie
gesehen hat (Rauschen-Ziehungen mit anderem Zufall bzw. frische Instanzen), und mit Contract Net und dem
zentralen CP-SAT-Optimum verglichen. Die Zahl "nominal" gilt für die gezeigte Instanz ohne Rauschen.

**Drei Schwächen, alle gemessen**: *Trainingsaufwand* (wie viele Entscheidungen, wie viele Sekunden),
*keine Garantie* (Verteilung der Ergebnisse über Held-out-Instanzen und frische Szenarien, schlechtester
Fall) und *Nicht-Stationarität* (Seed-Lotterie, Cross-Play und eine exakt durchgerechnete 2x2-Miniatur):
für jeden Agenten ändert sich die Umgebung, weil die anderen gleichzeitig lernen - die beste Antwort auf
das Team von gestern ist nicht die beste Antwort auf das Team von heute. Das ist der Aufhänger für das
nächste Stück der Linie (MAPPO/CTDE mit zentralem Kritiker im Training).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = C.PRESET_HELP
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_jobs = st.slider("Anzahl Aufträge", *bounds("n_jobs_slider"), key="n_jobs_slider")
    n_agents = st.slider("Anzahl Agenten (Kräne)", *bounds("n_agents_slider"), key="n_agents_slider")
    duration_variability = st.slider(
        "Streuung der Auftragsdauer", *bounds("duration_variability_slider"), key="duration_variability_slider",
    )
    travel_time_per_unit = st.slider(
        "Anfahrtszeit pro Positionseinheit", *bounds("travel_time_per_unit_slider"),
        key="travel_time_per_unit_slider",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.button(
        "🎲 Neue Instanz generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für Auftragspositionen und -dauern.",
    )

    st.markdown("**Training**")
    env_mode = st.radio(
        "Trainingsumgebung", options=list(C.ENV_LABELS), format_func=lambda k: C.ENV_LABELS[k],
        key="env_mode_radio",
        help="Wiederkehrend: immer dieselben Aufträge, nur die Dauern schwanken. Zufällig: jede Episode "
        "eine neue Instanz.",
    )
    if env_mode == C.ENV_RECURRING:
        sigma = st.slider(
            "Rauschen σ der Dauern", C.SIGMA_MIN, C.SIGMA_MAX, key="sigma_slider", step=0.05,
            help="Jede Dauer wird pro Trainings-Episode mit exp(N(0, σ)) multipliziert. σ = 0: Training "
            "und Test identisch.",
        )
    else:
        # Ein nicht gerenderter Widget-Key verliert sonst seinen Wert (Permalink, Rückwechsel).
        st.session_state["sigma_slider"] = st.session_state["sigma_slider"]
        sigma = st.session_state["sigma_slider"]
    episodes = st.select_slider(
        "Trainings-Episoden", options=C.EPISODES_CHOICES, key="episodes_slider",
        format_func=lambda x: f"{x:,}".replace(",", "."),
        help="Mehr Episoden = mehr Erfahrung, aber auch längere Rechenzeit.",
    )
    train_seed = st.number_input(
        "Trainings-Seed", *bounds("train_seed_input"), key="train_seed_input", step=1,
        help="Zufall des Lernens (Exploration, Rauschen) - unabhängig vom Szenario-Seed.",
    )

sync_query_params(
    n_jobs, n_agents, duration_variability, travel_time_per_unit, seed, env_mode, episodes, sigma, train_seed,
)

scenario_key = (int(n_jobs), int(n_agents), duration_variability, travel_time_per_unit, int(seed))
tkey = training_key(*scenario_key, env_mode, sigma, episodes, train_seed)

with st.spinner("Führe Contract Net Protocol aus..."):
    instance, cnp_result = _compute_cnp(*scenario_key)
with st.spinner("Trainiere die Agenten und werte aus (bei vielen Episoden einige Sekunden)..."):
    train = _compute_training(tkey)
    evaluation = _compute_evaluation(tkey)
cmp = evaluation["cmp"]
ortools_makespan = cmp["ortools_nominal"] if cmp["ortools_feasible"] else None
obs_fn, _ = make_observer(instance.n_jobs, instance.n_agents)

# --- Phase 1: Contract Net Rekapitulation -----------------------------------

st.markdown("## 🎯 Phase 1: Contract Net Protocol (Rekapitulation)")

if "cn_step" not in st.session_state or st.session_state.get("cn_step_owner") != scenario_key:
    st.session_state["cn_step"] = instance.n_jobs - 1
    st.session_state["cn_step_owner"] = scenario_key

max_step = instance.n_jobs - 1
step_col, play_col = st.columns([5, 1])
with step_col:
    if max_step == 0:
        step = 0
        st.caption("Nur ein Auftrag - kein Regler nötig.")
    else:
        step = st.slider("Schritt (Auftragsvergabe)", 0, max_step, key="cn_step")
with play_col:
    auto_play_cnp = st.button("▶️ Abspielen", width="stretch", key="cnp_play")

chart_col, bid_col = st.columns([3, 2])
schedule_slot = chart_col.empty()
bid_slot = bid_col.empty()


def _render_cnp(current_step):
    schedule_slot.plotly_chart(
        build_schedule_figure(instance, cnp_result, current_step, ortools_makespan),
        width="stretch", key=f"cnp_schedule_{current_step}",
    )
    bid_slot.plotly_chart(
        build_bid_chart(cnp_result.steps[current_step]),
        width="stretch", key=f"cnp_bids_{current_step}",
    )


if auto_play_cnp:
    for s in range(0, max_step + 1):
        _render_cnp(s)
        time.sleep(0.4)
    step = max_step
else:
    _render_cnp(step)

live = stats_up_to_step(cnp_result, step)
lm1, lm2 = st.columns(2)
lm1.metric("Aufträge bisher vergeben", f"{live['jobs_awarded']} / {instance.n_jobs}")
lm2.metric("Aktuell schlechteste freie Zeit", f"{live['worst_agent_free_time']:.1f} min")

st.markdown("---")

# --- Phase 2a: Lernkurve ------------------------------------------------------

st.markdown("## 📈 Phase 2a: Was die Agenten beim Training lernen")
st.caption(
    "Jeder Punkt ist die gierige (nicht mehr explorierende) Policy nach so vielen Episoden. Frühe Stände "
    "haben kaum Erfahrung gesammelt - ihr Verhalten liegt nahe an Contract Net oder darüber."
)

curve = evaluation["curve"]
learn_max = len(curve) - 1
if "learn_step" not in st.session_state or st.session_state.get("learn_owner") != tkey:
    st.session_state["learn_step"] = learn_max
    st.session_state["learn_owner"] = tkey

lstep_col, lplay_col = st.columns([5, 1])
with lstep_col:
    if learn_max == 0:
        learn_step = 0
        st.caption("Nur ein Zwischenstand - kein Regler nötig.")
    else:
        learn_step = st.slider(
            "Trainingsstand", 0, learn_max, key="learn_step",
            format="%d",
            help="0 = frühester Zwischenstand, letzter Wert = fertig trainiert.",
        )
with lplay_col:
    auto_play_learn = st.button("▶️ Abspielen", width="stretch", key="learn_play")

learn_left, learn_right = st.columns([3, 2])
learn_curve_slot = learn_left.empty()
learn_gantt_slot = learn_right.empty()
learn_caption_slot = st.empty()


def _render_learn(idx):
    ep, nominal, _heldout_mean = curve[idx]
    q = train.snapshots.get(ep, train.q)
    dispatch = run_dispatch(instance, greedy_policy(q), obs_fn)
    learn_curve_slot.plotly_chart(
        build_learning_curve(curve, cmp["cnp_nominal"], ortools_makespan, cmp["heldout_cnp_mean"], marker_episodes=ep),
        width="stretch", key=f"learn_curve_{idx}",
    )
    learn_gantt_slot.plotly_chart(
        build_schedule_figure(instance, dispatch.protocol_result, instance.n_jobs - 1, ortools_makespan),
        width="stretch", key=f"learn_gantt_{idx}",
    )
    pct = (nominal - cmp["cnp_nominal"]) / cmp["cnp_nominal"] * 100.0
    learn_caption_slot.caption(
        f"Nach {_fmt_int(ep)} Episoden (ε war dann {epsilon_at(ep, train.n_episodes):.2f}): nominal "
        f"**{nominal:.1f} min** ({pct:+.1f} % gegenüber Contract Net {cmp['cnp_nominal']:.1f} min)."
    )


if auto_play_learn:
    for i in range(learn_max + 1):
        _render_learn(i)
        time.sleep(0.7)
    learn_step = learn_max
else:
    _render_learn(learn_step)

st.markdown("---")

# --- Phase 2b: Die gelernte Policy Schritt für Schritt -----------------------

st.markdown("## 🔍 Phase 2b: Die fertig trainierte Policy im Einsatz")
st.caption(
    "Pro Auftrag: was jeder Agent beobachtet, welche Aktion seine Q-Tabelle wählt und wie sich sein Gebot "
    "dadurch gegenüber dem normalen Contract-Net-Gebot verschiebt."
)

final_dispatch = run_dispatch(instance, greedy_policy(train.q), obs_fn)

if "policy_step" not in st.session_state or st.session_state.get("policy_step_owner") != tkey:
    st.session_state["policy_step"] = instance.n_jobs - 1
    st.session_state["policy_step_owner"] = tkey

pstep_col, pplay_col = st.columns([5, 1])
with pstep_col:
    if max_step == 0:
        policy_step = 0
        st.caption("Nur ein Auftrag - kein Regler nötig.")
    else:
        policy_step = st.slider("Schritt (Auftragsvergabe)", 0, max_step, key="policy_step")
with pplay_col:
    auto_play_policy = st.button("▶️ Abspielen", width="stretch", key="policy_play")

pchart_col, pinfo_col = st.columns([3, 2])
pgantt_slot = pchart_col.empty()
pbid_slot = pinfo_col.empty()
pinfo_slot = pinfo_col.empty()


def _render_policy(s):
    pgantt_slot.plotly_chart(
        build_schedule_figure(instance, final_dispatch.protocol_result, s, ortools_makespan),
        width="stretch", key=f"policy_gantt_{s}",
    )
    pbid_slot.plotly_chart(
        build_policy_step_chart(final_dispatch, s), width="stretch", key=f"policy_bids_{s}",
    )
    lines = []
    for agent in range(instance.n_agents):
        job_no, free_text, travel_text = describe_state(final_dispatch.states[s][agent], instance.n_jobs, instance.n_agents)
        action = C.ACTION_NAMES[final_dispatch.actions[s][agent]]
        lines.append(f"- **Agent {agent + 1}**: Auftrag {job_no}, {free_text}, {travel_text} → *{action}*")
    pinfo_slot.markdown("\n".join(lines))


if auto_play_policy:
    for s in range(0, max_step + 1):
        _render_policy(s)
        time.sleep(0.6)
    policy_step = max_step
else:
    _render_policy(policy_step)

st.markdown("---")

# --- Vergleich -----------------------------------------------------------------

st.subheader("📐 Der Preis des Lernens")

n_pct = cmp["iql_vs_cnp_nominal_pct"]
h_pct = cmp["heldout_iql_vs_cnp_pct"]
gen = evaluation["gen"]

lottery_key = tkey
lottery_active = (
    st.session_state.get("lottery_owner") == lottery_key
    or st.session_state.get("lottery_preset_key") == lottery_key
)
lot = None
if lottery_active:
    with st.spinner("Trainiere mehrere Läufe für die Seed-Lotterie..."):
        lot = _compute_lottery(lottery_key)

st.markdown("**Die gezeigte Instanz (nominal, ohne Rauschen)**")
c1, c2, c3 = st.columns(3)
c1.metric("Contract Net (roh)", f"{cmp['cnp_nominal']:.1f} min")
_delta_metric(c2, "Gelernte Policy (IQL)", cmp["iql_nominal"], cmp["cnp_nominal"], "Contract Net roh")
if cmp["ortools_feasible"]:
    _delta_metric(
        c3, "Zentrale Optimierung (CP-SAT)", cmp["optimum_reference"], cmp["iql_nominal"], "gelernt",
        help_text=f"Echter industrieller Solver, {cmp['ortools_wall_time']:.2f}s - "
        + ("beweist Optimalität." if cmp["ortools_optimal"] else "Zeitlimit erreicht, beste gefundene Lösung.")
        + " CP-SAT rundet Zeiten auf; der Wert ist deshalb nie größer als eine zulässige Lösung angesetzt.",
    )
else:
    c3.metric("Zentrale Optimierung (CP-SAT)", "kein Ergebnis im Zeitlimit")

st.markdown(
    f"**Held-out: Mittel über {C.N_HELDOUT} Instanzen, die das Training nie gesehen hat**"
)
h1, h2, h3 = st.columns(3)
h1.metric("Contract Net (roh)", f"{cmp['heldout_cnp_mean']:.1f} min")
_delta_metric(h2, "Gelernte Policy (IQL)", cmp["heldout_iql_mean"], cmp["heldout_cnp_mean"], "Contract Net roh")
if cmp["heldout_opt_mean"] is not None:
    _delta_metric(h3, "Zentrale Optimierung (CP-SAT)", cmp["heldout_opt_mean"], cmp["heldout_iql_mean"], "gelernt")
else:
    h3.metric("Zentrale Optimierung (CP-SAT)", "nicht für alle im Zeitlimit")

if cmp["cnp_gap_pct"] is not None:
    st.caption(
        f"Lücke zum zentralen Optimum (nominal): Contract Net **{cmp['cnp_gap_pct']:.1f} %** → gelernt "
        f"**{cmp['iql_gap_pct']:.1f} %**. Gelernt vs. Contract Net: nominal **{n_pct:+.1f} %**, "
        f"Held-out **{h_pct:+.1f} %**."
    )

if n_pct > C.WORSE_THAN_CNP_THRESHOLD_PCT or h_pct > C.WORSE_THAN_CNP_THRESHOLD_PCT:
    reasons = []
    if env_mode == C.ENV_RANDOM:
        reasons.append(
            "Das Training sah immer neue Instanzen - ohne wiederkehrendes Muster gibt es nichts, was sich "
            "einprägen ließe (Umschalter auf *Wiederkehrendes Szenario* probieren)."
        )
    if episodes <= 1000:
        reasons.append("Mit so wenig Episoden ist die Q-Tabelle kaum gefüllt (mehr Episoden probieren).")
    if env_mode == C.ENV_RECURRING and sigma >= 0.5:
        reasons.append("Starkes Rauschen: jede Episode sieht ein anderes Szenario, das Gelernte passt schlechter.")
    if not reasons:
        reasons.append(
            "Die Agenten haben eine Konvention gelernt, die auf dieser Instanz oder ihren Varianten nicht trägt "
            "(siehe Seed-Lotterie unten)."
        )
    st.warning(
        f"⚠️ **Die gelernte Policy ist schlechter als Contract Net**: nominal {n_pct:+.1f} %, Held-out "
        f"{h_pct:+.1f} %. Gelernt heißt nicht besser - es gibt keine Garantie. " + " ".join(reasons)
    )
elif n_pct <= -C.CLEARLY_BETTER_THRESHOLD_PCT and h_pct <= -C.CLEARLY_BETTER_THRESHOLD_PCT:
    fresh_text = (
        f"auf frischen Szenarien gleicher Größe liegt dieselbe Policy bei {gen['fresh_iql_vs_cnp_pct']:+.1f} % "
        f"gegenüber Contract Net."
    )
    if lot is not None and lot["n_worse_than_cnp"] > 0:
        st.warning(
            f"⚠️ **Dieser Lauf schlägt Contract Net** (nominal {n_pct:.1f} %, Held-out {h_pct:.1f} %) - aber in der "
            f"Seed-Lotterie unten sind {lot['n_worse_than_cnp']} von {len(lot['seeds'])} Trainingsläufen schlechter "
            f"als Contract Net. Dieser hier hatte Glück; {fresh_text}"
        )
    else:
        stability = (
            "Wie stabil das über andere Trainings-Seeds ist, zeigt die Seed-Lotterie unten"
            if lot is None else
            f"In der Seed-Lotterie unten sind alle {len(lot['seeds'])} Trainingsläufe besser als Contract Net"
        )
        st.success(
            f"✅ **Lernen zahlt sich hier aus**: nominal {n_pct:.1f} %, Held-out {h_pct:.1f} % gegenüber Contract Net. "
            f"{stability}; {fresh_text}"
        )
else:
    st.info(
        f"Kaum Unterschied zu Contract Net (nominal {n_pct:+.1f} %, Held-out {h_pct:+.1f} %) - oder der Vorteil "
        f"zeigt sich nur auf einem der beiden."
    )

tab_cost, tab_guarantee, tab_nonstat = st.tabs(
    ["⏱️ Trainingsaufwand", "🎯 Keine Garantie", "🔀 Nicht-Stationarität"]
)

with tab_cost:
    t1, t2, t3 = st.columns(3)
    t1.metric("Trainings-Episoden", _fmt_int(train.n_episodes))
    t2.metric("Agenten-Entscheidungen", _fmt_int(train.n_decisions))
    t3.metric("Trainingszeit", _fmt_seconds(train.wall_time_s), help="Auf diesem Rechner, ohne Auswertung.")
    st.markdown(
        f"Contract Net entscheidet in einem einzigen Durchlauf, CP-SAT löst diese Instanz in "
        f"**{cmp['ortools_wall_time']:.2f} s** exakt. Das Training braucht **{_fmt_seconds(train.wall_time_s)}** und "
        f"{_fmt_int(train.n_decisions)} Entscheidungen, bevor die Policy besser als Contract Net sein *kann* - und "
        f"selbst dann nur für ein Szenario, das wiederkehrt. Der Aufwand lohnt sich nur, wenn dasselbe "
        f"Szenario oft genug wiederkommt; ein zentraler Planer löst jedes neue Szenario einfach neu."
    )
    st.caption(
        f"Zum Vergleich - zufällige Aufschläge statt gelernter: nominal im Mittel "
        f"**{cmp['random_bias_nominal']:.1f} min** "
        f"({(cmp['random_bias_nominal'] - cmp['cnp_nominal']) / cmp['cnp_nominal'] * 100:+.0f} % gegenüber Contract Net) - "
        f"der Gewinn (falls vorhanden) kommt also nicht von irgendwelchen Aufschlägen, sondern vom Lernen."
    )

with tab_guarantee:
    g1, g2, g3 = st.columns(3)
    g1.metric(
        "Held-out-Instanzen besser als Contract Net", f"{cmp['heldout_beat_frac'] * 100:.0f} %",
        help=f"Anteil der {C.N_HELDOUT} Held-out-Instanzen, auf denen die gelernte Policy einen kleineren Makespan hat.",
    )
    g2.metric("Held-out-Instanzen schlechter als Contract Net", f"{cmp['heldout_lose_frac'] * 100:.0f} %")
    g3.metric("Schlechteste Instanz", f"{cmp['heldout_worst_pct']:+.0f} %", help="Größte Verschlechterung ggü. Contract Net.")
    st.plotly_chart(build_ratio_histogram(cmp["heldout_ratios_pct"]), width="stretch", key="ratio_hist")
    st.markdown(
        f"**Auf frischen Szenarien** ({C.N_FRESH_SCENARIOS} neue Instanzen gleicher Größe, neue Positionen und "
        f"Dauern): dieselbe Policy liegt im Mittel bei **{gen['fresh_iql_vs_cnp_pct']:+.1f} %** gegenüber Contract Net "
        f"(besser in {gen['fresh_beat_frac'] * 100:.0f} %, schlechter in {gen['fresh_lose_frac'] * 100:.0f} % der Fälle)."
    )
    if env_mode == C.ENV_RECURRING:
        st.caption(
            "Im wiederkehrenden Szenario lernt die Policy dieses Szenario auswendig statt eine allgemeine "
            "Dispatch-Regel - auf fremden Szenarien ist sie deshalb typischerweise schlechter als Contract Net. "
            "Ein Umschalter auf *Zufällige Instanzen* zeigt die andere Seite: dort gibt es nichts Wiederkehrendes zu lernen."
        )
    st.caption(
        f"Obergrenze für jede Policy dieser Art: **{evaluation['ceiling']:.1f} min** (bestmögliche Zuteilung "
        f"bei fester Ankündigungsreihenfolge) - eine gelernte Policy kann sie nie unterbieten, und CP-SAT "
        f"darf zusätzlich die Reihenfolge innerhalb eines Agenten wählen."
    )

with tab_nonstat:
    st.markdown("**Exakt durchgerechnet: die 2x2-Miniatur**")
    mini = _compute_miniature()
    st.markdown(
        "2 Aufträge (Position 5/Dauer 5 und Position 0/Dauer 10), 2 Agenten (Start bei 5 und 15), Anfahrt 1 min "
        f"pro Einheit. Eine Policy wählt pro Agent und Auftrag eine der 3 Aktionen: 3⁴ = **{mini['n_total']}** "
        f"gemeinsame Policies, alle durchgerechnet. Contract Net: **{mini['cnp']:.0f} min**, Optimum: "
        f"**{mini['optimum']:.0f} min** - erreicht von **{mini['n_optimal']}** der {mini['n_total']} Policies."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Team A (Agent 2 greift Auftrag 1)", f"{mini['a']:.0f} min")
    m2.metric("Team B (Agent 1 lehnt Auftrag 1 ab)", f"{mini['b']:.0f} min")
    m3.metric(
        "Agent 1 aus A + Agent 2 aus B", f"{mini['mixed']:.0f} min",
        delta=f"{mini['mixed'] - mini['a']:+.0f} min ggü. Team A", delta_color="inverse",
        help="Zwei einzeln optimale Konventionen, deren Mischung nicht mehr optimal ist.",
    )
    st.caption(
        "Beide Teams sind für sich optimal, ihre Mischung fällt auf Contract Nets Ergebnis zurück: Die beste Antwort "
        "eines Agenten hängt von den Policies der anderen ab. Lernen alle gleichzeitig, verschiebt sich diese "
        "beste Antwort ständig - das ist Nicht-Stationarität."
    )

    st.markdown("**Gemessen: die Seed-Lotterie**")
    if not lottery_active:
        est = C.N_LOTTERY_SEEDS * train.wall_time_s * min(1.0, C.LOTTERY_EPISODE_CAP / train.n_episodes)
        st.button(
            f"🎰 Seed-Lotterie starten ({C.N_LOTTERY_SEEDS} Trainingsläufe, ca. {est:.0f} s)",
            on_click=_start_lottery, args=(lottery_key,), key="lottery_start",
        )
        st.caption(
            "Trainiert dasselbe Setup mit anderen Trainings-Seeds und zeigt, wie stark das Ergebnis vom Zufall "
            "des Lernens abhängt - plus Cross-Play zwischen den Läufen."
        )
    else:
        st.plotly_chart(build_lottery_chart(lot), width="stretch", key="lottery_chart")
        l1, l2, l3 = st.columns(3)
        l1.metric("Läufe schlechter als Contract Net", f"{lot['n_worse_than_cnp']} von {len(lot['seeds'])}")
        l2.metric("Streuung (Std, Held-out)", f"{lot['heldout_std_pct']:.1f} % von CNP")
        l3.metric("Cross-Play-Strafe", f"{lot['crossplay_penalty_pct']:+.1f} % von CNP")
        if lot["episodes"] < train.n_episodes:
            st.caption(f"Aus Zeitgründen mit {_fmt_int(lot['episodes'])} statt {_fmt_int(train.n_episodes)} Episoden trainiert.")
        if lot["n_worse_than_cnp"] > 0 or lot["heldout_std_pct"] >= C.LOTTERY_SPREAD_WARNING_PCT:
            st.warning(
                f"⚠️ **Das Ergebnis hängt vom Trainings-Seed ab**: {lot['n_worse_than_cnp']} von {len(lot['seeds'])} "
                f"Läufen sind auf Held-out-Instanzen schlechter als Contract Net, die Spanne reicht von "
                f"{min(lot['heldout_pct']):+.1f} % bis {max(lot['heldout_pct']):+.1f} %. Wer nur einen Lauf sieht, "
                f"weiß nicht, ob er Glück hatte."
            )
        if lot["crossplay_penalty_pct"] >= C.CROSSPLAY_PENALTY_WARNING_PCT:
            st.warning(
                f"⚠️ **Cross-Play**: setzt man Agent 1 aus einem Trainingslauf und die übrigen Agenten aus einem "
                f"anderen ein, wird es im Mittel {lot['crossplay_penalty_pct']:.1f} % (von Contract Net) schlechter als "
                f"das gemeinsam trainierte Team - obwohl alle Policies gleich gut trainiert sind. Die Agenten haben "
                f"Konventionen gelernt, die nur zusammen mit ihren Trainingspartnern funktionieren."
            )
        else:
            st.info(
                f"Cross-Play: {lot['crossplay_penalty_pct']:+.1f} % (von Contract Net) gegenüber dem gemeinsam "
                f"trainierten Team - hier greifen die Konventionen kaum ineinander."
            )
        st.plotly_chart(
            build_crossplay_heatmap(lot["crossplay_matrix"], lot["seeds"], cmp["heldout_cnp_mean"]),
            width="stretch", key="crossplay_heatmap",
        )
        st.caption(
            "Zeilen: Trainingslauf von Agent 1, Spalten: Trainingslauf der übrigen Agenten. Die Diagonale ist das "
            "gemeinsam trainierte Team; alles daneben sind kombinierte Teams. Farbe = Held-out-Makespan relativ zu "
            "Contract Net."
        )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gebot und Aktion.** Agent $a$ bietet für Auftrag $j$ die Fertigstellungszeit
$b_a(j) = f_a + \tau\,|p_a - q_j| + d_j$ (eigene Freizeit $f_a$, Position $p_a$, Auftragsposition $q_j$,
Dauer $d_j$, Anfahrtszeit pro Einheit $\tau$). Mit Aktion $u_a \in \{0, +\Delta, -\Delta\}$ ($\Delta = 25$ min)
lautet das abgegebene Gebot $\tilde b_a(j) = b_a(j) + u_a$. Zuschlag: $\arg\min_a (\tilde b_a(j), a)$.
Für $u \equiv 0$ ist das exakt Contract Net.

**Beobachtung.** $s_a = (j,\ \text{Bucket}(f_a),\ \text{Bucket}(\tau|p_a - q_j|))$ mit 4 Freizeit-Buckets
(Kanten $0.3,\ 0.7,\ 1.1$ der fairen Last $n \cdot 10 / k$) und 3 Anfahrt-Buckets (Kanten 3 und 8 min):
$|S| = 12\,n$.

**Belohnung.** Gemeinsam, nur am Episodenende: $R = -\text{Makespan} / 10$.

**Q-Learning (jeder Agent für sich, $\gamma = 1$).**

$$
Q_a(s, u) \leftarrow Q_a(s, u) + \alpha\,\big(y - Q_a(s, u)\big), \qquad
y = \begin{cases} R & \text{letzter Auftrag} \\ \max_{u'} Q_a(s', u') & \text{sonst} \end{cases}
$$

mit $\alpha = 0.1$ und $\varepsilon$-greedy-Exploration: $\varepsilon(e) = \max\!\big(0.05,\ 1 - 0.95\,e / (0.7\,N)\big)$
über $N$ Episoden. Gleichstände zwischen Aktionen gehen an Aktion 0 (*normal*), deshalb verhält sich
eine untrainierte Policy exakt wie Contract Net.

**Obergrenze für jede Policy.** Alle Policies halten die Ankündigungsreihenfolge fest und wählen nur, welcher
Agent welchen Auftrag bekommt. Daher gilt

$$
\text{Makespan(Policy)} \ \ge\ \text{Bestes bei fester Reihenfolge} \ \ge\ \text{CP-SAT-Optimum}.
$$

**Nicht-Stationarität.** Aus Sicht von Agent $a$ ist die Übergangs- und Belohnungsfunktion
$P(s', R \mid s, u_a, \pi_{-a})$ von den Policies $\pi_{-a}$ der anderen abhängig. Ändern diese sich beim Lernen,
gilt die Konvergenzgarantie des tabellarischen Q-Learning (stationäre Umgebung) nicht mehr - es gibt keine
Optimalitäts- oder Konvergenzgarantie für IQL.

Implementiert in `marl_env.py` (Umgebung), `marl_obs.py` (Beobachtung), `marl_iql.py` (Lernverfahren),
`marl_evaluation.py` (alle gezeigten Kennzahlen) und `marl_exact.py` (exakte Referenzen).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
