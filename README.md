# Multi-Agenten-Reinforcement-Learning (unabhängiges Q-Learning) an der Kran-Auftragsvergabe – Streamlit-Demo

Viertes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning", **Multi-Agenten-Koordinations-Linie** - ein
**unabhängiger Zweig** direkt vom [contract-net-demo](../contract-net-demo)-Root (neben
[task-swap-demo](../task-swap-demo) und [dcop-demo](../dcop-demo)).

## Was dieses Stück von den anderen unterscheidet

- **contract-net-demo**: festes Gebotsprotokoll, myopisch, endgültig.
- **task-swap-demo**: repariert die Zuteilung nachträglich per lokaler Suche.
- **dcop-demo**: löst ein Summen-Modell exakt - das Modell ist aber nicht der Makespan.
- **marl-demo**: die Agenten **lernen** aus wiederholter Interaktion, wie sie ihre Gebote
  anpassen. **MARL** ist das Feld (mehrere lernende Agenten in einer gemeinsamen Umgebung),
  **unabhängiges Q-Learning (IQL)** das einfachste Lernverfahren darin - jeder Agent lernt für
  sich und behandelt die anderen als Teil der Umgebung. Die Schwächen sind gemessene Zahlen,
  keine Behauptungen: **Trainingsaufwand**, **keine Optimalitätsgarantie**,
  **Nicht-Stationarität** (Aufhänger für das nächste Stück, MAPPO/CTDE).

Ehrliche Linien-Rahmung: das zentrale CP-SAT bleibt der praktische Industriestandard;
dezentrale Verfahren tauschen globale Optimalität gegen Dezentralität.

## Das Lernproblem

Dieselbe Ankündigen-Bieten-Zuschlag-Schleife wie Contract Net, nur wählt jeder Agent pro Auftrag
eine **Aktion**: einen Aufschlag auf sein eigenes Gebot - `normal` (0), `hoch` (+25 min, lehnt
eher ab) oder `niedrig` (−25 min, greift eher). Den Zuschlag vergibt weiter Contract Nets Regel.
Ein untrainierter Agent bietet immer normal ⇒ **exakt Contract Net** (getestet).

- **Beobachtung** (streng lokal): Auftrags-Index (öffentlicher Zähler), eigene Freizeit (4 Buckets),
  eigene Anfahrt (3 Buckets) - höchstens 120 Zustände. Keine Gebote anderer Agenten.
- **Belohnung**: gemeinsam, nur am Episodenende: `−Makespan / 10` (echtes Credit-Assignment).
- **Verfahren**: tabellarisches Q-Learning, α=0.1, γ=1, ε linear 1.0→0.05 über die ersten 70 % der
  Episoden, eine Tabelle pro Agent, numpy-frei (Python-Listen sind hier schneller).
- **Trainingsumgebung (Umschalter)**: *Wiederkehrendes Szenario* (dieselben Aufträge, Dauern ×
  `exp(N(0,σ))`) vs. *Zufällige Instanzen* (jede Episode eine neue). Das ist eine der Lehren: **gelernt
  wird nur, was wiederkehrt.**

## Empirische Befunde (vor dem Bau gemessen)

Der erste Ansatz (IQL, nur lokale Sicht, zufällige Instanzen) **scheitert**: im Prototyp im Mittel
+1…+14 % schlechter als rohes Contract Net, auch nach 300k Episoden; mit dem fertigen Code
nachgemessen (16 Szenarien x 4 Trainings-Seeds, 40 000 Episoden) ist die gelernte Policy auf
Held-out-Instanzen in **allen 64 Kombinationen** schlechter als Contract Net (+3…+24 %). Echtes
Lernen entsteht nur im wiederkehrenden Szenario (σ=0.3, gleiche Messung): nominal bis −30 %
gegenüber Contract Net, aber je nach Szenario auch kein Vorteil oder schlechter - und auf
**frischen** Szenarien ist dieselbe Policy +2…+16 % *schlechter* als Contract Net (sie merkt sich
den Plan statt eine allgemeine Dispatch-Regel zu lernen). Beides ist in der App sichtbar. Die Presets sind gegen den
echten Code kalibriert (n=8, k=3, Streuung 0.3, Anfahrt 1.0; % gegenüber Contract Net):

| Preset | nominal | Held-out | Aussage |
|---|---|---|---|
| Lernen schlägt Contract Net | −21.6 | −14.7 | 6 von 6 Trainings-Seeds besser |
| Kaum Vorteil | +1.1 | −2.0 | wenig Spielraum |
| Trainings-Seed entscheidet | −6.2 | −5.8 | Lotterie: −12.7…+39.3 nominal, 4 von 6 Läufen schlechter |
| Schlechter als Contract Net | +39.4 | +14.4 | σ=0.6, alle 6 Seeds schlechter |
| Zu wenig Training | +38.7 | +35.3 | 300 statt 40000 Episoden, gleiches Szenario wie Preset 1 |
| Zufällige Instanzen: kein Vorteil | +21.5 | +10.3 | Trainingsumgebung "zufällig", alle 6 Seeds schlechter |

## Die drei Schwächen als Zahlen

1. **Trainingsaufwand**: Episoden, Agenten-Entscheidungen (960 000 bei 40 000 Episoden, 8 Aufträge,
   3 Agenten), Trainingszeit vs. CP-SAT-Lösungszeit.
2. **Keine Garantie**: Verteilung über 30 Held-out-Instanzen, schlechteste Instanz, Anteil
   besser/schlechter, frische Szenarien; Obergrenze für jede Policy dieser Art
   (`fixed_order_ceiling`).
3. **Nicht-Stationarität**: **Seed-Lotterie** (6 Trainingsläufe), **Cross-Play** (Agent 1 aus Lauf a,
   die übrigen aus Lauf b - konfundierungsfrei, weil alle Policies gleich gut trainiert sind) und
   eine exakt durchgerechnete **2x2-Miniatur**: von 3⁴ = 81 gemeinsamen Policies erreichen 18 das
   Optimum (15 min statt Contract Nets 20); zwei einzeln optimale Teams A und B geben gemischt
   wieder 20 - die beste Antwort eines Agenten hängt von den Policies der anderen ab.

## Verifikation

- **Untrainiert = Contract Net** (50 Zufallsinstanzen), **Float-Kernel = `run_protocol`** (300 Instanzen),
  **Vehikel-Pfad = Kernel** für zufällige Q-Tabellen.
- **Q-Update und ε-Schedule** an handgerechneten Beispielen; Determinismus; Trainings-Seed vom
  Szenario-Seed entkoppelt; Held-out-Menge unabhängig vom Trainings-Seed.
- **2x2-Miniatur**: 81 Policies enumeriert, Optimum 15, 18 Optimal-Policies, Nicht-Stationarität als
  exakter Fakt; IQL findet das Optimum in 10/10 Trainings-Seeds.
- **Dominanz** (300 Instanzen): jede Policy ≥ `fixed_order_ceiling` ≥ CP-SAT-Optimum (bis auf Rundung);
  Decke = memoisierte DP = Enumeration.
- **Preset-Bänder**, **Lotterie-Streuung** und **Cross-Play-Strafe** als Regressionstests.
- **AppTest-Rauchtests**: Default, jedes Preset, Umschalter (σ-Wert bleibt erhalten), Kleinstinstanz.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, Lernkurve, Policy-Schritte, "Preis des Lernens" mit drei Tabs |
| `cn_constants.py` | Defaults, Regler-Grenzen, Lern-Konstanten, `PRESETS`, Bänder |
| `cn_presets.py` | `SettingSpec`/Permalink-Logik (erweitert um Umgebung, Episoden, σ, Trainings-Seed) |
| `cn_scenario.py`, `cn_bidding.py`, `cn_protocol.py`, `cn_schedule.py` | Vehikel + CNP (unverändert aus dcop-demo) |
| `cn_ortools_reference.py`, `cn_bruteforce.py`, `cn_evaluation.py`, `cn_visualization.py` | Referenzlöser und CNP-Charts (unverändert) |
| `marl_env.py` | Dispatch-Umgebung: schneller Float-Kernel + Vehikel-Pfad, Trainingsumgebungen, Held-out |
| `marl_obs.py` | Lokale Beobachtung in Buckets |
| `marl_iql.py` | Q-Update, ε-Schedule, `train_iql` |
| `marl_evaluation.py` | Vergleich, Lernkurve, Generalisierung, Cross-Play, Seed-Lotterie |
| `marl_exact.py` | Obergrenze, memoisierte DP, 2x2-Miniatur samt Enumeration |
| `marl_visualization.py` | Lernkurve, Gebots-Schritt, Heatmap, Lotterie, Histogramm |
| `tests/` | Handrechnungen, Kernel-Äquivalenz, Miniatur, Dominanz, Preset-Bänder, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
