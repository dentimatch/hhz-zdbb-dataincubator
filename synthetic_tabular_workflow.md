# Cursor Vibe: Synthetic Tabular Workflow

## 1. Open-Source Landscape for Tabular Synthesis

| Library | License | Stärken | Grenzen | Hinweise |
| --- | --- | --- | --- | --- |
| SDV (Synthetic Data Vault) | MIT | Reife Bibliothek mit Modulen für Single-/Multi-Table, Zeitreihen; SDMetrics-Evaluierung integriert; aktiver Support | Training kann bei großen Tabellen GPU/TPU erfordern; Dokumentation verteilt | https://sdv.dev
| YData Synthetic | Apache-2.0 | GAN-/VAE-Modelle mit Fokus auf Datenqualität; einfache API, Tutorials; CLI & Notebook-Vorlagen | Vergleichsweise junges OSS-Projekt; Evaluierungen weniger umfangreich als SDV | https://github.com/ydataai/ydata-synthetic
| Gretel Synthetics | Apache-2.0 | CLI/SDK, Differential-Privacy-Optionen, SaaS-Integration möglich; Beispiel-Notebooks | OSS-Kern liefert weniger Komfortfunktionen als SaaS; Abhängigkeit von TensorFlow | https://github.com/gretelai/gretel-synthetics
| DataSynthesizer | MIT | Minimalistisch, schnell, einfache „mode/independent/correlated“-Strategien; gut für Prototyping | Weniger präzise bei komplexen Beziehungen; kaum automatische Evaluierung | https://github.com/DataResponsibly/DataSynthesizer
| MOSTLY AI (Community) | Proprietär (Free Tier) | UI-geführte Pipeline, Out-of-the-box RLS; kostenlose Community-Edition für kleinere Datensätze | Kein Voll-Open-Source; Limits beim Datenvolumen; Account erforderlich | https://mostly.ai

Für KMUs, die Cursor Vibe einsetzen möchten, empfiehlt sich ein Python-zentrierter Stack (SDV oder YData Synthetic). Beide lassen sich lokal ausführen, versionieren und automatisieren.

## 2. Empfohlene Stack-Auswahl

- **Primär: SDV (Single Table → `CTGANSynthesizer`)**
  - Läuft vollständig lokal, Python ≥3.9.
  - Enthält automatische Metadata-Erkennung (`SingleTableMetadata.detect_from_dataframe`).
  - SDMetrics liefert sofort nutzbare Utility-/Privacy-Metriken.
  - Große Community, langfristige Wartung.

- **Optional: YData Synthetic**
  - Alternative, wenn GAN-/TimeGAN-basierte Modelle mit GPU geplant sind.
  - Notebook-First, gute Ergänzung für explorative Experimente.

## 3. Implementation Guide (Cursor Vibe)

### 3.1 Umgebung vorbereiten

1. Virtuelle Umgebung anlegen (PowerShell):
   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install "sdv[all]" pandas pyarrow streamlit
   ```
2. Optional: Abhängigkeiten in `requirements.txt` einfrieren (`pip freeze > requirements.txt`).

### 3.2 Beispielskript (`sdv_tabular_example.py`)

Das Skript liest eine Beispieldatei, trainiert `CTGANSynthesizer` und schreibt ein synthetisches Dataset. Die Bewertung erfolgt über `evaluation_utils.run_quality_report`, das SDMetrics *QualityReport* kapselt.

- `synthesis_runner.py` stellt eine gemeinsame Laufzeit-Schicht bereit (Metadaten-Erkennung, Dauerheuristik, Fortschritts-Callbacks), die sowohl CLI als auch Streamlit nutzen.
- CLI zeigt vor dem Start eine grobe Laufzeitabschätzung und während des Trainings einen aktualisierten Status (Heuristik).

```text
python sdv_tabular_example.py \
  --csv data/input.csv \
  --rows 1000 \
  --output data/synthetic.csv
```

Wichtige Anpassungspunkte:

- Datenschema: optional eigenes `metadata.json` nutzen (siehe SDV-Dokumentation).
- Sampling: `--rows` steuert die Anzahl synthetischer Zeilen (Standard = Originalgröße).
- Modellwahl: für sehr kleine Datensätze kann `GaussianCopulaSynthesizer` stabiler sein; Modell wird im Skript parametrisiert.
- Reproduzierbarkeit: `--random-seed` setzt Modell-seitig `random_state` (CTGAN/TVAE). Falls das Modell keinen Parameter bietet (z. B. GaussianCopula), werden globale RNGs (Python/Numpy, optional Torch) gesät.
- Streamlit UI lädt Datensatz und Metadaten gemeinsam gecacht (`load_dataset`), wodurch wiederholte Aufrufe schneller werden.

### 3.3 Notebooks und Tests

- Cursor Vibe unterstützt Jupyter: `python -m ipykernel install --user --name sdv-env` und Notebook `.ipynb` erstellen.
- Für Unit-Tests: `pytest` + Fixtures mit kleinen CSV-Dateien; SDV kann deterministische Seeds (`synthesizer.set_random_state`) setzen.

### 3.4 Streamlit UI (Optional)

- App starten:
  ```powershell
  streamlit run streamlit_app.py
  ```
- Funktionen: CSV aus `data/` auswählen, Vorschau/Statistiken ansehen, gewünschte zusätzliche Zeilen festlegen, synthetische Daten generieren, Evaluation anzeigen, Ergebnis herunterladen.
- Ausgabedateien werden im selben Verzeichnis wie das Original mit Suffix (Standard `_synthetic`) gespeichert; Beispiel: `patients_synthetic.csv`.
- Evaluation nutzt SDMetrics *QualityReport* (via `evaluation_utils.run_quality_report`); Score, Detailmetriken und numerische Vergleichstabellen erscheinen unter „Ergebnisse & Validierung“.
- Während der Synthese zeigt ein Fortschrittsbalken eine heuristische Restzeit an; nach Abschluss werden geschätzte und tatsächliche Dauer in der Zusammenfassung notiert. Der Tab „Infos“ speist sich aus `docs/explainers.md` und erklärt Modelle, Seeds & Kennzahlen.
- Über den Expander „Trainingseinstellungen“ können bei CTGAN/TVAE optional `epochs` und `batch_size` angepasst werden (defaults: 300/500 bzw. 300/256).
- Upload/Änderungen: CSV-Dateien können per Drag & Drop hochgeladen werden (werden in `data/` abgelegt). Externe Änderungen an der gewählten Datei werden anhand des Zeitstempels erkannt und führen zu einem Hinweis inkl. Neu-Laden der Daten.
- Seeding: Die Streamlit-Oberfläche setzt den angegebenen Seed beim Instanziieren des Synthesizers; Modelle ohne `random_state` nutzen globales Seeding.

## 4. Qualität & Compliance

- **Utility-Metriken**
  - `evaluation_utils.run_quality_report(real, synthetic, metadata)` → liefert Utility-Score [0,1] + Detailmetriken.
  - Visualisierungen: `sdmetrics.visualization.get_column_plot` für Verteilungen.

- **Privacy Checks**
  - Attribute Disclosure: `sdmetrics.single_table.PrivacyReport` signalisiert Reidentifikationsrisiken.
  - Distance-to-Closest-Record (DCR) prüfen und Schwellen (z. B. >0.1) dokumentieren.
  - Differential Privacy: Falls erforderlich, `clamp` + `noise_multiplier` über DP-fähige Varianten (z. B. Gretel DP, OpenDP).

- **Governance**
  - Logging: Modellparameter, Trainingszeitpunkte, Seeds in YAML protokollieren.
  - Data Minimization: Quellsamples pseudonymisieren/anonymisieren vor dem Training.

## 5. Next Steps für KMU-Rollout

- **Produktisierung**
  - Script in CLI verpacken (`typer`/`click`), Artefakte in `artifacts/` ablegen.
  - CI-Workflow (GitHub Actions/Azure DevOps) mit Tests + Qualitätsreporten automatisieren.

- **Deployment**
  - Für Batch-Jobs: geplanter Task (Windows Scheduler) oder Docker-Container.
  - Für Self-Service: FastAPI- oder Flask-Endpoint, der eingehende CSVs verarbeitet (nur in sicheren Umgebungen!).

- **Monitoring**
  - Kennzahlen (Utility/Privacy Scores) in zentralem Dashboard halten.
  - Regelmäßige Retrain-Zyklen definieren (z. B. monatlich, bei Schemaänderungen ad hoc).

- **Compliance**
  - Datenschutz-Folgenabschätzung dokumentieren (DSGVO Art. 35, falls nötig).
  - Verträge: definieren, ob synthetische Daten als Betriebsgeheimnis gelten und wie Vertrieb erfolgt (SaaS vs. On-Premises).

Damit steht ein Cursor-kompatibler Fahrplan: lokale Entwicklung in Python, reproduzierbare Scripts, evaluierte Qualität und klare nächste Schritte zur Produktivsetzung.

