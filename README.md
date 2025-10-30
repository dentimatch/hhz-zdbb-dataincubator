# Synthetic Data Workbench

Ein leichtgewichtiger Werkzeugkasten für KMUs, um tabellarische Beispieldaten in synthetische Datensätze zu überführen. Er kombiniert ein Python-Skript für die Kommandozeile und eine interaktive Streamlit-Oberfläche.

## Inhalt
- `data/` – Beispiel-CSV-Dateien (eigene Dateien hier ablegen)
- `evaluation_utils.py` – Gemeinsame Hilfsfunktionen für Qualitätsmetriken
- `sdv_tabular_example.py` – CLI-Skript zur Erzeugung synthetischer Daten
- `streamlit_app.py` – Streamlit-Web-App für Exploration, Synthese und Evaluation
- `synthetic_tabular_workflow.md` – Detailleitfaden zu Tools, Evaluierung und Rollout

## Voraussetzungen
- Windows 10/11 (getestet) oder kompatibles OS
- Python 3.13 (getestet mit 3.13.5); kompatibel mit 3.9–3.13
- Internetzugang für das erstmalige Installieren der Python-Pakete

## Einrichtung
1. Repository im gewünschten Ordner platzieren (z. B. `C:\Cursor_Git\hhz-zdbb-dataincubator`).
2. PowerShell öffnen und ins Projektverzeichnis wechseln:
   ```powershell
   cd C:\Cursor_Git\hhz-zdbb-dataincubator
   ```
3. Virtuelle Umgebung anlegen und aktivieren:
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   > Falls der `py`-Launcher fehlt, alternativ `python -m venv .venv` verwenden.
4. Abhängigkeiten installieren:
   ```powershell
   python -m pip install --upgrade pip
   pip install "sdv[all]" pandas pyarrow streamlit
   ```
5. Optional: Installation einfrieren, um sie zu teilen:
   ```powershell
   pip freeze > requirements.txt
   ```

## Daten vorbereiten
- Legen Sie Ihre CSV-Dateien in den Ordner `data/`. Alle Dateien mit Endung `.csv` werden automatisch erkannt.
- Jede Datei sollte eine Kopfzeile mit Spaltennamen enthalten.

## SDV-Version & Modellunterstützung
- Getestet mit Python 3.13.5 und der aktuellen SDV-Version (siehe `pip show sdv`).
- `CTGANSynthesizer` und `TVAESynthesizer` akzeptieren `random_state`; `GaussianCopulaSynthesizer` nutzt globale Zufallsquellen (Seed wird über `seed_everything` gesetzt).
- Bei abweichenden Versionen bitte die SDV-Release Notes prüfen; Änderungen am API-Verhalten (z. B. entfernte Methoden) sind in den Verbesserungsnotizen dokumentiert.

## Nutzung über die Kommandozeile
Das Skript `sdv_tabular_example.py` trainiert den SDV-CTGAN-Synthesizer (oder Alternativen) auf einer ausgewählten CSV-Datei und erzeugt eine synthetische Variante.

### Beispielaufruf
```powershell
python sdv_tabular_example.py `
  --csv data\patients.csv `
  --rows 1500 `
  --output data\patients_synth.csv `
  --report artifacts\patients_report.json
```

- `--csv`: Pfad zur Quelldatei.
- `--rows`: Anzahl der Zeilen im synthetischen Datensatz (Standard = Originalgröße).
- `--output`: Speicherort für das neue CSV.
- `--report`: Optionaler JSON-Bericht mit Metadaten, Utility-Score und Detailmetriken (SDMetrics QualityReport).
- `--model`: (optional) `ctgan`, `gaussiancopula`, `tvae`.
- `--random-seed`: (optional) Zufallsseed für reproduzierbare Ergebnisse. Modelle mit `random_state`-Parameter (CTGAN, TVAE) erhalten ihn direkt; für `GaussianCopula` wird intern auf globale RNG-Seeds zurückgegriffen.
- `--epochs`, `--batch-size`: (optional) Überschreiben die Standardwerte für Modelle, die diese Parameter unterstützen (z. B. CTGAN/TVAE).
- Vor dem Training erscheint eine grobe Zeitschätzung, während des Trainings aktualisiert die CLI fortlaufend den Fortschrittsstatus (Heuristik basierend auf Datensatzgröße/Modell).

## Nutzung über die Streamlit-Oberfläche
Die Webanwendung bietet eine geführte Bedienung für nicht-technische Anwender:innen.

### Starten – Schritt für Schritt
1. PowerShell öffnen und ins Projektverzeichnis wechseln:
   ```powershell
   cd C:\Cursor_Git\hhz-zdbb-dataincubator
   ```
2. Virtuelle Umgebung aktivieren (falls noch nicht aktiv):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Streamlit-App starten:
   ```powershell
   streamlit run streamlit_app.py
   ```
4. Der Browser öffnet sich automatisch. Falls nicht, den angezeigten lokalen Link (z. B. `http://localhost:8501`) in die Adresszeile kopieren.

### Funktionen im Überblick
1. **Datei wählen** – Dropdown zeigt alle CSV-Dateien aus `data/`.
2. **Deskriptive Statistik** – Vorschau der ersten Zeilen, numerische Kennzahlen, Top-Kategorien.
3. **Synthese konfigurieren** – Auswahl des SDV-Modells, Seed (für reproduzierbare Runs), Anzahl zusätzlicher Zeilen, Ausgabesuffix.
4. **Generieren & Speichern** – Synthetischer Datensatz entsteht und wird als CSV mit Suffix (Standard `_synthetic`) neben dem Original abgelegt. Download-Button inklusive.
5. **Evaluation** – SDMetrics *QualityReport* liefert Utility-Score (0–1) plus Detailmetriken; numerische Kennzahlen werden gegenübergestellt.
6. **Infos** – Tab mit erklärenden Inhalten aus `docs/explainers.md` (Modelle, Seeds, Kennzahlen, Workflow-Tipps).

### Typischer Ablauf in der App
1. CSV-Datei im Sidebar-Dropdown auswählen.
2. Statistiken prüfen (Tabs „Vorschau“, „Numerische Statistik“, „Kategoriale Werte“).
3. Formular ausfüllen (Modell, Seed, zusätzliche Zeilen, Suffix) und auf „Generieren“ klicken.
4. Optional: Im Expander „Trainingseinstellungen“ Epochen/Batch-Größe anpassen (für CTGAN/TVAE).
5. Synthese abwarten: Fortschrittsbalken zeigt dynamisch eine Heuristik der verbleibenden Zeit an. Danach erscheinen Kennzahlen, Vergleichstabellen und Download-Link.

## Ergebnisse interpretieren
- **Utility Score** (0–1): Maß dafür, wie ähnlich sich Original und synthetische Daten verhalten. Werte >0.7 gelten oft als gut, genaue Schwellen hängen vom Einsatzzweck ab. Die Berechnung erfolgt über SDMetrics *QualityReport*.
- **Vergleichstabellen**: Prüfen Sie, ob Verteilungen plausibel bleiben (z. B. Mittelwerte, Standardabweichung).
- **Detailmetriken**: Im Streamlit-Expander „Detailierte Qualitätsmetriken“ sowie im JSON-Report (`quality_report`) stehen Einzelmetriken und Ergebnisse je Eigenschaft.
- **Reproduzierbarkeit**: Sowohl CLI als auch Streamlit setzen den angegebenen Seed. Für Modelle ohne expliziten `random_state` (z. B. GaussianCopula) werden Python/Numpy (und optional Torch) global gesät.
- **Dateiausgabe**: Synthetische CSV befindet sich im selben Verzeichnis; der Dateiname erhält automatisch das gewählte Suffix.

## Fehlerbehebung
- *Keine CSV gefunden*: Sicherstellen, dass Dateien mit Endung `.csv` im Ordner `data/` liegen.
- *ImportError / Modul fehlt*: Prüfen, ob die virtuelle Umgebung aktiv ist und `pip install ...` erfolgreich war.
- *SDV hängt oder dauert sehr lange*: Bei sehr kleinen Datenmengen auf `--model gaussiancopula` wechseln oder die zusätzliche Zeilenanzahl reduzieren.
- *Schlechter Utility Score*: Metadaten prüfen, zusätzliche Vorverarbeitung (z. B. Encoding, Ausreißer) vornehmen oder alternatives Modell testen.

## Weiterführendes
- Ausführlicher Leitfaden: `synthetic_tabular_workflow.md`
- Eigene Automatisierung: CLI-Skript in Batch-Jobs oder CI-CD integrieren.
- Erweiterungen: Mehrere Tabellen (relational), Differential Privacy, Dashboarding der Qualitätsmetriken.
- Modelle & Konzepte: Grundlagen zu Modellen, Seeds und Kennzahlen in `synthetic_tabular_workflow.md` (Abschnitte 2–3) sowie vertiefende Erklärungen in `docs/explainers.md`.
- Implementierungsdetails: `synthesis_runner.py` kapselt Metadaten-Erkennung, Dauerheuristiken und Fortschrittscallbacks, die von Streamlit-UI und CLI geteilt werden.

## Verbesserungsnotizen

### Abgeschlossen
- **T1** Synthesizer-Registry-Lookup wird zwischengespeichert; `build_synthesizer` nutzt nun einen Cache für normalisierte Schlüssel.
- **T2** `seed_everything` fallback sichtbar machen (Logging oder UI-Hinweis), damit nachvollziehbar bleibt, wann globale Seeds greifen.
- **T3** Bei fehlenden QualityReport-Properties ein Signal geben (Log oder UI-Hinweis), statt sie still zu ignorieren.
- **T4** Metadata-Erkennung samt DataFrame-Caching (z. B. via `st.cache_data`) kombinieren, um größere CSVs schneller zu verarbeiten.
- **T5** `SDV`-Version & Modellunterstützung im README dokumentiert; separates Notiz-Dokument entfernt.
- **T6** Altes Planungsdokument `synt.plan.md` entfernt (Repo aufgeräumt).
- **F1** Qualitätsmetriken im UI nach Kategorie trennen (Tabs für Column Shapes vs. Column Pair Trends) und erläutern.
- **F2** Optional konfigurierbare Hyperparameter (Epochs, Batchgröße) mit sinnvollen Defaults anbieten.
- **F3** Zusammenfassende Ergebniskarte mit Seed, Modell und Zeilenanzahl; Kopierfunktion über bereitgestellten Text.
- **F4** Fehlermeldungen liefern nun Handlungsempfehlungen (UI und CLI zeigen kontextsensitive Tipps).
- **F5a** CSVs per Drag-and-Drop in die App ziehen und auswählen (Uploader speichert Dateien in `data/`).
- **F5b** Erkennen, wenn CSVs außerhalb der App geändert werden, inkl. Hinweis und automatischer Reload.
- **F7a** Kontext-Hover: Kennzahlen im UI erhalten Tooltip-Definitionen für unerfahrene Nutzer:innen.
- **F7b** Interpretation: Für die wichtigsten Qualitätskennzahlen erscheinen kurze Bewertungstexte (z. B. Utility-Score).
- **F7c** Wissensseite: Eigene Seite/Sektion mit Erklärungen zu Modellen, Seeds, Kennzahlen und empfohlenen Workflows bereitgestellt (`docs/explainers.md`).
- **F8a** Grobe Dauerabschätzung vor dem Training (Heuristik basierend auf Datensatzgröße und Modell, UI/CLI).
- **F8b** Während des Trainings wird eine dynamische Fortschrittsanzeige mit verbleibender Heuristikzeit dargestellt (UI & CLI).

### Offen
- **F6** Notification System, das bei erfolgreichen oder abgebrochenen Synthesen informiert (optional erweiterbar).
- **O1** (frei zur Definition weiterer Optimierungsideen)

Viel Erfolg beim Erstellen synthetischer Datensätze! Bei Fragen oder Verbesserungswünschen einfach melden.

