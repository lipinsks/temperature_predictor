
## Etap 0 – Przygotowanie środowiska i repozytorium

1. **Utwórz repozytorium** (GitHub/GitLab) z szablonem projektu Python:

   * Użyj Cookiecutter-a albo `poetry new`.
   * Struktura:

     ```
     weather_forecast/
     ├── src/weather_forecast/      # kod źródłowy
     │   ├── __init__.py
     │   ├── data_ingest.py
     │   ├── feature_engineering.py
     │   ├── modeling.py
     │   ├── serving.py
     │   └── utils.py
     ├── tests/                      # testy jednostkowe
     ├── pyproject.toml / setup.py   # zależności i metadata
     ├── README.md
     ├── .github/workflows/          # CI
     └── docs/                       # dokumentacja
     ```
2. **Konfiguracja dependency management**:

   * `poetry init` (lub `pipenv`) – zadeklaruj podstawowe pakiety: `pandas`, `numpy`, `scikit-learn`, `torch`, `fastapi`, `pytest`, itp.
3. **Konfiguracja CI**:

   * GitHub Actions: lint (flake8/mypy), testy (pytest), budowanie pakietu.

---

## Etap 1 – Moduł „data\_ingest” (zależność, nie serce)

1. **Interfejs źródeł danych**:

   * Zaimplementuj klasę bazową `BaseDataSource` z metodami `load()` → DataFrame.
   * Podklasy np. `PostgresSource`, `CSVSource`, `S3Source`.
2. **Testy integracyjne**:

   * Mockowane połączenie do bazy (pytest-mock).
3. **Konfiguracja**:

   * Użyj Pydantic Settings lub Hydra do zarządzania connection stringami.

---

## Etap 2 – Moduł „feature\_engineering”

1. **Funkcje czyszczenia**:

   * Usuwanie duplikatów, obsługa braków (interpolacja).
2. **Wyliczanie cech**:

   * Czasowe (rok, miesiąc, dzień, pora dnia, odległość od równika).
   * Przestrzenne (grupowanie w siatkę, średnie lokalne z okolicznych punktów).
3. **Pipeline**:

   * Zaimplementuj klasę `FeaturePipeline`, która w metodzie `transform(df)` zwraca nowe cechy.
4. **Testy jednostkowe**.

---

## Etap 3 – Moduł „modeling”

1. **Abstrakcja modelu**:

   * Bazowa klasa `BaseForecastModel` z `fit()`, `predict()`, `save()`, `load()`.
2. **Baseline’y**:

   * SARIMA (statsmodels), Prophet.
3. **ML & DL**:

   * XGBoost / LightGBM z scikit-learn API.
   * LSTM/TCN w PyTorch, opakowane w `ForecastDataset` + DataLoader.
   * (opcjonalnie) Transformer-based (Temporal Fusion Transformer).
4. **Śledzenie eksperymentów**:

   * MLflow (lub DVC + MLflow), logowanie parametrów, wyników, artefaktów.
5. **Hyperparameter tuning**:

   * Optuna / Ray Tune.

---

## Etap 4 – Pakiet jako moduł Pythonowy

1. **Packaging**:

   * Uzupełnij `pyproject.toml` (entry points).
2. **CLI**:

   * Dodaj `typer`-owy interface: `wf ingest`, `wf train`, `wf predict`.
3. **Dokumentacja**:

   * Sphinx / MkDocs – opis instalacji, przykłady użycia.

---

## Etap 5 – Serwowanie modelu („serving”)

1. **API REST**:

   * FastAPI z endpointem `/forecast?lat=&lon=&horizon=`.
2. **Containerization**:

   * Dockerfile: bazowy obraz Python, instalacja pakietu, uruchomienie Uvicorn.
3. **Kubernetes (opcjonalnie)**:

   * Manifesty Deployment + Service; albo Helm chart.

---

## Etap 6 – CI/CD & Infra-as-Code

1. **CI**:

   * Lint → Testy → Build → Publish pakiet (PyPI lub prywatny registry).
2. **CD**:

   * Automatyczne wdrożenie na dev/staging po merge do main.
3. **Infra-as-Code**:

   * Terraform: zasoby DB (RDS/Postgres), klaster Kubernetes, tajne zmienne (AWS Secrets Manager).

---

## Etap 7 – Monitoring i utrzymanie

1. **Logowanie i metryki**:

   * Python logging + Prometheus client w aplikacji.
2. **Dashboard**:

   * Grafana: wykorzystaj dane Prometheusa.
3. **Drift detection**:

   * Evidently/AIF360: monitoruj rozkład cech i błędy prognoz.
4. **Alerty**:

   * Alertmanager → e-mail/Slack w razie spadku jakości.

---

## Dodatkowe usprawnienia

* **Plugin-owa architektura**: pozwoli dodawać nowe modele czy źródła danych bez modyfikacji rdzenia.
* **Feature Store**: rozważ Feast, by centralnie przechowywać i podawać cechy.
* **Testy E2E**: wykorzystaj `pytest-docker` do odpalania bazy w kontenerze i testowania full-stack.

---

### Co robić teraz?

1. Skonfiguruj repozytorium i środowisko (Etap 0).
2. Zaimplementuj i przetestuj `data_ingest` oraz `feature_engineering` (Etapy 1–2).
3. Przejdź do eksperymentów modelowych i śledzenia wyników (Etap 3).
4. Zbuduj CLI i API, spakuj cały kod w moduł (Etapy 4–5).
5. Podłącz CI/CD i IaC (Etap 6), a na koniec monitoring (Etap 7).




TODO 
wystawic api z modelu i podpiac je do uzywanego wczesniej javascript mapy pogodowej interaktywnej
dopisac jakis fragment html zeby wybierac date i czas i na klikniecie na obszar mamy prediction

