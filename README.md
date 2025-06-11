Temperature Predictor

Projekt służy do przewidywania temperatury na podstawie danych pogodowych, czasu oraz lokalizacji geograficznej.
Wstępnie wytrenowany model LightGBM operuje na ponad 1 250 000 rekordach historycznych, a przewidywania są dostępne przez prostą aplikację webową z interaktywną mapą.

Główne funkcjonalności
	•	Pobieranie i przetwarzanie danych pogodowych z bazy SQLite
	•	Featuryzacja danych (czasowe oraz opcjonalnie przestrzenne cechy)
	•	Trening modeli regresyjnych LightGBM (warianty: spatial=True lub spatial=False)
	•	Zapis wytrenowanego modelu na dysk
	•	API udostępniające predykcje temperatury dla wskazanej lokalizacji i daty
	•	Prosta aplikacja webowa z interaktywną mapą do wyboru lokalizacji

Krótka charakterystyka modelu
	•	Algorytm: LightGBM
	•	Liczba rekordów treningowych: ~1 250 000
	•	Błąd średniokwadratowy (RMSE): około 4.0°C (zależny od wariantu modelu i danych)

Struktura projektu
	•	weather_forecast/ — moduły do przetwarzania danych, featuryzacji, treningu i inferencji
	•	models/ — katalog na zapisane modele (lgbm_spatial.pkl, lgbm_nospatial.pkl)
	•	templates/ — pliki HTML do webappki (frontend mapy)
	•	predict.py — szybki interfejs CLI do predykcji
	•	app.py — backend API na Flasku

Szybki start
	1.	Klonuj repo:

git clone https://github.com/lipinsks/temperature_predictor.git
cd temperature_predictor
	2.	Utwórz i aktywuj środowisko:

python3 -m venv venv
source venv/bin/activate  (Windows: venv\Scripts\activate)
	3.	Zainstaluj zależności:

pip install -r requirements.txt
	4.	Upewnij się, że masz przygotowaną bazę danych data/weather.db i wytrenowany model w katalogu models/.
	5.	Uruchom serwer API:

python app.py

Serwis będzie dostępny pod http://localhost:5001/ z interaktywną mapą do przewidywań.

Przykładowe użycie z terminala

python predict.py 50.0 19.9 2025-06-11T15:00:00

Plany rozwoju
	•	Poprawa dokładności modelu (więcej danych / bardziej zaawansowane cechy)
	•	Dodanie walidacji geograficznej i zakresów dat
	•	Optymalizacja wydajności API
	•	Deployment online (Docker / chmura)

Autor: Szymon Lipiński
