# Szczegółowa instrukcja wdrożenia aplikacji GA Analytics AI na Streamlit Cloud

Ta instrukcja przeprowadzi Cię krok po kroku przez proces wdrożenia aplikacji GA Analytics AI na platformie Streamlit Cloud, wymagając minimalnego zaangażowania z Twojej strony.

## Spis treści
1. [Przygotowanie repozytorium GitHub](#1-przygotowanie-repozytorium-github)
2. [Konfiguracja konta Streamlit Cloud](#2-konfiguracja-konta-streamlit-cloud)
3. [Wdrożenie aplikacji](#3-wdrożenie-aplikacji)
4. [Konfiguracja sekretów](#4-konfiguracja-sekretów)
5. [Testowanie aplikacji](#5-testowanie-aplikacji)
6. [Rozwiązywanie problemów](#6-rozwiązywanie-problemów)

## 1. Przygotowanie repozytorium GitHub

### 1.1. Utwórz konto GitHub (jeśli jeszcze go nie masz)
1. Przejdź do [GitHub.com](https://github.com)
2. Kliknij "Sign up" i postępuj zgodnie z instrukcjami, aby utworzyć konto

### 1.2. Utwórz nowe repozytorium
1. Po zalogowaniu kliknij przycisk "+" w prawym górnym rogu, a następnie "New repository"
2. Wprowadź nazwę repozytorium: `ga-analytics-ai`
3. Wybierz opcję "Public" (publiczne)
4. Zaznacz opcję "Add a README file"
5. Kliknij "Create repository"

### 1.3. Prześlij pliki aplikacji do repozytorium
1. W swoim repozytorium kliknij przycisk "Add file" > "Upload files"
2. Przeciągnij lub wybierz wszystkie pliki aplikacji, które otrzymałeś:
   - `ga_integration.py`
   - `llm_integration.py`
   - `analysis_pipeline.py`
   - `streamlit_app.py`
   - `streamlit_cloud_app.py` (zmień nazwę na `app.py` przed przesłaniem)
   - `.streamlit/config.toml` (utwórz folder `.streamlit` i prześlij plik)
3. Dodaj wiadomość commit: "Initial commit with application files"
4. Kliknij "Commit changes"

### 1.4. Utwórz plik requirements.txt
1. W swoim repozytorium kliknij przycisk "Add file" > "Create new file"
2. Nazwij plik `requirements.txt`
3. Wklej następującą zawartość:
```
streamlit==1.31.0
pandas==2.1.0
numpy==1.24.3
plotly==5.18.0
google-analytics-data==0.16.2
google-auth-oauthlib==1.0.0
google-auth==2.23.0
openai==1.6.1
anthropic==0.5.0
requests==2.31.0
python-dotenv==1.0.0
tenacity==8.2.3
```
4. Dodaj wiadomość commit: "Add requirements.txt"
5. Kliknij "Commit changes"

## 2. Konfiguracja konta Streamlit Cloud

### 2.1. Utwórz konto Streamlit Cloud
1. Przejdź do [Streamlit Cloud](https://streamlit.io/cloud)
2. Kliknij "Sign up" lub "Get started"
3. Wybierz opcję logowania przez GitHub (zalecane)
4. Autoryzuj Streamlit do dostępu do Twojego konta GitHub

### 2.2. Połącz z repozytorium GitHub
1. Po zalogowaniu do Streamlit Cloud, kliknij przycisk "New app"
2. W sekcji "Repository" wybierz swoje repozytorium `ga-analytics-ai`
3. W sekcji "Branch" wybierz `main`
4. W sekcji "Main file path" wpisz `app.py`

## 3. Wdrożenie aplikacji

### 3.1. Skonfiguruj ustawienia wdrożenia
1. Ustaw nazwę aplikacji: `ga-analytics-ai` (lub inną wybraną nazwę)
2. Wybierz opcję "Public" (publiczna)
3. Kliknij "Deploy"

Streamlit Cloud automatycznie zainstaluje wszystkie zależności z pliku `requirements.txt` i uruchomi aplikację. Proces ten może potrwać kilka minut.

## 4. Konfiguracja sekretów

Aby aplikacja działała poprawnie, musisz skonfigurować sekrety (np. klucze API) w Streamlit Cloud:

1. Po wdrożeniu aplikacji, przejdź do jej ustawień klikając ikonę "⋮" obok nazwy aplikacji, a następnie "Settings"
2. Przejdź do zakładki "Secrets"
3. Kliknij "Edit secrets"
4. Wklej następujący szablon i dostosuj go do swoich kluczy API:
```yaml
openai:
  api_key: "sk-twój-klucz-api-openai"
```
5. Kliknij "Save"

## 5. Testowanie aplikacji

### 5.1. Otwórz aplikację
1. Po zakończeniu wdrożenia, kliknij przycisk "View app" lub użyj linku wygenerowanego przez Streamlit Cloud
2. Aplikacja powinna otworzyć się w nowej karcie przeglądarki

### 5.2. Pierwsze uruchomienie
1. Zaloguj się do aplikacji podając swój adres email i imię
2. Postępuj zgodnie z instrukcjami w aplikacji, aby skonfigurować połączenie z Google Analytics i dodać klucz API OpenAI

## 6. Rozwiązywanie problemów

### Problem: Aplikacja nie uruchamia się
**Rozwiązanie:**
- Sprawdź logi aplikacji w Streamlit Cloud (ikona "⋮" > "Manage app" > "Logs")
- Upewnij się, że wszystkie pliki zostały prawidłowo przesłane do repozytorium
- Sprawdź, czy plik `app.py` jest głównym plikiem aplikacji

### Problem: Błędy importu modułów
**Rozwiązanie:**
- Upewnij się, że wszystkie pliki aplikacji znajdują się w głównym katalogu repozytorium
- Sprawdź, czy plik `requirements.txt` zawiera wszystkie wymagane zależności

### Problem: Błędy autoryzacji Google Analytics
**Rozwiązanie:**
- Postępuj zgodnie z instrukcją konfiguracji Google Analytics API
- Upewnij się, że prawidłowo skonfigurowałeś OAuth w Google Cloud Console

---

## Dodatkowe wskazówki

- **Automatyczne aktualizacje**: Każda zmiana w repozytorium GitHub automatycznie uruchomi ponowne wdrożenie aplikacji
- **Monitorowanie**: Możesz monitorować wydajność i użycie aplikacji w panelu Streamlit Cloud
- **Udostępnianie**: Możesz udostępnić link do aplikacji innym osobom, aby mogły z niej korzystać

Jeśli napotkasz jakiekolwiek problemy podczas wdrażania, skontaktuj się ze mną, a pomogę Ci je rozwiązać.
