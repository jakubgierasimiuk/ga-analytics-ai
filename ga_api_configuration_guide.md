# Instrukcja konfiguracji Google Analytics API

Ta instrukcja przeprowadzi Cię przez proces konfiguracji Google Analytics API, który jest niezbędny do połączenia aplikacji GA Analytics AI z Twoimi danymi analitycznymi.

## Spis treści
1. [Wymagania wstępne](#wymagania-wstępne)
2. [Konfiguracja projektu Google Cloud](#konfiguracja-projektu-google-cloud)
3. [Włączenie Google Analytics Data API](#włączenie-google-analytics-data-api)
4. [Konfiguracja OAuth](#konfiguracja-oauth)
5. [Tworzenie poświadczeń OAuth](#tworzenie-poświadczeń-oauth)
6. [Uzyskanie ID właściwości GA4](#uzyskanie-id-właściwości-ga4)
7. [Rozwiązywanie problemów](#rozwiązywanie-problemów)

## Wymagania wstępne

Przed rozpoczęciem upewnij się, że posiadasz:
- Konto Google Analytics 4 (GA4) z danymi
- Konto Google Cloud Platform (możesz użyć tego samego konta Google)
- Uprawnienia administratora do swojej właściwości GA4

## Konfiguracja projektu Google Cloud

1. Przejdź do [Google Cloud Console](https://console.cloud.google.com/)
2. Zaloguj się na swoje konto Google
3. Utwórz nowy projekt:
   - Kliknij na listę rozwijaną projektów w górnym pasku
   - Kliknij "Nowy projekt"
   - Wprowadź nazwę projektu (np. "GA Analytics AI")
   - Kliknij "Utwórz"
4. Poczekaj, aż projekt zostanie utworzony, a następnie wybierz go z listy projektów

## Włączenie Google Analytics Data API

1. W konsoli Google Cloud przejdź do sekcji "APIs & Services" > "Library" (w menu bocznym)
2. W polu wyszukiwania wpisz "Google Analytics Data API"
3. Kliknij na wynik "Google Analytics Data API"
4. Kliknij przycisk "Włącz" (Enable)
5. Poczekaj, aż API zostanie włączone

## Konfiguracja OAuth

1. W konsoli Google Cloud przejdź do sekcji "APIs & Services" > "OAuth consent screen" (w menu bocznym)
2. Wybierz typ użytkownika:
   - Jeśli używasz zwykłego konta Google, wybierz "External"
   - Jeśli używasz konta Google Workspace, możesz wybrać "Internal"
3. Kliknij "Utwórz"
4. Wypełnij formularz:
   - Nazwa aplikacji: "GA Analytics AI"
   - Adres e-mail pomocy technicznej: Twój adres e-mail
   - Logo aplikacji: opcjonalne
   - Domena aplikacji: możesz pominąć
   - Dane kontaktowe dewelopera: Twój adres e-mail
5. Kliknij "Zapisz i kontynuuj"
6. W sekcji "Scopes" kliknij "Add or remove scopes"
7. Wyszukaj i zaznacz następujące zakresy:
   - `https://www.googleapis.com/auth/analytics.readonly`
8. Kliknij "Update" i "Save and Continue"
9. W sekcji "Test users" kliknij "Add Users"
10. Dodaj swój adres e-mail Google
11. Kliknij "Save and Continue"
12. Przejrzyj podsumowanie i kliknij "Back to Dashboard"

## Tworzenie poświadczeń OAuth

1. W konsoli Google Cloud przejdź do sekcji "APIs & Services" > "Credentials" (w menu bocznym)
2. Kliknij "Create Credentials" i wybierz "OAuth client ID"
3. Wybierz typ aplikacji: "Web application"
4. Wprowadź nazwę: "GA Analytics AI Web Client"
5. W sekcji "Authorized JavaScript origins" kliknij "ADD URI" i dodaj:
   - `https://localhost:8501` (dla testów lokalnych)
   - `https://share.streamlit.io` (dla aplikacji na Streamlit Cloud)
6. W sekcji "Authorized redirect URIs" kliknij "ADD URI" i dodaj:
   - `https://localhost:8501/` (dla testów lokalnych)
   - `https://share.streamlit.io/` (dla aplikacji na Streamlit Cloud)
7. Kliknij "Create"
8. Pojawi się okno z ID klienta i tajnym kluczem klienta. Kliknij "Download JSON"
9. Zapisz plik JSON - będzie potrzebny do konfiguracji aplikacji

## Uzyskanie ID właściwości GA4

1. Przejdź do [Google Analytics](https://analytics.google.com/)
2. Zaloguj się na swoje konto
3. Wybierz właściwość GA4, którą chcesz połączyć z aplikacją
4. Kliknij na "Admin" (ikona koła zębatego) w lewym dolnym rogu
5. W kolumnie "Property", znajdź i zapisz "Property ID" - to jest numer, który będzie potrzebny do konfiguracji aplikacji
   - ID właściwości to zwykle liczba w formacie "123456789"

## Rozwiązywanie problemów

### Problem: Błąd autoryzacji OAuth

**Rozwiązanie:**
- Upewnij się, że dodałeś prawidłowe URI przekierowania
- Sprawdź, czy wybrałeś odpowiednie zakresy (scopes)
- Upewnij się, że dodałeś swój adres e-mail jako użytkownika testowego

### Problem: Błąd "API not enabled"

**Rozwiązanie:**
- Upewnij się, że włączyłeś "Google Analytics Data API"
- Poczekaj kilka minut po włączeniu API - zmiany mogą nie być natychmiastowe

### Problem: Brak dostępu do danych

**Rozwiązanie:**
- Upewnij się, że masz uprawnienia do przeglądania danych w Google Analytics
- Sprawdź, czy używasz prawidłowego ID właściwości
- Upewnij się, że konto Google używane do autoryzacji ma dostęp do właściwości GA4

---

Po zakończeniu tych kroków będziesz mieć wszystkie niezbędne informacje do konfiguracji aplikacji GA Analytics AI:
1. Plik poświadczeń OAuth JSON
2. ID właściwości GA4

Te informacje będą potrzebne podczas pierwszego uruchomienia aplikacji online.
