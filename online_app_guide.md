# Instrukcja konfiguracji aplikacji GA Analytics AI Online

Ta instrukcja przeprowadzi Cię przez proces konfiguracji i korzystania z aplikacji GA Analytics AI wdrożonej online na platformie Streamlit Cloud.

## Spis treści
1. [Dostęp do aplikacji](#dostęp-do-aplikacji)
2. [Pierwsze logowanie](#pierwsze-logowanie)
3. [Konfiguracja Google Analytics](#konfiguracja-google-analytics)
4. [Konfiguracja OpenAI API](#konfiguracja-openai-api)
5. [Tworzenie pierwszej analizy](#tworzenie-pierwszej-analizy)
6. [Zarządzanie raportami](#zarządzanie-raportami)
7. [Biblioteka promptów](#biblioteka-promptów)
8. [Rozwiązywanie problemów](#rozwiązywanie-problemów)

## Dostęp do aplikacji

Aplikacja GA Analytics AI jest dostępna pod następującym adresem:
```
https://ga-analytics-ai.streamlit.app/
```

Możesz otworzyć ten link w dowolnej nowoczesnej przeglądarce internetowej (Chrome, Firefox, Safari, Edge).

## Pierwsze logowanie

1. Otwórz aplikację w przeglądarce
2. Na stronie logowania wprowadź:
   - Adres e-mail: Twój adres e-mail
   - Imię: Twoje imię
3. Kliknij przycisk "Login"
4. Zostaniesz przekierowany do panelu głównego aplikacji

## Konfiguracja Google Analytics

Aby połączyć aplikację z Twoimi danymi Google Analytics, wykonaj następujące kroki:

1. W aplikacji przejdź do zakładki "Settings" (kliknij przycisk w menu bocznym)
2. W sekcji "Google Analytics Accounts" kliknij "Add New Account"
3. Wprowadź:
   - Account Name: Nazwa dla Twojego konta (np. "Moja strona")
   - Property ID: ID właściwości GA4 (uzyskane zgodnie z [instrukcją konfiguracji Google Analytics API](ga_api_configuration_guide.md))
4. Kliknij "Upload Credentials File" i wybierz plik JSON z poświadczeniami OAuth pobrany z Google Cloud Console
5. Kliknij "Add Account"
6. Zostaniesz przekierowany do strony Google, gdzie musisz zalogować się i udzielić aplikacji dostępu do danych Google Analytics
7. Po udzieleniu zgody zostaniesz automatycznie przekierowany z powrotem do aplikacji

## Konfiguracja OpenAI API

Aby skonfigurować dostęp do OpenAI API:

1. W aplikacji przejdź do zakładki "Settings"
2. W sekcji "API Keys" wybierz "openai" z listy rozwijanej
3. Wprowadź swój klucz API OpenAI w polu "API Key"
4. Kliknij "Save API Key"

Twój klucz API jest przechowywany bezpiecznie i używany tylko do generowania analiz w aplikacji.

## Tworzenie pierwszej analizy

Po skonfigurowaniu Google Analytics i OpenAI API możesz utworzyć swoją pierwszą analizę:

1. Przejdź do zakładki "New Analysis" (kliknij przycisk w menu bocznym)
2. Wybierz skonfigurowane konto Google Analytics z listy rozwijanej
3. Wybierz zakres dat dla analizy:
   - "Last N Days": Ostatnie X dni
   - "Custom Range": Niestandardowy zakres dat
   - "Comparison": Porównanie dwóch okresów
4. Wybierz typ analizy:
   - "General Analysis": Ogólna analiza wszystkich kluczowych metryk
   - "Traffic Analysis": Analiza źródeł ruchu
   - "Conversion Analysis": Analiza konwersji
   - "User Behavior Analysis": Analiza zachowań użytkowników
   - "Anomaly Detection": Wykrywanie anomalii w danych
5. Dostosuj metryki i wymiary (lub pozostaw domyślne)
6. Wybierz szablon promptu z biblioteki
7. Wprowadź tytuł i opis raportu
8. Kliknij "Run Analysis"

Analiza może potrwać od kilkudziesięciu sekund do kilku minut, w zależności od ilości danych i złożoności analizy.

## Zarządzanie raportami

Wszystkie wygenerowane raporty są zapisywane w historii:

1. Przejdź do zakładki "Report History"
2. Przeglądaj, filtruj i wyszukuj swoje raporty
3. Kliknij "View" przy dowolnym raporcie, aby zobaczyć szczegóły
4. W widoku szczegółowym możesz:
   - Pobrać raport w formacie Markdown lub HTML
   - Dodać raport do ulubionych
   - Uruchomić podobną analizę z tymi samymi parametrami

## Biblioteka promptów

Aplikacja zawiera bibliotekę predefiniowanych promptów, które możesz dostosować do swoich potrzeb:

1. Przejdź do zakładki "Prompt Library"
2. Przeglądaj dostępne szablony promptów
3. Aby utworzyć nowy prompt:
   - Przejdź do zakładki "Create New Prompt"
   - Wprowadź tytuł i opis
   - Wybierz kategorię
   - Wprowadź szablon promptu (możesz użyć zmiennych w formacie {nazwa_zmiennej})
   - Dodaj parametry i ich domyślne wartości
   - Kliknij "Create Prompt"
4. Twoje niestandardowe prompty będą dostępne podczas tworzenia nowych analiz

## Rozwiązywanie problemów

### Problem: Błąd autoryzacji Google Analytics

**Rozwiązanie:**
- Upewnij się, że prawidłowo skonfigurowałeś Google Cloud Project zgodnie z instrukcją
- Sprawdź, czy ID właściwości GA4 jest poprawne
- Spróbuj ponownie dodać konto Google Analytics w ustawieniach

### Problem: Błąd API OpenAI

**Rozwiązanie:**
- Upewnij się, że Twój klucz API jest aktualny i ma wystarczające środki
- Sprawdź, czy nie przekroczyłeś limitów API
- Wprowadź klucz API ponownie w ustawieniach

### Problem: Analiza trwa zbyt długo

**Rozwiązanie:**
- Zmniejsz zakres dat analizy
- Ogranicz liczbę wybranych metryk i wymiarów
- Wybierz mniej złożony typ analizy

### Problem: Brak danych w raporcie

**Rozwiązanie:**
- Upewnij się, że w wybranym zakresie dat istnieją dane w Google Analytics
- Sprawdź, czy wybrane metryki i wymiary są dostępne w Twojej właściwości GA4
- Upewnij się, że konto Google używane do autoryzacji ma dostęp do danych

---

W przypadku innych problemów lub pytań, skontaktuj się z nami przez funkcję wiadomości w aplikacji lub bezpośrednio przez e-mail.

Życzymy owocnych analiz!
