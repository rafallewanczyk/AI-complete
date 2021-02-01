# Generowanie automatycznych podpoweidze w środowiskach programowania 

## Wtyczka do środowiska SublimeText 3 generująca podpowiedzi przy pomocy metod uczenia maszynowego. 

## Pliki 
- ```backend``` - moduł łączący wtyczkę z serwerem 
- ```chceckpoints``` - wszystkie wytrenowane sieci neuronowe, przebieg treningu oraz uzyskane wyniki w podczas walidacji
- ```data``` - wykorzystany zbiór danych 
- ```dokumentacja``` - praca naukowa
- ```drivers``` - notatniki python wykorzystywane w celu treningu sieci
- ```server``` - serwer serwujący podpowiedzi 
- ```src``` - implementacja modeli oraz narzędzi pomocniczych
- ```AI-complete.py``` - implementacja wtyczki 

## Uruchomianie 

### Trening 

- przy pomocy notatnika ```generate_data.ipynb`` wygenerować podzibiór treningowy oraz walidacyjny
- przy pomocy notatnike ```training_driver.ipynb``` wykonać trening modelu którego konfiguracja znajduje się w pliku ```src/model.py```. Trening zaczyna się od wygenerowania słownika o zadanej długości. Następnie należy dobrać odpowiednie hiperparametry, oraz rozpocząć trening

### Walidacja 
- przy pomocy notatnika ```prediction_driver.ipynb``` wczytać wytrenowany model oraz rozpocząć walidację. Istnieje możliwość nadania argumentowi ```html``` wartość ```true``` w celu generowania plików wyjściowych z zaznaczonymi poprawnymi oraz niepoprawnymi sugestiamii. 

### Instalacja oraz serwowanie predykcji 
Umieścić zawartość folderu w w folderze ```plugins``` w lokalizacji SublimeText 3. 
Uruchomić serwer znajdujący się w folderze ```server```. 
Serwer domyślnie działa pod adresem ```localhost:8000```
