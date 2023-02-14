# importuje niezbędne bibloteki
# open cv
import cv2 

# program ocr
import keras_ocr

# biblioteka do łączności z serwerem mqtt
import paho.mqtt.publish as publish

# biblioteka do obsługi xml
import xml.dom.minidom

# zegar
import time
from datetime import datetime, timedelta

# ustawienie zegara
current_time = datetime.now()
zbyt_szybki_wjazd = current_time

# otwarcie plików lista.xml, logi.xml
zmienna = xml.dom.minidom.parse('c:/Users/mikol/OneDrive/studia/Programowanie/Python/Projekt/lista.xml')
zmienna2 = xml.dom.minidom.parse('c:/Users/mikol/OneDrive/studia/Programowanie/Python/Projekt/logi.xml')
# załadowanie zawartości plików xml do programu
lista = zmienna.documentElement
logi = zmienna2.documentElement
plates = lista.getElementsByTagName('plate')
print("Baza tablic zaladowana")

# wyświetla listę pojazdów uprawnionych
for plate in plates:
   print ("*****Tablice rejestracyjne****")
   print (f"plate:  { plate.getAttribute ('number')}") 
   imie = plate.getElementsByTagName('imie')[0].childNodes[0].nodeValue
   nazwisko = plate.getElementsByTagName('nazwisko')[0].childNodes[0].nodeValue
   marka = plate.getElementsByTagName('marka')[0].childNodes[0].nodeValue
   model = plate.getElementsByTagName('model')[0].childNodes[0].nodeValue
   print (f"Imie: {imie}")
   print (f"nazwisko: {nazwisko}")
   print (f"Marka: {marka}")
   print (f"Model: {model}") 

# uruchamia moduł OCR
pipeline = keras_ocr.pipeline.Pipeline()

# OpenCV przechwytuje obraz z kamery (testowałem stream z kamery przemysłowej i kamerki internetowej)

#cap = cv2.VideoCapture(r'rtsp://admin:haslo@192.168.100.252:554/Streaming/Channels/102')
cap = cv2.VideoCapture(0)

while True:
  # Czytanie klatek z kamery
  ret, frame = cap.read()

  # Dokonujemy konwersji klatek do odcieni szarości
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Nakładamy filtr Gaussa oby odszumić obraz
  # W przypadku funkcji cv2.GaussianBlur() parametr frame to obraz, który ma zostać rozmyty. 
  # Drugi parametr (5, 5) określa wielkość maski filtra, która jest kwadratem o boku równym 5 pikseli. 
  # Trzeci parametr (0) to odchylenie standardowe dla rozkładu Gaussa, które określa, jak bardzo piksele są ważone. 
  # Im większe odchylenie, tym bardziej rozmyty jest otrzymany obraz. 
  # Wartość 0 oznacza, że odchylenie jest obliczane automatycznie na podstawie wielkości maski.
  frame = cv2.GaussianBlur(frame, (5, 5), 0)
  
  # W przypadku funkcji cv2.threshold() parametr frame to obraz, który ma zostać poddany binarzacji. 
  # Drugi parametr (0) to prog, który jest porównywany z wartościami pikseli. 
  # Trzeci parametr (255) to wartość, do której są konwertowane piksele o wartościach większych od progu. 
  # Czwarty parametr (cv2.THRESH_BINARY + cv2.THRESH_OTSU) to opcja, która określa sposób obliczania progu. 
  # THRESH_BINARY oznacza, że prog jest stały i podawany jako parametr (w tym przypadku 0). 
  # THRESH_OTSU oznacza, że prog jest obliczany automatycznie za pomocą algorytmu Otsu,
  # który dąży do minimalizacji sumy kwadratów odchyleń pomiędzy grupami pikseli o różnych wartościach po przekroczeniu progu.
  frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  
  # Wykrywanie tablicy rejestracyjnej 
  
  # license_plate_cascade jest zmienną, która jest przypisywana instancji klasy CascadeClassifier z biblioteki cv2. 
  # Instancja ta jest tworzona za pomocą metody CascadeClassifier z jednym argumentem,
  # ścieżką do pliku XML zawierającego wytrenowany model kaskady klasyfikatora. 
  # Plik XML jest używany do zainicjowania obiektu klasy CascadeClassifier i jest wymagany do działania tego obiektu.
  license_plate_cascade = cv2.CascadeClassifier('c:/Users/mikol/OneDrive/studia/Programowanie/Python/Projekt/plate.xml')
  
  # Następnie, metoda detectMultiScale jest wywoływana na obiekcie license_plate_cascade z dwoma argumentami: frame i scaleFactor, oraz trzecim argumentem minNeighbors.
  # Metoda ta przeszukuje obraz frame w poszukiwaniu obiektów zgodnych z modelem zawartym w pliku XML.
  # Argument scaleFactor określa, jak bardzo skalowany jest obraz podczas przeszukiwania. 
  # Wartość 1.4 oznacza, że obraz jest skalowany o 40% w każdym kroku przeszukiwania. 
  # Im większa wartość scaleFactor, tym szybciej przeszukiwanie jest wykonywane, ale też mniejsza jest dokładność wykrywania obiektów.
  # Argument minNeighbors określa, ile sąsiadujących obiektów musi być znalezionych, aby obiekt został uznany za poprawnie rozpoznany. 
  # Wartość 5 oznacza, że dla każdego potencjalnego obiektu musi być znalezionych co najmniej 5 sąsiadujących obiektów, aby został on uznany za poprawnie rozpoznany. 
  # Im większa wartość minNeighbors, tym większa jest dokładność wykrywania, ale też mniejsza jest szybkość przeszukiwania.
  #Metoda detectMultiScale zwraca listę prostokątów zawierających wykryte obiekty. Ta lista jest przypisywana do zmiennej license_plates. 
  # Prostokąty te są potem używane do wycięcia i dalszego przetwarzania obiektów.
  license_plates = license_plate_cascade.detectMultiScale(frame, 1.4, 5)
  
  # potwierdzenie znalezienia tablicy
  if len(license_plates)>0:
    print('znalazłem tablice ', len(license_plates))
      
  # Rozpoznawanie tekstu w wyciętym obrazie frame.jpg, który jest zapisywany jako pojedyńcze bloki
  for (x, y, w, h) in license_plates:
    license_plate = frame[y:y+h, x:x+w]
    cv2.imwrite("frame.jpg", license_plate)
    image = keras_ocr.tools.read('frame.jpg')
    prediction = pipeline.recognize([image])

    # Obróbka poszczególnych bloków usunięcie zbędnego tekstu, ustalenie odpowiedniej kolejności i połaczenie w numer rejestracyjny
    pred_img = prediction[0]
    x=[]
    for text, box in pred_img: 
        if text != 'pl':
         if str.isalpha(text):    
          x.insert(0,text)
          print('litery ',text)
         else:
          x.append(text)
          print('cyfry ',text)
    tablica_rej =''.join(x)
    print (tablica_rej)
    
    # wysłanie rozpoznanego numeru rejestracyjnego do serwera MQTT
    ltime=time.localtime()
    t = time.strftime("%Y-%m-%dT%H:%M:%S", ltime)
    msgs = [{'topic':"tele/python/TABLICA", 'payload':tablica_rej}]
    publish.multiple(msgs, hostname="192.168.100.8")
    
    # wysłanie rozpoznanego numeru rejestracyjnego do pliku z logami
    new_plate = zmienna2.createElement("plate")
    new_plate.setAttribute("numer",tablica_rej)
    czas = zmienna2.createElement("czas")
    czas.appendChild(zmienna.createTextNode(t))
    new_plate.appendChild(czas)
    logi.appendChild(new_plate)
    with open('c:/Users/mikol/OneDrive/studia/Programowanie/Python/Projekt/logi.xml','w') as f:
        f.write(zmienna2.toprettyxml())
        
    # weryfikacja czy rozpoznay numer rejestracyjny znajduje się na białej liście,
    # jeśli tak  następuje zapisanie czasu otwarcia aby zabezpieczyć się przed zbyt częstym wysłaniem sygnału i 
    # otwarcie bramy.  
    current_time = datetime.now()
    for plate in plates:
     if (plate.getAttribute ('number') == tablica_rej) and (current_time >= zbyt_szybki_wjazd):
      print('tablica rejestracyjna o nr', (tablica_rej) , ' znajduje się na białej liście')
      zbyt_szybki_wjazd = current_time + timedelta(minutes=1)     
      msgs = [{'topic':"tele/python/SWITCH", 'payload':"OFF"},
          ("tele/python/SWITCH", "ON", 0, False)]
      publish.multiple(msgs, hostname="192.168.100.8")
      msgs = [{'topic':"tele/python/SWITCH", 'payload':"ON"},
          ("tele/python/SWITCH", "OFF", 0, False)]
      publish.multiple(msgs, hostname="192.168.100.8")
    
  # Pokazuje przetworzony obraz na żywo
  cv2.imshow('frame', frame)
 
  # zatrzymanie kodu poprzes naciśnięcie "q"
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()

cv2.destroyAllWindows()
