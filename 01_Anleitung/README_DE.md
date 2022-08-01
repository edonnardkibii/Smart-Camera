# Smart-Kamera-Anleitung
Diese Anleitung wird Ihnen helfen, die Smart-Kamera-Software erfolgreich zu nutzen. Sollten Sie Fragen haben, wenden Sie sich bitte an edonnardkibii@gmail.com

## virtualenv (Optional)
In einer virtuellen Umgebung lässt sich das Projekt effizienter verwalten, insbesondere die Bibliotheken. <br />
Um eine virtuelle Umgebung betreiben zu können:
```
pip install virtualenv
```
Gehen Sie in das Projektverzeichnis und geben Sie ein:
```
py -m venv env
```
Aktivieren Sie die virtuelle Umgebung
```
.\env\Scripts\activate
```

## Bibliotheksinstallationen
Verwenden Sie den Paketmanager [pip](https://pip.pypa.io/en/stable/), um die folgenden erforderlichen Bibliotheken zu installieren.

Sender-Gerät:
```
pip install -r requirements_sender.txt
```

Empfänger-Gerät:
```
pip install -r requirements_receiver.txt
```
Die Text-Daten finden Sie in **03_Requirements**

### Scikit-learn und Tensorflow
So installieren Sie scikit-learn:
```
pip install -U scikit-learn
```
Um Tensorflow zu installieren:
```
pip install tensorflow
```
**Wichtig:** Es kann sein, dass Tensorflow beim ersten Versuch nicht richtig installiert wird. Falls es Probleme gibt, wird eine Lösung im abschließenden Projektbericht vorgeschlagen. (Seite 4-5)

## Training des ML-Modells
Ort: 02_Software/03_ML_Training <br />
**Wichtig:** Wenn Sie sich in einer virtuellen Umgebung befinden, müssen Sie möglicherweise matplotlib installieren. Trainieren Sie das Modell auch auf demselben Gerät, auf dem das Receiver-Programm ausgeführt wird. <br/>
```
pip install matplotlib
```
Um den Trainingsprozess zu starten, geben Sie in das Befehlsfenster:
```
python model_training.py --dataset dataset
```
Nach dem Training des Modells kopieren Sie das Modell **mask_detector.model** in den Receiver-Ordner

## Generierung des Geheimschlüssels
Führen Sie das Programm **encryption_key.py** aus
Es wird eine Config-Datei **secret_key.ini** erstellt. Stellen Sie sicher, dass sich eine Kopie des geheimen Schlüssels sowohl im Sender-Ordner als auch im Receiver-Ordner befindet.\
Dieser Schlüssel sollte ebenfalls geheim bleiben und nicht an andere Personen weitergegeben werden.
```
python encryption_key.py
```

## Ausführen des Sender- und Empfängerprogramms
Sender-Ordner: <br /> 
Ort 02_Software/01_Sender
```
python mqtt_video_sender.py
```
Receiver-Ordner: <br/>
Ort: 02_Software/02_Receiver
```
python mqtt_mask_detection.py
```
