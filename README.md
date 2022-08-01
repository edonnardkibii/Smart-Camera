# Smart-Camera guide
This guide will help you use the smart camera software successfully. If you have any questions, please contact edonnardkibii@gmail.com <br />
A German version of the README can be found in 01_Anleitung

## virtualenv (Optional)
In a virtual environment, the project can be managed more efficiently, especially the libraries. <br />
To operate a virtual environment:
```
pip install virtualenv
```
Go to the project directory and type:
```
py -m venv env
```
Activate the virtual environment
```
.\env\Scripts\activate
```

## Library installations
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following required libraries.

Sender Device:
```
pip install -r requirements_sender.txt
```

Receiver Device:
```
pip install -r requirements_receiver.txt
```
The text data can be found in **03_Requirements**

### Scikit learn and Tensorflow
To install scikit-learn:
```
pip install -U scikit-learn
```
To install Tensorflow:
```
pip install tensorflow
```
**Important:** Tensorflow may not install properly on the first try. If there are problems, a solution will be proposed in the final project report. (Page 4-5)

## Training of the ML model
Location: 02_Software/03_ML_Training <br />
**Important:** If you are in a virtual environment, you may need to install matplotlib. Also, train the model on the same device running the Receiver program. <br/>
```
pip install matplotlib
```
To start the training process, type in the command window:
```
python model_training.py --dataset dataset
```
After training the model, copy the **mask_detector.model** model to the receiver folder

## Secret key generation
Run the **encryption_key.py** program
A config file **secret_key.ini** is created. Make sure there is a copy of the secret key in both the sender and receiver folders.\
This key should also remain secret and not be passed on to other people.
```
python encryption_key.py
```

## Run the sender and receiver programs
Channel Folder: <br />
Location 02_Software/01_Sender
```
python mqtt_video_sender.py
```
Receiver folder: <br/>
Location: 02_Software/02_Receiver
```
python mqtt_mask_detection.py
```
