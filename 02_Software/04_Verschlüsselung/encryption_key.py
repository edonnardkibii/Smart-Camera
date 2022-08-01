"""

Author: James Kibii
Project: Security & IoT
Description: Create a symmetric encryption key

"""

from configparser import ConfigParser
import base64
from cryptography.fernet import Fernet
import time

secretKey = Fernet.generate_key()
current_time = time.strftime("%H:%M:%S", time.localtime())

config = ConfigParser()
config.read('secret_key.ini')
config.add_section('Symmetric Encryption')
config.set('Symmetric Encryption', 'Secret Key', base64.b64encode(secretKey).decode('utf-8'))

with open('secret_key.ini', 'w') as configfile:
    config.write(configfile)

print("Key was generated successfully at " +str(current_time))
print("Key Length: " +str(len(secretKey)))
print("To view how long it would take to crack your password visit")
print("https://www.security.org/how-secure-is-my-password/")
print("& enter your password")