import platform
import os
import time
import logging
from enum import Enum

def timer(func):
    def outer(*args, **kwargs):
        start_time = time.time()
        
        def inner():
            return func(*args, **kwargs)
        
        result = inner()
        end_time = time.time()
        logging.info(f"'{func.__name__}' fonksiyonu {end_time - start_time:.6f} saniye sürdü.")
        return result

    return outer

def notify(title, message):
    system = platform.system()

    if system == "Windows":
        pass
    else:
        os.system(f'notify-send "{title}" "{message}"')

class Colors(Enum): # BGR
    # Ana Renkler
    SNOWWHITE = (250, 250, 255)     # #FFFAFA
    RED_PRIMARY = (70, 57, 230)     # #E63946
    GRAY_DARK = (66, 45, 43)        # #2B2D42
    BLACK_TEXT = (27, 26, 26)       # #1A1A1B

    # Ekstra Renkler
    BLUE_STEEL = (157, 123, 69)     # #457B9D
    BLUE_LIGHT = (245, 243, 241)    # #F1F3F5

    # Ability level renkleri
    ABILITY_1 = (229, 136, 30)      # #1E88E5
    ABILITY_2 = (193, 172, 0)       # #00ACC1
    ABILITY_3 = (165, 191, 0)       # #00BFA5
    ABILITY_4 = (113, 204, 46)      # #2ECC71
    ABILITY_5 = (61, 255, 106)      # #6AFF3D

    # Reading
    READING_1 = (235, 243, 246)     # #F6F3EB
    READING_2 = (220, 233, 239)     # #EFE9DC