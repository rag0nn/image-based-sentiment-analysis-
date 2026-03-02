import platform
import os
import time

def timer(func):
    def outer(*args, **kwargs):
        start_time = time.time()
        
        def inner():
            return func(*args, **kwargs)
        
        result = inner()
        end_time = time.time()
        print(f"'{func.__name__}' fonksiyonu {end_time - start_time:.6f} saniye sürdü.")
        return result

    return outer

def notify(title, message):
    system = platform.system()

    if system == "Windows":
        pass
    else:
        os.system(f'notify-send "{title}" "{message}"')
