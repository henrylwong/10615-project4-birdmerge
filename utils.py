import time
from datetime import datetime

def get_time():
    timestamp = time.time()
    formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    return formatted_time