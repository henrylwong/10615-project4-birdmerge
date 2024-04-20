import datetime

import options

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(options.LOGGING_FILEPATH, "a") as file:
        file.write(f"{timestamp} || {message}\n")