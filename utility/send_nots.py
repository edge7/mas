import threading
from notifications.email import try_to_send_email
from notifications.wapp import send_wapp_mess
from pathlib import Path
import os

def notify_the_boss(message):
    try_to_send_email(message)
    send_wapp_mess(message)


def notify(msg):
    os.environ["PATH"] += os.pathsep + str(Path(__file__).parent.parent) + "/lib"

    th = threading.Thread(target=notify_the_boss, args=(msg,))
    th.start()



