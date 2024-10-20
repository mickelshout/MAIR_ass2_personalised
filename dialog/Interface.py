from dialog_system import Dialog_system
import os
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# The dialog is created and updates itself using the user's response to its responses
dialog = Dialog_system()
while not dialog.terminate:
    dialog.update_state()
    dialog.utterance = input(dialog.generate_response())
    if dialog.config.get("delay"):
        time.sleep(2)
