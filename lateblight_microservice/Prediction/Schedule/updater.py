from datetime import datetime
from django.core.management.base import BaseCommand
from apscheduler.schedulers.background import BackgroundScheduler
# from Prediction.cron import PrepareProbabilities
from Prediction.management.commands.run_inference import Command

scheduler = BackgroundScheduler()
# scheduled = False
scheduled = True  


def start():
    global scheduled
    if not scheduled:
        scheduler.add_job(Command(BaseCommand).handle, 'cron', hour=0, minute=15) #change the time later making it 00 in nepali time
        scheduler.start()
        scheduled = True