from django.core.management.base import BaseCommand
from apscheduler.schedulers.background import BackgroundScheduler
from Prediction.management.commands.run_inference import Command
from apscheduler.triggers.cron import CronTrigger


def start():
    scheduler = BackgroundScheduler()
    scheduler.add_job(Command(BaseCommand).handle, CronTrigger(hour=12,minute=5))
    scheduler.start()