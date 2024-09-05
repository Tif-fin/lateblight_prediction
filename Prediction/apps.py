from django.apps import AppConfig


class PredictionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Prediction'
    def ready(self):
        import Prediction.Schedule.updater as updater
        updater.start()
