from worker.celery_app import celery_app
from worker import tasks  # noqa: F401


if __name__ == "__main__":
    celery_app.worker_main(["worker", "--loglevel=info", "-Q", "celery"])
