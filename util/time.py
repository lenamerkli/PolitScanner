from datetime import datetime


def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
