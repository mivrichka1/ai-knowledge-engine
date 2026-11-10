from .config import Settings
from .main import create_app

app = create_app(Settings())  # type: ignore[call-arg]
