from pathlib import Path
from subprocess import Popen


def load_jupyter_server_extension(nbapp):
    apps = Path('apps').glob('*.py')
    Popen(["bokeh", "serve", "--allow-websocket-origin=*", *(app.name for app in apps)])
