from pathlib import Path
from subprocess import Popen

NOT_APPS = {'__init__.py', 'common.py'}

def load_jupyter_server_extension(nbapp):
    apps = Path('apps').glob('*.py')
    Popen(["bokeh", "serve", "--allow-websocket-origin=*", *(str(app) for app in apps if str(app) not in NOT_APPS)])
