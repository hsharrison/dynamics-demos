from subprocess import Popen


def load_jupyter_server_extension(nbapp):
    Popen(["bokeh", "serve", "hkb.py", "--allow-websocket-origin=*"])
