import os
import subprocess
import threading
import signal
import time

def start_or_restart_streamlit_app(app_path="app.py", port=8501, username=None):
    """
    Startet eine Streamlit-App auf dem angegebenen Port in einem separaten Thread.
    Falls bereits eine Instanz läuft, wird sie vorher beendet.
    Am Ende wird die JupyterHub-Proxy-URL ausgegeben.

    :param app_path: Pfad zur Streamlit-Datei (Standard: 'app.py')
    :param port: Port, auf dem die App laufen soll (Standard: 8501)
    :param username: JupyterHub-Username für Proxy-Link (optional, wird sonst aus ENV gelesen)
    """
    if username is None:
        username = os.getenv("JUPYTERHUB_USER", "unknown-user")

    def kill_existing_streamlit():
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if "streamlit" in line and f"--server.port={port}" in line:
                    parts = line.split()
                    pid = int(parts[1])
                    print(f"🔁 Beende existierenden Streamlit-Prozess (PID {pid}) auf Port {port}...")
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(1)
        except Exception as e:
            print(f"⚠️ Fehler beim Beenden von Streamlit-Prozessen: {e}")

    def run_streamlit():
        kill_existing_streamlit()
        print(f"🚀 Starte Streamlit-App '{app_path}' auf Port {port}...\n")

        # Setze ENV Variablen zur Unterdrückung von Startup-Meldungen
        env = os.environ.copy()
        env["STREAMLIT_BROWSER_GATHERUSAGESTATS"] = "false"
        env["STREAMLIT_SERVER_HEADLESS"] = "true"
        env["STREAMLIT_LOGGER_LEVEL"] = "error"

        url = f"https://jupyterlab.aai-dfine.de/user/{username}/proxy/{port}/"
        print(f"\n✅ Streamlit ist jetzt erreichbar unter:\n👉 {url}")
        # Starte die App und leite Ausgabe durch
        process = subprocess.Popen(
            ["streamlit", "run", app_path, f"--server.port={port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )

        try:
            for line in process.stdout:
                print(line, end="")
        except Exception as e:
            print(f"⚠️ Fehler bei der Streamlit-Ausgabe: {e}")

    threading.Thread(target=run_streamlit).start()