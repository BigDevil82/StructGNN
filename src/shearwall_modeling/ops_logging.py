from pathlib import Path

import openseespy.opensees as ops


def redirect_ops_output(log_path: str | Path = "outputs/logs/ops.log") -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ops.logFile(str(path), "-noEcho")
