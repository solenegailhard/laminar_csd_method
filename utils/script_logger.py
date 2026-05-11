# utils/sim_logger.py

import ast
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT     = Path(__file__).parent.parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def _week_label() -> str:
    """Retourne ex: '2025-W11'"""
    now = datetime.now()
    return f"{now.year}-W{now.strftime('%W')}"


def _parse_functions(script_path: str) -> List[Dict]:
    """
    Extrait via ast les fonctions définies dans le script :
    nom + noms des arguments (sans exécuter).
    """
    try:
        source = Path(script_path).read_text()
        tree   = ast.parse(source)
    except Exception:
        return []

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            functions.append({"name": node.name, "args": args})
    return functions


def _load_week_log(label: str) -> Dict:
    path = LOGS_DIR / f"{label}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"week": label, "runs": []}


def _save_week_log(label: str, log: Dict) -> None:
    path = LOGS_DIR / f"{label}.json"
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=str)


def log_run(
    script:     str,
    config:     Dict,
    params:     Dict,
    status:     str            = "completed",
    outputs:    Optional[List[str]] = None,
    error:      Optional[str]  = None,
    duration_s: Optional[float] = None,
) -> None:
    """
    Enregistre un run dans le log hebdomadaire.

    Args:
        script     : __file__ du script lancé
        config     : dict retourné par load_config()
        params     : dict des paramètres propres au script
        status     : "completed" | "failed"
        outputs    : liste de paths de fichiers produits
        error      : message d'erreur si failed
        duration_s : durée totale en secondes
    """
    label    = _week_label()
    week_log = _load_week_log(label)

    entry = {
        "timestamp":   datetime.now().isoformat(),
        "script":      Path(script).name,
        "script_path": str(Path(script)),
        "status":      status,
        "duration_s":  round(duration_s, 1) if duration_s else None,
        "config":      config,
        "params":      params,
        "functions":   _parse_functions(script),
        "outputs":     outputs or [],
        "error":       error,
    }

    week_log["runs"].append(entry)
    _save_week_log(label, week_log)
    print(f"[log] Entry added to logs/{label}.json")