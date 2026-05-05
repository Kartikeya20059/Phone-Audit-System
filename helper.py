import os
import subprocess
import re
import shlex
import shutil
from typing import List, Dict, Any, Optional

# Try to pick adb from environment or PATH, fallback to a common Windows location
ADB_PATH = os.environ.get("ADB_PATH") or shutil.which("adb") or r"C:\platform-tools\adb.exe"

def run_adb_cmd(cmd_args: List[str], timeout: int = 30, quiet: bool = True) -> str:
    """
    Run adb using ADB_PATH. cmd_args: list like ["devices"] or ["shell", "content", "query", ...]
    Returns decoded stdout or raises RuntimeError on failure.
    """
    if not ADB_PATH:
        raise RuntimeError("adb not found. Set ADB_PATH env var or ensure adb is on PATH.")
    cmd = [ADB_PATH] + cmd_args
    if not quiet:
        print("Running:", " ".join(shlex.quote(c) for c in cmd))
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
        return out.decode(errors="ignore")
    except subprocess.CalledProcessError as e:
        out = e.output.decode(errors="ignore") if hasattr(e, "output") else str(e)
        raise RuntimeError(f"ADB command failed (rc={e.returncode}):\n{out}")
    except FileNotFoundError:
        raise RuntimeError(f"adb not found at {ADB_PATH}. Please verify the path or set ADB_PATH.")
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ADB command timed out after {timeout}s: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to run adb command: {e}")

def adb_devices(timeout: int = 10) -> str:
    """Return output of `adb devices` for diagnostics."""
    return run_adb_cmd(["devices"], timeout=timeout)

def fetch_call_log_raw(timeout: int = 60) -> str:
    """Fetch raw call log using content provider query."""
    return run_adb_cmd(["shell", "content", "query", "--uri", "content://call_log/calls"], timeout=timeout)

# --- Device info ---
def fetch_device_props(timeout: int = 20) -> str:
    """Return the raw output of `adb shell getprop`."""
    return run_adb_cmd(["shell", "getprop"], timeout=timeout)

def fetch_battery(timeout: int = 20) -> str:
    """Return the raw output of `adb shell dumpsys battery`."""
    return run_adb_cmd(["shell", "dumpsys", "battery"], timeout=timeout)

def fetch_meminfo(timeout: int = 30) -> str:
    """Return the raw output of `adb shell dumpsys meminfo all`."""
    return run_adb_cmd(["shell", "dumpsys", "meminfo", "all"], timeout=timeout)

def fetch_uptime(timeout: int = 10) -> str:
    """Return the raw output of `adb shell uptime`."""
    return run_adb_cmd(["shell", "uptime"], timeout=timeout)

# --- Parser for `content query` output ---
_TOKEN_RE = re.compile(r'''
    (?P<key>[a-zA-Z0-9_]+)\s*=\s*
    (?P<val>
        "(?:[^"\\]|\\.)*"   |    # double-quoted string
        '(?:[^'\\]|\\.)*'   |    # single-quoted string
        [^,]+                     # unquoted value until comma
    )
''', re.VERBOSE)

def _strip_quotes(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def _try_cast_number(k: str, v: str) -> Any:
    int_fields = {"duration", "type", "date", "_id", "presentation", "subscription_id"}
    v_clean = str(v).strip()
    if k in int_fields:
        m = re.search(r"-?\d+", v_clean)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return v
    return v

def parse_content_query(output: str) -> List[Dict[str, Any]]:
    """
    Parse output from `adb shell content query --uri ...`.
    Returns a list of dicts (one dict per Row). Numeric fields (duration, type, date) are converted to ints when possible.
    """
    rows = []
    if not output:
        return rows

    parts = re.split(r'\bRow:\s*\d+\b', output)
    for part in parts[1:]:
        record = {}
        text = part.strip()
        if not text:
            continue

        for m in _TOKEN_RE.finditer(text):
            k = m.group("key").strip()
            v = _strip_quotes(m.group("val").strip())
            if v.upper() == "NULL":
                v = ""
            v = _try_cast_number(k, v)
            record[k] = v

        if not record:
            tokens = [t.strip() for t in text.replace("\n", " ").split(",") if t.strip()]
            for token in tokens:
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                v = v.strip().strip(",").strip()
                if v.upper() == "NULL":
                    v = ""
                v = _try_cast_number(k.strip(), v)
                record[k.strip()] = v

        if record:
            rows.append(record)

    return rows
