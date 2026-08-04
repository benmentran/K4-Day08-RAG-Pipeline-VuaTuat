"""Test tasks 6-10."""
import sys
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + ENV.get("PYTHONPATH", "")
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["PYTHONUTF8"] = "1"

TASKS = [
    "task6_lexical_search",
    "task7_reranking",
    "task8_pageindex_vectorless",
    "task9_retrieval_pipeline",
    "task10_generation",
]

results = []
for module in TASKS:
    print(f"\n{'='*70}\nTESTING: {module}\n{'='*70}")
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", f"src.{module}"],
            cwd=str(ROOT),
            env=ENV,
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - start
        log_path = LOG_DIR / f"test_{module}.txt"
        log_path.write_text(f"=== STDOUT ===\n{proc.stdout}\n=== STDERR ===\n{proc.stderr}\n", encoding="utf-8")
        results.append((module, "OK" if proc.returncode == 0 else "FAIL", elapsed, proc.returncode))
        print(proc.stdout)
        if proc.stderr.strip():
            print("--- STDERR ---")
            print(proc.stderr)
        print(f">>> exit={proc.returncode} in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - start
        results.append((module, f"ERROR: {e}", elapsed, -1))
        print(f">>> {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for m, s, e, c in results:
    print(f"  {m:35s} {s:10s} {e:6.1f}s (exit={c})")
