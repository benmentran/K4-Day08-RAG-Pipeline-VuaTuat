"""
Test runner - chạy các task 4-10 (trừ 1-3) với PYTHONPATH=src để relative import hoạt động.
"""
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
    ("task4_chunking_indexing", "Chunking & Indexing"),
    ("task5_semantic_search", "Semantic Search"),
    ("task6_lexical_search", "Lexical Search (BM25)"),
    ("task7_reranking", "Reranking"),
    ("task8_pageindex_vectorless", "PageIndex (vectorless)"),
    ("task9_retrieval_pipeline", "Retrieval Pipeline"),
    ("task10_generation", "Generation with Citation"),
]

results = []

for module, label in TASKS:
    print(f"\n{'='*70}")
    print(f"TESTING: {module} ({label})")
    print('='*70)
    log_path = LOG_DIR / f"test_{module}.txt"
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
        # Save full output
        log_path.write_text(
            f"=== STDOUT ===\n{proc.stdout}\n=== STDERR ===\n{proc.stderr}\n",
            encoding="utf-8",
        )
        status = "OK" if proc.returncode == 0 else "FAIL"
        results.append((module, label, status, elapsed, proc.returncode))
        # Print last 30 lines of stdout
        lines = proc.stdout.strip().splitlines()
        print("\n".join(lines[-30:]))
        if proc.stderr.strip():
            print("[STDERR]")
            print("\n".join(proc.stderr.strip().splitlines()[-15:]))
        print(f"\n>>> {status} in {elapsed:.1f}s (exit={proc.returncode})")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        results.append((module, label, "TIMEOUT", elapsed, -1))
        print(f">>> TIMEOUT after {elapsed:.0f}s")
    except Exception as e:
        elapsed = time.time() - start
        results.append((module, label, f"ERROR: {e}", elapsed, -1))
        print(f">>> ERROR: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for module, label, status, elapsed, code in results:
    print(f"  {module:35s} {status:10s} {elapsed:6.1f}s (exit={code})")
