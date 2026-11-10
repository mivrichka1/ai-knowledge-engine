import os
import subprocess
import sys
from pathlib import Path


def test_migrations_on_empty_sqlite_database(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path}/migration.db"}
    for args in (["upgrade", "head"], ["check"]):
        result = subprocess.run(
            [sys.executable, "-m", "alembic", *args],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
