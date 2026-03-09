import subprocess


def test_cli_manifest_smoke():
    cmd = ["python", "Data Analysis for Hospitals/task/cli.py", "manifest"]
    assert subprocess.run(cmd, check=False).returncode == 0
