import subprocess
from time import sleep
import sys
import hashlib
from _hashlib import HASH as Hash
from pathlib import Path

def run_command_with_live_output(command, shell=False, capture_output=False):
    if not capture_output:                                                                                                                                                            
        result = subprocess.run(command, shell=shell, stdout=sys.stdout, stderr=sys.stderr, text=True)
        return result, None
    else:
        result = subprocess.run(command, shell=shell, capture_output=True, text=True)
        return result, result.stdout.split('\n')

def get_available_conda_envs():
    result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
    envs = list(filter(None, map(lambda x: x.split(' ', 1)[0], result.stdout.split('\n')[2:])))
    return envs

def setup_conda_env_from_file(environment_file, force=False):
    print("Setting up new Conda environment, this could take a while...")
    run_command_with_live_output(['conda', 'env', 'create', '-f', environment_file, '--force' if force else ''])
    print("Setup finished.")

def launch_python_script_in_conda_env(environment_name, environment_file, script, args = []):
    if environment_name not in get_available_conda_envs():
        print(f"Required environment {environment_name} not found on this machine, setting it up.")
        setup_conda_env_from_file(environment_file)
        print("Running script now.")
    
    # run_command_with_live_output(["conda" , "run" , "--no-capture-output", "-n", environment_name, "python", script] + list(map(lambda x: str(x), args)))
    run_command_with_live_output(["conda" , "run", "-n", environment_name, "python", script] + list(map(lambda x: str(x), args)))

def md5_update_from_dir(directory, hash):
    assert Path(directory).is_dir()
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash.update(chunk)
        elif path.is_dir():
            hash = md5_update_from_dir(path, hash)
    return hash

def md5_dir(directory):
    return md5_update_from_dir(directory, hashlib.md5()).hexdigest()
