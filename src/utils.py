import os
import yaml
import tarfile
import subprocess


def load_yaml_config(config_path: str):
    """Load project constants from yaml"""
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


def extract_key_from_file(file_path):
    with open(file_path, 'r') as file:
        key = file.read()
    return key

def run_shell_cmd(cmd, verbose = False) -> None:
    """Run bash commands with Python"""
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Error happened: {std_err}")
    if verbose:
        print(std_out, std_err)


def untar_archive(acrhive_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with tarfile.open(acrhive_file, "r:gz") as tar:
        tar.extractall(output_dir)

