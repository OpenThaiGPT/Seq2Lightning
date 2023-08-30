from setuptools import setup, find_packages

setup(
    name = "seq2lightning",
    version = "0.0.dev1",
    packages = find_packages(exclude=[]),
    license="Apache License, Version 2.0",
    include_package_data = True,
    install_requires = [
        "packaging>=21.0",
        "dask>=2023.8.1",
        "torch>=2.0.1",
        "numpy>=1.23.5",
        "transformers>=4.32.1",
        "sentencepiece>=0.1.99",
        "datasets>=2.14.4",
        "pytorch-lightning>=2.0.7",
        "rouge>=1.0.1",
        "wandb>=0.15.9",
        "PyYAML>=6.0"
    ]
)