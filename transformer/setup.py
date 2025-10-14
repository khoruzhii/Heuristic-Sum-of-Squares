from setuptools import setup, find_packages

setup(
    name="transformer",
    version="0.1.0",
    description="Transformer for Polynomial Tasks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "wandb",
        "tqdm",
    ],
    python_requires=">=3.8",
) 