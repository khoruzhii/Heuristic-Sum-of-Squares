from setuptools import setup, find_packages

setup(
    name="sos",
    version="0.1.0",
    description="Sum of Squares Transformer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "wandb",
        "tqdm",
        "cvxpy",
    ],
    python_requires=">=3.8",
) 