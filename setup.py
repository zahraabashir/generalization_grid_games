from setuptools import setup, find_packages

setup(
    name="generalization-grid-games",
    version="0.1.0",
    packages=find_packages(include=["generalization_grid_games", "generalization_grid_games.*"]),
    install_requires=["gym>=0.26"],  # or gymnasium compatibility layer if you prefer
)