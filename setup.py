from setuptools import setup, find_packages

setup(
    name="generalization-grid-games",
    version="0.1.0",
    packages=find_packages(include=["generalization_grid_games", "generalization_grid_games.*"]),
    include_package_data=True,
    package_data={
        "generalization_grid_games": [
            "envs/assets/*",      # images like matchstick.png
        ]
    },
    install_requires=[
        "gym>=0.26", 
        "matplotlib", 
        "Pillow", 
        "prpl_utils @ git+https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono.git@2d89490#subdirectory=prpl-utils"
        ],
)