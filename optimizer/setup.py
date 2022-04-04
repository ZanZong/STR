from setuptools import setup

# Installation of pkgs in requirements.txt is required
setup(
    name='stropt',
    version='0.1.0',
    description='Memory optimization for DNN training',
    packages=['stropt'],
    python_requires='>=3.6',
)