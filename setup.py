# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="ml_project",
    version="development",
    description="Machine Learning project: reproducing Noise2Noise",
    author="Ivan Prosperi",
    author_email="ivan.prosperi@stud.unifi.it",
    url="https://github.com/ivan94fi/ml_project",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
