from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["tensorflow>=2"]

setup(
    name="RLPy",
    version="0.0.1",
    author="Samit Ahlawat",
    author_email="samit.ahlawat@gmail.com",
    description="Reinforcement Learning Concepts and Algorithms",
    long_description=readme,
    url="https://github.com/samit-ahlawat/RLPy",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)