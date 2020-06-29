from setuptools import setup, find_packages


setup(
    name="kinpy",
    version="0.1",
    description="Tools for post processing of data acquired of DLC session",
    url="https://github.com/caniko2/DeepLabCutAnalysis",
    author_email="canhtart@gmail.com",
    packages=find_packages(),
    zip_safe=False,
    install_requires=["numpy", "pandas", "matplotlib"],
)
