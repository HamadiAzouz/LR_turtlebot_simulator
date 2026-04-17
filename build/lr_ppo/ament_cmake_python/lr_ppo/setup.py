from setuptools import find_packages
from setuptools import setup

setup(
    name='lr_ppo',
    version='1.0.0',
    packages=find_packages(
        include=('lr_ppo', 'lr_ppo.*')),
)
