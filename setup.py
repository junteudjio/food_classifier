# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='food_classifier',
    version='0.0.1',
    description='Food classifier: Determine if an image is a sushi or a sandwich',
    long_description=readme,
    author='Junior Teudjio Mbativou',
    author_email='jun.teudjio@gmail.com',
    url='https://github.com/junteudjio',
    license=license,
    packages=find_packages(exclude=('tests')),
    install_requires=[

    ]
)

