#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pip._internal.req import parse_requirements
from setuptools import setup, find_packages


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

parsed_requirements = parse_requirements(
    'requirements/prod.txt',
    session='workaround',
)

parsed_test_requirements = parse_requirements(
    'requirements/test.txt',
    session='workaround',
)


requirements = [str(ir.requirement) for ir in parsed_requirements]
test_requirements = [str(tr.requirement) for tr in parsed_test_requirements]


setup(
    name='position-independent-embeddings',
    version='0.2.0',
    description='A python package that allows you to train, use, and evaluate position-independent word embeddings',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    author="Vít Novotný",
    author_email='witiko@mail.muni.cz',
    url='https://github.com/MIR-MU/pine',
    packages=find_packages(exclude=['tests', 'tests.*']),
    include_package_data=True,
    setup_requires=['setuptools', 'pip'],
    install_requires=requirements,
    license='LGPLv2+',
    zip_safe=False,
    keywords='pine',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Environment :: GPU :: NVIDIA CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Utilities',
        'Typing :: Typed',
        'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    tests_require=test_requirements,
)
