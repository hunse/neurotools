#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
    except Exception as e:
        print("Forget setuptools, trying distutils...")
        from distutils.core import setup

setup(
    name="neurotools",
    version="0.0.1",
    author="Eric Hunsberger",
    author_email="ehunsber@uwaterloo.ca",
    scripts=[],
    url="https://github.com/hunse/neurotools",
    license="LICENSE",
    description="Tools for computational neuroscientists",
    long_description=open('README.md').read(),
    requires=[],
    # test_suite='neurotools.tests',
)
