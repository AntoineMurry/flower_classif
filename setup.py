from setuptools import find_packages
from setuptools import setup

# Python version used for development: 3.6.7

requirements = """
pip>=9
setuptools>=26
wheel>=0.29
pytest
coverage
flake8
matplotlib
numpy
keras
tensorflow=1.3
"""

setup(name='flower_classif',
      setup_requires=['setuptools_scm'],
      use_scm_version={'write_to': 'MLP_from_scratch/version.txt'},
      description="package with MLP coded from numpy",
      packages=find_packages(),
      test_suite = 'tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
