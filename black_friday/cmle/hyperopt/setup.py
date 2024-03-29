from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['google-cloud-storage','pandas','gcsfs', 'xgboost', 'cloudml-hypertune']

setup(
    name='trainer',
    version='0.2',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
