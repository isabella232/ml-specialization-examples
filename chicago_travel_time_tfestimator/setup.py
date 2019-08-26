from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['google-cloud-bigquery==1.8.1', 'google-api-python-client']

setup(
    name='trainer',
    version='0.2',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
