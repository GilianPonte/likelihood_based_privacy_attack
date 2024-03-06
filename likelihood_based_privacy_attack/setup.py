from setuptools import setup, find_packages

setup(
    name='likelihood_based_privacy_attack',
    version='0.1',
    packages=find_packages(),
    author='Gilian',
    author_email='ponte@rsm.nl',
    description='Privacy attack',
    url='https://github.com/GilianPonte/likelihood_based_privacy_attack',
    install_requires=[
        'numpy',
        'scikit-learn',
        # Add any other dependencies here
    ],
)
