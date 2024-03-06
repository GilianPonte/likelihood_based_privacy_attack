from setuptools import setup, find_packages

setup(
    name='privacy_attack',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'privacy_attack=privacy_attack:main',
        ],
    },
)
