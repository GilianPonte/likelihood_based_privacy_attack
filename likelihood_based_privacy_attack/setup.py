from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='likelihood-based privacy attack',
    url='',
    author='',
    author_email='',
    # Needed to actually package something
    packages=['likelihood-based privacy attack'],
    # Needed for dependencies
    install_requires=['numpy', 'pandas', 'sklearn', 'random', 'math'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Likelihood-based privacy attack',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
