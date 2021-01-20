import setuptools

with open('requirements.txt', 'r') as file:
    requirements = file.read()

setuptools.setup(
    name='vcboost',
    install_requires=requirements
)
