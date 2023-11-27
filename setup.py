from setuptools import setup

def get_requirements():
    with open("requirements.txt", encoding="utf8") as f:
        return f.read().splitlines()

setup(
    name='sahi_batched',
    version='0.1',
    description='SAHI batched inference library',
    packages=['sahi_batched'],
    install_requires=get_requirements(),
)
