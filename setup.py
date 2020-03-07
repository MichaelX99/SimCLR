from setuptools import setup, PEP420PackageFinder
import os

version = {}
with open("simclr/__version__.py") as fp:
    exec(fp.read(), version)

with open('README.md', encoding='utf-8') as f:
    long_desc = f.read()

REQUIRED = [
    #'torch',
    #'torchvision',
]

EXTRAS = {
    'test': ['pylint', 'pytest']
}

setup(
    name="simclr",
    version=version['__version__'],
    author="Michael Person",
    #url="",
    packages=PEP420PackageFinder.find(include=['simclr']),
    description="Jax/Pytorch implementation of SimCLR",
    long_description=long_desc,
    #long_description_content_type='text/markdown',
    python_requires=">=3.6",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
)