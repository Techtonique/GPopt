from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = "0.2.2"

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(
    path.join(here, "requirements.txt"), encoding="utf-8"
) as f:
    all_reqs = f.read().split("\n")

install_requires = [
    x.strip() for x in all_reqs if "git+" not in x
]
dependency_links = [
    x.strip().replace("git+", "")
    for x in all_reqs
    if x.startswith("git+")
]

setup(
    name="GPopt",
    version=__version__,
    description="Bayesian Optimization using Gaussian Process Regression",
    long_description="Bayesian Optimization using Gaussian Process Regression",
    url="https://github.com/thierrymoudiki/GPopt",
    download_url="https://github.com/thierrymoudiki/GPopt/tarball/"
    + __version__,
    license="BSD",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    keywords="",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    author="Thierry Moudiki",
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email="thierry.moudiki@gmail.com",
)
