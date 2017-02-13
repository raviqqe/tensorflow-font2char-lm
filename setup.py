import re
import setuptools
import sys


if not sys.version_info >= (3, 5):
    exit("Sorry, Python must be later than 3.5.")


setuptools.setup(
    name="tensorflow-font2char-lm",
    version=re.search(r'__version__ *= *"([0-9]+\.[0-9]+\.[0-9]+)" *\n',
                      open("font2char_lm/__init__.py").read()).group(1),
    description="Simple character or font-based language model in TensorFlow",
    long_description=open("README.md").read(),
    license="Public Domain",
    author="Yota Toyama",
    author_email="raviqqe@gmail.com",
    url="https://github.com/raviqqe/tensorflow-font2char_lm/",
    packages=["font2char_lm"],
    install_requires=[
        "char2image",
        "tensorflow-qnd<=1",
        "tensorflow-extenteten<=1",
        "tensorflow-qndex<=1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Public Domain",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking",
    ],
)
