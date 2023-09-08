from setuptools import setup

name = "bbr"
version = "0.1.0"
description = 'Official implementation of the paper: "Box-based Refinement for Weakly Supervised and Unsupervised Localization Tasks"'
author = "Eyal Gomel"
author_email = "eyalgomel12@gmail.com"
url = "https://github.com/eyalgomel/box-based-refinement"
license = "MIT"


setup(
    name=name,
    version=version,
    description=description,
    author=author,
    author_email=author_email,
    url=url,
    license=license,
    packages=["bbr", "CLIP", "BLIP", "detr", "dino", "LOST", "moveseg", "TokenCut"],
)
