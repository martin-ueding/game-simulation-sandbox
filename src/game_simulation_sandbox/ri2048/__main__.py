#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse

from . import environment
from . import training


def main():
    parser = argparse.ArgumentParser()
    options = parser.parse_args()

    environment.validate()

    training.make_agent()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options = parser.parse_args()
    main(options)
