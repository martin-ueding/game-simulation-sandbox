#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse

from . import environment


def main():
    parser = argparse.ArgumentParser()
    options = parser.parse_args()

    environment.validate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options = parser.parse_args()
    main(options)
