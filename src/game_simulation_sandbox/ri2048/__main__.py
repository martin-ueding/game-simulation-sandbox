#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse

import matplotlib.pyplot as pl
 
import ri2048.game
import ri2048.training


def main():
    parser = argparse.ArgumentParser()
    options = parser.parse_args()

    ri2048.training.make_agent()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options = parser.parse_args()
    main(options)
