#!/bin/bash

coverage run -m unittest discover almiky
coverage report -m
coverage html