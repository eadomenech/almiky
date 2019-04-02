#!/bin/bash

coverage run -m unittest discover almiky
coverage report
coverage html