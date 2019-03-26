#!/bin/sh

cd /app
pip install -r requirements.txt --proxy=$http_proxy
python /app/medgate_trial.py /app/letter_directory
