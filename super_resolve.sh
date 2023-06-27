#!/bin/bash

if [[ $1 == "model_1000_750.pth" ]]; then
    # check the number of command line arguments
    if [ "$#" -eq 2 ]; then
    /bin/python3 "1000_750/predict.py" --model "models/$1" --low_res_file $2
    else
    /bin/python3 "1000_750/predict.py" --model "models/$1" --low_res_file $2 --high_res_file $3
    fi
fi

if [[ $1 == "model_750_300.pth" ]]; then
    if [ "$#" -eq 2 ]; then
    /bin/python3 "750_300/predict.py" --model "models/$1" --low_res_file $2
    else
    /bin/python3 "750_300/predict.py" --model "models/$1" --low_res_file $2 --high_res_file $3
    fi
fi

if [[ $1 == "model_1000_300.pth" ]]; then
    if [ "$#" -eq 2 ]; then
    /bin/python3 "1000_300/predict.py" --model "models/$1" --low_res_file $2
    else
    /bin/python3 "1000_300/predict.py" --model "models/$1" --low_res_file $2 --high_res_file $3
    fi
fi
