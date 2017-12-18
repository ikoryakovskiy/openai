#!/bin/bash

echo -en "\033[1;32m --- Download results --- \033[0m\n"
mkdir -p logs
ssh ikoryakovskiy@calcutron "cd openai && tar --use-compress-program=pbzip2 --exclude CMake*.txt -cf results.tar.bz2 *.csv tmp/*.yaml"
scp ikoryakovskiy@calcutron:~/openai/results.tar.bz2 ./
tar --use-compress-program=pbzip2 -xvf results.tar.bz2 -C ./logs/
#rm ./results.tar.bz2


