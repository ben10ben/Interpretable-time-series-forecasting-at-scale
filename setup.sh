#!/bin/bash
mkdir data
mkdir data/electricity

cd data/electricity
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip" -O temp.zip
unzip temp.zip
rm temp.zip
cd ..
cd ..

conda create -n yourenvname python=3.8 anaconda
source activate yourenvname
conda install pip
pip3 install -r requirements.txt
