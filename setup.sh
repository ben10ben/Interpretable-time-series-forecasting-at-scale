virtualenv myenv
mkdir data
mkdir models
mkdir lightning_logs
cd data
mkdir electricity
cd electricity

wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip" -O temp.zip
unzip temp.zip
rm temp.zip
cd ..
cd ..

source myenv/bin/activate

#pip3 install -r requirements.txt

#python3 load_datasets.py

#python3 prepare_datasets.py

#python3 train_model.py