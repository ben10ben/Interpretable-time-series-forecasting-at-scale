conda create -n yourenvname python=3.9 anaconda#mkdir data
#mkdir models
#mkdir lightning_logs
#cd data
#mkdir electricity
#mkdir retail
#cd electricity

#wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip" -O temp.zip
#unzip temp.zip
#rm temp.zip
#cd ..
#cd retail
#wget "https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data?select=oil.csv.7z" -O retail_tmp.zip
#unzip retail_tmp.zip
#rm retail_tmp.zip
#cd ..
#cd ..

source activate yourenvname
conda install pip
pip3 install -r requirements.txt
python3 electricity_tft.py
python3 retail_tft.py
cat output.txt