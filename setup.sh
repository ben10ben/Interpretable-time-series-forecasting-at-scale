virtualenv myenv
source myenv/bin/activate

pip3 install -r requirements.txt

python3 load_datasets.py

python3 prepare_datasets.py

python3 train_model.py