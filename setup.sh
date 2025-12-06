conda create -n lsnet python==3.8
conda activate lsnet
pip install -r requirements.txt

curl -L -o dataset/vnfood-30-100.zip \
    https://www.kaggle.com/api/v1/datasets/download/meowluvmatcha/vnfood-30-100

unzip dataset/vnfood-30-100.zip