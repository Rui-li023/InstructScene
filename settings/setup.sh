conda create -n instructscene python=3.9
conda activate instructscene

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install -r settings/requirements.txt

python3 -c "import nltk; nltk.download('cmudict')"
