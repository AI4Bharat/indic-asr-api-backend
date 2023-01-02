# Indic ASR API

## Installation Instructions

```
conda create -n ai4b_asr python=3.8
conda activate ai4b_asr
pip install torch torchvision torchaudio
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
bash reinstall.sh
cd ..
git clone https://github.com/AI4Bharat/indic-asr-api-backend.git
cd indic-asr-api-backend
pip install -r requirements.txt
```

## Download models
```
mkdir -p models
wget -P models https://objectstore.e2enetworks.net/indic-asr-public/external/checkpoints/english/en-conformer-ctc.nemo

```

## Usage

#### Starting the server:

```
python api.py
```

This will start the server on port 4992. Keep this process running. You can modify the PORT in `api.py`.

#### Running inference:
Here we show how to access the API using an example (on a new terminal) -
```
python example_ai4b_asr_rest_api.py
```
Output:

[Play Audio](https://objectstore.e2enetworks.net/indic-asr-public/sample_audio.wav)
```
Response: <Response [200]>
Text: deposit five thousand rupees in my bank account
```