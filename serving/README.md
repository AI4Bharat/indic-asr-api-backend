# Start Triton server inside container

- For indo-aryan(ia):
```
docker run --shm-size=16g --gpus device=0 -it --rm -p8000:8000 -p8001:8001 -p8002:8002 ai4bharat/triton-indo-aryan-asr:latest tritonserver --model-repository=/models/
```
- For dravidian(dr): replace the docker image with `ai4bharat/triton-dravidian-asr:latest`
- For hindi(hi): replace the docker image with `ai4bharat/triton-hindi-asr:latest`
- For english(en): replace the docker image with `ai4bharat/triton-whisper-asr:latest`

## Test using client code

> Install requirements: `pip install -r client/requirements.txt`
> Run client code:
```
cd client
python client_ensemble.py <lang-id> #lang-id to language mapping is given below
```
# Languages supported

| Language | Lang id | Triton base |
|----------|-------|-------|
| Hindi | hi | hi |
| Bengali | bn | ia |
| Gujarati | gu | ia |
| Kannada | kn | dr |
| Malayalam | ml | dr |
| Marathi | mr | ia |
| Odia | or | ia |
| Punjabi | pa | ia |
| Tamil | ta | ta |
| Telugu | te | dr |
| Indian English | en | en |

# (Optional) Export .nemo checkpoint

*Note: This step converts .nemo checkpoint to jitted .pt format and is required since .nemo ckpt format is not supported in triton serving*

> 1. Install requirements: `pip install -r export/requirements.txt`
> 2. Export to .pt format
```
cd export
python nemo2pt.py <nemo-checkpoint-path> # Path to custom nemo checkpoint
```
> 3. Copy the converted checkpoint to `<triton-base>/asr_am/1` directory and rename it to `model.pt`.
