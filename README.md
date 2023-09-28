
## Serveur docker pour tritton 


```bash
cd server
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v%cd%:/workspace/ -v%cd%/model_repository:/models nvcr.io/nvidia/tritonserver:23.07-py3 bash

pip install torch torchvision
pip install transformers
pip install Image
pip install s3fs
pip install transformers
pip install accelerate
pip install gradio
pip install auto-gptq
pip install unidecode
tritonserver --model-repository=/models
```

## Préparation du Client avec docker 

```bash
cd client
docker run -it --net=host -p8501:8501 -v%cd%:/workspace/ nvcr.io/nvidia/tritonserver:23.07-py3-sdk bash

pip install torch torchvision
pip install transformers
pip install Image
pip install scikit-learn
pip install -U sentence-transformers
python3 test_it
```

