# Triton Server

### Doc

[https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md](https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md)

---

## Dans un premier temps il faut se login sur GNC nvidia pour pouvoir pull l’image docker de triton

Une fois son compte créer etc on génére une clé API 

on fais 

```bash
docker [ncvr.io](http://ncvr.io) login
```

En username on met :

- $oauthtoken

et en password on mets son API KEY

### Getting started

Il faut aussi mettre à jour le PATH pour le stockage du modèle 

Dans le cas du tuto le mieux c’est de DL le modèle directement ici je prend le ONNX densenet 


```bash
# ONNX densenet
mkdir -p model_repository/densenet_onnx/1
wget -O model_repository/densenet_onnx/1/model.onnx \
     https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx
```

Je le télécharge à la main et je le mets dans le bon dossier

- Je crée un dossier /1 et ensuite je mets le modèle en renommant modèle.onnx


- Triton version : [nvcr.io/nvidia/tritonserver:23.07-py3](http://nvcr.io/nvidia/tritonserver:23.07-py3)

### Lancer et préparer le docker du serveur triton

Serveur docker pour tritton 

`docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $%cd%:/workspace/ -v $%cd%/model_repository:/models nvcr.io/nvidia/tritonserver:23.07-py3 bash`

```bash
# Pull and run the Triton container & replace yy.mm 
# with year and month of release. Eg. 23.05
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v%cd%:/workspace/ -v%cd%/model_repository:/models nvcr.io/nvidia/tritonserver:23.07-py3 bash
# Install dependencies
pip install torch torchvision
pip install transformers
pip install Image
pip install sckikit-learn

# Launch the server
tritonserver --model-repository=/models
```

### Préparation du Client avec docker toujours

```bash
docker run -it --net=host -v%cd%:/workspace/ nvcr.io/nvidia/tritonserver:23.07-py3-sdk bash
# Install dependencies
pip install torch torchvision
pip install transformers
pip install Image
pip install sckikit-learn


# Run the client

python3 client.py --model_name "model_name"
```