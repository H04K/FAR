import numpy as np
from PIL import Image
import tritonclient.http as httpclient
import requests
import argparse
from transformers import BertTokenizer,AutoTokenizer
import numpy as np
from unidecode import unidecode
from sklearn.metrics.pairwise import cosine_similarity
client = httpclient.InferenceServerClient(url="localhost:8000")
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ", use_fast=True)


def infer__llama(text):
    
    np_input_data = np.asarray([str.encode(text)])
    # Set Inputs
    input_tensors = [
        httpclient.InferInput("input_data", [1], datatype="BYTES"),
    ]
    input_tensors[0].set_data_from_numpy(np_input_data.reshape([1]), binary_data=False)
    # Set outputs
    output_llama = [
        httpclient.InferRequestedOutput("output_data")
    ]
    # Query
    query_response = client.infer(model_name="llama2",inputs=input_tensors,outputs=output_llama)
    # Output    
    embedding = query_response.as_numpy("output_data")
    return  embedding


def main(model_name):
    if model_name == "llama2":
        text = """['atendeu as minhas necessidades gostei muito \r\n\r\n\r\n ',
        'As ferramentas do aplicativo não são eficases. Solicitei o concelamento de uma compra antes da nota fiscal ser emitida a solicitacao foi aceita porém não foi efetivada. O produto chegou .',
        'ADORO ESSA LOA, SUPER RECOMENDO',
        'Não a loja melhor de se comprar rápido e ágil nas entregar parabéns ',
        'Só estou esperando a outra mercadoria que ainda não chegou foi comprada primeiro do que a que chegou. ',
        'de parabéns lannister e baratheon',
        'amei ',
        'tudo que eu compro sempre chegou sem problema algum aqui na minha residência. o carteiro reclamou de não ter ninguém em casa, e que na rua não passa carro. nada procede...covarde!!! ele não veio mesmo',
        'muito boa, entrega rapido e sem complicações',
        'produto entregue no prazo recomendo o site',
        'comprei, já paguei e ainda nã recebi',
        'Fico muito satisfeita em comprar mas lojas lannister',
        'Ainda não recebi chegou. Ultimamente os correios estão atrasando muito']"""
        text = unidecode(text)
        print(infer__llama(text))
    else:
        print("Please select between python_vit and python_bert")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Select between enemble_model and python_vit")
    args = parser.parse_args()
    main(args.model_name)