import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "model"
from unidecode import unidecode

use_triton = False
import numpy as np
class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=use_triton,
            quantize_config=None)

        self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.9,
                top_p=0.6,
                repetition_penalty=1.15
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            input_data = pb_utils.get_input_tensor_by_name(request, "input_data").as_numpy()
            if input_data is not None:
                input_text = input_data[0].decode('utf-8')
                # Appeler la fonction predict pour générer une réponse en fonction de l'entrée
                input_text = self.unicode_to_ascii(input_text)
                output_text = self.predict(input_text)
                output_text = self.unicode_to_ascii(output_text)
                print(output_text)
                
                #encode the output
                output_text_bytes = output_text.encode('utf-8')
                # Créer une réponse
                output_tensors = [
                pb_utils.Tensor("output_data", np.array([output_text_bytes], dtype=object))
                ]

                inference_response = pb_utils.InferenceResponse(output_tensors)
                responses.append(inference_response)

        return responses

    def predict(self, input_text):
        #input_prompt = f"[INST] <<SYS>>\nTu es un assistant de vente et d'analyse répond le plus mieux possible à la question suivante en analysant les avis pour les produits suivant\n<</SYS>>\n\n "
        #input_prompt = input_prompt + str("Qu'en pense les clients? ") + str(input_text) + " [/INST] "

        input_prompt = self.unicode_to_ascii(input_text)
        resp = self.pipe(input_prompt)
        resp = resp[0]['generated_text']
        

        return resp

    def unicode_to_ascii(self, s):
        return unidecode(s)
    