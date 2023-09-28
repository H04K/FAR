import os
import argparse

def create_model_folder(model_name):
    # Create the model_repository folder if it doesn't exist
    if not os.path.exists("model_repository"):
        os.mkdir("model_repository")
    
    # Create a folder with the given model_name
    model_folder = os.path.join("model_repository", model_name)
    os.mkdir(model_folder)

    # Create a config.pbtxt file with model_name inserted
    config_file_content = """
name: "{}"
backend: "python"
input [
  {{
    name: "input_data"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }}
]
output [
  {{
    name: "embedding"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }}
]
instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
""".format(model_name)

    config_file_path = os.path.join(model_folder, "config.pbtxt")
    with open(config_file_path, "w") as config_file:
        config_file.write(config_file_content)

    # Create a subfolder "1"
    subfolder_path = os.path.join(model_folder, "1")
    os.mkdir(subfolder_path)

    # Create a Python file inside the subfolder
    python_file_content = """\
import torch
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        pass
    
    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "input_data")

            if inp is not None:
                # Do stuff (you can add your logic here)
                sentence_embedding = np.zeros((768,), dtype=np.float32)
                inference_response = pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor(
                        "embedding",
                        sentence_embedding
                    )
                ])
                responses.append(inference_response)

        return responses

    def finalize(self):
        pass
"""
    python_file_path = os.path.join(subfolder_path, "model.py")
    with open(python_file_path, "w") as python_file:
        python_file.write(python_file_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Triton model folder structure.")
    parser.add_argument("model_name", type=str, help="Name of the model")
    args = parser.parse_args()

    create_model_folder(args.model_name)
    print(f"Model folder '{args.model_name}' created successfully in 'model_repository'.")
