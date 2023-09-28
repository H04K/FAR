import numpy as np
from PIL import Image
import tritonclient.http as httpclient
import requests
import argparse
import numpy as np
from unidecode import unidecode
client = httpclient.InferenceServerClient(url="localhost:8000")

import pandas as pd
import os
import json
import gradio as gr
import requests
import re

# Define the S3 bucket and file paths
s3_bucket = "s3://nan-olist-data/"
s3_data = {
    'customers': f"{s3_bucket}olist_customers_dataset.csv",
    'geolocation': f"{s3_bucket}olist_geolocation_dataset.csv",
    'order_items': f"{s3_bucket}olist_order_items_dataset.csv",
    'order_payments': f"{s3_bucket}olist_order_payments_dataset.csv",
    'order_reviews': f"{s3_bucket}olist_order_reviews_dataset.csv",
    'orders': f"{s3_bucket}olist_orders_dataset.csv",
    'products': f"{s3_bucket}olist_products_dataset.csv",
    'sellers': f"{s3_bucket}olist_sellers_dataset.csv",
    'product_cat_translation': f"{s3_bucket}product_category_name_translation.csv",
}

date_cols = [
    'review_creation_date',
    'review_answer_timestamp'
]

# Load the necessary datasets
order_reviews_df = pd.read_csv(s3_data['order_reviews'], parse_dates=date_cols)
products_df = pd.read_csv(s3_data["order_items"])
product_infos_df = pd.read_csv(s3_data["products"])


# Fusionner order_reviews_df et products_df sur la colonne 'order_id'
merged_df = order_reviews_df.merge(products_df, on='order_id')

# Fusionner merged_df avec product_infos_df sur la colonne 'product_id'
merged_df = merged_df.merge(product_infos_df, on='product_id')

# Supprimer les lignes avec des avis manquants (NaN) dans 'review_comment_message'
merged_df = merged_df.dropna(subset=['review_comment_message'])

# Sélectionner uniquement les colonnes 'product_id', 'review_comment_message' et 'product_category_name'
merged_df = merged_df[['product_id', 'review_comment_message', 'product_category_name']]

# Group by 'product_id' and aggregate reviews
product_reviews = merged_df.groupby('product_id').agg({
    'review_comment_message': list,
    'product_category_name': 'first',  # Select the first category name, assuming it's the same for all rows of a product
}).reset_index()

# Count the number of reviews for each product
product_reviews['review_count'] = product_reviews['review_comment_message'].apply(len)

# Filter products with more than 5 reviews
product_reviews = product_reviews[product_reviews['review_count'] > 5]

# Drop the 'review_count' column if you no longer need it
product_reviews.drop('review_count', axis=1, inplace=True)

# Reset the index
product_reviews.reset_index(drop=True, inplace=True)

# Display the result
print(product_reviews)

system_message ="""Vous êtes un assistant utile, respectueux et honnête. Répondez toujours de manière aussi utile que possible tout en étant sûr. Incorporez les informations principales du texte pour expliquer le ressenti du client à propos du produit. Mettez en évidence les sentiments, les opinions et les expériences clés exprimées par le client. Assurez-vous de résumer les éléments qui décrivent le produit de manière positive ou négative, les caractéristiques qui suscitent des émotions particulières et les points forts ou les faiblesses mentionnés. Fournissez une vue d'ensemble concise et cohérente du ressenti global du client concernant le produit
GENERE EN FRANCAIS
"""



def get_reviews_by_product_id(product_id, product_reviews):
    # Filter the DataFrame to include only reviews for the specified product_id
    filtered_df = merged_df[merged_df['product_id'] == product_id]

    # Get all review comments for the specified product_id
    review_comments = filtered_df['review_comment_message'].tolist()

    return review_comments

def get_product_category(product_id):
    product = product_reviews[product_reviews['product_id'] == product_id]
    if not product.empty:
        return product['product_category_name'].values[0]
    else:
        return "Product not found"
def display_category(product_id):
    selected_product_id = product_id
    category = get_product_category(selected_product_id)
    name = "A définir"
    desc = "A definir"
    return category,name,desc
def predict(message, chatbot):
  input_prompt = f"[INST] <<SYS>>\nTu es un assistant pour des employés dans une entreprise de Ecommerce répond à leur demande et leur instruction en restant poli courtois et le plu sjuste possible"
  for interaction in chatbot:
    input_prompt = input_prompt
    input_prompt = input_prompt + str(message) + " [/INST] "
    input_prompt = unidecode(input_prompt)

  resp = infer_llama(input_prompt)
  resp = resp[0][len(input_prompt):].strip()
  return resp

def predict_analyse(product_id):
  input_prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n "
  input_prompt = input_prompt
  reviews = get_reviews_by_product_id(product_id,product_reviews)
  input_prompt = input_prompt + str(reviews) + " [/INST] "
  input_prompt = unidecode(input_prompt)
  resp = infer_llama(input_prompt)
  resp = resp[0][len(input_prompt):].strip()

  return resp
def get_reviews_string_by_category(category_name):
    # Filtrer le dataframe product_reviews pour obtenir les avis des produits de la catégorie spécifiée
    category_products = product_reviews[product_reviews['product_category_name'] == category_name]

    # Créer une chaîne de caractères qui regroupe les ID de produit avec leurs avis
    reviews_string = ""
    for index, row in category_products.iterrows():
        product_id = row['product_id']
        reviews = row['review_comment_message']
        reviews_string += f"Product ID: {product_id} [Reviews: {', '.join(reviews)}]\n"

    return reviews_string[:1000]
def predict_analyse_cat(categories,quest):
  input_prompt = f"[INST] <<SYS>>\nTu es un assistant de vente et d'analyse répond le plus mieux possible à la question suivante en analysant les avis pour les produits suivant\n<</SYS>>\n\n "
  input_prompt = input_prompt
  reviews= get_reviews_string_by_category(categories)
  input_prompt = input_prompt + str(quest) + str(reviews) + " [/INST] "
  input_prompt = unidecode(input_prompt)

  resp = infer_llama(input_prompt)
  resp = resp[0][len(input_prompt):].strip()

  return resp



# Defining Ids etc
ids = product_reviews["product_id"].apply(lambda x: str(x))
ids = list(ids)
categories = product_reviews["product_category_name"].apply(lambda x: str(x))
categories = list(categories)


# Defining Ids etc
ids = product_reviews["product_id"].apply(lambda x: str(x))
ids = list(ids)
categories = product_reviews["product_category_name"].apply(lambda x: str(x))
categories = list(categories)


title = "Llama2 13 Chatbot"
description = """
"""
css = """.toast-wrap { display: none !important } """

# on peut mettre ici des exemples de phrases / instructions pour généraliser les tests
examples=[
"""hello"""]
def infer_llama(text):
    text = unidecode(text)
    print(text)
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
with gr.Blocks() as demo:
  with gr.Tab("Afficher les informations du produits"):
    product_id = gr.Dropdown(ids,label = "Id des produit")
    btn = gr.Button(value='Afficher les informations')
    name = gr.Textbox(value="", label="Nom")
    cat = gr.Textbox(value="", label="Catégorie")
    desc = gr.Textbox(value="", label="Description")
    btn.click(display_category,inputs = [product_id],outputs=[cat,name,desc])
    analyze = gr.Button(value="Qu'en pense les clients")
    Analyse = gr.Textbox(value='',label="Analyse")
    analyze.click(predict_analyse,inputs = [product_id],outputs=[Analyse])
  with gr.Tab("Outils Analyse Avis"):
      gr.ChatInterface(predict, css=css)
  with gr.Tab("labo"):
    #categories
    categories = gr.Dropdown(categories,label = "catégories")
    quest = gr.Textbox(value='',label="question")
    analyze = gr.Button(value="Envoyer question")
    Analyse = gr.Textbox(value='',label="Analyse")
    analyze.click(predict_analyse_cat,inputs = [categories,quest],outputs=[Analyse])


demo.launch(debug=True,share=True)