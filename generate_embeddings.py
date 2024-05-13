#use this to generate embedings for the prompts and responses in the dataset using SBERT 
import numpy as np 
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import csv


"""
Function to generate embeddings
input:
    data: list of strings
    embedding_model_name: name of the SBERT model to use
"""
def gen_embeddings(data, embedding_model_name):
    # Load the SBERT model
    model = SentenceTransformer(embedding_model_name)
    # Compute embeddings
    embeddings = model.encode(data)
    return embeddings
    


"""
Create a new csv file with embedded versions of the prompt and response 
change the :100 index, still a work in progress, generated embeddings for only the first 100 rows
"""
def create_embedded_dataset(input_file_path, output_file_path, prompt_emb_model, response_emb_model):
    df = pd.read_csv(input_file_path)
    
    prompts_embeddings = gen_embeddings(df['prompt'].values, prompt_emb_model)
    responses_a_embeddings = gen_embeddings(df['response_a'].values, response_emb_model)
    response_b_embeddings = gen_embeddings(df['response_b'].values, response_emb_model)

    ids = df['id'].values[:100]
    output = [[ids[i], prompts_embeddings[i], responses_a_embeddings[i], response_b_embeddings[i]] for i in range(len(ids))]
    
    fields = ['id', 'prompt', 'response_a_emb', 'response_b_emb']
    with open(output_file_path, 'w') as f:
     
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(output)
    

def main():
    create_embedded_dataset('train.csv', "output2.csv", "all-MiniLM-L6-v2", "all-MiniLM-L6-v2")



if __name__ == "__main__":
    main()