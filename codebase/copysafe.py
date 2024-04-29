from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    AutoModel,
    RobertaTokenizer,
    RobertaModel,
    XLNetTokenizer,
    XLNetModel,
    MarianMTModel,
    MarianTokenizer,
)
from sklearn.metrics import f1_score
from langdetect import detect
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import torch


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data = data["abstract"].dropna()
    return data


# Load and preprocess data
source_data = load_and_preprocess_data(filepath="../dataset/data.csv")

# def create_source_vectors(data, model, tokenizer, max_length=512, device=None):
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Move the model to the specified device
#     model.to(device)

#     vectors = []

#     # Set the model to evaluation mode
#     model.eval()

#     with torch.no_grad():
#         for text in tqdm(data[:5000]):
#             # Tokenize and handle sequences longer than max_length
#             input_ids = tokenizer.encode(
#                 text, add_special_tokens=True, max_length=max_length, truncation=True
#             )
#             input_ids = torch.tensor([input_ids]).to(device)

#             # Generate document vector
#             vector = model(input_ids)[0].mean(dim=1).squeeze().detach().cpu().numpy()
#             vectors.append(vector)

#     return vectors

def generate_document_vector(text, model, tokenizer, max_chunk_length=512, overlap=50):
    # Tokenize the text into chunks with specified overlap
    tokens = tokenizer.tokenize(
        tokenizer.decode(tokenizer.encode(text, add_special_tokens=True))
    )
    chunks = [
        tokens[i : i + max_chunk_length]
        for i in range(0, len(tokens), max_chunk_length - overlap)
    ]

    # Initialize an empty tensor to store the embeddings
    embeddings = torch.zeros((1, len(tokens), model.config.hidden_size))

    # Iterate through chunks and generate embeddings
    for i, chunk in enumerate(chunks):
        input_ids = tokenizer.convert_tokens_to_ids(chunk)
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids]))
        last_hidden_states = outputs.last_hidden_state
        embeddings[:, i : i + len(chunk), :] = last_hidden_states

    # Average the embeddings over all tokens
    document_vector = embeddings.mean(dim=1).squeeze().detach().numpy()
    return document_vector


def analysis(document_vector, source_vectors, source_data, threshold):
    similarity_scores = cosine_similarity([document_vector], source_vectors)

    most_similar_index = np.argmax(similarity_scores[0])
    most_similar_score = similarity_scores[0][most_similar_index]
    most_similar_article = source_data[most_similar_index]

    is_matched = most_similar_score > threshold

    return [is_matched, most_similar_score, most_similar_article]


def report_results(is_matched, similarity_scores, document, most_similar_article):
    print("Analysis Results:\n")
    print(f"Similarity Score: {similarity_scores}")
    print(f"Decision: {'Match Detected' if is_matched else 'No match Detected'}")
    print(f"Article submitted: \n{document}")
    if is_matched:
        print(f"Most Similar Article:\n{most_similar_article}")
    else:
        print("\nNo evidence of similar document.")


# Load BERT model and tokenizer
# model = BertModel.from_pretrained("bert-base-uncased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load sciBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


# Load Roberta model and tokenizer
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model = RobertaModel.from_pretrained('roberta-base')


# Load XLNet model and tokenizer
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# model = XLNetModel.from_pretrained('xlnet-base-cased')


# if vector repository is not available then use source_vectors = create_bert_vectors(source_data, model, tokenizer)

# Load the pickle file containing the top 5000 abstracts
# provide the path
with open("embeddings_sciBERT.pkl", "rb") as f:
    source_vectors = pickle.load(f)


def translate_text(input_text, source_lang, target_lang):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    # Load pre-trained translation model and tokenizer
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Tokenize and translate
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    translation_ids = model.generate(input_ids)

    # Decode the translated text
    translated_text = tokenizer.decode(translation_ids[0], skip_special_tokens=True)

    return translated_text


def check_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "unknown"

