# %%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import os

# %%
load_dotenv()
MAX_ENTRIES = 250
QUERIES = 10
BATCH_SIZE = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

wikidata = load_dataset("wikipedia", "20220301.simple")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(DEVICE)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# %%
def get_embeddings(text: str) -> torch.Tensor:
    """
    Get the embeddings for the given text using a BERT model.

    Args:
        text (str): The input text.

    Returns:
        torch.Tensor: The embeddings of the text.
    """
    with torch.no_grad():
        inputs = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        outputs = model.forward(**inputs)

    return outputs.last_hidden_state


def reduce_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Reduce the input embeddings tensor for multiple sentences to a single embedding
    by taking the mean along the specified dimensions.

    Args:
        embeddings (torch.Tensor): The input embeddings tensor.

    Returns:
        torch.Tensor: The reduced embeddings tensor.

    """
    return torch.mean(torch.mean(embeddings, dim=1, keepdim=True), dim=0, keepdim=True)


def process_entry(entry: dict) -> dict:
    """
    Process an entry and return a record with the entry's ID, title embedding, and text embedding.

    Args:
        entry (dict): The entry to be processed, containing the keys "id", "title", and "text".

    Returns:
        dict: A dictionary with the entry's ID, title embedding, and text embedding.
    """
    title_embedding = reduce_embeddings(get_embeddings(entry["title"]))
    text = [x for x in entry["text"].split("\n") if x != ""]
    sentence_embeddings = get_embeddings(text)
    text_embedding = reduce_embeddings(sentence_embeddings)
    return {"id": entry["id"], "title": title_embedding, "text": text_embedding}


def get_embeddings_dataset(dataset: dict) -> list:
    """
    Retrieves embeddings dataset from the given dictionary limiting to the number of entries
    to the amount we want in the experiment.

    Args:
        dataset (dict): The input dictionary containing the dataset.

    Returns:
        list: The embeddings dataset.

    """
    data = []
    for index, entry in tqdm(enumerate(dataset)):
        if index >= MAX_ENTRIES:
            return data
        data.append(process_entry(entry))
    return data


def get_embedding_angles(queries: list, documents: list) -> list:
    """
    Calculates the embedding angles between queries and documents.

    Args:
        queries (list): A list of query objects.
        documents (list): A list of document objects.

    Returns:
        list: A list of records (dicts) containing the query ID, document ID, query embedding,
              document text, and the embedding angles phi_q_t and phi_q_d.
    """
    records = []
    cos = torch.nn.CosineSimilarity(dim=-1)
    for q in tqdm(queries):
        for d in documents:
            records.append(
                {
                    "query_id": q["id"],
                    "document_id": d["id"],
                    "x_q": q["embedding"],
                    "x_t": d["text"].to("cpu"),
                    "phi_q_t": torch.arccos(
                        cos(q["embedding"].to(DEVICE), d["title"].to(DEVICE))
                    ).to("cpu"),
                    "phi_q_d": torch.arccos(
                        cos(q["embedding"].to(DEVICE), d["text"].to(DEVICE))
                    ).to("cpu"),
                }
            )
    return records


def process_query(idx: int, text: str) -> dict:
    """
    Processes a query by removing a prefix the LLM generates at a time then applying the method to obtain its embedding.

    Args:
        idx (int): The ID of the query.
        text (str): The text of the query.

    Returns:
        dict: A dictionary containing the query ID, the original text, and the text embedding.
    """
    text = text.replace("Summary: ", "")
    sentence_embeddings = get_embeddings(text)
    text_embedding = reduce_embeddings(sentence_embeddings)
    return {"id": idx, "query": text, "embedding": text_embedding.to("cpu")}


def build_queries(n=QUERIES):
    """
    Builds a list of queries using OpenAI API.

    Args:
        n (int): The number of queries to build. Defaults to QUERIES.

    Returns:
        list: A list of dictionaries representing the queries. Each dictionary
              contains an 'idx' key representing the index and a 'text' key
              representing the query text.
    """
    queries = [
        {"idx": idx, "text": get_query(wikidata["train"]["text"][idx])}
        for idx in torch.randint(low=0, high=MAX_ENTRIES, size=(n,)).tolist()
    ]
    return queries


def get_query(text: str) -> str:
    """
    Creates a query from the given text by summarizing it using the OpenAI API.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized response.

    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that summarizes the text sent to you by the user into a single sentence. You respond very succintly with only the answer from to the request.",
            },
            {"role": "user", "content": f"{text}"},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":

    # Get all document embeddings
    embeds = get_embeddings_dataset(wikidata["train"])
    pretty = [
        {
            "documnent_id": e["id"],
            "x_t": e["title"].to("cpu"),
            "x_d": e["text"].to("cpu"),
        }
        for e in embeds
    ]
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(pretty, f)

    # Get all angles between embeddings
    all_angles = []
    cos = torch.nn.CosineSimilarity(dim=-1)
    for t in pretty:
        for d in pretty:
            all_angles.append(
                {
                    "id_t": t["documnent_id"],
                    "id_d": d["documnent_id"],
                    "phi_d_t": torch.arccos(
                        cos(t["x_t"].to(DEVICE), d["x_d"].to(DEVICE))
                    ).to("cpu"),
                }
            )
    with open("title_document_empirical_dist.pkl", "wb") as f:
        pickle.dump(all_angles, f)

    # Genereate queries and get their embeddings
    queries = build_queries()
    query_embeddings = [process_query(idx=q["idx"], text=q["text"]) for q in queries]

    with open("queries.pkl", "wb") as f:
        pickle.dump(query_embeddings, f)

    angle_data = get_embedding_angles(query_embeddings, embeds)
    with open("angles.pkl", "wb") as f:
        pickle.dump(angle_data, f)
