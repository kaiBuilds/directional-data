from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from tqdm import tqdm

MAX_ENTRIES = 2_500
BATCH_SIZE = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

wikidata = load_dataset("wikipedia", "20220301.simple")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(DEVICE)


def get_embeddings(text: str) -> torch.Tensor:
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
    return torch.mean(embeddings, dim=1, keepdim=True)


def process_entry(entry: dict) -> dict:
    title_embedding = reduce_embeddings(get_embeddings(entry["title"]))
    text = [x for x in entry["text"].split("\n") if x != ""]
    sentence_embeddings = get_embeddings(text)
    text_embedding = reduce_embeddings(sentence_embeddings)
    return {"id": entry["id"], "title": title_embedding, "text": text_embedding}


def get_embeddings_dataset(dataset: dict) -> list:
    data = []
    for index, entry in tqdm(enumerate(dataset)):
        if index >= MAX_ENTRIES:
            return data
        data.append(process_entry(entry))
    return data


def get_embedding_angles(dataset: dict, MAX_ENTRIES: int = MAX_ENTRIES) -> list:
    data = []
    cos = torch.nn.CosineSimilarity(dim=-1)
    for title_idx in tqdm(range(len(dataset))):
        for text_idx in range(title_idx, min(len(dataset), MAX_ENTRIES)):
            title_embedding = dataset[title_idx]["title"]
            text_embedding = dataset[text_idx]["text"]
            angle = torch.arccos(cos(title_embedding, text_embedding))
            data.append(
                {
                    "title_id": dataset[title_idx]["id"],
                    "text_id": dataset[text_idx]["id"],
                    "angle": angle,
                }
            )
    return data


if __name__ == "__main__":

    embeds = get_embeddings_dataset(wikidata["train"])
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeds, f)

    angle_data = get_embedding_angles(embeds)
    with open("angles.pkl", "wb") as f:
        pickle.dump(angle_data, f)
