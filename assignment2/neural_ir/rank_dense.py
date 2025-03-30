import argparse
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from neural_ir.index import Faiss
from neural_ir.models.dense_encoder import DenseBiEncoder
from neural_ir.utils import write_trec_run
from neural_ir.utils.dataset_utils import read_pairs
from neural_ir.utils.preprocess import preprocess_text, query_expansion

parser = argparse.ArgumentParser(description="Ranking with BiEncoder")
parser.add_argument(
    "--c", type=str, default="data/collection.tsv", help="path to document collection"
)
parser.add_argument(
    "--q", type=str, default="data/test_queries.tsv", help="path to queries"
)
parser.add_argument(
    "--device", type=str, default="cuda", help="device to run inference"
)
parser.add_argument("--bs", type=int, default=16, help="batch size")
parser.add_argument(
    "--checkpoint",
    default="output/dense/model",
    type=str,
    help="path to model checkpoint",
)
parser.add_argument(
    "--o",
    type=str,
    default="output/dense/test_run.trec",
    help="path to output run file",
)
parser.add_argument(
    "--preprocess_d",
    action="store_true",
    help="Preprocess document text",
)
parser.add_argument(
    "--preprocess_q",
    action="store_true",
    help="Preprocess query text",
)
# NOTE: query expansion implies query preprocessing!
parser.add_argument(
    "--query_expansion",
    action="store_true",
    help="Use query expansion",
)
parser.add_argument(
    "--normalize_enc",
    action="store_true",
    help="normalize the encoder output",
)
args = parser.parse_args()
# print args
print(args)

docs = read_pairs(args.c)
queries = read_pairs(args.q)

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = DenseBiEncoder.from_pretrained(args.checkpoint, args.normalize_enc).to(
    args.device
)
model.eval()
query_embs = []
docs_embs = []
doc_ids = []
for idx in tqdm(
    range(0, len(docs), args.bs), desc="Encoding documents", position=0, leave=True
):
    batch = docs[idx : idx + args.bs]
    docs_texts = [e[1] for e in batch]
    if args.preprocess_d:
        docs_texts = [
            preprocess_text(text) for text in docs_texts
        ]  # Preprocess document text
    doc_ids.extend([e[0] for e in batch])
    docs_inps = tokenizer(
        docs_texts, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.cuda.amp.autocast(), torch.no_grad():
        batch_embs = model.encode(**docs_inps).to("cpu")
        docs_embs.append(batch_embs)

index = Faiss(d=docs_embs[0].size(1))
docs_embs = torch.cat(docs_embs, dim=0).numpy().astype("float32")
index.add(docs_embs)
# ?for batch_embds in tqdm(docs_embs, desc="Indexing document embeddings"):
# index.add(batch_embs.numpy().astype("float32"))

run = defaultdict(list)
queries_embs = []
for idx in tqdm(
    range(0, len(queries), args.bs),
    desc="Encoding queries and search",
    position=0,
    leave=True,
):
    batch = queries[idx : idx + args.bs]
    query_texts = [e[1] for e in batch]
    if args.preprocess_q or args.query_expansion:
        print("Preprocessing query text")
        print("Example query text: ", query_texts[0], query_texts[1])
        query_texts = [
            preprocess_text(text) for text in query_texts
        ]  # Preprocess query text
        print("Preprocessed query text: ", query_texts[0], query_texts[1])
    if args.query_expansion:
        query_texts = [query_expansion(text) for text in query_texts]
        print("Expanded query text: ", query_texts[0], query_texts[1])

    query_inps = tokenizer(
        query_texts, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.cuda.amp.autocast(), torch.no_grad():
        batch_query_embs = (
            model.encode(**query_inps).to("cpu").numpy().astype("float32")
        )
    scores, docs_idx = index.search(batch_query_embs, 1000)
    for idx in range(len(batch)):
        query_id = batch[idx][0]
        for i, score in zip(docs_idx[idx], scores[idx]):
            if i < 0:
                continue
            doc_id = doc_ids[i]
            run[query_id].append((doc_id, score))

write_trec_run(run, args.o)
