import argparse
from collections import defaultdict

import ir_measures
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from neural_ir.index import Faiss
from neural_ir.models.cross_encoder import CrossEncoder
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
    "--run",
    type=str,
    default="data/test_bm25.trec",
    help="path to the run file of a first-stage ranker (BM25, Dense, Sparse)",
)
parser.add_argument(
    "--device", type=str, default="cuda", help="device to run inference"
)
parser.add_argument("--bs", type=int, default=16, help="batch size")
parser.add_argument(
    "--checkpoint",
    default="output/ce/model",
    type=str,
    help="path to model checkpoint",
)
parser.add_argument(
    "--o",
    type=str,
    default="output/ce/test_run.trec",
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
args = parser.parse_args()
# print args
print(args)

docs = dict(read_pairs(args.c))
queries = dict(read_pairs(args.q))
if args.preprocess_d:
    docs = [(docid, preprocess_text(doc)) for docid, doc in docs.items()]
    docs = dict(docs)
if args.preprocess_q or args.query_expansion:
    print("Preprocessing queries")
    print("Example query:", list(queries.items())[0])
    queries = [(qid, preprocess_text(query)) for qid, query in queries.items()]
    queries = dict(queries)
    print("Example query after preprocessing:", list(queries.items())[0])
if args.query_expansion:
    queries = [(qid, query_expansion(query)) for qid, query in queries.items()]
    queries = dict(queries)
    print("Example query after query expansion:", list(queries.items())[0])

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = CrossEncoder.from_pretrained(args.checkpoint).to(args.device)
model.eval()

pairs_text = []
pairs_id = []
run = ir_measures.read_trec_run(args.run)
count_docs = defaultdict(lambda: 0)
for pair in tqdm(run, desc=f"Reading pairs from {args.run}", position=0, leave=True):
    if count_docs[pair.query_id] < 100:
        pairs_id.append((pair.query_id, pair.doc_id))
        pairs_text.append((queries[pair.query_id], docs[pair.doc_id]))
        count_docs[pair.query_id] += 1

results = defaultdict(list)
for idx in range(0, len(pairs_text), args.bs):
    batch_pairs_text = pairs_text[idx : idx + args.bs]
    batch_pairs_id = pairs_id[idx : idx + args.bs]
    batch_inps = tokenizer(
        batch_pairs_text, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        scores = model.score_pairs(batch_inps).to("cpu").tolist()
    for pairid, score in zip(batch_pairs_id, scores):
        qid, did = pairid
        results[qid].append((did, score))
write_trec_run(results, args.o)
