import os
import pickle
import json
from argparse import Namespace
import torch
from ltr.utils import seed, create_results
from ltr.dataset_new import (
    Queries, Preprocess, Documents, FeatureExtraction, GenerateFeatures, DataSet
)
from ltr.model import LTRModel
from ltr.train import train_pointwise, train_pairwise_spedup, train_listwise

COLLECTION_PATH = "./analysis_data/NUTRITION/collection.tsv"
QUERIES_PATH = "./analysis_data/NUTRITION/queries.tsv"
TRAIN_PATH = "./analysis_data/NUTRITION/train_pairs_graded.tsv"
DEV_PATH = "./analysis_data/NUTRITION/dev_pairs_graded.tsv"
TEST_PATH = "./analysis_data/NUTRITION/test_pairs_graded.tsv"
STOP_WORDS_PATH = "./data/common_words"
DOC_JSON = "./datasets/doc.pickle"
OUTPUT_JSON = "./outputs/analysis_res_n.json"

MODEL_SAVE_DIR = "./analysis_data/outputs_nutrition_new/"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


seed(42)

prp = Preprocess(STOP_WORDS_PATH)
queries = Queries(prp)
queries.preprocess_queries(QUERIES_PATH)

RESET = True
if os.path.exists(DOC_JSON) and not RESET:
    with open(DOC_JSON, "rb") as file:
        documents = pickle.load(file)
else:
    documents = Documents(prp)
    documents.process_documents(COLLECTION_PATH)
    with open(DOC_JSON, "wb") as file:
        pickle.dump(documents, file)

N_FEATURES = 17
feature_ex = FeatureExtraction({}, documents, queries)
feat_gen = GenerateFeatures(feature_ex)
args = {"k1": 1.5, "b": 0.75, "idf_smoothing": 0.5}

# Uncomment and run when adding a new feature
feat_gen.run(TRAIN_PATH, TRAIN_PATH + "g", **args)
feat_gen.run(DEV_PATH, DEV_PATH + "g", **args)
feat_gen.run(TEST_PATH, TEST_PATH + "g", **args)

fold_paths = ["./analysis_data/NUTRITION/"]
data = DataSet(
    "ir1-2023", fold_paths,
    num_rel_labels=5,  
    num_features=N_FEATURES,
    num_nonzero_feat=N_FEATURES
)

data = data.get_data_folds()[0]
data.read_data()

params_regr = Namespace(epochs=11, lr=1e-3, batch_size=1, metrics={"ndcg", "precision@05", "recall@05"})
pointwise_model = LTRModel(data.num_features)
pointwise_results = create_results(
    data, pointwise_model, train_pointwise, pointwise_model, 
    os.path.join(MODEL_SAVE_DIR, "pointwise_res.json"), params_regr
)
torch.save(pointwise_model.state_dict(), os.path.join(MODEL_SAVE_DIR, "pointwise_model"))

params_pairwise = Namespace(epochs=11, lr=1e-3, batch_size=1, metrics={"ndcg", "precision@05", "recall@05"})
pairwise_model = LTRModel(N_FEATURES)
pairwise_results = create_results(
    data, pairwise_model, train_pairwise_spedup, pairwise_model, 
    os.path.join(MODEL_SAVE_DIR, "pairwise_spedup_res.json"), params_pairwise
)
torch.save(pairwise_model.state_dict(), os.path.join(MODEL_SAVE_DIR, "pairwise_spedup_model"))

params_listwise = Namespace(epochs=11, lr=1e-4, batch_size=1, metrics={"ndcg", "precision@05", "recall@05"})
listwise_model = LTRModel(N_FEATURES)
listwise_results = create_results(
    data, listwise_model, train_listwise, listwise_model, 
    os.path.join(MODEL_SAVE_DIR, "listwise_res.json"), params_listwise
)
torch.save(listwise_model.state_dict(), os.path.join(MODEL_SAVE_DIR, "listwise_model"))

final_results = {
    "pointwise": pointwise_results,
    "pairwise_spedup": pairwise_results,
    "listwise": listwise_results,
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(final_results, f, indent=4)

print(f"All results saved to {OUTPUT_JSON}")
print(f"Models saved in {MODEL_SAVE_DIR}")

