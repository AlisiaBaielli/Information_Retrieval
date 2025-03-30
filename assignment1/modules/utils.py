import os
import requests
from tqdm import tqdm
import zipfile


def download_dataset():
    folder_path = os.environ.get("IR1_DATA_PATH")
    if not folder_path:
        folder_path = "./datasets/"
    os.makedirs(folder_path, exist_ok=True)

    file_location = os.path.join(folder_path, "cacm.zip")

    # download file if it doesn't exist
    if not os.path.exists(file_location):

        url = "https://surfdrive.surf.nl/files/index.php/s/M0FGJpX2p8wDwxR/download"

        with open(file_location, "wb") as handle:
            print(f"Downloading file from {url} to {file_location}")
            response = requests.get(url, stream=True)
            for data in tqdm(response.iter_content()):
                handle.write(data)
            print("Finished downloading file")

    if not os.path.exists(os.path.join(folder_path, "train.txt")):
        # unzip file
        with zipfile.ZipFile(file_location, "r") as zip_ref:
            zip_ref.extractall(folder_path)


def print_results(docs, docs_by_id, len_limit=50):
    for i, (doc_id, score) in enumerate(docs):
        doc_content = (
            docs_by_id[doc_id].strip().replace("\n", "\\n")[:len_limit] + "..."
        )
        print(f"Rank {i}({score:.2}): {doc_content}")


def write_trec_run(results, outfn, tag="trecrun"):
    with open(outfn, "wt") as outf:
        qids = sorted(results.keys())
        for qid in qids:
            rank = 1
            for docid, score in sorted(results[qid], key=lambda x: x[1], reverse=True):
                print(f"{qid} Q0 {docid} {rank} {score} {tag}", file=outf)
                rank += 1
