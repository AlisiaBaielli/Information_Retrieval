[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nbywSnZM)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=18335923)
# Information Retrieval 1 - Assignment 2  
*Course from the Master Artificial Intelligence at the University of Amsterdam (Edition 2025)*  

## Introduction  
Welcome to IR1 Assignment 2. This assignment is worth 30% of your assignment grade.
In this assignment, you will explore **Ad hoc retrieval**, a task that identifies documents relevant to a (usually new) query, and **Neural Information Retrieval** (NIR) methods that use deep learning to match the content of queries and documents.  

All the methods considered in this assignment are designed for content-based retrieval. That means that the relevance of the documents/passages is ranked based on their content with respect to a given query, as opposed to considering other types of signals such as page rank, user clicks, etc. Queries are typically short and may only contain keywords, while documents/passages are longer and contain more detailed information that may cover different topics. Moreover, documents relevant to a query may match semantically without containing many exact matches with query terms. Bridging the lexical gap between queries and documents is a long-standing challenge for lexical-based methods that rely on exact matches, such as BM25. However, recent advancements in neural methods, aided by modern representation learning techniques like pre-trained transformers, have brought significant improvements over prior methods, such as [query expansion using pseudo-relevance feedback](https://en.wikipedia.org/wiki/Relevance_feedback).

Retrieval methods typically need to handle large document collections containing millions or even billions of documents -- or even the entire public Internet in the case of web search. To ensure low retrieval latency, the retrieval process is typically split into multiple stages. Figure 1 shows a typical pipeline of a modern information retrieval system, which includes two stages: ranking and re-ranking. At the ranking stage, an efficient retrieval system (such as BM25, Dense Retrieval, or Learned Sparse Retrieval) is used to retrieve a set of document candidates, which are then re-ranked by a more expensive but more accurate system (such as Cross-Encoder) in the re-ranking stage.

<img src="images/two-stage-pipeline.svg" alt="Two-stage retrieval pipeline" style="height: 300px; margin-top: -20px; margin-bottom: -20px"/>

In the previous assignment, you implemented BM25, which is one of the most popular ranking methods. In this homework, we will delve into more advanced neural ranking and re-ranking systems, including:

<ul>
    <li>Cross-Encoder for re-ranking</li>
    <li>Dense Bi-Encoder for ranking</li>
    <li>(Learned) Sparse Bi-Encoder for ranking</li>
</ul>

**Objectives**:
- Neural Information Retrieval:
    - Setting up a modern deep learning pipeline for training and evaluation.
    - Implementing state-of-the-art neural information retrieval models.
    - Indexing and efficiently searching documents using vector indexing and inverted indexing methods.

## Scoring and Submission Guidelines for Assignment 2  

To achieve a full score on **Assignment 2**, you must complete both the **implementation** and **analysis** components. The weight distribution is as follows:  

- The **implementation component** accounts for **1/3 of your Assignment 2 grade**. Your score for this component is determined by the number of autograding tests you pass. To maximize your score, ensure that your implementation meets all specified requirements.  
- The **analysis component** constitutes **2/3 of your Assignment 2 grade**. This part will be **manually graded** based on the clarity, depth, and correctness of your explanations and insights. To maximize your score, ensure that your analysis is thorough and well-organized. You will need to have the implementation component finished to complete the analysis.


### Submission Requirements
- You will implement your **code in the provided Python files**.
- Push your code to your **GitHub repository**.
- Write a **concise report in LaTeX** summarizing your findings.  
- Submit your **compiled PDF of the report** via **Canvas** before the deadline.  


## Guidelines

### How to proceed?

The structure of the `neural_ir` package is shown below, containing various sub-packages and modules that you need to implement. For the files that require changes, a :pencil2: icon is added after the file name.

We recommend that you start with the [dataset utils](neural_ir/utils/) module, which contains helper functions for loading TSV files, such as a document/query collection. Then, you can move on to the [dataset](neural_ir/dataset/) modules, which contain the implementation of the pair and triplet datasets. These inherit the PyTorch Dataset class, and are used when training a cross-encoder and a bi-encoder, respectively. Finally, you can implement the actual neural [models](neural_ir/models/), which are a cross-encoder, a dense bi-encoder, and a sparse bi-encoder. For these classes, we are mainly interested in creating a scoring function and the model's forward pass.

This assignment requires knowledge of [PyTorch](https://pytorch.org/). If you are unfamiliar with PyTorch, you can go over [these series of tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### Analysis  

The goal of this analysis is to evaluate **Neural Information Retrieval (NIR)** techniques and improve ranking quality based on relevance metrics. Your report should summarize key findings, justify improvements, and use visual aids such as tables and graphs. The report must be concise and not exceed **1000 words** (excluding tables and visuals).  

You will first evaluate retrieval performance on four variations of `corrupted_data` using all three models—Cross-Encoder, Dense Bi-Encoder, and Sparse Bi-Encoder—without retraining. Analyze how different corruptions affect retrieval performance and discuss model robustness. Next, propose modifications to improve precision and recall while ensuring generalization, saving results in `output/your_creativity/test_run_P.trec` and `output/your_creativity/test_run_R.trec`. Compare these results to the baseline and assess their effectiveness.  

Finally, provide a **concise comparison and discussion** of trade-offs between precision and recall, supported by tables and graphs. Submit the **compiled PDF** via **Canvas** before the deadline.  

Refer to the **complete instructions in the LaTeX template on Canvas** for detailed intructions and submission guidelines.

### Grading
When you push to your repository on GitHub, unit tests will automatically run as in the previous parts. You can view the total points available and the points awarded to your current implementation following the same process before (i.e., click the 'x' or 'checkmark', and then view the output inside the 'Autograding' section).

**Structure of the ``neural_ir`` package:**

📦neural_ir\
 ┣ 📂dataset\
 ┃ ┣ 📜__init__.py\
 ┃ ┣ 📜pair_collator.py\
 ┃ ┣ 📜pair_dataset.py :pencil2: \
 ┃ ┣ 📜triplet_collator.py\
 ┃ ┗ 📜triplet_dataset.py :pencil2: \
 ┣ 📂index\
 ┃ ┣ 📜__init__.py\
 ┃ ┗ 📜vector_index.py\
 ┣ 📂models\
 ┃ ┣ 📜__init__.py\
 ┃ ┣ 📜cross_encoder.py :pencil2: \
 ┃ ┣ 📜dense_encoder.py :pencil2: \
 ┃ ┗ 📜sparse_encoder.py :pencil2: \
 ┣ 📂public_tests\
 ┣ 📂trainer\
 ┃ ┣ 📜__init__.py\
 ┃ ┗ 📜hf_trainer.py\
 ┣ 📂utils\
 ┃ ┣ 📜__init__.py\
 ┃ ┗ 📜dataset_utils.py :pencil2:\
 ┣ 📜rank_dense.py\
 ┣ 📜rank_sparse.py\
 ┣ 📜rerank.py\
 ┗ 📜train.py


The `neural_ir` package is comprised of several sub-packages and modules that are crucial to its functionality. In the following, we'll provide an overview of these components:

- `neural_ir.dataset`: This package contains modules for loading, preprocessing, sampling, and batching datasets for training and inference. The datasets come in two forms: (1) triplet datasets, which consist of (query, positive document, negative document) triplets and are used for training with contrastive loss and (2) pair datasets, which consist of (query, document) pairs and are used for inference.

- `neural_ir.models`: This package contains the implementation of various neural IR models. Each model implementation includes two important methods, `score_pairs()` and `forward()`. The `score_pairs()` method calculates the relevance score of a batch of (query, document) pairs, while the `forward()` method takes in the sampled training triplets and returns the estimated loss for training. It's important to strictly follow the logic described in the code comments to pass all unit tests. You're encouraged to be creative with new implementations to improve the results in your run files, but make sure to keep any new implementation inside separate, different functions that will not be called by unit tests. 

- `neural_ir.trainer`: This component connects all the other components (dataset, collator, model, loss, optimizer) and handles common training routines, such as logging, evaluation, model selection, and checkpoint saving. This assignment uses a customized version of the HuggingFace trainer.

- `neural_ir.train`: This is the entry point for training and can be run by typing `python -m neural_ir.train` in the command line.

- `neural_ir.rerank`, `neural_ir.rank_dense`, `neural_ir.rank_sparse`: These modules handle different modes of inference, including re-ranking run files with a cross-encoder, vector indexing and searching with a dense bi-encoder, and inverted indexing and searching with a sparse bi-encoder. Each of these modules will generate a run file, which you need to save and commit for grading.

### Training and Evaluation Instructions  

When your implementation is ready, you can run training and evaluation on **Snellius** or **Google Colab** to take advantage of GPU acceleration. We provide a notebook ([neural_ir_colab.ipynb](/neural_ir_colab.ipynb)) demonstrating how to set up and run a dense bi-encoder in Colab. If using Colab, be sure to **select GPU in the runtime settings**, as the default is CPU. Follow the instructions in the notebook to generate an access token and clone your repository in Colab.  

On **Snellius**, you have **75.000 SBU** available. If each team uses one A100 GPU, this allows for **694 SBU per team** (75.000 / 108). If each person uses one A100 individually, this allows for **360 SBU per person** (75.000 / 208). Since resources are limited, first validate that your model trains and passes all tests before using Snellius for full-scale experiments.  

Avoid running computationally expensive experiments in **GitHub Codespaces**, as these will not finish in a reasonable time and may limit your resource quota. Use **Codespaces** only for developing and debugging your code, while **Snellius or Colab** should be used for running experiments. This setup mirrors standard practices in industry and academia, where development and experimentation are handled separately.


**Install necessary packages:**

To install necessary packages for this assignment. Please run the following command in your terminal:
```console
pip install -r requirements.txt
```

#### Section 1. Training a Cross-Encoder
##### To train
**Important**: Please use the distilbert-base-uncased 2 model for the Dense Bi-Encoder Experiments, with at least 1000 `max_steps` and `train_batch_size` 32. You can also experiment with different parameters, and then submit it to see if it passes the tests, but with these configurations from the command below, you should get a trained model that is passing the unit tests.
```console
python -m neural_ir.train --pretrained distilbert-base-uncased --max_steps 1000 --train_batch_size 32 ce
```
The best model's checkpoint will be save at `output/ce/model`.
In Python, you can load and use this model's checkpoint as follows:

```python
from neural_ir.models import CrossEncoder
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.pretrained("output/ce/model")
ce = CrossEncoder.from_pretrained("output/ce/model")
pairs = [["This is a query","This is a document"]]
pairs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
scores = ce.score_pairs(pairs)
```

##### To predict and generate a run file
```console
python -m neural_ir.rerank
```
This command will re-rank the candidates generated by BM25 (found in `data/test_bm25.trec`). The re-ranked run file will be saved in `output/ce/test_run.trec`. You need to commit this re-ranked run file for grading.

#### Section 2. Training a Dense Bi-Encoder
##### To train
**Important**: Please use the sentence-transformers/paraphrase-MiniLM-L3-v2 model for the Dense Bi-Encoder Experiments, with at least 1000 `max_steps` and `train_batch_size` 32. You can also experiment with different parameters, and then submit it to see if it passes the tests, but with these configurations from the command below, you should get a trained model that is passing the unit tests.
```console
python -m neural_ir.train --pretrained sentence-transformers/paraphrase-MiniLM-L3-v2 --max_steps 1000 --train_batch_size 32 dense
```
The best model's checkpoint will be save at `output/dense/model`.
In Python, you can load and use this model's checkpoint as follows:
```python
from neural_ir.models import DenseBiEncoder
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.pretrained("output/dense/model")
dense_model = CrossEncoder.from_pretrained("output/dense/model")
queries = ["queries 1", "queries 2"]
docs = ["docs1", "docs2"]
queries = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
docs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
scores = dense_model.score_pairs(queries, docs)
```
##### To predict and generate a run file
```console
python -m neural_ir.rank_dense
```

This command encodes all documents and test queries into dense vectors and utilizes the [Faiss](https://github.com/facebookresearch/faiss) library for indexing and searching in vector space. The resulting run file will be written to `output/dense/test_run.trec`. You must commit this ouptut file for grading.

#### Section 3. Training a Sparse Bi-Encoder
##### To train
**Important**: Please use the sentence-transformers/paraphrase-MiniLM-L3-v2 model for the Sparse Bi-Encoder Experiments, with at least 1000 `max_steps` and `train_batch_size` 32. You can also experiment with different parameters, and then submit it to see if it passes the tests, but with these configurations from the command below, you should get a trained model that is passing the unit tests.. 
```console
python -m neural_ir.train --pretrained sentence-transformers/paraphrase-MiniLM-L3-v2 --max_steps 1000 --train_batch_size 32 sparse
```
The best model's checkpoint will be save at `output/sparse/model`.
In Python, you can load and use this model's checkpoint as follows:
```python
from neural_ir.models import SparseBiEncoder
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.pretrained("output/sparse/model")
sparse_model = CrossEncoder.from_pretrained("output/sparse/model")
queries = ["queries 1", "queries 2"]
docs = ["docs1", "docs2"]
queries = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
docs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
scores = sparse_model.score_pairs(queries, docs)
```
##### To predict and generate a run file
```console
python -m neural_ir.rank_sparse
```

This command encodes textual documents and queries into sparse vectors (bag-of-words) and stores these vectors at `output/sparse/docs` and `output/sparse/queries`. Once these files are ready, the [pyserini](https://github.com/castorini/pyserini) library is used for inverted indexing and search. The resulting run file for test queries will be written to `output/sparse/test_run.trec`. You must commit this run file for grading.

### HuggingFace API
In this assignment, we use the :hugs: Transformers library to load and train our neural models. This library is a powerful tool for natural language processing (NLP) and is widely used in the industry. If needed, you can find an extensive documentation [here](https://huggingface.co/transformers/).

The main components that you will be interacting with are the [Tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html) class and the [Model](https://huggingface.co/transformers/main_classes/model.html) class. The former is used to tokenize a text into a sequence of token ids, while the latter encodes a sequence of such token ids into a sequence of embeddings.

The following code snippet shows how to use these two classes to encode a query and a few documents into a sequence of embeddings. Note that we use the `distilbert-base-uncased` model and tokenizer for this example, but you can use any other model and tokenizer that you want:
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

query = "This is a query"
docs = [
    "This is a document",
    "This is another document",
    "This is yet another document",
]

query = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
docs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")

query_embeddings = model(**query).last_hidden_state
doc_embeddings = model(**docs).last_hidden_state
```

Behind the scenes, the tokenizer returns an object of type `BatchEncoding`, which is a dictionary containing the token ids and other information. For our example collection of documents, it looks like this:
```python
{
    'input_ids': [
        [101, 2023, 2003, 1037, 6254, 102, 0],
        [101, 2023, 2003, 2178, 6254, 102, 0],
        [101, 2023, 2003, 2664, 2178, 6254, 102]
    ],
    'attention_mask': [
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1]
    ],
    ...
}
```
Other than the `input_ids`, notice the `attention_mask` key which is used to indicate which tokens are "real" and which are "padding" tokens. The model expects this mask as input, and uses it to ignore the padding tokens when computing the embeddings.

Subsequently, the model takes these parameters as input (the `**` operator is used to "unpack" the dictionary and pass the values as keyword arguments) and performs a forward pass. The final output contains the desired embeddings of the input sequences, and looks like this:
```python
{
    'last_hidden_state': [
        [
            [-0.1659, -0.1084, ...],
            [-0.5257, -0.4065, ...],
            ...
        ],
        [
            [-0.1647, -0.1342, ...],
            [-0.5539, -0.3975, ...],
            ...
        ],
        [
            [-0.1048, -0.1085, ...],
            [-0.3779, -0.2081, ...],
            ...
        ]
    ],
    ...
}
```
We can then use these embeddings to compute the similarity between the query and the documents. This is practically what you are called to implement in this assignment for the Neural IR models.


### General remarks on assignment organization
At this point, you should already be familiar with Git, the organization of directories and files in the assignments, and how to submit. We recommend you to review the `README` of the pre-assignment to refresh you mind about file organization and directories.

**General Submission instructions**:
- Only the last commit of the main git branch will be graded
- Submission happens automatically at the deadline; after the deadline passes, you can no longer push to your repository, and its current contents will be graded
- If you're using Codespaces and receive a disk usage warning, you can free up some space with `rm -rf ~/.cache`
- Please use Python 3.8+ and `pip install -r requirements.txt` to avoid version issues. (This will happen automatically if using Codespaces. When using your own laptop, Snellius, Colab, etc you will need to install requirements.txt.)
- You can create additional helper functions / classes as long as the API we expect does not change.
- One repository per each group needs to be submitted. You should collaborate with each other within the same repository.
- You are not allowed to change the definitions of the functions that you need to implement. Doing so will lead to failed tests and point deductions
- If you work with the notebook, it's helpful to [find a way to reload modules without reloading the notebook](https://stackoverflow.com/questions/5364050/reloading-submodules-in-ipython/54440220#54440220).
- **You are not allowed to change the contents of .github directory!** Changes to anything in this directory will result in an automatic zero on the assignment, and we will additionally consider it cheating
- Do not forget to finish the analysis component and write a concise report in LaTeX summarizing your findings. Then move your results and conclusions to the LaTeX report.