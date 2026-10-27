# LADR

LADR models conversational retrieval at two levels: choosing a document and selecting a passage within that document. A session encoder produces topic and focus representations so that the retrieval scores can use both the conversation's broader subject and its current question.

## Code layout

| Path | Purpose |
| --- | --- |
| `code/prism.py` | Query encoders and hierarchical model components |
| `code/train.py` | Training, evaluation and model configuration |
| `code/retrieval.py` | Passage and document indexes; retrieval metrics |
| `code/prep_data.py` | TopiOCQA and OR-QuAC conversion |
| `code/build_docs.py` | Document embeddings and passage-to-document mapping |
| `code/build_dstore.py` | Grouped document store for training |

Some modules retain their development names. The hierarchical configuration is selected with `--kind concat --latent`; `prism.py` also contains the other query encoders used for comparison.

## Environment and data

Use CUDA-enabled PyTorch with Transformers, NumPy, pandas, PyArrow and scikit-learn. The query encoder uses `BAAI/bge-base-en-v1.5`.

Download TopiOCQA and OR-QuAC from their providers and prepare the corresponding passage collections. Run `code/prep_data.py` from the repository root after setting up its input files. Training and full-collection evaluation also require the frozen passage-index shards expected by `code/retrieval.py`.

For a prepared TopiOCQA passage index:

```bash
python code/prep_data.py topiocqa
python code/build_docs.py topiocqa
python code/build_dstore.py topiocqa
```

`build_docs.py` adds document IDs and document embeddings. `build_dstore.py` groups passages by document and prepares the training store. Inspect their input paths alongside `code/retrieval.py` when arranging a new local collection.

## Train and evaluate

```bash
python code/train.py --kind concat --latent --expert centroid --cond 1 --learn_tau 1 --data topiocqa --name ladr_seed0 --seed 0
python code/train.py --kind concat --latent --expert centroid --cond 1 --learn_tau 1 --data topiocqa --name ladr_seed0 --seed 0 --eval_only
```

`--expert centroid` selects the document-level representation; `--cond 1` normalizes passage scores within a document, and `--learn_tau 1` learns the topic and focus temperatures. Use the same model options and run name when loading a checkpoint for evaluation.

Run artifacts are written to `runs/<name>/`, including `model.pt`, evaluation JSON files and ranked-passage arrays. Cross-dataset evaluation is available through `--eval_data` and `--eval_split` after preparing the target collection.

This repository contains implementation and experiment code. Benchmark collections, indexes, checkpoints and manuscript materials are not included.

## License

A license for the project-specific code has not yet been specified.
