# Contextualization With SPLADE For High Recall Retrieval

This repository contains the scripts for encoding the collection with a trained SPLADE model and running 
TAR experiments with [`TARexp`](https://github.com/eugene-yang/tarexp)

To setup the environment, please install the requirement packages through pip install. 
```bash
pip install -r requirements.txt
```

## Encode documents with SPLADE

The `splade_encode.py` script encodes the documents into `scipy.sparse` matrix for `TARexp`.
When encoding with multiple GPUs, please use [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) through `torchrun`. 

The following command is an example using `naver/splade-cocondenser-ensembledistil` to encode the Jeb Bush collection.

```bash
torchrun --nproc_per_node=4 splade_encode.py \
--text ./jb_info/raw_text.pkl \
--model naver/splade-cocondenser-ensembledistil \
--topk 0.1 \
--output ./splade_jb/splade.10p.pkl
```

The `--text` flags accept both pickle and csv files that can be read by `pandas.read_csv` or `pandas.read_pickle` into a 
`pandas.DataFrame`. The `--text_column` indicates the column containing the text to be encoded (default `raw`). 
Please see `python splade_encode.py --help` for more information on the arguments.

## TAR experiments

The `exp.py` script execute the TAR experiments through `TARexp`. 
File `jb_cates.txt` and `rcv1_cates.py` specify the categories used in the paper. 

The following example command runs an experiment on category `100` in the Jeb Bush collection 
with the matrix decoded from the SPLADE model and a matrix with BM25 saturated tf without idf. 

```bash
python exp.py \
--ranker lr \
--rel_info ./rcv1_info/rel_info.pkl \
--experiment jb_test \
--matrix ./splade_jb/splade.10p.pkl ./splade_jb/bm25woidf.pkl \
--exp_topic 100 \
--output_root ./tarexp_experiments_to_80
```

Please see `python exp.py --help` for more information on the arguments.

## Citation

Please consider citing the following paper. 
```bibtex
@inproceedings{splade-tar,
    author = {Eugene Yang},
    title = {Contextualization with SPLADE for High Recall Retrieval},
    booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) (Short Paper)},
    year = {2024},
    doi = {10.1145/3626772.3657919}
}
```