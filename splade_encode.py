import os
import math
import argparse
from pathlib import Path
from typing import List
import pandas as pd
import tempfile
import more_itertools
import pickle

import datetime

from tqdm.auto import tqdm

import torch
from torch import distributed as dist
from transformers import AutoModelForMaskedLM, AutoTokenizer

from scipy import sparse as sp

def print_message(*s):
    s = ' '.join([str(x) for x in s])
    msg = "[{}][{}] {}\n".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), os.environ['LOCAL_RANK'], s)
    print(msg, flush=True, end='')

    return msg

def _loader(text: List[str], batch_size: int, rank: int, nrank: int):
    n_input = len(text)
    nfilling = math.ceil(n_input/batch_size/nrank)*batch_size*nrank - n_input
    text = text + ['']*nfilling

    for chunk_id, batch in enumerate(more_itertools.batched(enumerate(text), batch_size)):
        if chunk_id % nrank == rank:
            yield (
                chunk_id, 
                [ t for i, t in batch], # actual text
                [ i < n_input for i, t in batch ] # is real input
            )


def main(args, rank: int, nrank: int):
    working_dir = Path(args.working_dir)

    print_message("Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(args.model).eval().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device: torch.device = model.device

    text = (pd.read_csv if args.text.name.endswith('.csv') else pd.read_pickle)(args.text)[args.text_column].tolist()
    n_batch = math.ceil(len(text) / args.batch_size / nrank)

    loader = tqdm(
        _loader(text, args.batch_size, rank, nrank),
        total=n_batch, dynamic_ncols=True, disable=rank>1
    )
    
    dist.barrier()
    res = {}
    with torch.inference_mode():
        for chunk_id, input_text, real_mask in loader:
            tokenized = tokenizer(
                input_text, return_tensors='pt', 
                max_length=args.max_length, truncation='longest_first', padding='longest'
            ).to(device)

            # basically SPLADE decoding 
            encoded = torch.topk(
                torch.max(torch.log( torch.relu(model(**tokenized).logits)+1 ) * tokenized.attention_mask.unsqueeze(-1), dim=1).values, 
                k=args.topk, dim=1
            )

            rm = torch.tensor(real_mask).expand(args.topk, -1).T.reshape(-1).to(device)

            rows: torch.Tensor = torch.arange(encoded.values.shape[0]).expand(args.topk, -1).T.reshape(-1).to(device)
            cols: torch.Tensor = encoded.indices.reshape(-1)
            vals: torch.Tensor = encoded.values.reshape(-1)
            assert rows.shape == cols.shape == vals.shape, f"{rows.shape}, {cols.shape}, {vals.shape}"

            rows = rows[ rm & (vals>0) ].cpu().numpy()
            cols = cols[ rm & (vals>0) ].cpu().numpy()
            vals = vals[ rm & (vals>0) ].cpu().numpy()

            res[chunk_id] = sp.csr_matrix((vals, (rows, cols)), shape=(sum(real_mask), len(tokenizer)))

    with (working_dir / f"{rank}.pkl").open('wb') as fw:
        pickle.dump(res, fw)
    
    print_message("Done processing text, waiting...")

    dist.barrier()

    if rank < 1:
        chunks = {}
        for i in tqdm(range(nrank), desc='Merging'):
            chunks.update(pd.read_pickle(working_dir / f"{i}.pkl"))
        matrix = sp.vstack([ chunks[i] for i in range(max(chunks.keys())+1) ])
        pickle.dump(matrix, open(args.output, 'wb'))


if __name__ == '__main__':
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=Path, required=True, help="Pickle or quoted csv file for the text")
    parser.add_argument('--text_column', type=str, default='raw', help="Column of the text to be encoded.")

    parser.add_argument('--model', type=str, required=True, help="Huggingface model or local model path")
    parser.add_argument('--max_length', type=int, default=512, help="Maximum document length")
    parser.add_argument('--topk', type=float, default=0.05, help="Percentage of the top-scored tokens to retain")

    parser.add_argument('--batch_size', '--bs', type=int, default=64, help="Per GPU batch size")

    parser.add_argument('--output', type=Path, required=True, help="Output path")
    parser.add_argument('--working_dir', type=Path, default=None, help="Working directory")

    parser.add_argument('--overwrite', action='store_true', default=False, help="Overwrite existing output file")

    args = parser.parse_args()

    rank, nrank = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])

    if args.output.exists():
        print(f"Output file {args.output} already exists")
        if not args.overwrite:
            raise FileExistsError()
    
    if args.working_dir is None:
        ws = [None for _ in range(nrank)]
        dist.all_gather_object(ws, tempfile.mkdtemp())
        args.working_dir = ws[0]
    print_message(f"Using working dir `{args.working_dir}`")

    if args.topk < 1.: 
        args.topk = len(AutoTokenizer.from_pretrained(args.model))*args.topk
    args.topk = int(args.topk)

    main(args, rank, nrank)