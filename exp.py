import argparse

from pathlib import Path

import pickle
import random
from typing import List
import pandas as pd
import numpy as np

import tarexp
from tarexp import component
from tarexp.util import stable_hash

from sklearn.linear_model import LogisticRegression
import ir_measures as irms

import logging


class MultipleDataset(tarexp.SparseVectorDataset):
    
    def __init__(self, dss: List[tarexp.SparseVectorDataset]):
        super().__init__()

        assert len(dss) > 0
        assert all( ds.n_docs == dss[0].n_docs for ds in dss )

        self.all_ds = dss
    
    @property
    def n_docs(self):
        return self.all_ds[0].n_docs

    @property
    def identifier(self):
        if self._labels is None:
            return (self.n_docs, stable_hash([ m.identifier for m in self.all_ds ]), None)    
        return (self.n_docs, stable_hash([ m.identifier for m in self.all_ds ]), stable_hash(self.labels))
    
    def ingest(self, text, force=False):
        raise NotImplemented

    def getAllData(self, copy=False):
        return [ ds._vectors.copy() if copy else ds._vectors for ds in self.all_ds ]

    def getTrainingData(self, ledger: tarexp.Ledger):
        annt = ledger.annotation
        mask = ~np.isnan(annt)
        return [ ds._vectors[mask] for ds in self.all_ds], annt[mask].astype(bool)
    
    def duplicate(self, deep=False):
        return MultipleDataset([ ds.duplicate(deep=deep) for ds in self.all_ds ])
    

class CommitteeSklearnRanker(component.Ranker):

    def __init__(self, modules, **kwargs):
        super().__init__()
        self.rankers = [
            component.SklearnRanker(m, **config, **kwargs)
            for m, config in modules
        ]
    
    @property
    def nrankers(self):
        return len(self.rankers)
    
    def reset(self):
        for ranker in self.rankers:
            ranker.reset()
    
    def trainRanker(self, Xs, y, **kwargs):
        return [ 
            ranker.trainRanker(X, y, **kwargs) 
            for X, ranker in zip(Xs, self.rankers)
        ]
    
    def scoreDocuments(self, Xs, **kwargs):
        probs = []
        for X, ranker in zip(Xs, self.rankers):
            probs.append(ranker.scoreDocuments(X, **kwargs))
        
        summed = probs[0]
        for prob in probs[1:]:
            summed += prob
        
        return summed / self.nrankers


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def main(args):

    output_path: Path = args.output_dir / f"{args.exp_topic}.{args.i_seed}.dump.pkl"

    if output_path.exists():
        if not args.overwrite:
            print(f"{output_path} already exists -- skip")
            return 
        print(f"{output_path} already exists -- will overwrite")
        
    Xs = []
    for matrix in args.matrix:
        X = pd.read_pickle(matrix)
        if isinstance(X, dict):
            X = X['X']
        Xs.append(X)
    
    logger.info("Loading matrix")
    if len(Xs) == 1:
        ds = tarexp.SparseVectorDataset.from_sparse(Xs[0])
    else:
        ds = MultipleDataset([
            tarexp.SparseVectorDataset.from_sparse(X) for X in Xs
        ])

    assert args.ranker == 'lr'
    if len(args.matrix) == 1:
        ranker = component.SklearnRanker(LogisticRegression, solver='liblinear')
    else:
        ranker = CommitteeSklearnRanker([
            (LogisticRegression, {'solver': 'liblinear'}) for _ in range(len(args.matrix))
        ])

    rel_info: pd.DataFrame = pd.read_pickle(args.rel_info)
    labels: pd.Series = rel_info[args.exp_topic]
    n_pos_total = labels.sum()

    random.seed(12345)
    np.random.seed(12345)
    labels_shuffled = labels.sample(labels.size, replace=False)

    seeds = [
        labels_shuffled[labels_shuffled].index[args.i_seed], 
        labels_shuffled[~labels_shuffled].index[args.i_seed]
    ]

    setting = component.combine(ranker, 
                                component.PerfectLabeler(), 
                                component.RelevanceSampler() if args.sampler == 'relevance' else component.UncertaintySampler())()

    workflow = tarexp.OnePhaseTARWorkflow(
        ds.setLabels(labels),
        setting, 
        seed_doc=seeds, 
        batch_size=args.batch_size, 
        random_seed=12345
    )
    
    recording_metrics = [irms.RPrec, irms.P@200, tarexp.OptimisticCost(target_recall=args.target_recall, cost_structure=(1,1,1,1))]
    results = []
    after_target_met = 0

    logger.info("Workflow started")
    for ledger in workflow:
        current_recall = ledger.n_pos_annotated / n_pos_total
        
        logger.info(f"Round {ledger.n_rounds}: found {ledger.n_pos_annotated} positives (R={current_recall:.4f}) in total {ledger.n_annotated}") 
        logger.info(f"metric: {workflow.getMetrics(recording_metrics)}")
        
        results.append({
            **workflow.getMetrics(recording_metrics),
            'n_pos_annotated': ledger.n_pos_annotated,
            'n_annotated': ledger.n_annotated,
            'recall': current_recall
        })
        
        if ledger.n_rounds % args.save_freqency == 0:
            logger.info(f"Saving...")
            pickle.dump( (results, ledger), output_path.open('wb') )
        
        # experimental stopping
        if ledger.n_rounds >= args.rounds_min and current_recall > args.target_recall:
            after_target_met += 1
            if after_target_met >= args.rounds_over_recall:
                break

    logger.info("Save at the end")
    pickle.dump( (results, ledger), output_path.open('wb') )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--arg_file', default=None, help="argument files for specifying experiments")
    parser.add_argument('--task_id', type=int, default=None, help="task id")
    parser.add_argument('--arg_keys', nargs='+', default=[], help="arguments specifying through arg_file")

    parser.add_argument('--ranker', choices=['lr'], help="Classification/Ranking model")
    parser.add_argument('--rel_info', type=str, required=True, help="rel_info pickle file. see tarexp documentations for more details")
    parser.add_argument('--matrix', nargs='+', type=str, required=True, help="One or more matrix files")
    parser.add_argument('--sampler', choices=['relevance', 'uncertainty'], default='relevance', help="Sampling strategy")
    parser.add_argument('--batch_size', type=int, default=200, help="Batch size for each TAR round")

    parser.add_argument('--target_recall', type=float, default=0.8, help="Target recall")
    parser.add_argument('--rounds_over_recall', type=int, default=5, help="Rounds continue to run after hitting target recall")
    parser.add_argument('--rounds_min', type=int, default=5, help="Minimum number of TAR runs to execute")

    parser.add_argument('--exp_topic', type=str, help="Topic/category for experiment")
    parser.add_argument('--i_seed', type=int, default=0, help="Experiment seed for replication")
    
    parser.add_argument('--output_root', type=Path, default=Path('./tarexp_experiments'), help="Experiment output root directory")
    parser.add_argument('--experiment', type=str, help="Experiment name")
    parser.add_argument('--overwrite', action='store_true', default=False, help="Overwrite existing experiment")

    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--save_freqency', type=int, default=10)

    args = parser.parse_args()
    
    if args.arg_file is not None:
        assert len(args.arg_keys) > 0 and args.task_id is not None
        tasks = [ line.strip().split('\t') for line in open(args.arg_file) ]
        current_task = tasks[args.task_id]
        assert len(args.arg_keys) == len(current_task)
        parser.parse_args(sum([
            [f"--{argname}", *val.split()]
            for argname, val in zip(args.arg_keys, current_task)
        ], []), namespace=args)
        

    for required in ['ranker', 'matrix', 'exp_topic', 'experiment']:
        assert getattr(args, required) is not None

    args.output_dir = args.output_root / args.experiment
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(args)

    main(args)