import torch
import pandas as pd
from typing import List, Dict
from numpy.typing import ArrayLike
from py_models.data import Data
from py_models.model import BasicNeuralNet
from py_models.pipeline import ClsTrainingPipeline
from benchmark.benchmark import BenchmarkResult
from torch import Tensor


def default_best_guess_benchmark(src_paper_df: pd.DataFrame, train_idx: ArrayLike, valid_idx: ArrayLike,
                                 test_idx: ArrayLike) -> Dict[str, float]:
    paper_df = src_paper_df.copy()

    train_df = paper_df.loc[paper_df.nodeId.isin(train_idx), ['nodeId', 'subjectId']]
    valid_df = paper_df.loc[paper_df.nodeId.isin(valid_idx), ['nodeId', 'subjectId']]
    test_df = paper_df.loc[paper_df.nodeId.isin(test_idx), ['nodeId', 'subjectId']]

    cnt_df = train_df.groupby('subjectId').count().reset_index().rename(columns={'nodeId': 'cnt'})
    pred = cnt_df.subjectId[cnt_df.cnt == cnt_df.cnt.max()].sort_values().iloc[0]

    return {'train_acc': sum(train_df.subjectId == pred) / train_df.shape[0],
            'valid_acc': sum(valid_df.subjectId == pred) / valid_df.shape[0],
            'test_acc': sum(test_df.subjectId == pred) / test_df.shape[0]}


def run_model(x: Tensor, y: Tensor, train_idx: ArrayLike, valid_idx: ArrayLike, test_idx: ArrayLike,
              hidden_dims: List[int] = [64], dropout: int = 0.3, batch_normalize: bool = True,
              epochs: int = 100, tolerance: float = 0.001, patience: int = 2, learning_rate: float = 0.01,
              weight_decay: float = 5e-4, verbose: bool = True):
    num_classes = y.unique().shape[0]
    data = Data(x, y, train_idx, valid_idx, test_idx)

    model = BasicNeuralNet(data.x.shape[1], num_classes, hidden_dims, dropout=dropout, batch_normalize=batch_normalize)
    training_pipeline = ClsTrainingPipeline(data, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs_df = training_pipeline.train(device, epochs=epochs, tolerance=tolerance, patience=patience,
                                        learning_rate=learning_rate, weight_decay=weight_decay, verbose=verbose)
    return BenchmarkResult(training_pipeline), epochs_df