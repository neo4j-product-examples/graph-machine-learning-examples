import copy
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
import torch
import pandas as pd
from py_models.model import BaseModel
from py_models.data import Data
import py_models.quality_check as qc


class TrainingPipeline(ABC):
    def __init__(self, data: Data, model: BaseModel) -> None:
        self._data = data
        self._model = model

        self._bestModel: BaseModel = None
        self._bestModelStats: Dict[str, Any] = dict()
        self._converged: bool = False
        self.qualityChecks: Dict[str, Any] = dict()

    def get_best_model(self):
        return copy.deepcopy(self._bestModel)

    def get_best_model_stats(self):
        return copy.deepcopy(self._bestModelStats)

    def did_converge(self):
        return self._converged

    @abstractmethod
    def run_quality_checks(self) -> Dict[str, Any]:
        pass


class ClsTrainingPipeline(TrainingPipeline, ABC):
    def __init__(self,
                 data: Data,
                 model: BaseModel) -> None:
        super().__init__(data, model)

        self._bestModel: BaseModel = None
        self.bestValidAcc: float = 0
        self._bestModelStats: Dict[str, Any] = dict()

    def run_quality_checks(self) -> Dict[str, Any]:
        return self._data.x_apply(qc.zeros_feat_vector_check)

    @staticmethod
    def _train_epoch(data: Data, model: BaseModel, optimizer):
        x_tr = data.x_train()
        y_tr = data.y_train()
        model.train()
        optimizer.zero_grad()
        out = model.forward(x_tr)
        loss = model.loss(out, y_tr)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss.item()

    @staticmethod
    def _eval_acc(y_true, y_pred):
        correct = torch.eq(y_pred, y_true)
        return float(correct.sum()) / correct.shape[0]

    def _test_epoch(self, data: Data, model: BaseModel):
        model.eval()
        out = model.forward(data.x)
        y_pred = out.argmax(dim=-1)
        train_acc = self._eval_acc(data.y_train(), y_pred[data.train_idx()])
        valid_acc = self._eval_acc(data.y_valid(), y_pred[data.valid_idx()])
        test_acc = self._eval_acc(data.y_test(), y_pred[data.test_idx()])
        return train_acc, valid_acc, test_acc

    def train(self, device: torch.device, epochs: int = 100, tolerance: float = 0.001, patience: int = 2,
              learning_rate: float = 0.01, weight_decay: float = 5e-4, verbose: bool = True) -> pd.DataFrame:
        data = self._data.to(device)
        model = self._model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_valid_acc = 0.0
        last_loss = np.Inf
        patient_count = 0
        test_results = []
        epoch = 0
        terminate = False
        while not terminate:
            loss = self._train_epoch(data, model, optimizer)
            train_acc, valid_acc, test_acc = self._test_epoch(data, model)
            stats = {'epoch': epoch, 'loss': loss, 'train_acc': train_acc, 'valid_acc': valid_acc,
                     'test_acc': test_acc}
            if valid_acc > best_valid_acc:
                self._bestModelStats = stats
                self._bestModel = copy.deepcopy(model)
                best_valid_acc = valid_acc
            if verbose:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
            test_results.append(stats)

            epoch += 1
            if (last_loss - loss) < tolerance:
                if patient_count >= patience:
                    terminate = True
                    self._converged = True
                else:
                    patient_count += 1
            elif epoch >= epochs:
                terminate = True
                warnings.warn('Model did not converge. It terminated on max epochs condition.')
            else:
                patient_count = 0
                last_loss = loss

        return pd.DataFrame.from_records(test_results)
