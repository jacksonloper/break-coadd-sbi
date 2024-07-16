import lightning as L
import lightning.pytorch.loggers.logger
import pandas as pd

class DataFrameLogger(lightning.pytorch.loggers.logger.Logger):
    def __init__(self,name='torch_checkpoints',version=0):
        super().__init__()
        self._series = []
        self.param_info=[]
        self._name=name
        self._version=version

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    def log_metrics(self, metrics, step):
        self._series.append(metrics | {"step": step})

    @property
    def metrics(self):
        return pd.DataFrame(self._series)

    @lightning.pytorch.utilities.rank_zero_only
    def log_hyperparams(self, params):
        self.param_info.append(params)

    def save(self):
        pass

    def finalize(self, status):
        pass