import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast as T5Tokenizer
import pytorch_lightning as pl 


'''
Extension of PyTorch LightningDataModule, where the setup and dataloader methods
are overwritten to be used for a summary dataset.
'''
class SummaryDataModule(pl.LightningDataModule):

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer: T5Tokenizer, batch_size = 8):
        super().__init__()

        # instantiate parameters
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.FULL_TEXT_MAX_LEN = 512
        self.SUMMARY_MAX_LEN = 128

    
    '''
    Overwriting the setup method.
    - Creates SumamryDataset classes for training and testing data.
    '''
    def setup(self, stage=None):
        # create dataset for training use
        self.train_data = SummaryDataset(
            self.train_df,
            self.tokenizer,
            self.FULL_TEXT_MAX_LEN,
            self.SUMMARY_MAX_LEN
        )

        # create dataset for validation use
        self.test_data = SummaryDataset(
            self.test_df,
            self.tokenizer,
            self.FULL_TEXT_MAX_LEN,
            self.SUMMARY_MAX_LEN
        )

    '''
    Create and return a DataLoader object for the training data
    '''
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2 # number of subprocesses 
        )
    
    '''
    Create and return a DataLoader object for the validation data
    '''
    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2 # number of subprocesses 
        )
    
    '''
    Create and return a DataLoader object for the testing data
    '''
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2 # number of subprocesses 
        )