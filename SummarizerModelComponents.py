import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast as T5Tokenizer



'''
An extension of PyTorch's Dataset class. 
Extended for the purpose of being used for the specific dataset 
that is being used for training/testing/validation.
'''
class SummaryDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: T5Tokenizer):

        self.data = data
        self.tokenizer = tokenizer
        self.FULL_TEXT_MAX_LEN = 512
        self.SUMMARY_MAX_LEN = 128

        super().__init__()

    # overwrite: return size of dataset
    def __len__(self):
        return len(self.data)
    
    # overwrite: get data at a given index within the dataframe -> need to return the encoded text and summary 
    def __getitem__(self, index):

        # encode text at index requested
        text = self.data.iloc[index]["text"]
        tokenized_text = self.tokenizer(text, max_length=self.FULL_TEXT_MAX_LEN, truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors="pt")

        # encode summary at index requested
        summary = self.data.iloc[index]["summary"]
        tokenized_summary = self.tokenizer(text, max_length=self.SUMMARY_MAX_LEN, truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors="pt")

        # Get the input_ids, attention_mask, and labels for the text
        text_input_ids = tokenized_text["input_ids"].flatten()
        text_attention_mask = tokenized_text["attention_mask"].flatten()

        # get the labels for the summary
        labels = tokenized_summary["input_ids"].flatten()
        labels[labels == 0] = -100 # replace paddning tokens (currently 0), with -100 -> ensure correct labels needed for T5

        # Get the attention_mask for the labels
        labels_attention_mask = tokenized_summary["attention_mask"].flatten()

        return dict(
            text=text, # raw text
            summary=summary, # raw summary
            text_input_ids=text_input_ids, # represents raw text as tokens
            text_attention_mask=text_attention_mask, # binary representation of text tokens showing model where to not pay attention
            labels = labels, # raw summary as tokens
            labels_attention_mask=labels_attention_mask # binary representation of summary tokens showing model where to not pay attention
        )
    


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