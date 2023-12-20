import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ( AdamW, T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer)
import pytorch_lightning as pl 


'''
Overwrite PyTorch's LightningModule class.
''' 
class SummaryModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)

    ''' 
    Define a forward pass
    '''
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):

        # pass given parameters throght the T5 model
        output = self.model(
            input_ids, 
            attention_mask = attention_mask, 
            labels = labels,
            decoder_attention_mask = decoder_attention_mask
        )

        # get the calculated loss and logits from the output 
        return output.loss, output.logits

    ''' 
    Define a single step in the training process.
    '''
    def training_step(self, batch, batch_index):
        batch_input_ids = batch["text_input_ids"] # tokenized text from the batch
        batch_attention_mask = batch["text_attention_mask"] # attention mask for the batch
        batch_labels = batch["labels"] # correct output
        batch_labels_attention_mask = batch_labels_attention_mask["labels_attention_mask"] # attention mask for output

        # pass batch data through the forward pass and return the loss
        results = self(
            input_ids = batch_input_ids,
            attention_mask = batch_attention_mask,
            decoder_attention_mask = batch_labels_attention_mask,
            labels = batch_labels
        )[1]


        # add log?
        return results
    
    ''' 
    Define a single step in the validation process.
    '''
    def validation_step(self, batch, batch_index):
        batch_input_ids = batch["text_input_ids"] # tokenized text from the batch
        batch_attention_mask = batch["text_attention_mask"] # attention mask for the batch
        batch_labels = batch["labels"] # correct output
        batch_labels_attention_mask = batch_labels_attention_mask["labels_attention_mask"] # attention mask for output

        # pass batch data through the forward pass and return the loss
        results = self(
            input_ids = batch_input_ids,
            attention_mask = batch_attention_mask,
            decoder_attention_mask = batch_labels_attention_mask,
            labels = batch_labels
        )[1]


        # add log?
        return results
    
    ''' 
    Define a single step in the test process.
    '''
    def test_step(self, batch, batch_index):
        batch_input_ids = batch["text_input_ids"] # tokenized text from the batch
        batch_attention_mask = batch["text_attention_mask"] # attention mask for the batch
        batch_labels = batch["labels"] # correct output
        batch_labels_attention_mask = batch_labels_attention_mask["labels_attention_mask"] # attention mask for output

        # pass batch data through the forward pass and return the loss
        results = self(
            input_ids = batch_input_ids,
            attention_mask = batch_attention_mask,
            decoder_attention_mask = batch_labels_attention_mask,
            labels = batch_labels
        )[1]


        # add log?
        return results
    
    '''
    Define optimizers and learning rate
    '''
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr = 0.0001)



