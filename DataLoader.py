import pandas as pd

'''
Class used to load and allow access to CSV file containing data used for training 
'''
class DataLoader:

    '''
    Initialize the dataset being used as a pandas dataframe.
    '''
    def __init__(self) -> None:
        self.filePath = "Data/news_summary.csv"
        self.encoding = "latin-1"

        self.dataframe = self.csvToDataFrame()

    '''
    Convert the csv file to a pandas dataframe and clean it 
    for easy use.
    '''
    def csvToDataFrame(self):
        # read csv file
        df = pd.read_csv(self.filePath, encoding = self.encoding)

        # only keep the full text and summary -> rename for ease of use
        df = df[["text", "ctext"]] #text = summary, ctext = complete text
        df.columns = ["summary", "text"]

        # clean missing values and return df
        df.dropna()
        return df
