from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

'''
BigBird citation
@misc{zaheer2021big,
      title={Big Bird: Transformers for Longer Sequences}, 
      author={Manzil Zaheer and Guru Guruganesh and Avinava Dubey and Joshua Ainslie and Chris Alberti and Santiago Ontanon and Philip Pham and Anirudh Ravula and Qifan Wang and Li Yang and Amr Ahmed},
      year={2021},
      eprint={2007.14062},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
'''


class TextSummarizerModel:

    def __init__(self):
        self.model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-bigpatent")
        self.tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-bigpatent")

    def summarize(self, text):

        chunk_size = 4096 # number of tokens to pass to model at once
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        # generate summaries for each chunk
        summaries = []

        '''
        BigBird comes with 2 implementations: original_full & block_sparse. For the sequence length < 1024, 
        using original_full is advised as there is no benefit in using block_sparse attention.
        '''
        if len(text_chunks[0]) < 1024:
            print("using original full")
            inputs = self.tokenizer(text_chunks[0], return_tensors='pt', truncation=True)
            prediction = self.model.generate(**inputs)
            summary = self.tokenizer.batch_decode(prediction, skip_special_tokens=True, attention_type = "original_full")[0]
            summaries.append(summary)
        else:
            for chunk in text_chunks:
                inputs = self.tokenizer(chunk, return_tensors='pt', truncation=True)
                prediction = self.model.generate(**inputs, no_repeat_ngram_size=3)
                summary = self.tokenizer.batch_decode(prediction, skip_special_tokens=True)[0]
                summaries.append(summary)

        # return summarizes of each chunk
        return summaries