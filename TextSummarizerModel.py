from transformers import pipeline

class TextSummarizerModel:

    def __init__(self):
        self.model = pipeline('summarization') # import a summarization model

    def summarize(self, text):
        
        #  add indicators for end of sentence to split
        text = text.replace('.', ".<eos>")
        text = text.replace('!', "!<eos>")
        text = text.replace('?', "?<eos>")
        text = text.replace('\n', "<eos>")

        # split text into list of sentences
        text_sentences = [sentence.strip() for sentence in text.split('<eos>') if sentence.strip()]

        # create chunks of sentences with cumulative word length < 500 to send to model
        max_chunk_len = 500
        sentence_chunks = []
        chunk_index = 0
        
        for sentence in text_sentences:
            if len(sentence_chunks) == chunk_index + 1: # check if there is a current chunk
                if len(sentence_chunks[chunk_index]) + len(sentence.split(' ')) <= max_chunk_len: # if sentence can be added to current chunk do so
                    sentence_chunks[chunk_index].extend(sentence.split(' ')) # add words to current chunk
                else: # create new chunk
                    chunk_index += 1
                    sentence_chunks.append(sentence.split(' '))
            else: # create a new chunk if there is not a current one
                sentence_chunks.append(sentence.split(' ')) # protects agains no sentences in inital read


        # join indidual words back into consecutive text for each chunk
        for curr_chunk in range(len(sentence_chunks)):
            sentence_chunks[curr_chunk] = ' '.join(sentence_chunks[curr_chunk])

        # pass sentence chunks on to summarizer for work
        text_summary = self.model(sentence_chunks, max_length = 120, min_length = 30, do_sample = False) 

        # Check if the summary is a valid JSON object with the required "summary_text" field
        if not all(isinstance(item, dict) and 'summary_text' in item for item in text_summary):
            raise ValueError("Invalid summary format")

        # Concatenate all the "summary_text" fields
        concatenated_summary = ' '.join(item['summary_text'] for item in text_summary) # need to add error check in case of bad returns

        return concatenated_summary