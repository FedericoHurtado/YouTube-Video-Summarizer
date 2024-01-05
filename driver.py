from TextSummarizerModel import TextSummarizerModel
from youtube_transcript_api import YouTubeTranscriptApi
from deepmultilingualpunctuation import PunctuationModel


def get_video_transcript(video_url):

    # note: video has v= URL parameter, shorts do not
    # TODO: add youtube short accessbility

    # obtain video id from video URL ("?v=[video_id]"")
    video_parts = video_url.split("?v=")

    
    if len(video_parts) != 2:
        # raise error
        raise ValueError("Invalid video URL")
    
    video_id = video_parts[1]
    
    print("VIDEO ID: ", video_id)


    # get the raw transcript using YouTubeTranscriptApi (No punctuation)
    try:
        transcript_raw = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        # Catch any exceptions and print the error message
        print(f"Error retrieving transcript: {e}")
        raise ValueError("Issue retrieving transcript from this URL")  # Return None to signal an error condition

    # concatenate the groups of text sent by YouTubeTranscriptApi
    transcript_raw = ' '.join(item['text'] for item in transcript_raw) # need to add error check in case of bad returns

    # add punctuation to the transcript using the deepmultilingualpunctuation package
    punctuation_model = PunctuationModel()

    # return the punctuated transcript
    return punctuation_model.restore_punctuation(transcript_raw)





if __name__ == "__main__":

    # step 1: instantiate summarizer model
    print("Initializing model.....")
    text_summarizer_model = TextSummarizerModel()
    print("Summarizer model created")

    while True:
        user_input = input("Enter YouTube URL for summarization. Enter 'Exit' if you wish to exit the program: ")

        if user_input.lower() == "exit":
            exit()  # exit program command

        try:
            # step 2: get text from the YouTube URL
            url_transcript = get_video_transcript(user_input)

            # step 3: pass transcript through summarizer model
            summary = text_summarizer_model.summarize(url_transcript)
            print("-----------------------------")
            print("SUMMARY: ", summary)
        except ValueError as e:  # catch errors
            print(f"Error: {e}")
            url_transcript = ""
        
    





