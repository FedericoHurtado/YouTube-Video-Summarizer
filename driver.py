from TextSummarizerModel import TextSummarizerModel
from youtube_transcript_api import YouTubeTranscriptApi
from deepmultilingualpunctuation import PunctuationModel


def get_video_transcript(video_url):

    # note: video has v= URL parameter, shorts do not
    # TODO: add youtube short accessbility

    # obtain video id from video URL ("?v=[video_id]"")
    video_id = video_url.split("?v=")[1]

    print("VIDEO ID: ", video_id)

    if video_url == None:
        # raise error
        return 

    # get the raw transcript using YouTubeTranscriptApi (No punctuation)
    transcript_raw = YouTubeTranscriptApi.get_transcript(video_id)

    # concatenate the groups of text sent by YouTubeTranscriptApi
    transcript_raw = ' '.join(item['text'] for item in transcript_raw)

    # add punctuation to the transcript using the deepmultilingualpunctuation package
    punctuation_model = PunctuationModel()

    # return the punctuated transcript
    return punctuation_model.restore_punctuation(transcript_raw)





if __name__ == "__main__":
    # step 1: instantiate summarizer model
    text_summarizer_model = TextSummarizerModel()

    # step 2: get text from the youtube url
    # TODO: make user input
    dev_url = "https://www.youtube.com/watch?v=jNsBh6_jUDM"
    url_transcript = get_video_transcript(dev_url)

    # step 3: pass transcript through summarizer model
    sumamry = text_summarizer_model.summarize(url_transcript)
    print("-----------------------------")
    print("SUMMARY: ", sumamry)
    





