from flask import Flask, render_template, request, jsonify
from src.TextSummarizerModel import TextSummarizerModel
from youtube_transcript_api import YouTubeTranscriptApi
from deepmultilingualpunctuation import PunctuationModel

app = Flask(__name__)


def format_output(summaries):
    output = ""
    for i in range(0, len(summaries)):
        output += "- " + summaries[i] + "\n"
    return output


def get_video_transcript(video_url, punctuation_model):

    # note: video has v= URL parameter, shorts do not
    # TODO: add youtube short accessbility
    # note: video can have an "&t" parameter for time, does not affect api call, but should be told to the user

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
        raise ValueError("Issue retrieving transcript from this URL")

    # concatenate the groups of text sent by YouTubeTranscriptApi
    transcript_raw = ' '.join(item['text'] for item in transcript_raw) # need to add error check in case of bad returns

    # return the punctuated transcript
    return punctuation_model.restore_punctuation(transcript_raw)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    return jsonify({'success': True, 'summary': "Test summary"})

    # video_url = request.form.get('video_url')
    
    # try:
    #     transcript = get_video_transcript(video_url, punctuation_model)
    #     summary = text_summarizer_model.summarize(transcript)
    #     formatted_summary = format_output(summary)
    #     return jsonify({'success': True, 'summary': formatted_summary})
    # except ValueError as e:
    #     return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Initialize your summarizer and punctuation models here
    text_summarizer_model = TextSummarizerModel()
    punctuation_model = PunctuationModel()
    
    app.run(debug=True)
