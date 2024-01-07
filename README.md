# YouTube Video Summarizer

Design Doc: https://quip.com/JoG6A3vjmYez/YouTube-Summarizer

## Prerequisites
- pip install youtube-transcript-api
- pip install deepmultilingualpunctuation
- pip install transformers

## Current Functionality
- Leveraged youtube_transcript_api, deepmultilingualpunctuation libraries to retrive and pre process a text transcript from a YouTube video URL.
- Used transformers library to create a comprehensive summary of the transcript that was created.
- Thoroughly checked for any possible exceptions and to prevent crashing (assuming prerequisite libraries are installed)


## Timeline
12/18/2023 - Project started

12/20/2023 - 1/1/2024 - Experimented and researched possible solutions

1/4/24 -Finalized inital model using a TextSummarizerModel and a helper method to extract transcripts from youtube url


## Next steps
- Fine tune summarizing model (in progress)
- Create front end
- Find ways to improve output format
