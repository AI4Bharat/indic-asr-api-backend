import requests
import os
from pydub import AudioSegment
import base64
import json


def download_file(url):
  '''Download the file locally'''
  local_filename = url.split('/')[-1]
  # NOTE the stream=True parameter below
  with requests.get(url, stream=True) as r:
      r.raise_for_status()
      with open(local_filename, 'wb') as f:
          for chunk in r.iter_content(chunk_size=8192):
              f.write(chunk)
  return local_filename

if __name__ == "__main__":

  # URL to the audio file
  MEDIA_URL = "https://objectstore.e2enetworks.net/indic-asr-public/sample_audio.wav"
  # API URL
  API_URL = "http://0.0.0.0:4992/recognize/en"
  
  # Download the audio file
  file = download_file(MEDIA_URL)
  
  # Preprocess the audio file (change sample rate and bitrate)
  given_audio = AudioSegment.from_wav(file)
  given_audio = given_audio.set_frame_rate(16000)
  given_audio.export("temp.wav",format="wav", codec="pcm_s16le")
  os.remove(file)
  
  # Load the wav file into the base64 format
  with open("temp.wav", "rb") as wav_file:
      encoded_string = base64.b64encode(wav_file.read())
  #Encode the file.
  encoded_string = str(encoded_string,'ascii','ignore')
  
  # POST request data format
  data = {
    "config": {
      "language": {
        "sourceLanguage": "en"
      },
      "transcriptionFormat": {
        "value": "transcript"
      },
      "audioFormat": "wav",
      "samplingRate": "16000",
      "postProcessors": None
    },
    "audio": [
      {
        "audioContent": encoded_string
      }
    ]
  }
  
  # Send the API request
  x = requests.post(API_URL, data=json.dumps(data))
  print("Response:", x)
  print("Text:", json.loads(x.text)["output"][0]["source"])
