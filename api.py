import os, subprocess
import base64, json
import traceback
import urllib
import time
import io

import torch
import numpy as np
from pydub import AudioSegment
import nemo.collections.asr as nemo_asr

from flask import Flask, request
from flask import jsonify
from flask_sockets import Sockets
from flask_cors import CORS, cross_origin

PORT = 4992

# Load config file
with open('conformer.json','r') as j:
    config = json.load(j)

# Load models to GPU / CPU
name2model_dict = dict()
for k,m in config.items():
    # Load from checkpoint
    model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=m['model_path'])
    model.freeze() # inference mode
    model = model.to(m["device"]) # transfer model to device
    name2model_dict[k] = model

# Create the flask app
app = Flask(__name__)
sockets = Sockets(app) # enable sockets
cors = CORS(app) # enable cross-origin
app.config['CORS_HEADERS'] = 'Content-Type'


def softmax(logits):
    '''Perform the softmax operation on logits'''
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

def transcribe(wav_array, model):
    '''Transcribe a given audio as a wav array and a model path'''
    feature = torch.from_numpy(wav_array).float()
    lengths = torch.Tensor([len(feature)])

    feature = feature.to(model.device)
    lengths = lengths.to(model.device)
    model.freeze()
    with torch.no_grad():
        logits, logits_len, _ = model.forward(input_signal=feature.unsqueeze(0), input_signal_length=lengths)
        current_hypotheses, _ = model.decoding.ctc_decoder_predictions_tensor(
            logits, decoder_lengths=logits_len, return_hypotheses=True,
        )
        text = current_hypotheses[0].text
    return text

def predict_sample(wav_array, model):
    '''
    Run inference a given audio as a wav array and a model path
    This includes preprocessing steps like audio normalization, framerate conversion
    '''
    song=AudioSegment.from_raw(io.BytesIO(wav_array), sample_width=2, frame_rate=16000, channels=1)
    samples = song.get_array_of_samples()
    arr = np.array(samples).T.astype(np.float64)
    print(song.duration_seconds, "s, Duration of audio")
    arr /= np.iinfo(samples.typecode).max
    arr = arr.reshape(-1)
    op = transcribe(arr,model)

    return op
    
def load_data(wavpath,of='raw',**extra):
    '''Read an audio into an array from a file or a URL'''
    if of == 'raw':
        orig = wavpath
        tmp_file = orig + "_temp.wav"
        subprocess.call(['ffmpeg', '-i', orig,'-ar', '16k', '-ac', '1', '-hide_banner', '-loglevel','error',tmp_file])
        os.rename(tmp_file,orig)
        wav = AudioSegment.from_file(orig, sample_width=2, frame_rate=16000, channels=1)
        return wav.raw_data
    elif of == 'url': 
        lang = extra['lang']
        if not os.path.exists('downloaded/'+lang+'/'):
            os.makedirs('downloaded/'+lang+'/')
        urllib.request.urlretrieve(wavpath, 'downloaded/'+lang+'/'+os.path.split(wavpath)[1])
        print('url downloaded')
        return load_data('downloaded/'+lang+'/'+os.path.split(wavpath)[1])
    elif of == 'bytes':
        lang = extra['lang']
        name = extra['bytes_name']
        if not os.path.exists('downloaded/'+lang+'/'):
            os.makedirs('downloaded/'+lang+'/')

        with open('downloaded/'+lang+'/'+name, 'wb') as file_to_save:
            decoded_image_data = base64.b64decode(wavpath)
            file_to_save.write(decoded_image_data)
        return load_data('downloaded/'+lang+'/'+name)
    else:
        raise "Not implemented"


@app.route("/recognize/en",methods=['POST'])
@cross_origin()
def infer_ulca_en():
    req_data = json.loads(request.data)
    status = "SUCCESS"
    preds = []
    number_mode = float(req_data.get('number_mode',False))
    lang = 'en'
    for f in req_data['audio']:
        audio_uri, audio_bytes = f.get('audioUri',None),f.get('audioContent',None)
        la = req_data['config']['language']['sourceLanguage']
        af = req_data['config']['audioFormat']
        if audio_uri in [None,''] and audio_bytes in [None,'']:
            status = 'ERROR'
            print(traceback.format_exc())
            continue
        try:
            if audio_bytes == None:
                fp_arr = load_data(audio_uri,of='url',lang=la)
            else:
                nm = str(round(time.time() * 1000))
                fp_arr = load_data(audio_bytes,of='bytes',lang=la,bytes_name=nm+"."+af)
        except:
            status = 'ERROR'
            print(traceback.format_exc())
            continue
            
        model = name2model_dict[lang]
        res = predict_sample(fp_arr,model)
        preds.append({'source':res})
    return jsonify({"status":status, "output":preds})



if __name__ == "__main__":
    
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('0.0.0.0', PORT), app, handler_class=WebSocketHandler)
    print("Server listening on: http://0.0.0.0:" + str(PORT))
    server.serve_forever()
