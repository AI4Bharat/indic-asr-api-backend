# python3 client_ensemble.py ta
import numpy as np
import tritonclient.http as http_client
import sys
from datasets import load_dataset

language = sys.argv[1]

triton_http_client = http_client.InferenceServerClient(url='localhost:8000', verbose=False)
results = []

audio_dataset = load_dataset("google/fleurs", f"{language}_in", split="test", streaming=True)
print("Ground Truth :", next(iter(audio_dataset))['transcription'])
audio_signal = np.asarray([next(iter(audio_dataset))["audio"]["array"]]).astype("float32")
audio_len = np.asarray(audio_signal.shape[1]).reshape(1,-1).astype("int32")

input0 = http_client.InferInput("AUDIO_SIGNAL", audio_signal.shape, "FP32")
input1 = http_client.InferInput("NUM_SAMPLES", audio_len.shape, "INT32")
input2 = http_client.InferInput("LANG_ID", audio_len.shape, "BYTES")

input0.set_data_from_numpy(audio_signal)
input1.set_data_from_numpy(audio_len.astype('int32'))
lang_id = [language]*len(audio_len)
input2.set_data_from_numpy(np.asarray(lang_id).astype('object').reshape(audio_len.shape))

output0 = http_client.InferRequestedOutput('TRANSCRIPTS')
if language == "hi" or language == "en" or language == "ta":
    response = triton_http_client.infer("asr_am_ensemble", model_version='1',inputs=[input0, input1], outputs=[output0])
else:
    response = triton_http_client.infer("asr_am_ensemble", model_version='1',inputs=[input0, input1, input2], outputs=[output0])
    
result_response = response.get_response()
batch_result = response.as_numpy("TRANSCRIPTS")
print("Prediction :", batch_result[0].decode("utf-8"))

