import numpy as np
import json
from swig_decoders import map_batch

import triton_python_backend_utils as pb_utils
import multiprocessing

class TritonPythonModel:

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        parameters = model_config["parameters"]
        self.subsampling_rate = 2
        self.blank_id = 0
        for li in parameters.items():
            key, value = li
            value = value['string_value']
            if key == "subsampling_rate":
                self.subsampling_rate = int(value)
            elif key == "blank_id":
                self.blank_id = int(value)
        self.vocab = ['<unk>', 'ா', 'ி', 'ு', 'வ', 'க', '▁ப', 'ை', 'ன', 'ர', 'ன்', '்', '▁க', 'ம்', 'த', 'ே', 'ய', 'ல்', '▁அ', 'ர்', 'க்க', '▁வ', 'ல', '▁ம', 'து', 'ட', 'ப்ப', 'ம', '▁த', 'ப', '▁', 'ச', 'ட்ட', 'ண', '▁ச', '▁இ', 'ும்', 'ிய', 'ோ', '▁எ', 'ெ', 'த்த', 'ூ', 'ங்க', '▁ந', 'ழ', 'ொ', 'ரு', 'தி', 'ற', 'ள', 'த்தி', 'ந்த', 'க்கு', 'ீ', 'டி', 'டு', 'ார்', 'த்து', '▁ஆ', 'ரி', 'ற்ற', 'ட்', 'கள்', '▁உ', 'ஸ்', 'வி', 'று', 'ுள்ள', '▁மு', 'லை', 'ந்து', 'ண்ட', 'ல்ல', 'க்', 'ச்ச', 'ள்', 'ளி', 'ன்ற', '▁இரு', 'ங்கள', 'யி', '▁இந்த', '▁வி', 'ட்டு', '▁செ', '▁நா', 'யில்', 'றி', 'மா', 'ந', 'ப்', 'யா', '▁கு', 'ஜ', 'டை', '▁போ', 'ற்க', '▁தொ', '▁ர', '▁நி', 'றை', 'ப்பு', 'ண்டு', '▁ஒ', '▁செய்த', 'ஷ', 'ஐ', 'ஞ', 'ஹ', 'ஓ', 'ஃ', 'ங', 'ஊ', 'ஈ', 'எ', 'ஸ', 'ௌ', 'ஆ', 'இ', 'ஏ', 'அ', 'ஒ', 'உ', 'ஔ', 'ஶ', '௧', '௭']
        
        if self.blank_id == -1:
            self.blank_id = len(self.vocab)
        self.num_processes = multiprocessing.cpu_count()
        if args["model_instance_kind"] == "GPU":
            print("GPU GREEDY DECODER IS NOT SUPPORTED!")
        
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSCRIPT")
        
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        """
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        batch_transcript = []
        request_batch_size = []
        responses = []
        for request in requests:
            # B X T
            in_0 = pb_utils.get_input_tensor_by_name(request, "TRANSCRIPT_ID")
            transcript_id = in_0.as_numpy().tolist()
            cur_batch_size = len(transcript_id)
            request_batch_size.append(cur_batch_size)
            # B X 1
            in_1 = pb_utils.get_input_tensor_by_name(request, "NUM_TIME_STEPS")
            timesteps = in_1.as_numpy()
            for i in range(cur_batch_size):
                cur_len = (timesteps[i][0] + 1) // self.subsampling_rate
                batch_transcript.append(transcript_id[i][0:cur_len])

        num_processes = min(self.num_processes, len(batch_transcript))
        res_sents = map_batch(batch_transcript, self.vocab, num_processes, True, self.blank_id)
        start = 0
        for b in request_batch_size:
            sent = res_sents[start:start+b]
            sent = np.array([s.replace("▁", " ").strip() for s in sent])
            out_tensor_0 = pb_utils.Tensor("TRANSCRIPT", sent.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
            start = start + b
        return responses
