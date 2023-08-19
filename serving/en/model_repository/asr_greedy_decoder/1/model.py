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
        # self.vocab = ['<unk>', 'ा', 'े', 'र', 'न', 'ी', 'ि', 'ल', 'क', 'स', 'म', '्', '▁स', 'त', 'ो', 'ट', 'ं', '▁द', 'प', '▁क', 'ह', '▁', 'ु', '▁ब', '▁अ', 'व', '▁प', '▁म', '्र', '▁ज', 'ू', '▁व', '▁आ', 'द', '▁है', 'ग', 'ड', '▁र', 'ज', 'र्', '्य', 'य', '▁ह', 'ै', '▁न', 'श', 'ब', '▁ल', '▁के', 'ज़', '▁में', '▁इ', '▁य', 'च', '्ट', 'ए', '▁प्र', '▁त', 'ों', 'ाइ', '्स', '▁उ', 'ख', '▁का', 'भ', 'ॉ', '▁ग', '▁की', 'ध', '▁से', '▁को', 'फ', '▁और', '▁फ', 'थ', '▁टू', 'न्ट', '▁वि', 'स्ट', 'िक', '▁इन', '▁च', '▁ड', '▁कर', '▁अन्ड', 'ई', '्व', '▁कि', 'ता', '्ल', '▁अव', '▁हैं', 'ने', '▁हो', '▁श', '▁ट', 'ंग', 'ण', 'ेश', '▁ए', '▁थ', '▁इस', 'िया', '्त', 'उ', 'ष', 'इ', 'ौ', 'अ', 'छ', 'ओ', 'ठ', '़', 'आ', 'ृ', 'ढ', 'घ', 'ँ', 'झ', 'ऊ', 'ऐ', 'ः', 'औ', 'ऑ', 'ङ', 'ऋ', 'ञ', 'ॅ']
        # self.vocab = ['<unk>', 's', '▁', 'e', 't', 'u', 'd', 'a', 'o', 'n', 'i', '▁the', '▁a', 'm', 'y', 'l', 'h', 'p', 're', '▁s', 'g', 'r', '▁to', '▁i', 'ing', '▁and', 'f', '▁p', 'an', 'c', 'w', 'er', 'ed', '▁of', '▁in', 'k', "'", '▁w', 'ar', 'or', '▁f', 'b', '▁b', 'en', '▁you', 'al', 'le', 'in', 'll', '▁that', '▁he', 'ro', '▁t', 'es', '▁it', '▁be', 've', 'v', 'ly', '▁c', 'th', '▁o', 'ent', 'ch', 'ur', '▁we', '▁re', '▁n', 'it', '▁so', '▁co', '▁g', '▁on', '▁for', 'on', 'ce', 'ri', '▁do', '▁is', '▁ha', '▁ma', 'ver', 'li', 'ra', '▁was', 'ic', 'la', '▁e', 'se', 'ter', 'ct', 'ion', '▁ca', '▁st', '▁me', 'ir', '▁mo', '▁with', '▁but', '▁have', '▁go', '▁de', '▁ho', '▁di', '▁not', '▁know', '▁lo', '▁this', 'ation', 'ther', 'ate', '▁com', '▁like', '▁uh', 'ck', '▁his', 'j', '▁yeah', '▁my', '▁ex', '▁what', '▁will', '▁mi', 'q', 'ight', 'x', 'z', '-']
        # self.vocab = ['<unk>', '▁', 's', 't', 'e', 'd', 'o', '▁the', 'a', 'i', '▁a', 'u', 'y', 'm', 'l', 'n', 'p', 're', 'c', 'h', 'r', '▁s', 'g', '▁to', 'er', 'ing', 'f', '▁and', 'an', '▁i', 'k', '▁that', "'", '▁of', '▁in', 'w', '▁p', 'ed', 'or', 'al', 'ar', '▁f', 'en', 'in', 'b', '▁you', '▁w', '▁b', 'le', 'll', 'es', '▁it', 've', 'ur', '▁we', '▁re', '▁be', 'ly', '▁is', '▁he', '▁o', '▁c', 'it', '▁n', '▁on', 'un', '▁t', 'on', 'se', 'th', 'ce', '▁do', 'ic', '▁for', '▁th', 'ion', 'ch', '▁was', 'ri', 'ent', '▁g', 'ver', '▁co', 'li', '▁ha', '▁ma', 'la', 'ro', 'v', 'us', '▁ca', '▁di', '▁this', 'ra', '▁st', '▁e', '▁not', '▁so', '▁de', '▁have', 'ter', 'ir', '▁go', 'ation', '▁with', 'ate', '▁me', '▁mo', 'ment', '▁con', '▁but', 'vi', '▁pro', '▁ho', 'j', '▁com', 'ight', '▁know', '▁what', 'ect', '▁ex', '▁some', '▁would', '▁like', 'x', '▁his', 'q', 'z']
        self.vocab = ["<unk>", "s", "t", "e", "▁the", "d", "▁", "▁a", "i", "n", "a", "m", "▁to", "y", "o", "ing", "▁and", "er", "p", "u", "▁in", "▁of", "'", "▁i", "▁that", "ed", "re", "r", "c", "h", "al", "ar", "f", "▁you", "▁s", "▁f", "an", "b", "▁it", "l", "w", "▁is", "▁p", "in", "▁we", "▁re", "▁be", "es", "g", "or", "▁he", "▁c", "ly", "le", "k", "en", "▁for", "▁w", "ll", "ur", "ic", "ri", "▁e", "▁so", "on", "ct", "ve", "▁b", "▁g", "▁st", "it", "▁t", "▁do", "ra", "▁on", "▁was", "▁this", "ent", "th", "ro", "ce", "▁have", "▁de", "▁o", "ter", "▁ma", "▁se", "▁co", "▁di", "ation", "▁with", "▁not", "▁m", "il", "▁me", "us", "ir", "▁are", "v", "▁but", "▁pro", "▁th", "ch", "▁con", "ate", "me", "at", "la", "li", "▁they", "ver", "▁go", "▁what", "▁ha", "vi", "▁ne", "▁or", "ive", "▁as", "▁there", "▁know", "ment", "un", "lo", "▁su", "▁can", "is", "▁ex", "▁ch", "▁mo", "ck", "ul", "▁like", "tion", "el", "▁le", "▁one", "ng", "ci", "▁ca", "▁an", "▁all", "ne", "ge", "▁lo", "x", "ut", "▁la", "▁if", "▁at", "▁un", "ol", "qu", "▁no", "▁fa", "as", "▁ho", "ity", "▁just", "▁would", "▁about", "▁from", "▁ba", "▁v", "mp", "▁think", "▁my", "z", "co", "ad", "▁us", "▁will", "▁li", "end", "▁by", "ight", "▁some", "▁po", "▁his", "ig", "ry", "▁your", "▁our", "▁out", "▁pa", "ff", "▁don", "ru", "▁had", "▁te", "▁up", "j", "▁when", "▁because", "▁which", "▁da", "▁get", "age", "▁sp", "▁two", "▁bo", "▁say", "sion", "ction", "▁pre", "▁were", "ence", "▁how", "▁time", "▁k", "▁who", "▁mi", "▁right", "▁comp", "able", "▁she", "▁any", "▁more", "ugh", "▁now", "▁other", "▁yeah", "▁app", "ance", "▁uh", "▁also", "▁people", "▁part", "▁want", "▁very", "ound", "▁work", "▁look", "▁comm", "port", "▁year", "▁case", "▁court", "▁really", "▁said", "side", "▁where", "▁could", "▁make", "▁even", "▁dr", "▁every", "▁those", "▁take", "▁ju", "▁three", "▁good", "▁first", "▁should", "▁point", "q"]
        
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
