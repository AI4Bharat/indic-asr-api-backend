import numpy as np
import json
from swig_decoders import map_batch

from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
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
        # self.vocab = ['<unk>', '்த', '்க', '▁ப', '▁க', '▁ம', '்ட', 'த்த', '்ப', 'ல்', '▁அ', '▁ந', '▁வ', '▁த', 'ரு', 'க்க', 'ங்', 'ந்', 'ம்', 'ர்', '்ர', 'ிய', '▁ச', '▁இ', '▁ஸ', 'ங்த', 'ிக', 'ன்', 'ப்ப', 'து', 'ட்ட', 'ாக', '▁எ', 'ள்', 'ந்ந', '்ய', 'ார', 'ங்க', 'தி', 'ந்த', '▁ஆ', 'டு', 'ல்ல', 'ரி', 'று', '▁உ', '்ச', 'ித', 'த்து', 'ில்', '▁ர', 'ற்', 'டி', 'ுக', 'ெய', 'ள்ள', 'ாய', 'ும்', '▁ப்ர', 'ால', 'ில', 'ாத', 'ாந', 'ைய', 'ச்ச', 'ண்ட', 'வி', '▁ஒ', '▁ஹ', '▁வி', 'க்கு', 'ங்ட', 'வு', 'ார்', 'த்தி', 'ந்நு', '▁ஜ', 'ஸ்', '▁மு', 'ிர', 'கள', 'க்', 'ல்லி', 'த்', '▁கு', 'ிஸ', '▁தி', 'நி', 'லு', 'கு', 'ட்', 'ான', '▁அவ', 'ாண', '▁நி', 'றி', 'ர்க', '▁ல', 'ன்று', 'ங்ச', 'ேச', '▁ஈ', 'ங்ப', 'ட்டு', 'ாவ', 'ஸ்த', 'ாம', 'ின்', 'நு', '▁அத', 'ளு', 'ிற', 'ற்ற', '▁கொ', '▁கோ', '▁செய', 'ாப', '▁நா', 'டெ', 'ங்து', 'ற்று', 'ங்தி', '்வ', 'மை', '▁ஏ', 'ாரு', 'க்ஷ', '▁மா', 'லோ', '▁ய', '▁தெ', 'ாயி', '▁இந்த', '▁தொ', '▁போ', 'ியு', 'ிகெ', '▁கூ', 'ப்', 'ங்ங', '▁நீ', 'ளி', 'ங்த்ர', 'மி', '▁வே', '▁ரா', '▁மூ', '▁நட', 'ாகி', 'ற்க', 'ப்பு', '▁சே', '▁இரு', 'ந்து', 'வே', '▁பெ', 'ிதெ', 'ிந', 'ுவ', 'ஷ்ட', '▁ட', 'ால்', 'ோக', 'ம்ப', '▁பு', '▁பி', '▁கே', 'ட்டி', 'ர்த', '▁திரு', 'நா', '▁வெ', 'ரெ', '▁ஒரு', 'லி', '▁செ', 'லை', 'ளை', '▁மே', '▁பா', '▁என', 'லா', 'தெ', 'ஸி', 'ண்டு', '▁கா', 'த்தில்', 'ம்ம', 'ன்ற', 'த்ய', 'ருவ', 'னை', 'பி', '்', '▁', 'த', 'ி', 'க', 'ு', 'ர', 'ா', 'ப', 'ந', 'ட', 'ம', 'ல', 'வ', 'ய', 'ங', 'ச', 'ெ', 'ள', 'ஸ', 'ற', 'ே', 'ன', 'ை', 'ோ', 'அ', 'ண', 'இ', 'ீ', 'ூ', 'எ', 'ொ', 'ஜ', 'ஹ', 'ஆ', 'ஷ', 'உ', 'ழ', 'ஒ', 'ൽ', 'ർ', 'ஈ', 'ஞ', 'ஏ', 'ൾ', 'ൻ', 'ஐ', 'ௌ', 'ஓ', 'ஊ', 'ൺ', 'ஃ', 'ஔ', 'ஶ', 'र', 'क', 'ह', 'ा', 'ी', 'ट', '्', 'ग', 'च']
        self.vocab = ['<unk>', '്ത', '്ക', '▁പ', '▁ക', '▁മ', 'ല്', 'ത്ത', 'ര്', '▁വ', '▁ന', '▁അ', '്ട', 'രു', 'ക്ക', 'മ്', '▁ത', 'ന്', '്ര', '്പ', 'ിയ', '▁സ', '▁ഇ', '▁ച', 'ഩ്', 'പ്പ', 'ട്ട', 'രി', 'ള്', '▁എ', 'ന്ന', '്യ', 'തു', 'ംദ', 'ില്', 'ിക', 'ന്ത', '▁ആ', 'ുമ്', 'ാര', '്ച', '▁ഉ', 'ലി', '▁ര', 'തി', 'റു', 'ള്ള', '▁വി', 'ാക', 'ായ', 'ത്തു', 'വി', 'റ്', 'െയ', 'ടു', '▁പ്ര', 'ാര്', 'ങ്ക', 'ിദ', 'ാന', 'ൈയ', 'ാഗ', 'ലു', 'ച്ച', '▁ഹ', 'ക്കു', 'സ്', '▁ബ', '▁മു', 'ണ്ട', '▁നി', 'വു', 'ടി', '▁ജ', 'ത്തി', '▁ഒ', 'ാല', 'ത്', 'ളി', 'നി', 'ക്', 'റ്റ', 'ന്നു', 'ല്ലി', 'കു', '്ദ', '▁മാ', 'ലാ', 'ഗള', 'ലോ', 'ിരു', '▁ദ', 'മാ', 'ംത', 'ിര', '▁അവ', 'ട്ടു', 'വാ', 'ഗെ', 'ിസ', '▁കു', 'ഡു', '▁ഗ', 'ര്ക', 'ംഗ', '▁നാ', 'ട്', 'ംച', '▁രാ', 'നു', '▁ഈ', 'ണ്', 'റി', 'ംദു', 'ളു', 'ഩ്റു', '▁ശ', 'ിന', 'ലൈ', 'ക്ഷ', 'േശ', 'ിറ', '▁കോ', '▁കൊ', '▁ചെയ', 'മൈ', '▁കാ', 'ഩ്റ', 'നാ', '▁സം', '▁തെ', 'മി', 'ാരു', '▁ഭ', 'ുക', '▁ഏ', 'രാ', '▁യ', 'ങ്', '▁തൊ', 'ഡി', '▁ഇന്ത', '▁കൂ', 'ിദ്ദ', 'ായി', 'റ്ക', '▁തിരു', '▁നട', 'ംദി', 'ളൈ', 'പ്പു', '▁കേ', 'ന്തു', '്രി', 'ില', 'ിയു', 'ല്ല', '▁വേ', '▁നീ', '്വ', 'ിയാ', 'ങ്ങ', 'ദു', '▁ല', '▁മൂ', 'സ്ത', 'ഷ്ട', '▁പു', 'ദി', '▁പോ', '▁ചേ', '▁പാ', 'ുവ', 'മു', 'പ്', 'ാഩ', 'വേ', 'ിദെ', 'ംഡ', '▁വെ', 'ത്തില്', 'ദ്ദ', '▁ഇരു', 'ഩ്ഩ', 'ടെ', '▁ചെ', '▁മേ', 'ാരി', '്യാ', 'ിവ', 'താക', 'മ്പ', 'ാല്', 'രെ', 'ണി', 'ണ്ടു', 'ഡ്', 'സി', 'ദല്ലി', 'റൈ', 'ോഗ', 'ട്ടി', 'ദ്', 'മെ', 'ത്ര', 'പ്പി', 'തിക', 'ിയി', 'ാവ', '▁പി', '▁തി', '▁തേ', '▁രാജ', '▁പേ', '▁സാ', '▁നേ', 'കാ', 'ാഗി', '▁പെ', '▁ഒരു', 'ൈയില്', 'ില്ല', 'ഩൈ', '▁അര', 'ള്ളതു', '▁പരി', 'സ്ഥ', 'ര്ത', 'റ്റു', '▁വാ', 'ക്കപ്പ', 'ഩ്പ', '▁മറ്റ', 'ും', 'ാരെ', 'മ്മ', '▁പൊ', 'പി', '▁ഐ', '▁ആയ', 'നെ', 'ന്ന്', '▁സ്', 'വൈ', 'ധി', '▁എംദു', 'തിയ', 'രണ', 'പു', 'ക്കെ', 'ംഗാ', 'ഴു', '▁തെരി', 'യി', 'ാമ', '▁മത്തു', '▁മറ്റുമ്', 'ികി', 'ംബ', '▁തു', '▁നിര്', 'ളില്', '▁മീ', 'ംദ്ര', 'ദെ', 'ധാന', 'ടൈ', 'ച്ചു', '▁അമൈ', 'രോ', 'ഡെ', '▁പിര', 'ട്ച', 'കി', 'റ്പ', 'സു', 'ത്തിയ', '്ഞ', '▁തെരിവി', 'ദ്യ', 'ക്കുമ്', '▁തീ', '▁ചെയ്ത', 'ഗളു', 'രേ', '▁ചി', '▁വൈ', 'ാര്യ', 'ങ്കള്', '▁അരച', 'പാ', '▁ഉള്ള', '▁എഩ്റ', 'ദേശ', 'ക്ത', 'ംലോ', 'വെ', 'താ', 'ലെ', '▁ഇര', '▁എഩ്റു', 'ഖ്യ', 'ധ്യ', '്', '▁', 'ി', 'ു', 'ക', 'ത', 'ര', 'ാ', 'ന', 'പ', 'മ', 'ല', 'വ', 'ട', 'യ', 'െ', 'ദ', 'ം', 'ള', 'ച', 'സ', 'റ', 'േ', 'ൈ', 'ഩ', 'ഗ', 'ോ', 'അ', 'ണ', 'ഡ', 'ഇ', 'ീ', 'ൂ', 'ൊ', 'എ', 'ജ', 'ബ', 'ഹ', 'ങ', 'ഷ', 'ആ', 'ശ', 'ഉ', 'ധ', 'ഴ', 'ഭ', 'ഒ', 'ൽ', 'ർ', 'ഥ', 'ഖ', 'ഈ', 'ഞ', 'ഏ', 'ൻ', 'ൾ', 'ഫ', 'ൃ', 'ഐ', 'ഘ', 'ൌ', '1', '0', '2', 'ഓ', 'ഠ', 'ഊ', 'ൺ', '9', '3', '5', '4', '7', '6', 'ഃ', '8', 'ഛ', 'ഢ', 'ഔ', 'a', 'ഋ', 'i', 'c', 'r', 'p', 't', 'n', 'ഝ', 'e', 'b', 'o', 's', 'd', 'v', 'm', 'x', 'l', 'f', 'y', 'g', 'h', 'u', 'k', '൦', 'q', 'z', 'j', 'w', 'α', 'δ', 'π', 'φ', 'ω', 'θ', '഼', 'μ', 'ψ', '০', 'λ', 'β', '൨', 'τ', '×', 'ρ', '൧', '൯', 'ഌ', 'ε', 'ൕ', '൩', 'ഁ', '̂', '−', '∞', '°', 'ൖ', '൪', '൫', '൭', '′', 'σ', 'ʼ', 'η', '൮', '൬', '•', 'ν', '∆', 'γ', '√', 'ɵ', 'õ', 'र', 'ह', '→', 'ß', 'ô', 'ɸ', 'ൄ', '⃗', '∂', '∅', '∇', '+', 'á', '÷', 'ħ', 'ʹ', 'ф', 'ѱ', 'ा', '൞', '∗', '⍴', '▽', 'û', 'ˆ', '̇', 'ζ', 'χ', 'ी', '౼', 'ൡ', 'ൿ', '⁄', '≠', '⊕', '⌀', '⌉', '±', '¿', 'à', 'â', 'ü', 'ý', 'ŝ', '̈', '̤', 'ξ', 'च', 'ल', '௸', 'ഽ', '\u0d50', 'ൠ', 'ṡ', 'ẋ', '∩', '∼']
        # self.vocab = ['<unk>', 's', '▁', 'e', 't', 'u', 'd', 'a', 'o', 'n', 'i', '▁the', '▁a', 'm', 'y', 'l', 'h', 'p', 're', '▁s', 'g', 'r', '▁to', '▁i', 'ing', '▁and', 'f', '▁p', 'an', 'c', 'w', 'er', 'ed', '▁of', '▁in', 'k', "'", '▁w', 'ar', 'or', '▁f', 'b', '▁b', 'en', '▁you', 'al', 'le', 'in', 'll', '▁that', '▁he', 'ro', '▁t', 'es', '▁it', '▁be', 've', 'v', 'ly', '▁c', 'th', '▁o', 'ent', 'ch', 'ur', '▁we', '▁re', '▁n', 'it', '▁so', '▁co', '▁g', '▁on', '▁for', 'on', 'ce', 'ri', '▁do', '▁is', '▁ha', '▁ma', 'ver', 'li', 'ra', '▁was', 'ic', 'la', '▁e', 'se', 'ter', 'ct', 'ion', '▁ca', '▁st', '▁me', 'ir', '▁mo', '▁with', '▁but', '▁have', '▁go', '▁de', '▁ho', '▁di', '▁not', '▁know', '▁lo', '▁this', 'ation', 'ther', 'ate', '▁com', '▁like', '▁uh', 'ck', '▁his', 'j', '▁yeah', '▁my', '▁ex', '▁what', '▁will', '▁mi', 'q', 'ight', 'x', 'z', '-']
        # self.vocab = ['<unk>', '▁', 's', 't', 'e', 'd', 'o', '▁the', 'a', 'i', '▁a', 'u', 'y', 'm', 'l', 'n', 'p', 're', 'c', 'h', 'r', '▁s', 'g', '▁to', 'er', 'ing', 'f', '▁and', 'an', '▁i', 'k', '▁that', "'", '▁of', '▁in', 'w', '▁p', 'ed', 'or', 'al', 'ar', '▁f', 'en', 'in', 'b', '▁you', '▁w', '▁b', 'le', 'll', 'es', '▁it', 've', 'ur', '▁we', '▁re', '▁be', 'ly', '▁is', '▁he', '▁o', '▁c', 'it', '▁n', '▁on', 'un', '▁t', 'on', 'se', 'th', 'ce', '▁do', 'ic', '▁for', '▁th', 'ion', 'ch', '▁was', 'ri', 'ent', '▁g', 'ver', '▁co', 'li', '▁ha', '▁ma', 'la', 'ro', 'v', 'us', '▁ca', '▁di', '▁this', 'ra', '▁st', '▁e', '▁not', '▁so', '▁de', '▁have', 'ter', 'ir', '▁go', 'ation', '▁with', 'ate', '▁me', '▁mo', 'ment', '▁con', '▁but', 'vi', '▁pro', '▁ho', 'j', '▁com', 'ight', '▁know', '▁what', 'ect', '▁ex', '▁some', '▁would', '▁like', 'x', '▁his', 'q', 'z']
        
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
        request_batch_lang = []
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
            in_2 = pb_utils.get_input_tensor_by_name(request, "LANG_ID")
            # print([s.decode("utf-8") for s in in_2.as_numpy()])
            # NOTE: Assumption is that the "lang" in a requrest-batch remains same
            request_batch_lang.append(in_2.as_numpy()[0].decode("utf-8"))
            for i in range(cur_batch_size):
                cur_len = (timesteps[i][0] + 1) // self.subsampling_rate
                batch_transcript.append(transcript_id[i][0:cur_len])

        num_processes = min(self.num_processes, len(batch_transcript))
        res_sents = map_batch(batch_transcript, self.vocab, num_processes, True, self.blank_id)
        start = 0
        for i,b in enumerate(request_batch_size):
            sent = res_sents[start:start+b]
            lang = request_batch_lang[i]
            sent = np.array([UnicodeIndicTransliterator.transliterate(s.replace("▁", " ").strip(),"ml",lang) for s in sent])
            out_tensor_0 = pb_utils.Tensor("TRANSCRIPT", sent.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
            start = start + b
        return responses
