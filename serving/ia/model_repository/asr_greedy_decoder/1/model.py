import numpy as np
import json

import triton_python_backend_utils as pb_utils
import multiprocessing
from indictrans import Transliterator
from swig_decoders import map_batch

from inverse_text_normalization.hi.run_predict import inverse_normalize_text as hi_itn
from inverse_text_normalization.gu.run_predict import inverse_normalize_text as gu_itn
from inverse_text_normalization.mr.run_predict import inverse_normalize_text as mr_itn
from inverse_text_normalization.bn.run_predict import inverse_normalize_text as bn_itn
from inverse_text_normalization.ori.run_predict import inverse_normalize_text as or_itn
from inverse_text_normalization.pa.run_predict import inverse_normalize_text as pa_itn

from punctuate.punctuate_text import Punctuation
all_punct = { 
    'hi': Punctuation('hi'),
    'gu': Punctuation('gu'),
    'mr': Punctuation('mr'),
    'pa': Punctuation('pa'),
    'bn': Punctuation('bn'),
    'or': Punctuation('or')
}
# print("Doneee"*100)

def format_numbers_with_commas(sent, lang):
    words = []
    for word in sent.split(' '):
        word_contains_digit = any(map(str.isdigit, word))
        currency_sign = ''
        if word_contains_digit:
            if len(word) > 4 and ':' not in word:
                pos_of_first_digit_in_word = list(map(str.isdigit, word)).index(True)

                if pos_of_first_digit_in_word != 0:  # word can be like $90,00,936.59
                    currency_sign = word[:pos_of_first_digit_in_word]
                    word = word[pos_of_first_digit_in_word:]

                s, *d = str(word).partition(".")
                # getting [num_before_decimal_point, decimal_point, num_after_decimal_point]
                if lang == 'hi':
                    # adding commas after every 2 digits after the last 3 digits
                    r = "".join([s[x - 2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
                else:
                    r = "".join([s[x - 3:x] for x in range(-3, -len(s), -3)][::-1] + [s[-3:]])

                word = "".join([r] + d)  # joining decimal points as is

                if currency_sign:
                    word = currency_sign + word
                words.append(word)
            else:
                words.append(word)
        else:
            words.append(word)
    return ' '.join(words)


def inverse_normalize_text(text_list, lang):
    if lang == 'hi':
        itn_results = hi_itn(text_list)
        itn_results_formatted = [format_numbers_with_commas(sent=sent, lang=lang) for sent in itn_results]
        return itn_results_formatted
    elif lang == 'gu':
        itn_results = gu_itn(text_list)
        itn_results_formatted = [format_numbers_with_commas(sent=sent, lang='hi') for sent in itn_results]
        return itn_results_formatted
    elif lang == 'mr':
        itn_results = mr_itn(text_list)
        itn_results_formatted = [format_numbers_with_commas(sent=sent, lang='hi') for sent in itn_results]
        return itn_results_formatted
    elif lang == 'pa':
        itn_results = pa_itn(text_list)
        itn_results_formatted = [format_numbers_with_commas(sent=sent, lang='hi') for sent in itn_results]
        return itn_results_formatted
    elif lang == 'bn':
        itn_results = bn_itn(text_list)
        itn_results_formatted = [format_numbers_with_commas(sent=sent, lang='hi') for sent in itn_results]
        return itn_results_formatted
    elif lang == 'or':
        itn_results = or_itn(text_list)
        itn_results_formatted = [format_numbers_with_commas(sent=sent, lang='hi') for sent in itn_results]
        return itn_results_formatted
    
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
        self.vocab = ['<unk>', '▁क', '▁स', '्य', '▁प', '्र', '▁म', 'ार', '▁ह', '▁आ', '▁न', '▁ब', '▁द', '▁अ', 'ान', '▁ज', '्या', '▁व', '▁र', '▁त', '▁के', 'र्', 'या', '्त', '▁कर', 'ला', 'ें', '▁प्र', 'ां', 'िक', '▁ग', 'ना', '▁है', '▁श', 'ों', '▁ल', 'ित', '▁उ', '▁ए', 'ाल', '▁में', 'ात', '▁भ', '▁च', 'ने', '▁को', 'न्', '▁इ', '▁की', 'ाम', 'क्', '▁कि', '▁पर', 'य़', '▁से', '्री', 'ाज', 'िल', 'ास', '▁य', 'ता', 'ंत', '▁वि', '▁ने', '्ट', 'हे', 'ेश', '▁सं', '▁हो', 'रे', '▁का', 'वा', 'हा', '▁या', '▁एक', 'नी', '▁औ', 'क्ष', '▁नि', '▁और', 'ुर', 'ही', '्थ', '▁सम', 'बा', 'ती', 'ड़', '्क', '्व', '्य़', 'री', '▁छ', '▁मु', 'ति', 'ले', 'ाव', '▁ख', '▁राज', '▁ट', '▁इस', 'कार', 'न्त', 'ली', '▁आहे', 'िय', '▁फ', 'च्या', 'िस', 'रा', 'ोग', 'हि', 'ाग', 'ते', '▁दे', '▁पा', 'ून', 'ाय', 'ॆर', '▁दि', '्प', '▁अस', '▁बि', '▁कार', 'ंद', '▁थ', '▁घ', 'ल्या', 'ष्ट', 'चा', 'का', '▁ड', '▁ऎ', '▁सा', 'ीस', 'ारी', 'द्', 'िए', 'ंग', 'देश', 'न्द', 'ंत्री', '▁ला', '▁हैं', '▁आज', 'धान', '▁ते', 'ेत', 'सा', 'णि', '▁उप', '▁अध', 'ण्या', 'दी', '▁सु', '▁सर', '▁जा', 'ोज', 'ची', 'ङ्क', '्म', 'कर', 'त्त', '▁आप', '▁बा', '▁अन', 'ीन', '▁हज', '्ष', '▁मा', 'वि', '▁रा', 'मां', 'क्त', '▁दो', '्च', 'रो', '▁', 'ा', 'र', '्', 'क', 'े', 'ि', 'न', 'त', 'स', 'ी', 'य', 'म', 'ं', 'ल', 'ह', 'प', 'ो', 'द', 'व', 'ब', 'ज', 'ु', 'ग', 'च', 'श', 'आ', 'ट', 'ण', 'अ', '़', 'ध', 'ड', 'भ', 'ॆ', 'ू', 'ष', 'ए', 'ै', 'इ', 'थ', 'ख', 'उ', 'छ', 'ठ', 'ळ', 'फ', 'ई', 'औ', 'ौ', 'ँ', 'घ', 'ओ', 'ृ', 'ङ', 'ऎ', 'झ', 'ढ', 'ॊ', 'ॉ', 'ञ', 'ऒ', 'ऊ', 'ऐ', 'ः', 'ऱ', 'ऑ', 'ऋ', 'র', 'ऽ', 'ऍ', 'ॅ', 'ॐ', 'm', 'ऩ', 'ॄ', 'ব', 'ऴ', 'z', 'ॠ']
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
        request_batch_size = []
        request_batch_lang = []
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
        map2tgt = {"bn": "ben", "gu": "guj", "mr":"mar", "or":"ori", "pa":"pan", "ur":"urd", "hi":"hin", "en":"eng"}
        start = 0
        for i,b in enumerate(request_batch_size):
            sent = res_sents[start:start+b]
            lang = request_batch_lang[i]
            if lang=="hi" or lang=="sa":
                # sent = np.array(inverse_normalize_text([s.replace("▁", " ").strip() for s in sent], lang="hi"))
                sent = [s.replace("▁", " ") for s in sent]
            else:
                trn = Transliterator(source='hin', target=map2tgt[lang], build_lookup=True)
                sent = [trn.transform(s.replace("▁", " ").strip()) for s in sent]
            
            sent = all_punct[lang].punctuate_text(sent)
            sent = np.array(inverse_normalize_text(sent, lang=lang))
            
            out_tensor_0 = pb_utils.Tensor("TRANSCRIPT", sent.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
            start = start + b
        return responses
