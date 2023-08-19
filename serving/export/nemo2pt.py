# python3 src/nemo2pt.py <path-to-nemo-file>

import torch
import nemo.collections.asr as nemo_asr
import sys
path = sys.argv[1]
model = nemo_asr.models.EncDecCTCModelBPE.restore_from(path)

out_path = path.replace(".nemo", ".pt")
with torch.autocast(device_type="cuda", enabled=False):
    model.export(out_path)
print(model.decoder.vocabulary)