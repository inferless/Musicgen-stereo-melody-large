import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import base64

class InferlessPythonModel:
    
    def initialize(self):
        self.model = MusicGen.get_pretrained('facebook/musicgen-stereo-melody-large')
        self.model.set_generation_params(duration=8)

        
    def infer(self, inputs):
        
        descriptions = [inputs["prompt"]]
        wav = self.model.generate(descriptions)
        
        for idx, one_wav in enumerate(wav):
            audio_write("temp", one_wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
        
        with open("temp.wav", "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {"generated_audio_base64": audio_base64}

    def finalize(self,args):
        self.model = None
