from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/hi_speaker_2"

inputs = processor("Hello Mera Naam Shirish Waghmode", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.gpu().numpy().squeeze()

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)