from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from cog import Path


class InferlessPythonModel:

    # Implement the Load function here for the model
    def initialize(self):
        pass

    # Function to perform inference
    def infer(self, inputs):
        output_name = "output"
        output_name_with_format = "output.{}".format(inputs.get("audio_format", "wav"))
        model = MusicGen.get_pretrained(inputs.get("model_type", "medium"))
        model.set_generation_params(
            duration=inputs.get("duration", 5),
            use_sampling=inputs.get("use_sampling", True),
            top_k=inputs.get("top_k", 250),
            top_p=inputs.get("top_p", 0.0),
            temperature=inputs.get("temperature", 1.0),
            cfg_coef=inputs.get("cfg_coef", 3.0),
            two_step_cfg=inputs.get("two_step_cfg", False),
        )

        descriptions = [inputs.get("prompt", "A dynamic blend of hip-hop and orchestral elements, with sweeping strings and brass, evoking the vibrant energy of the city.")]

        output_data = model.generate(descriptions)
        wav = output_data[0]
        audio_write(
            stem_name=output_name,
            wav=wav.cpu(),
            sample_rate=model.sample_rate,
            format=inputs.get("audio_format", "wav"),
            mp3_rate=inputs.get("audio_mp3_rate", 320),
            normalize=inputs.get("audio_normalize", True),
            strategy=inputs.get("audio_strategy", "peak"),
            peak_clip_headroom_db=inputs.get("audio_peak_clip_headroom_db", 1.0),
            rms_headroom_db=inputs.get("audio_rms_headroom_db", 18.0),
            loudness_headroom_db=inputs.get("audio_loudness_headroom_db", 14.0),
            log_clipping=inputs.get("audio_log_clipping", True),
        )

        return {"output_audio": Path(output_name_with_format)}

    # perform any cleanup activity here
    def finalize(self):
        pass


# def main():
#     inferless_python_model = InferlessPythonModel()
#     res = inferless_python_model.infer(
#         inputs={
#             "prompt": "A dynamic blend of hip-hop and orchestral elements, with sweeping strings and brass, evoking the vibrant energy of the city.",
#             "model_type": "small",
#             "duration": 5,
#             "use_sampling": True,
#             "top_k": 250,
#             "top_p": 0.0,
#             "temperature": 1.0,
#             "cfg_coef": 3.0,
#             "two_step_cfg": False,
#             "audio_format": "wav",
#             "audio_mp3_rate": 320,
#             "audio_normalize": True,
#             "audio_strategy": "peak",
#             "audio_peak_clip_headroom_db": 1.0,
#             "audio_rms_headroom_db": 18.0,
#             "audio_loudness_headroom_db": 14.0,
#             "audio_log_clipping": True,
#         }
#     )
#     print(res)


# if __name__ == "__main__":
#     main()
