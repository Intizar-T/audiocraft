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
        output_name_with_format = "output.{}".format(inputs["audio_format"])
        model = MusicGen.get_pretrained(inputs["model_type"])
        model.set_generation_params(
            duration=inputs["duration"],
            use_sampling=inputs["use_sampling"],
            top_k=inputs["top_k"],
            top_p=inputs["top_p"],
            temperature=inputs["temperature"],
            cfg_coef=inputs["cfg_coef"],
            two_step_cfg=inputs["two_step_cfg"],
        )

        descriptions = [inputs["prompt"]]

        output_data = model.generate(descriptions)
        wav = output_data[0]
        audio_write(
            stem_name=output_name,
            wav=wav.cpu(),
            sample_rate=model.sample_rate,
            format=inputs["audio_format"],
            mp3_rate=inputs["audio_mp3_rate"],
            normalize=inputs["audio_normalize"],
            strategy=inputs["audio_strategy"],
            peak_clip_headroom_db=inputs["audio_peak_clip_headroom_db"],
            rms_headroom_db=inputs["audio_rms_headroom_db"],
            loudness_headroom_db=inputs["audio_loudness_headroom_db"],
            log_clipping=inputs["audio_log_clipping"],
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
