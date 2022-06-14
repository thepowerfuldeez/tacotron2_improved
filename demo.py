# run with streamlit run demo.py --server.port 8080

import streamlit as st

from pathlib import Path
import json
import torch
import soundfile as sf
import numpy as np

from inference import text2mel_traced, plt
from utils.utils import vocoder_infer

# currently, cpu is not working
DEVICE = "cuda"


def clean_text(text):
    """
    Cleaning text for demo, just before other cleaning text utilities, in order to prevent corner cases
    """
    from cleantext import clean

    return clean(text,
                 fix_unicode=True,  # fix various unicode errors
                 to_ascii=True,  # transliterate to closest ASCII representation
                 lower=False,  # lowercase text
                 no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                 no_urls=False,  # replace all URLs with a special token
                 no_emails=False,  # replace all email addresses with a special token
                 no_phone_numbers=False,  # replace all phone numbers with a special token
                 no_numbers=False,  # replace all numbers with a special token
                 no_digits=False,  # replace all digits with a special token
                 no_currency_symbols=False,  # replace all currency symbols with a special token
                 no_punct=False,  # remove punctuations
                 replace_with_punct="",  # instead of removing punctuations you may replace them
                 replace_with_url="<url>",
                 replace_with_email="<email>",
                 replace_with_phone_number="<phone>",
                 replace_with_number="<NUMBER>",
                 replace_with_digit="0",
                 replace_with_currency_symbol="<CUR>",
                 lang="en"  # set to 'de' for German special handling
                 )


@st.cache(hash_funcs={torch.jit._script.RecursiveScriptModule: lambda _: None}, allow_output_mutation=True)
def load_models(checkpoint_dir):
    """
    Function to load traced models with specific naming from inference folder
    """
    model = torch.jit.load(str(checkpoint_dir / "tacotron2.jit"), map_location=DEVICE)
    generator = torch.jit.load(str(checkpoint_dir / "generator.pth"), map_location=DEVICE)
    return model, generator


def inference(model, text, input_phones, generator, sampling_rate,
              cleaners=("flowtron_cleaners",), transition_agent_bias=0.0):
    """
    Main inference function, use this in every testing script.
    Model should be already traced. See README.md

    :params:
    model: initialised model
    text: line of text to infer
    generator: initialised instance of supported vocoder, e.g. FreGAN, should be traced
    sampling_rate: 22050 by default
    cleaners: text dependent set of functions, see config for this specific experiment
    transition_agent_bias: if model was trained with ForwardAttention, using this value you can specify speed of speech
    """
    audios = []
    new_text_full = ""
    new_phones_full = ""
    mels = []

    if input_phones == "":
        input_phones = None

    # for text_cut in text.split(". "):
    for text_cut in [text]:
        with torch.no_grad():
            text_cut = clean_text(text_cut)
            # TODO: add double stop-token at inference for short phrases
            mel_outputs_postnet, new_text, new_phones, *_ = text2mel_traced(
                model, text_cut,
                input_phones=input_phones,
                verbose=False,
                use_g2p=True, cleaners=cleaners, transition_agent_bias=transition_agent_bias
            )
            audio = vocoder_infer(generator, mel_outputs_postnet)
            audios.append(np.append(audio, np.zeros_like(audio, shape=(1, 550))))
            mels.append(mel_outputs_postnet.float().cpu().numpy()[0])
        print(new_text)
        new_text_full += new_text
        new_phones_full += new_phones
        if len(audios) > 1:
            new_text_full += ". "
            new_phones_full += ". "
    sf.write("temp.wav", np.concatenate(audios).reshape(-1), sampling_rate)
    return new_text_full, new_phones_full, np.concatenate(mels, -1)


if __name__ == "__main__":
    # this json file contains parameters for inference model
    speakers = json.load(open("demo_dictors.json"))
    speaker_name = st.selectbox("Select model name", list(speakers))
    speaker_params = speakers[speaker_name]

    st.write("CMUDict: http://www.speech.cs.cmu.edu/cgi-bin/cmudict?in=dog&stress=-s")

    model, generator = load_models(Path(speaker_params['checkpoint_dir']))
    text = st.text_input(
        'Type text to synthesize (recommended min length 10 characters, max len 200 characters, one sentence!)',
        max_chars=1000)
    input_phones = st.text_input("phones (leave empty to predict from text)")
    if speaker_params.get("speed_control"):
        speed = st.slider("Choose speed", -1.5, 1.5, value=0.0, step=0.1)
    else:
        speed = 0.0
    if text.strip() and 5 <= len(text) < 1000:
        new_text, new_phones, mel = inference(model, text.strip(), input_phones.strip(),
                                              generator, sampling_rate=speaker_params['sampling_rate'],
                                              cleaners=speaker_params.get("cleaners", ["flowtron_cleaners"]),
                                              transition_agent_bias=speed)
        st.write("Cleaned text: ", new_text)
        st.write("Phones: ", new_phones)
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(mel, aspect='auto', origin='lower', interpolation='none')
        plt.title("predicted mel-spectrogram")
        plt.xlabel("decoder steps")
        st.write(fig)
        with open("temp.wav", "rb") as f:
            st.audio(f.read())
