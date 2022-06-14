from pathlib import Path

import tgt

from text.cleaners import english_cleaners


def read_textgrid_file(tg_path):
    return tgt.io.read_textgrid(tg_path, include_empty_intervals=True)


def get_words_from_tg_obj(tg_obj):
    return [[it.start_time, it.end_time, it.text] for it in tg_obj.get_tier_by_name("words")._objects]


def get_phones_from_tg_obj(tg_obj):
    return [[it.start_time, it.end_time, it.text] for it in tg_obj.get_tier_by_name("phones")._objects]


def prepare_textgrid_name(path):
    """mfa changes name of files, replacing _ to - and appending parent folder, so we need to turn it back"""
    audio_path = Path(path)
    if audio_path.parent.stem == "wavs":
        # in previous versions of MFA it changed names of files
        # audio_id = f"{audio_path.stem}".replace("_", "-").replace(" ", "-")
        audio_id = f"{audio_path.stem}".replace(" ", "-")
    else:
        # n = f"{audio_path.parent.stem}-{audio_path.stem}".replace("_", "-").replace(" ", "-")
        n = f"{audio_path.parent.stem}-{audio_path.stem}".replace(" ", "-")
        audio_id = f"{audio_path.parent.stem}/{n}"
    return audio_id


def read_alignment(orig_text, alignments_path, audio_path):
    """
    Read TextGrid alignment file and map cleaned text from it and original one from dataset
    """
    audio_id = prepare_textgrid_name(audio_path)

    orig_words = orig_text.split()
    clean_words = []
    clean_idx2orig = []

    # mapping from cleaned words to orig
    j = 0
    for i, word in enumerate(orig_words):
        clean_text_ = english_cleaners(word)
        clean_text_ = list(filter(
            lambda s: s,
            [w.strip(".,;:?!-'\"()[]@&%").replace(",", "").replace("..", "").replace("...", "").replace(
                "-", " ").strip() for w in clean_text_.split()])
        )
        clean_text = []
        for s in clean_text_:
            clean_text.extend(s.split())
        clean_words.extend(clean_text)
        clean_idx2orig.extend([j] * len(clean_text))
        j += 1

    # read alignments and detect pauses
    tg_path = alignments_path / f"{audio_id}.TextGrid"
    textgrid = read_textgrid_file(tg_path)
    words = get_words_from_tg_obj(textgrid)
    phones = get_phones_from_tg_obj(textgrid)

    # check if all words mapped correctly
    err = False
    for w1, w2 in zip([it[-1] for it in words if it[-1]], clean_words):
        try:
            assert w1 == w2 or w1 == "<unk>" or w1.endswith("'")
        except:
            print(w1, w2)
            err = True
            break
    return err, orig_words, words, phones, clean_words, clean_idx2orig
