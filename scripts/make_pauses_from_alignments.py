import argparse
from collections import Counter
from pathlib import Path

import soundfile as sf
from librosa.effects import trim

from utils.mfa_alignment_utils import read_alignment
from text.cleaners import english_cleaners

# text / audio_len ratio for filtering
MIN_TEXT_LEN_RATIO = 6.5
MAX_TEXT_LEN_RATIO = 25

# change if you think these thresholds are not enough, more on L162
P0_THRESH = 0.04
P1_THRESH = 0.2  # 0.16
P2_THRESH = 0.36  # 0.32


def trim_audio(audio, top_db=30):
    # usually, trim using top_db=25, but here we are inside some phrases, so better increase threshold
    # trim audio, return indices
    ind = trim(audio, top_db=top_db, frame_length=256, hop_length=64)[1]
    ind = [max(0, ind[0] - 275), min(len(audio), ind[1] + 275)]
    return audio[ind[0]:ind[1]]


def maybe_split_audio(clean_words, words, clean_idx2orig, thresh=0.64):
    """
    Return new list with sub-sequences of original audio file splitted on silence

    clean_words – list of words after english_cleaners
    words – original words from dataset
    clean_idx2orig – mapping to preserve ordering as cleaners might increase/decrease length of lists
    """
    clean_words_ = clean_words.copy()
    split_audio_time_word_ind = []
    find_running_ind = 0
    for i, (s, e, t) in enumerate(words[1:], 1):
        word_start, word_end, word_text = words[i - 1]

        if word_text != "<unk>" and t == "" and (e - s) >= thresh:
            try:
                j = clean_words_.index(word_text)
                try:
                    if Counter(clean_words_)[word_text] > 1:
                        j1 = clean_words_[find_running_ind:].index(word_text) + find_running_ind
                        j = j1
                except:
                    print("counter prob")

            except Exception as e:
                if word_text.endswith("'"):
                    j = clean_words_.index(word_text[:-1])
                else:
                    continue

            clean_words_ = ["#"] * j + clean_words_[j:]
            split_audio_time_word_ind.append((clean_idx2orig[j], s + 0.01, e - 0.01, i - 1))
        else:
            if t != "":
                find_running_ind += 1
    if thresh != 0.64:
        if len(split_audio_time_word_ind) >= 4:
            split_audio_time_word_ind = split_audio_time_word_ind[1::2]
        elif len(split_audio_time_word_ind) >= 2:
            split_audio_time_word_ind = [split_audio_time_word_ind[len(split_audio_time_word_ind) // 2]]
    return split_audio_time_word_ind


def do_read_alignments(lines, alignments_path):
    """
    1. read alignments in specific format
    """
    read_alignments_list = []
    for line in lines:
        path, orig_text, *_ = line.split("|")
        audio_path = Path(path)
        try:
            err, orig_words, words, phones, clean_words, clean_idx2orig = read_alignment(
                orig_text.replace("<p0>", "").replace("<p1>", "").replace("<p2>", ""),
                alignments_path, audio_path)
        except Exception as e:
            print(e)
            continue
        read_alignments_list.append((err, orig_words, words, phones, clean_words, clean_idx2orig, audio_path))
    return read_alignments_list


def do_clean_alignments(read_alignments, min_audio_threshold=6):
    """
    2. process read alignments in specified format, split into shorter subsequences
    """
    clean_alignments = []
    cleaned_count = 0
    for (err, orig_words, words, phones, clean_words, clean_idx2orig, audio_path) in read_alignments:
        wav, sr = sf.read(audio_path)
        length = len(wav) / sr

        # split one alignment into smaller ones if silence length longer than predefined threshold
        if length > min_audio_threshold:
            splits = maybe_split_audio(clean_words, words, clean_idx2orig, P2_THRESH if length > 9 else P2_THRESH * 2)
        else:
            splits = None

        if splits:
            text_inds, word_inds, audio_time_inds = [0], [0], [0]
            for s in splits:
                text_inds += [s[0], s[0] + 1]
                audio_time_inds += [s[1], s[2]]
                word_inds += [s[3], s[3] + 2]
            text_inds += [len(orig_words)]
            audio_time_inds += [len(wav) / sr]
            word_inds += [len(words)]

            for i in range(0, len(text_inds), 2):
                if text_inds[i] == text_inds[i + 1] or word_inds[i] == word_inds[i + 1]:
                    continue
                orig_words_split = orig_words[text_inds[i]:text_inds[i + 1] + 1]
                words_split = words[word_inds[i]:word_inds[i + 1] + 1]
                wav_segment = wav[int(sr * audio_time_inds[i]):int(sr * audio_time_inds[i + 1])]
                wav_segment = trim_audio(wav_segment)
                try:
                    if words_split[0][-1] == "":
                        words_split[0] = (audio_time_inds[i], words_split[0][1], "")
                    if words_split[-1][-1] == "":
                        words_split[-1] = (words_split[-1][1], audio_time_inds[i + 1], "")
                except:
                    print("err indicing words_split[0]")
                    continue

                # expect only 0.75 sec splits or more
                if len(wav_segment) >= 0.7 * sr:
                    new_name = audio_path.stem + f"_{i + 1}" + audio_path.suffix
                    new_audio_path = audio_path.parent / new_name
                    sf.write(new_audio_path, wav_segment, sr)
                    clean_alignments.append((err, orig_words_split, words_split, phones, clean_words,
                                             clean_idx2orig, new_audio_path))
                    cleaned_count += 1
        else:
            # before it was splitting long sequences if pause is too long, but better is to skip such audios
            # since we are not splitting by pauses, just skip audios with silence len > 1.0
            c = 0
            # 2 SCENARIOS WHEN WE SKIP AUDIO: 1. when one silence longer than 0.8s or 3 silences each longer than 0.45s
            for s, e, word_text in words:
                if word_text == "" and (e - s) >= 0.8:
                    err = True
                    print("pause > 800ms")
                if word_text == "" and (e - s) >= 0.45:
                    c += 1
                    if c >= 3:
                        err = True
                        print("3 pauses > 450ms")
            if not err:
                clean_alignments.append((err, orig_words, words, phones, clean_words, clean_idx2orig, audio_path))
    return clean_alignments, cleaned_count


def insert_pauses(clean_words, orig_words, words, clean_idx2orig,
                  p0_thresh=0.04, p1_thresh=0.16, p2_thresh=0.32):
    """
    3. latest function to insert pauses

    <p0> lies in [p0_thresh, p1_thresh)
    <p1> lies in [p1_thresh, p2_thresh)
    <p2> lies in [p2_thresh, MAX)
    """
    add_pause1, add_pause2, add_pause3 = [], [], []
    clean_words_ = clean_words.copy()
    for i, (s, e, t) in enumerate(words[1:], 1):
        word_start, word_end, word_text = words[i - 1]

        if word_text != "<unk>" and t == "" and (e - s) >= p0_thresh:
            try:
                j = clean_words_.index(word_text)
            except:
                try:
                    if word_text.endswith("'"):
                        j = clean_words_.index(word_text[:-1])
                    else:
                        continue
                except:
                    continue

            # YOU CAN TWEAK THESE PARAMS TO YOUR WISH
            # <p0>
            if p0_thresh <= (e - s) < p1_thresh:
                add_pause1.append(clean_idx2orig[j])
            # <p1>
            elif p1_thresh <= (e - s) < p2_thresh:
                add_pause2.append(clean_idx2orig[j])
            # <p2>
            else:
                add_pause3.append(clean_idx2orig[j])

            clean_words_ = ["#"] * j + clean_words_[j:]

    tokens = []
    for i, s in enumerate(orig_words):
        if i in add_pause1:
            tokens.append(f"{s}<p0>")
        elif i in add_pause2:
            tokens.append(f"{s}<p1>")
        elif i in add_pause3:
            tokens.append(f"{s}<p2>")
        else:
            tokens.append(s)
    new_text = " ".join(tokens)
    return new_text, len(add_pause1), len(add_pause2), len(add_pause3)


def get_result_dict(clean_alignments):
    result_dict = {}
    has_pause = 0
    all_p0_count = 0
    all_p1_count = 0
    all_p2_count = 0
    for (err, orig_words, words, phones, clean_words, clean_idx2orig, audio_path) in clean_alignments:
        name = str(audio_path)
        orig_text = " ".join(orig_words)
        s = sf.info(audio_path).duration

        # heuristic
        ratio = len(english_cleaners(orig_text)) / s
        if ratio < MIN_TEXT_LEN_RATIO or ratio > MAX_TEXT_LEN_RATIO:
            continue

        if not err:
            new_text, p0_count, p1_count, p2_count = insert_pauses(
                clean_words, orig_words, words, clean_idx2orig,
                P0_THRESH, P1_THRESH, P2_THRESH
            )
            all_p0_count += p0_count
            all_p1_count += p1_count
            all_p2_count += p2_count
            if (p0_count + p1_count + p2_count) > 0:
                has_pause += 1
            result_dict[name] = new_text
        else:
            result_dict[name] = orig_text
    return result_dict, has_pause, all_p0_count, all_p1_count, all_p2_count


def main():
    """Insert pause token <p> into text. Creates new dir parallel to filelists with text inputs for training"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, default='filelists/')
    parser.add_argument("--alignments_path", type=Path, default='../data/ljspeech/alignments/')
    parser.add_argument("--min_audio_threshold", type=int, default=6,
                        help='minimum audio len (in seconds) to split on pauses')
    args = parser.parse_args()

    alignments_path = Path(args.alignments_path)
    input_dir = Path(args.input_dir)
    if not len(list(input_dir.glob("*.txt"))):
        print("input dir has no files!")
    else:
        out_dir = input_dir.parent / (input_dir.stem + "_pauses")
        out_dir.mkdir(exist_ok=True, parents=True)
        has_pause, all_p0_count, all_p1_count, all_p2_count, total_count = 0, 0, 0, 0, 0

        for labels_path in input_dir.glob("*.txt"):
            lines = labels_path.read_text().strip('\n').split("\n")

            read_alignments = do_read_alignments(lines, alignments_path)
            clean_alignments, cleaned_count = do_clean_alignments(read_alignments, args.min_audio_threshold)

            print(f"there was {cleaned_count} segments with pauses longer than 1 sec, splitted and resaved")

            result_dict, has_p, p0, p1, p2 = get_result_dict(clean_alignments)
            all_p0_count += p0
            all_p1_count += p1
            all_p2_count += p2
            has_pause += has_p
            out_dir.joinpath(labels_path.name).write_text("\n".join(["|".join([k, v]) for k, v in
                                                                     result_dict.items()]))
            total_count += len(result_dict)
        print(f"finished processing {total_count} items. {has_pause} now has pauses")
        print(f"total p0 count {all_p0_count} items. total p1 count {all_p1_count} items, "
              f"total p2 count {all_p2_count} items")


if __name__ == "__main__":
    main()
