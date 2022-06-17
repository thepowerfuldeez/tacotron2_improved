""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import symbols, _letters, _punctuation as punctuation_symbols
try:
    from text.acronyms import normalize_acronyms
except:
    pass

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

# for arpabet with apostrophe
_apostrophe = re.compile(r"(?=\S*['])([a-zA-Z'-]+)")


def text_to_sequence(text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        The text can optionally have ARPAbet sequences enclosed in curly braces embedded
        in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."
        Args:
            text: string to convert to a sequence
            cleaner_names: names of the cleaner functions to run the text through
        Returns:
            List of integers corresponding to the symbols in the text
    '''
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(text)
            break
        sequence += _symbols_to_sequence(m.group(1))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return [_symbol_to_id['<bos>']] + sequence + [_symbol_to_id['<eos>']]


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    i = 0
    id_sequence = []
    while i < len(symbols):
        if symbols[i:i + 7] == "<pzero>":
            id_sequence.append(_symbol_to_id["<p0>"])
            i += 7
        elif symbols[i:i + 6] == "<pone>":
            id_sequence.append(_symbol_to_id["<p1>"])
            i += 6
        elif symbols[i:i + 6] == "<ptwo>":
            id_sequence.append(_symbol_to_id["<p2>"])
            i += 6
        else:
            if _should_keep_symbol(symbols[i]):
                id_sequence.append(_symbol_to_id[symbols[i]])
            i += 1
    return id_sequence


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'


def _clean_word_before_phones(word):
    # start of the string AND all the punctuation
    re_start_punc = r"^\W+"
    # all the punctuation till the end of the string
    re_end_punc = r"(\W+|\W*<pzero>|\W*<pone>|\W*<ptwo>)$"

    start_symbols = re.findall(re_start_punc, word)
    if len(start_symbols):
        start_symbols = start_symbols[0]
        word = word[len(start_symbols):]
    else:
        start_symbols = ''

    end_symbols = re.findall(re_end_punc, word)
    if len(end_symbols):
        end_symbols = end_symbols[0]
        word = word[:-len(end_symbols)]
    else:
        end_symbols = ''

    arpabet_suffix = ''
    if _apostrophe.match(word) is not None and word.lower() != "it's" and word.lower()[-1] == 's':
        word = word[:-2]
        arpabet_suffix = ' Z'
    return start_symbols, word, arpabet_suffix, end_symbols


def get_phones(g2p, cmudict, clean_text):
    """
    Infer g2p model from g2p-en library
    clean_text might contain pause tokens, we need to preserve that
    but clean_text must not contain ARPAbet
    """
    word_pieces = re.findall(r'\S*\{.*?\}\S*|\S+', clean_text)
    word_pieces = [normalize_acronyms(w) if not w.startswith("{") else w for w in word_pieces]
    words = []
    for word_piece in word_pieces:
        if word_piece.startswith("{"):
            start_symbols, word, arpabet_suffix, end_symbols = word_piece, "thisisabbreviation", "", ""
        # preserve pauses in word
        else:
            start_symbols, word, arpabet_suffix, end_symbols = _clean_word_before_phones(word_piece)
            if not cmudict.lookup(word):
                start_symbols, word, arpabet_suffix, end_symbols = word_piece, "thisisoov", "", ""
        words.append([start_symbols, word, arpabet_suffix, end_symbols])

    phones = " ".join(g2p(" ".join([w for _, w, *_ in words]))).split("  ")
    result_text = ""
    for i, ((start_symbols, word, arpabet_suffix, end_symbols), phone) in enumerate(zip(words, phones)):
        if word == "thisisabbreviation":
            word_rec = start_symbols
        elif word == "thisisoov":
            word_rec = start_symbols
        else:
            word_rec = start_symbols + '{%s}' % (phone.strip() + arpabet_suffix) + end_symbols

        if i < len(words) - 1:
            result_text += (word_rec + " ")
        else:
            result_text += word_rec
    return result_text


def get_arpabet(word, cmudict, skip_heteronyms=True, index=0):
    start_symbols, word, arpabet_suffix, end_symbols = _clean_word_before_phones(word)
    arpabet = None if (word.lower() in HETERONYMS and skip_heteronyms) else cmudict.lookup(word)

    if arpabet is not None:
        return start_symbols + '{%s}' % (arpabet[index] + arpabet_suffix) + end_symbols
    else:
        return start_symbols + word + end_symbols


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

try:
    HETERONYMS = set(files_to_list('data/heteronyms'))
except:
    pass
