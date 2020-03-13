import random

from ds.ml.neural.checkpoints import load_or_create

from lib import read_langs, filter_pairs


# preparing the data
from lib.model import EncoderRNN, AttnDecoderRNN
from lib.train import fit


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse=reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepare_data('fra', 'eng', True)
print(random.choice(pairs))

hidden_size = 256
encoder = load_or_create(EncoderRNN, model_params={'hidden_size': hidden_size, 'input_size': input_lang.n_words},
                         model_name='encoder')
decoder = load_or_create(AttnDecoderRNN,
                         model_params={'dropout': 0.1, 'hidden_size': hidden_size, 'output_size': output_lang.n_words},
                         model_name='decoder')

fit(encoder, decoder, 75000, print_every=5000, input_lang=input_lang, output_lang=output_lang, pairs=pairs,
    optim_params={'lr': 0.01})
