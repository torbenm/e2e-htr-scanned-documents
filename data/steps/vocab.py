def getVocab(fulltext):
    vocab = list(set(fulltext))
    vocab.append("")  # ctc blank label
    vocab_size = len(vocab)
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
    return (idx_to_vocab, vocab_to_idx)
