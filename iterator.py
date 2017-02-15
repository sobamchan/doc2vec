class Iterator(object):

    def __init__(self, batch_size, texts, labels, window_size):
        '''
        input:
            batch_size: int
            texts: [[word_id, word_id, ...], [word_id, word_id, ...], ...]
            lebels: [doc_id, doc_id, ...]
            window_size: int
        '''
        self._i = 0
        self._j = 0
        self._j_end = None
        self._N = len(texts)
        self.batch_size = batch_size
        self.texts = texts
        self.labels = labels
        self.window_size = window_size

    def __iter__(self):
        return self

    def __next__(self):
        if self._j_end is not None and self._j + self.window_size >= self._j_end+1:
            self._j_end = None
            self._i += 1

        if self._i >= self._N:
            raise StopIteration

        i = self._i
        window_size = self.window_size
        words = self.texts[i]

        if self._j_end is None:
            self._j_end = len(words)
            self._j = window_size


        j = self._j
        self._j += 1

        context = words[j-window_size : j+window_size]
        center = words[j]
        label = self.labels[i]

        return label, center, context
