


class Iterator(object):

    def __init__(self, batch_size, vocabulary, documents, window_size):
        '''
        documents: [['id:int', 'text: str']]
        '''
        self._i = 0
        self.vocabulary = vocabulary
        self._N = len(document_ids)
        self.window = window
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= self._N:
            raise StopIteration
        i = self._i
