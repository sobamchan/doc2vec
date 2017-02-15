from iterator import Iterator

batch_size = 10
texts = [[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20]]
labels = [1,2]
window_size = 3

iterator = Iterator(batch_size, texts, labels, window_size)

for b in iterator:
    print(b)
