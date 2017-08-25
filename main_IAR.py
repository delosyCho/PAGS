import tensorflow as tf
import numpy as np
import Improved_AoA_Reader
import data_processor

IAR_model = Improved_AoA_Reader.Improved_AoA_Reader()
"""
emb = IAR_model.dataset.get_glove('bbc')
sum_v = 0
for i in range(IAR_model.dataset.embedding_size):
    sum_v = sum_v + emb[i]
print(sum_v)

emb = IAR_model.dataset.get_glove('asjdnas')
sum_v = 0
for i in range(IAR_model.dataset.embedding_size):
    sum_v = sum_v + emb[i]
print(sum_v)
input()
"""
"""
while True:
    word = input()
    emb = IAR_model.dataset.get_glove(word)
    sum_v = 0
    for i in range(IAR_model.dataset.embedding_size):
        sum_v = sum_v + emb[i]
    print(sum_v)
"""
"""
a = 0
while a != -1:
    a = int(input())

    i = IAR_model.dataset.paragraph_index[a]

    print(IAR_model.dataset.paragraph_arr[i])
    print(IAR_model.dataset.question_batch[a])
    print(IAR_model.dataset.paragraph_arr[i, int(IAR_model.dataset.start_index_batch[a])], int(IAR_model.dataset.start_index_batch[a]))
    print(IAR_model.dataset.paragraph_arr[i, int(IAR_model.dataset.stop_index_batch[a])], int(IAR_model.dataset.stop_index_batch[a]))
"""

IAR_model.training_prediction_start_stop(training_epoch=5, is_continue=True)
