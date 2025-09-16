import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_dir = "E:/DogBreedClassification/dogImages/train"
test_dir = "E:/DogBreedClassification/dogImages/test"
valid_dir = "E:/DogBreedClassification/dogImages/valid"

breed_freq = {}
for breed in os.listdir(train_dir):
    breed_path = os.path.join(train_dir,breed)
    if os.path.isdir(breed_path):
        freq = len(os.listdir(breed_path))
        breed_freq[breed] = freq

breed_freq_df = pd.DataFrame(list(breed_freq.items()), columns=['Breed', 'Frequency'])
print(breed_freq_df)
