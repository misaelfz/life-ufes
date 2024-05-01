#!/bin/bash

wget https://tinyurl.com/mnistdatasetpkl
mv mnistdatasetpkl mnist_dataset.pkl

wget https://tinyurl.com/augmenteddatasetpkl
mv augmenteddatasetpkl augmented_dataset.pkl

wget https://tinyurl.com/classifieddatasetpkl
mv classifieddatasetpkl classified_dataset.pkl

wget https://tinyurl.com/randomdatasetpkl
mv randomdatasetpkl random_dataset.pkl

wget https://tinyurl.com/maxentropydataset
mv maxentropydataset max_entropy_dataset.pkl

wget https://tinyurl.com/minentropydataset
mv minentropydataset min_entropy_dataset.pkl

wget https://tinyurl.com/euclideandataset
mv euclideandataset euclidean_dataset.pkl

wget https://tinyurl.com/cosinedataset
mv cosinedataset cosine_dataset.pkl

# source: https://drive.google.com/drive/folders/1-IbfrovOGF3ePisBpyFFQ-5zULRXVhat