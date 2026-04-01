# drishti
## A note on how the files are downloaded and how data is organized.
1. Data for different museums will be organized into different csvs. Currently we are processing 2.
    a. Cleveland museum of art (museumid=cma)
    b. Metropolitan museum of art (museumid=met)
2.There might be errors, some uninitialized fields necessary for the bookkeeping for the training. So these datafiles are first normalized and merged into a single file. The command for this is `python -m datasets.normalize_data`. This generates `datasets/csvs/training/normalized_output.csv`
3. Once the data is normalized the process is not repeated unless there is a new museum for which the model needs a training.

## Preparing the training data.
1. To prepare the training data from normalized data, code first finds out the image url by calling the museum's objet endpoint. ObjectId from the normalized data is used for fetching the collection-object information. This results into a training file `datasets/csvs/training/intermediate/training_source.csv`. It has following columns
`image_url,title,museum_index,id,instruments`.
2. Here `instruments` is a list of the instruments present in the current paintings. Usually single painting contains multiple instrument and this is crucial to keep in mind.
3. Training file is not generated every time. It is generated only if the file is not present on the path `datasets/csvs/training/intermediate/training_source.csv`
4. The training file is then loaded into memory as a dataframe and for every instrument, this dataframe is divided into 2 sets. One is training set and other is validation set.
5. Training set contains 80% of the total instances and a validation set contains 20% of the total instances for a particular instrument. In other words, if the dataset has 10 pakhawaj depictions, 8 will be used in training and 2 will be used in validation.
6. It is important to keep in mind that images may get downloaded multiple times in the first run. This is because a single image contains multiple instruments. So it will get downloaded for that particular instruments' training set.
7. Assuming that the first run was successful and it downloaded all the images for the consecuitive runs, it only checks if the file is present on the path.
6. Another important aspect is how the data is divided between training and validation. It is important to note that an image in context of a instrument can not be present in both training and validation set. i.e
image['i1']['objectid1'] can not exist simultaniously in the training and validation set for instrument i1. But it could exist for the instrument i2.

# Running the pipeline
```bash
python -m src.run
```