import sys
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

MULTIMODAL_SDK_PATH = "/home/dstratton/PycharmProjects/InterpretableMultimodal/CMU-MultimodalSDK"
sys.path.append(MULTIMODAL_SDK_PATH)
import mmsdk
from mmsdk import mmdatasdk as md


# loads the dataset in the form from the datasets library
def load_dataset():
    DATASET = md.cmu_mosi
    DATA_PATH = "cmumosi"

    # if needed, download the data
    try:
        md.mmdataset(DATASET.highlevel, DATA_PATH)
    except RuntimeError:
        print("High-level features have been downloaded previously.")

    try:
        md.mmdataset(DATASET.raw, DATA_PATH)
    except RuntimeError:
        print("Raw data have been downloaded previously.")

    try:
        md.mmdataset(DATASET.labels, DATA_PATH)
    except RuntimeError:
        print("Labels have been downloaded previously.")

    # load the dataset based on what you want, in this case text and labels
    dataset = md.mmdataset({
        'CMU_MOSI_TimestampedWords': DATA_PATH + '/CMU_MOSI_TimestampedWords.csd',
        'CMU_MOSI_Opinion_Labels': DATA_PATH + '/CMU_MOSI_Opinion_Labels.csd'
    })

    # align to labels
    dataset.align('CMU_MOSI_Opinion_Labels')

    segment_ids = list(dataset['CMU_MOSI_TimestampedWords'].keys())

    train_split = DATASET.standard_folds.standard_train_fold
    dev_split = DATASET.standard_folds.standard_valid_fold
    test_split = DATASET.standard_folds.standard_test_fold
    data_dict = {}

    for split_name, split in [('train', train_split), ('val', dev_split), ('test', test_split)]:
        segment_ids_for_split = [vid for vid in segment_ids if any(substring in vid for substring in split)]

        sentences = []
        labels = []
        for video_id in segment_ids_for_split:
            sentence = []
            for word in dataset['CMU_MOSI_TimestampedWords'][video_id]['features']:
                if word[0] != b'sp':
                    sentence.append(word[0].decode('utf-8'))
            sent = ' '.join(sentence)
            sentences.append(sent)
            labels.append(dataset['CMU_MOSI_Opinion_Labels'][video_id]['features'][0][0])
            # you can also store interval information from dataset['CMU_MOSI_TimestampedWords'][video_id]['intervals'] if needed

        text_data = pd.DataFrame({'text': sentences, 'labels': labels})

        # we are choosing to make the labels binary here
        text_data['labels'] = np.sign(text_data['labels']).astype('int32')
        text_data['labels'][text_data['labels'] == -1] = 0

        data_dict[split_name] = Dataset.from_pandas(text_data)
    return DatasetDict(data_dict)
