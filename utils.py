import sys
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

MULTIMODAL_SDK_PATH = "/home/dstratton/PycharmProjects/InterpretableMultimodal/CMU-MultimodalSDK"
sys.path.append(MULTIMODAL_SDK_PATH)
import mmsdk
from mmsdk import mmdatasdk as md


# loads the dataset in the form from the datasets library
def load_dataset(sequences_to_load):
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

    seq_dict = {'CMU_MOSI_Opinion_Labels': DATA_PATH + '/CMU_MOSI_Opinion_Labels.csd'}
    for seq_name in sequences_to_load:
        seq_dict[seq_name] = DATA_PATH + f"/{seq_name}.csd"

    # load the dataset based on what you want, in this case text and labels
    dataset = md.mmdataset(seq_dict)
    # dataset = md.mmdataset({
    #     'CMU_MOSI_TimestampedWords': DATA_PATH + '/CMU_MOSI_TimestampedWords.csd',
    #     'CMU_MOSI_Opinion_Labels': DATA_PATH + '/CMU_MOSI_Opinion_Labels.csd'
    # })

    # align to labels
    dataset.align('CMU_MOSI_Opinion_Labels')

    segment_ids = list(dataset['CMU_MOSI_Opinion_Labels'].keys())

    train_split = DATASET.standard_folds.standard_train_fold
    dev_split = DATASET.standard_folds.standard_valid_fold
    test_split = DATASET.standard_folds.standard_test_fold
    data_dict = {}

    for split_name, split in [('train', train_split), ('val', dev_split), ('test', test_split)]:
        segment_ids_for_split = [vid for vid in segment_ids if any(substring in vid for substring in split)]

        ### labels
        labels = []
        for video_id in segment_ids_for_split:
            labels.append(dataset['CMU_MOSI_Opinion_Labels'][video_id]['features'][0][0])
            # you can also store interval information from dataset['CMU_MOSI_TimestampedWords'][video_id]['intervals'] if needed
        df = pd.DataFrame({'labels': labels})

        # we are choosing to make the labels binary here
        df['labels'] = np.sign(df['labels']).astype('int32')
        df['labels'][df['labels'] == -1] = 0

        ### text
        if 'CMU_MOSI_TimestampedWords' in sequences_to_load:
            sentences = []
            for video_id in segment_ids_for_split:
                sentence = []
                for word in dataset['CMU_MOSI_TimestampedWords'][video_id]['features']:
                    if word[0] != b'sp':
                        sentence.append(word[0].decode('utf-8'))
                sent = ' '.join(sentence)
                sentences.append(sent)

            df['text'] = sentences

        # semi-slow solution but meh
        if 'CMU_MOSI_Visual_Facet_41' in sequences_to_load:
            feature_names = dataset['CMU_MOSI_Visual_Facet_41'].metadata['dimension names']
            all_features = []
            for video_id in segment_ids_for_split:
                features = []
                for feat in dataset['CMU_MOSI_Visual_Facet_41'][video_id]['features']:
                    features.append(feat)
                all_features.append(pd.DataFrame(features, columns=feature_names))

            for name in feature_names:
                # todo: idk, dimensions too high
                # currently just taking the mean i guess
                df[name] = [x[name].mean() for x in all_features]
            pass

        data_dict[split_name] = Dataset.from_pandas(df)

    return DatasetDict(data_dict)
