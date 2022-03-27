# downloaded from https://github.com/alanwuha/torchemotion
from torchemotion.datasets.IemocapDataset import IemocapDataset
import matplotlib.pyplot as plt

# i had to do this to the torchemotion iemocap datasetfile on line 41:
#                 f = open(os.path.join(path, file), 'r', encoding="utf8", errors='ignore')
# from this link
# https://stackoverflow.com/questions/42339876/error-unicodedecodeerror-utf-8-codec-cant-decode-byte-0xff-in-position-0-in

# Initialize IemocapDataset
iemocap_dataset = IemocapDataset('/home/dstratton/IEMOCAP_full_release')

# seems like speechbrain people do an 80/10/10 split with a random shuffle.
# i guess its dropped to 4 emotion classes (anger, happy, sad, neutral)

# the label distribution
iemocap_dataset.df["emotion"].unique()
iemocap_dataset.df["emotion"].hist()
plt.show()