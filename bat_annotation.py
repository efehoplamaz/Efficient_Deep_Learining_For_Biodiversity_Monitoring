from generate_spectrogram import get_spectrogram_sampling_rate, display_spectrogram
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
from matplotlib import pyplot as plt
from skimage import io, transform


class BatAnnotationDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.bat_anns = json.load(open(json_file))
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.bat_anns)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        wav_name = self.bat_anns[idx]['id']
        anns = self.bat_anns[idx]
        spec, sampling_rate = get_spectrogram_sampling_rate(self.root_dir + wav_name)
        sample = {'wav_file': wav_name, 'spectrogram': spec ,'sampling_rate': sampling_rate, 'annotations': anns}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def show_bat_annotation_batch(self, sample_batched):
	    for sample in sample_batched:
	            display_spectrogram(self.root_dir + sample['wav_file'], sample['spectrogram'], sample['sampling_rate'], sample['annotations'])

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        wav_file, spectrogram, sampling_rate, annotations  = sample['wav_file'], sample['spectrogram'], sample['sampling_rate'], sample['annotations']

        h, w = spectrogram.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        spec = transform.resize(spectrogram, (new_h, new_w))

        return {'wav_file': wav_file, 'spectrogram': spec ,'sampling_rate': sampling_rate, 'annotations': annotations}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        wav_file, spectrogram, sampling_rate, annotations  = sample['wav_file'], sample['spectrogram'], sample['sampling_rate'], sample['annotations']
                
        return {'wav_file': wav_file, 'spectrogram': torch.from_numpy(spectrogram), 'sampling_rate': sampling_rate, 'annotations': annotations}


def build(image_set, args):

    PATHS = {
        "train": ('C:\\Users\\ehopl\\Desktop\\Fourth Year\\Disseration\\bat_data_oct_2020_ug4\\annotations\\BritishBatCalls_MartynCooke_2018_1_sec_train_expert.json', 'C:\\Users\\ehopl\\Desktop\\Fourth Year\\Disseration\\bat_data_oct_2020_ug4\\audio\\mc_2018\\audio\\'),
        "val": ('C:\\Users\\ehopl\\Desktop\\Fourth Year\\Disseration\\bat_data_oct_2020_ug4\\annotations\\BritishBatCalls_MartynCooke_2019_1_sec_train_expert.json', 'C:\\Users\\ehopl\\Desktop\\Fourth Year\\Disseration\\bat_data_oct_2020_ug4\\audio\\mc_2019\\audio\\'),
    }

    ann_file, audio_file = PATHS[image_set]
    dataset = BatAnnotationDataSet(json_file = ann_file, root_dir= audio_file, transform=transforms.Compose([ToTensor(), Rescale((256, 1718))]))
    return dataset


# if __name__ == '__main__':
#     dataset = build('train', 0)
#     for i in range(len(dataset)):
#         sample = dataset[i]
#         print(sample['spectrogram'].shape)
#         display_spectrogram(dataset.root_dir + sample['wav_file'], sample['spectrogram'], sample['sampling_rate'], sample['annotations'])
#         if i == 1:
#             #plt.show()
#             break