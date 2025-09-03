import SimpleITK
import json

from typing import (
    Dict,
)
from abc import abstractmethod
import torch

from evalutils.evalutils import Algorithm
import csv
from monai.transforms import *

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def to_input_format(input_image):
    input_image = torch.Tensor(input_image)
    
    val_transforms = Compose([Resize(mode="trilinear", align_corners=False,
                                spatial_size=(128, 128, 128))
##                                spatial_size=(120, 120, 120))
                                ])
    input_image = val_transforms(input_image.unsqueeze(0)).squeeze(0).float()
    
    input_image = input_image.unsqueeze(0).unsqueeze(0).to(device)
    return input_image

def unpack_single_output(output):
    return output.cpu().numpy().astype(float)[0]


class MultiClassAlgorithm(Algorithm):

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Classify input_image image
        return self.predict(input_image=input_image)

    @abstractmethod
    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        raise NotImplementedError()

    def save(self):
#        if len(self._case_results) > 1:
#            raise RuntimeError("Multiple case prediction not supported with single-value output interfaces.")
        case_result = self._case_results#[0]
        '''
        for output_file, result in case_result.items():
            with open(str(self._output_path / output_file) + '.json', "w") as f:
                json.dump(result, f)
        '''
        keys = [key for key, value in case_result[0].items()]
        with open(self._output_path / 'result.csv', 'w', newline='') as csvfile:
            first = ['PatientID', 'probCOVID', 'probSevere']
            writer = csv.writer(csvfile)
            writer.writerow(first)
            for i in range(len(case_result)):
                PatientID = str(self._cases['input_image']['path'][i]).split('/')[-1][:-4]
                result = [PatientID, case_result[i][keys[0]], case_result[i][keys[1]]]
                writer.writerow(result)

if __name__ == "__main__":
    MultiClassAlgorithm()
