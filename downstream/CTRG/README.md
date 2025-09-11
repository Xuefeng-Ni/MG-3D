## Requirements
- `Python >= 3.6`
- `Pytorch >= 1.7`
- `torchvison`
- [Microsoft COCO Caption Evaluation Tools](https://github.com/tylin/coco-caption)

## Folder Structure
- config : setup training arguments and data path
- data : store IU and MIMIC dataset
- models: basic model and all our models
- modules: 
    - the layer define of our model 
    - dataloader
    - loss function
    - metrics
    - tokenizer
    - some utils
- pycocoevalcap: Microsoft COCO Caption Evaluation Tools

## Training and Testing
- The validation and testing will run after training.
- More options can be found in `config/opts.py` file.
- The model will be trained using command：
    1. full model
    
        ```
        python main.py --cfg config/{$dataset_name}_resnet.yml --expe_name {$experiment name} --label_loss --rank_loss --version 12
        ```
        
    2. basic model
    
        ```
        python main_basic.py --cfg config/{$dataset_name}_resnet.yml --expe_name {$experiment name} --label_loss --rank_loss --version 91
        ```
        
    3. our model without the learned knowledge base
    
        ```
        python main.py --cfg config/{$dataset_name}_resnet.yml --expe_name {$experiment name} --label_loss --rank_loss --version 92
        ```
        
    4. for the model without multi-modal alignment
        You remove `--label_loss` or `--rank_loss` from the commonds.

