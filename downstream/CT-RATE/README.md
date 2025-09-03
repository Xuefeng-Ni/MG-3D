## Requirements

Before you start, you must install the necessary dependencies. To do so, execute the following commands:

```setup
# Navigate to the 'transformer_maskgit' directory and install the required packages
cd transformer_maskgit
pip install -e .

# Return to the root directory
cd ..

# Navigate to the 'CT-RATE' directory and install its required packages
cd CT-RATE
pip install -e .

# Return to the root directory
cd ..
```

You can download the dataset used in this work via the [Hugging Face repository](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE). 

Data used to finetune and validate the text classifier model can be accessed [here](text_classifier/data).


