import os
import json


def get_config(filename):
    # load base config
    base_filename = os.path.join(os.path.dirname(filename), 'config.json')
    with open(base_filename, 'r') as f:
        config = json.load(f)
    config["filename"] = filename

    config["experiment_name"] = os.path.splitext(os.path.basename(filename))[0]

    # load experiment-specific config
    with open(filename, 'r') as f:
        experiment_config = json.load(f)

    # replace values
    for key, value in experiment_config.items():
        if key in config.keys():
            config[key] = value
        else:
            raise ValueError(f"Config item: {key} not implemented.")

    return config
