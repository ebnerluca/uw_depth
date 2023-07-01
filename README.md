# Metrically Scaled Monocular Depth Estimation through Sparse Priors for Underwater Robots

Thesis link: [Google Drive](https://drive.google.com/file/d/14l2fhPkZmZFd02KGrkmKGr-atHMf_0nd/view?usp=sharing)  
Video link: [Google Drive](https://drive.google.com/file/d/1KoIy49MqRIfAvJXvllrXJwZ92Vnmrgh8/view?usp=sharing)

<video height="480" src="data/readme/rgb_heatmap_demo.mp4" controls title="Title"></video>
---

## Install
Clone the repository, and navigate into its root folder. From there:
```
# create venv and activate
python3 -m venv venv
source venv/bin/activate

# install pip dependencies
pip3 install -r dependencies.txt

# add repo root to pythonpath
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Demo
While in the repository root folder, run
```
python3 inference.py
```
The results will be available under `data/out`

---

## Documentation

### Training, Test & Inference
The training scripts are made available in the repository root folder and serve as a examples on how you can train and monitor your custom training runs.

### Depth Estimation
The `depth_estimation` module contains python packages with the code for setting up the model as well as utils to load data, compute losses and visualize data during training.

### Custom Datasets
`data/example_dataset` folder contains an example dataset which can be used as inspiration on how to setup your own custom dataset. Inside, the `dataset.py` script contains a conventient `get_example_dataset()` method which is reading a list of path tuples from `dataset.csv`.

### Preprocessing
The `helper_scripts` folder contains useful scripts which can be used for preprocessing of datasets, such as extracting visual features for usage as sparse depth measurements or creating train/test splits.






---

## Acknowledgements
https://github.com/shariqfarooq123/AdaBins  
https://github.com/uf-robopi/UDepth

