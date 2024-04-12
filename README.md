# Metrically Scaled Monocular Depth Estimation through Sparse Priors for Underwater Robots

## Description
This is an accepted paper at [**ICRA 2024**](https://2024.ieee-icra.org).

In short, this depth prediction model estimates depth maps from underwater RGB images. To solve the problem of scale ambiguity, the network additionally fuses sparse depth priors, e.g. coming from a SLAM pipeline.

**Paper document:** [arXiv](https://arxiv.org/abs/2310.16750), [arXiv PDF](https://arxiv.org/pdf/2310.16750.pdf), [Google Drive](https://drive.google.com/file/d/1iVhY4Fepr0NKsXlnEsKGzqN_aFS8brBi/view?usp=sharing)  
**Paper video (3min):** [Google Drive](https://drive.google.com/file/d/1gjsTty9ybdq1y9jcAU-WVLKuEU3IfC1v/view?usp=sharing)  
**Paper graphic:** [Google Drive](https://drive.google.com/file/d/1CKoh1t7I11uhEjTj_gEW5-1O-NIelhbd/view?usp=sharing)  

https://github.com/ebnerluca/depth_estimation/assets/48278846/24c51208-c357-4ffc-91d6-d1a83b60e995

**Video:** RGB (left) vs. Depth Prediction (right) on FLSea dataset at 10 Hz (Full Video: [Google Drive](https://drive.google.com/file/d/1KoIy49MqRIfAvJXvllrXJwZ92Vnmrgh8/view?usp=sharing)).

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
---

## Demo
While in the repository root folder, run
```
python3 inference.py
```
The results will be available under `data/out`

---

## Documentation

### Training, Test & Inference
The training, test and inference scripts are made available in the repository root folder and serve as examples on how you can train and monitor your custom training runs.

### Depth Estimation
The `depth_estimation` module contains python packages with the code for setting up the model as well as utils to load data, compute losses and visualize data during training.

### Custom Datasets
`data/example_dataset` folder contains an example dataset which can be used to run the demo as well as an inspiration on how to setup your own custom dataset. Inside, the `dataset.py` script provides a convenient `get_example_dataset()` method which is reading a list of path tuples from `dataset.csv`.

### Preprocessing
The `helper_scripts` folder contains useful scripts which can be used for preprocessing of datasets, such as extracting visual features for usage as sparse depth measurements or creating train/test splits. In general, every data point in a dataset needs:
- RGB image (see `data/example_dataset/rgb`)
- keypoint location with corresponding depth (see `data/example_dataset/features`) *
- depth image ground truth (for training / evaluation only, see `data/example_dataset/depth`)

\* check out `helper_scripts/extract_dataset_features.py` for a simple example on how such features can be generated if ground truth is available. If not, you could use e.g. SLAM.

Then, the `.csv` file defines the tuples, see `data/example_dataset/dataset.csv`.

Make sure that you also load your data correctly via the dataloader, e.g. depending on your dataset, images can be in uint8, uint16 or float format (see `data/example_dataset/dataset.py`)







---

## Acknowledgements
[AdaBins](https://github.com/shariqfarooq123/AdaBins)  
[UDepth](https://github.com/uf-robopi/UDepth)  
[FLSea](https://arxiv.org/abs/2302.12772)

