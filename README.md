# NR-CRS

Source code of the submission 4147 **"Augmentation with Neighboring Information for Conversational Recommendation"** for peer review.

For anonymity reasons, we have replaced sensitive words in the implementation with ```XXXX_id``` and omitted the base ```crslab/crslab/``` folder. 
```crslab/crslab/``` contains basic data preprocessing, evaluation metrics and etc, can be easily found in [C2-CRS](https://github.com/Zyh716/WSDM2022-C2CRS).

### Dataset

We follow the experimental settings detailed in [C2-CRS](https://dl.acm.org/doi/abs/10.1145/3488560.3498514).

For dataset and external resources, please refer to the GitHub repo of [C2-CRS](https://github.com/Zyh716/WSDM2022-C2CRS).


### Experiments

We use Slurm (Simple Linux Utility for Resource Management) to submit and run all the experiments.

First, we generate Slurm scripts for experiments.
```
python generate_slurms.py
```

For the main experiment, we first train NR-CRS on the recommendation task and then the conversation task:
```
sbatch script/redial/experiment_slurms/nr-crs/base_0.slurm
```

### Requirements: 
- Python == 3.8
- Pytorch == 1.8.1
- CRSLab == 0.1.2

We use Anaconda to prepare the environment for NR-CRS:
```
conda install -n nrcrs-env -c pytorch faiss-cpu
```

Then, we install requirements via pip:
```
pip install -r requirements.txt
```