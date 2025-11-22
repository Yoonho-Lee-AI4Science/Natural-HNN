# Natural-HNN

## Note
This is the official repository for `Disentangling Hyperedges through the Lens of Category Theory`, which is accepted to `NeurIPS 2025`.

Note that, the paper `Capturing functional context of genetic pathways through hyperedge disentanglement` (accepted to ICLR 2025 workshop MLGenX) also uses the same code provided in this repository.

Official name of our model is `Natural-HNN`. However, it is named `disen_hgnn` in our implementation(code).

If you use our preprocessed data files or our model, please cite our paper.

## Environment

### conda with R package (download & process bio data)
For downloading and pre-processing, you need to use following R packages
* [TCGAbiolinks](https://bioconductor.org/packages/release/bioc/html/TCGAbiolinks.html)
* [gistic2](https://broadinstitute.github.io/gistic2/)
* [biomaRt](https://bioconductor.org/packages/release/bioc/html/biomaRt.html)
* [clusterProfiler](https://bioconductor.org/packages/release/bioc/html/clusterProfiler.html)
* [rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html)
* [GOSemsim](https://bioconductor.org/packages/release/bioc/html/GOSemSim.html)
  
You can simply create conda environment by executing the following:  

```console
(base) foo@bar:~/{location of the project folder}/Natural-HNN$ conda env create -f natural_hnn_r.yml
```

If it does not work well or cause errors, you can manually create environment by following:

```console
(base) foo@bar:~$ conda create -n natural_hnn_r
(base) foo@bar:~$ conda activate natural_hnn_r
(natural_hnn_r) foo@bar:~$ conda install conda-forge::r-base
(natural_hnn_r) foo@bar:~$ conda install bioconda::bioconductor-tcgabiolinks
(natural_hnn_r) foo@bar:~$ conda install hcc::gistic2
(natural_hnn_r) foo@bar:~$ conda install bioconda::bioconductor-biomart
(natural_hnn_r) foo@bar:~$ conda install bioconda::bioconductor-clusterprofiler
(natural_hnn_r) foo@bar:~$ conda install conda-forge::r-rcpp
(natural_hnn_r) foo@bar:~$ conda install bioconda::bioconductor-gosemsim
```


### conda environment for training the model
You need to create conda environment named `natural_hnn_model` that includes the following libraries
* pytorch
* pytorch_geometric
* torch-scatter
* networkx
* numpy
* scipy
* scikit-learn
* seaborn
* matplotlib
* pandas
* tensorboard
* wandb
* pdb
* tqdm
* pickle

You can simply create environment by following:
```console
(base) foo@bar:~/{location of the project folder}/Natural-HNN$ conda env create -f natural_hnn_model.yml
```

If it does not work well or cause errors, you can manually create environment by following:

```console
(base) foo@bar:~$ conda create -y -n natural_hnn_model python=3.7
(base) foo@bar:~$ conda activate natural_hnn_model
(natural_hnn_model) foo@bar:~$ conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
(natural_hnn_model) foo@bar:~$ conda install -y -c anaconda scikit-learn
(natural_hnn_model) foo@bar:~$ conda install -y -c anaconda numpy
(natural_hnn_model) foo@bar:~$ conda install -y -c conda-forge matplotlib
(natural_hnn_model) foo@bar:~$ conda install -y -c anaconda pandas
(natural_hnn_model) foo@bar:~$ conda install pyg -y -c pyg
(natural_hnn_model) foo@bar:~$ conda install -y -c anaconda networkx
(natural_hnn_model) foo@bar:~$ conda install -y -c anaconda seaborn
(natural_hnn_model) foo@bar:~$ conda install -y -c anaconda scipy
(natural_hnn_model) foo@bar:~$ conda install -y -c anaconda tqdm
(natural_hnn_model) foo@bar:~$ conda install -y -c conda-forge tensorboard
(natural_hnn_model) foo@bar:~$ conda install -y conda-forge::pdbpp
(natural_hnn_model) foo@bar:~$ conda install -y conda-forge::pickle5
(natural_hnn_model) foo@bar:~$ conda install -y conda-forge::wandb
```

## Traning the model
The description is written in Natura-HNN/src/README.md file.

## Additional experiments
The description is written in Natural-HNN/ablation/README.md file.


## ETC
If you are interested in GOSemSim library, visit [this website](https://yulab-smu.top/biomedical-knowledge-mining-book/). It will be helpful for understanding the R codes we used for data preprocessing.  



