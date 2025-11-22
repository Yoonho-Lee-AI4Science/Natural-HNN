# Processing Data
## Processing Cancer dataset
### Note
This README.md file describes the procedure of downloading and processing data for cancer subtype classification.  
Most of our downloading & preprocessing code were adopted from Pathformer.([paper](https://www.biorxiv.org/content/10.1101/2023.05.23.541554v2), [github](https://github.com/lulab/Pathformer))  
Note that preprocessing step can take several hours per data.  
Since it takes too long time ( maybe ~10 hours per cancer) for preprocessing, we also provide pre-processed data since the result can be different by the time at which the data is downloaded. If you just use our pre-processed data (for reproducing result) , you do not need to download or process anything. For pre-processed npy files, you need to unzip npy_files.zip file and place the contents under the {cancer_name}/raw/ directory.  
```
Unfortunately, anonymous github will not correctly provide some pre-processed files (npy files) due to the limit of size.    
```    

### download data
First, you need to download necessary files.  
Follow the instructions written in the README.md in 'Natural-HNN/dataset/cancer_subtype/raw' directory.

### processing data
Preprocessing step is quite complicated. Sometimes, the procedure requires the exection of R scripts during execution of python code. Thus, we recommend to use two separate shells, one for python with natural_hnn_model environment and the other for R script with natural_hnn_r environment.  

1. Execute `preprocess.py` file. You must give dataset name as argument. You can give `--skip` option to skip processing the files that are already created.  
```console
(natural_hnn_model) foo@bar:~/{location_of_the_project_folder}/Natural-HNN/dataset/cancer_subtype/src$ python3 preprocess.py
```
The execution will be halted by pdb with the following messsage.  
```
continue if you finished running gene_length.R or have gene_hgnc_lengths.csv in raw directory.
```
  
2. Calculate gene length. You will get `gene_hgnc_lengths.csv` file in `Natural-HNN/dataset/cancer_subtype/raw` directory after executing `gene_length.R` script just like below. After the execution is complete, then resume preprocess.py by typing `c`.  
```console
(natural_hnn_r) foo@bar:~/{location_of_the_project_folder}/Natural-HNN/dataset/cancer_subtype/src$ Rscript gene_length.R
```
After resuming preprocess.py, it will be halted again by pdb with the following message.
```
continue if you finished running gistic ~~.sh file for the dataset
```
  
3. Execute `gistic_cnv_gene_level_cancer.sh` with cancer name. It can take 0.5 ~ 4 hours depending on the data size. After the execution of the shell script is finished, then resume preprocess.py by typing 'c'. Cancer name should be one of (BRCA, STAD, SARC, HNSC, LGG, CESC, KIPAN, NSCLC)
```console
(natural_hnn_r) foo@bar:~/{location_of_the_project_folder}/Natural-HNN/dataset/cancer_subtype/src$ ./gistic_cnv_gene_level_cancer.sh {cancer name}
```
Now, it will take several hours (~7 hours) to finish preprocessing. When processing is finished, the following message will appear
```
The end of data pre-processing
```

## Processing GO and Calculating pathway similarity
Note that the calculated similarity is provided under `Natural-HNN/dataset/cancer_subtype/pre_processed` directory. (file : `go_sem_dist_bp_lin_bma.csv`)
However, for those who want to manually calculate pathway similarity, we provide the following procedure. This process also needs to use two different environments at the same time.
1. Execute `go_extraction.py` file with `natural_hnn_model` environment. You can give `--skip` option to skip the process that was already done before.
```console
(natural_hnn_model) foo@bar:~/{location of the project folder}/Natural-HNN/dataset/cancer_subtype/src$ python3 go_extraction.py
```
The procedure will be halted with the following message.
```
From the files generated above, you need to do enrichment analysis with R files.
If you finished enrichment analysis, then resume.
```

2. Execute `go_enrich.R` in `natural_hnn_r` environment. After the execution is over, then resume go_extraction.py by typing `c`.
```console
(natural_hnn_r) foo@bar:~/{location of the project folder}/Natural-HNN/dataset/cancer_subtype/src$ Rscript go_enrich.R
```
go_extraction.py will end its procedure with the following message:
```
The end of data go-processing
```

3. Execute `go_distance_bp_lin_bma.R` in `natural_hnn_r` environment. It will create `go_sem_dist_bp_lin_bma.csv` file under the cancer_subtpye/pre_processed directory. Based on what I remember, it took about 2~3 days.
```console
(natural_hnn_r) foo@bar:~/{location of the project folder}/Natural-HNN/dataset/cancer_subtype/src$ Rscript go_distance_bp_lin_bma.R
```