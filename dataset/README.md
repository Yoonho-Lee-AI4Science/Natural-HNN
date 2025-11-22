# Datasets
### Standard Hypergraph Benchmark Datasets
We adopted the [pre-processed files](https://github.com/jianhao2016/AllSet) used in the paper: [You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://arxiv.org/abs/2106.13264).


### Cancer Subtype Classification Task Datasets
Most of our downloading & preprocessing code were adopted from Pathformer([paper](https://www.biorxiv.org/content/10.1101/2023.05.23.541554v2), [github](https://github.com/lulab/Pathformer)).  
We provide pre-processing code in "Natural-HNN/dataset/cancer_subtype/" directory.  
Note that preprocessing step can take several hours.  
Since it takes too long time to perform preprocessing, we also provide pre-processed files. Pre-processed files are contained in the direct subdirectory of current directory with the cancer name. ( ex : "Natural-HNN/dataset/brca" ). If you just use our pre-processed data (for reproducing result) , you do not need to download with R or process anything. For pre-processed npy files, you need to unzip npy_files.zip file and place the contents under the {cancer_name}/raw/ directory.    
For more details, read the description in "Natural-HNN/dataset/cancer_subtype/src/README.md"


### Note on data privacy (For Cancer dataset)
We use publicly available data provided by TCGAbiolinks R package and data from Broad GDAC Firehose ([https://gdac.broadinstitute.org/](https://gdac.broadinstitute.org/)). Note that TCGAbiolinks (refer to : [tcgabiolink_1](https://bioconductor.org/packages/release/bioc/html/TCGAbiolinks.html),[tcga_biolink_2](https://academic.oup.com/nar/article/44/8/e71/2465925?login=true)) and Broad GDAC Firehose (refer to : [firehose_1](https://genomespace.org/support/guides/tool-guide/sections/firebrowse-GS-capabilities/), [firehose_2](https://broadinstitute.atlassian.net/wiki/spaces/GDAC/pages/844334036/FAQ)) provides open-access data. 