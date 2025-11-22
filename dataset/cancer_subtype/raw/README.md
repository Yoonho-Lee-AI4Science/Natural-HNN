# raw data files
### Note on data privacy
We use publicly available data provided by TCGAbiolinks R package and data from Broad GDAC Firehose ([https://gdac.broadinstitute.org/](https://gdac.broadinstitute.org/)). Note that TCGAbiolinks (refer to : [tcgabiolink_1](https://bioconductor.org/packages/release/bioc/html/TCGAbiolinks.html),[tcga_biolink_2](https://academic.oup.com/nar/article/44/8/e71/2465925?login=true)) and Broad GDAC Firehose (refer to : [firehose_1](https://genomespace.org/support/guides/tool-guide/sections/firebrowse-GS-capabilities/), [firehose_2](https://broadinstitute.atlassian.net/wiki/spaces/GDAC/pages/844334036/FAQ)) provides open-access data. 

### Additional Download
You need to download the following files. After downloading, put the files in this directory with the same name mentioned below.  
1. [go-basic.obo](https://geneontology.org/docs/download-ontology/)  
2. [snp6.na35.remap.hg38.subset.txt.gz](https://gdc.cancer.gov/about-data/gdc-data-processing/gdc-reference-files)  
3. [HM450.hg38.manifest.gencode.v36.tsv](https://gdc.cancer.gov/about-data/gdc-data-processing/gdc-reference-files)  
4. [goa_human.gaf](https://current.geneontology.org/products/pages/downloads.html)  
5. [Homo_sapiens.GRCh38.91.chr.gtf](https://ftp.ensembl.org/pub/release-91/gtf/homo_sapiens/Homo_sapiens.GRCh38.91.chr.gtf.gz) : download and unzip the gz file. Place 'Homo_sapiens.GRCh38.91.chr.gtf' file under the current directory.  

### Generating files
You need to execute get_marker.R in Natural-HNN/dataset/cancer_subtype/src directory to generate snp6.na35.remap.hg38.subset.marker_file.txt from snp6.na35.remap.hg38.subset.txt.gz. Use 'natural_hnn_r' environment.  
```console
(natural_hnn_r) foo@bar:~/{location_of_the_project_folder}/Natural-HNN/dataset/cancer_subtype/src$ Rscript get_marker.R
```

### Downloading cancer data
You need to download the data for the following cancers :  BRCA, STAD, SARC, CESC, HNSC, LGG, KIRC, KICH, KIRP, LUSC, LUAD
Do the following execution for all datasets above. Use 'natural_hnn_r' environment. It can take about 0.5 ~ 2 hours per cancer depending on the data size.  
```console
(natural_hnn_r) foo@bar:~/{location_of_the_project_folder}/Natural-HNN/dataset/cancer_subtype/src$ Rscript download_cancer.R {cancer name}
```

### Downloading label data
##### BRCA and STAD cancer
For BRCA, STAD labels, you can download by executing download_labels.R script. It will create 'labels.csv' under 'Natural-HNN/dataset/cancer_subtype/raw' directory. Use 'natural_hnn_r' environment.  
```console
(natural_hnn_r) foo@bar:~/{location_of_the_project_folder}/Natural-HNN/dataset/cancer_subtype/src$ Rscript download_labels.R
```


##### Other cancers
You need to download labels from [Broad GDAC Firehose](https://gdac.broadinstitute.org/). Click the 'Browse' button in 'Data' column for each cancer type ( SARC, CESC, HNSC, LGG, LUAD, LUSC, KIRP, KIRC, KICH). The cancer type is shown in 'Cohort' column. Click 'OK' button and download 'Clinical_Pick_Tier1 (MD5)' file. Find All_CDEs.txt and rename it to '{cancer name}_lables.txt' (ex: CESC_labels.txt) and place it under 'Natural-HNN/dataset/cancer_subtype/raw' directory.

### Summary
This directory must have the following files.
```
feature_type.txt
gene_all.txt
go-basic.obo
goa_human.gaf
hg38.UCSC.add_miR.160920.refgene.mat
HM450.hg38.manifest.gencode.v36.tsv
Homo_sapiens.GRCh38.91.chr.gtf
miRNA_id_new.txt
Pathformer_select_gene_name.txt
Pathformer_select_gene.txt
pathway_list.txt
snp6.na35.remap.hg38.subset.marker_file.txt
snp6.na35.remap.hg38.subset.txt.gz
```

As well as downloaded cancer files below. (BRCA, STAD, SARC, CESC, HNSC, LGG, KICH, KIRC, KIRP, LUSC, LUAD)

```
labels.csv
CESC_labels.txt
HNSC_labels.txt
KICH_labels.txt
KIRC_labels.txt
KIRP_labels.txt
LGG_labels.txt
LUAD_labels.txt
LUSC_labels.txt
SARC_labels.txt
STAD_labels.txt
BRCA.miRNA.csv
BRCA.mRNA.csv
BRCA.DNAmethy.csv
BRCA.CNV_seg.csv
BRCA.CNV_masked_seg.csv
STAD.miRNA.csv
STAD.mRNA.csv
STAD.DNAmethy.csv
STAD.CNV_seg.csv
STAD.CNV_masked_seg.csv
SARC.miRNA.csv
SARC.mRNA.csv
SARC.DNAmethy.csv
SARC.CNV_seg.csv
SARC.CNV_masked_seg.csv
CESC.miRNA.csv
CESC.mRNA.csv
CESC.DNAmethy.csv
CESC.CNV_seg.csv
CESC.CNV_masked_seg.csv
HNSC.miRNA.csv
HNSC.mRNA.csv
HNSC.DNAmethy.csv
HNSC.CNV_seg.csv
HNSC.CNV_masked_seg.csv
KIRC.miRNA.csv
KIRC.mRNA.csv
KIRC.DNAmethy.csv
KIRC.CNV_seg.csv
KIRC.CNV_masked_seg.csv
KIRP.miRNA.csv
KIRP.mRNA.csv
KIRP.DNAmethy.csv
KIRP.CNV_seg.csv
KIRP.CNV_masked_seg.csv
KICH.miRNA.csv
KICH.mRNA.csv
KICH.DNAmethy.csv
KICH.CNV_seg.csv
KICH.CNV_masked_seg.csv
LGG.miRNA.csv
LGG.mRNA.csv
LGG.DNAmethy.csv
LGG.CNV_seg.csv
LGG.CNV_masked_seg.csv
LUAD.miRNA.csv
LUAD.mRNA.csv
LUAD.DNAmethy.csv
LUAD.CNV_seg.csv
LUAD.CNV_masked_seg.csv
LUSC.miRNA.csv
LUSC.mRNA.csv
LUSC.DNAmethy.csv
LUSC.CNV_seg.csv
LUSC.CNV_masked_seg.csv
```