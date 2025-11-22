library(TCGAbiolinks)
library(SummarizedExperiment)


dir_name <- file.path(dirname(getwd()),"raw")
args <- commandArgs(trailingOnly = TRUE)
cancer_name <- args[1]

#mRNA
mRNA_query <- GDCquery(project = paste("TCGA", cancer_name, sep="-"),
                  data.category = "Transcriptome Profiling",
                  data.type = "Gene Expression Quantification",
                  workflow.type = "STAR - Counts")
GDCdownload(mRNA_query, method = "api",directory=dir_name)
mRNA_data <- GDCprepare(query = mRNA_query,
                   save = TRUE,
                   directory =  dir_name,
                   save.filename = file.path(dir_name, paste(cancer_name,"mRNA.RData", sep=".")))
load(file.path(dir_name, paste(cancer_name,"mRNA.RData", sep=".")))
data
expr = assay(data)
expr = as.data.frame(expr)
write.csv(expr,file.path(dir_name, paste(cancer_name,"mRNA.csv", sep=".")),row.names=TRUE)


#miRNA
miRNA_query <- GDCquery(project = paste("TCGA", cancer_name, sep="-"),
                  data.category = "Transcriptome Profiling",
                  data.type = "miRNA Expression Quantification",
                  workflow.type = "BCGSC miRNA Profiling")
GDCdownload(miRNA_query, method = "api",directory=dir_name)
miRNA_data <- GDCprepare(query = miRNA_query,
                   save = TRUE,
                   directory =  dir_name, 
                   save.filename = file.path(dir_name, paste(cancer_name,"miRNA.RData", sep=".")))
load(file.path(dir_name, paste(cancer_name,"miRNA.RData", sep=".")))
expr = as.data.frame(data)
write.csv(expr,file.path(dir_name, paste(cancer_name,"miRNA.csv", sep=".")),row.names=TRUE)


#DNA
DNA_query <- GDCquery(project = paste("TCGA", cancer_name, sep="-"),
                  data.category = "DNA Methylation",
                  platform = "Illumina Human Methylation 450",
                  data.type ='Methylation Beta Value')
GDCdownload(DNA_query, method = "api",directory=dir_name)
DNA_data <- GDCprepare(query = DNA_query,
                   save = TRUE,summarizedExperiment=FALSE,
                   directory =  dir_name,
                   save.filename = file.path(dir_name, paste(cancer_name,"DNAmethy.RData", sep=".")))
load(file.path(dir_name, paste(cancer_name,"DNAmethy.RData", sep=".")))
expr = as.data.frame(data)
write.csv(expr,file.path(dir_name, paste(cancer_name,"DNAmethy.csv", sep=".")),row.names=TRUE)

#CNV_seg
CNV_seg_query <- GDCquery(project = paste("TCGA", cancer_name, sep="-"),
                  data.category = "Copy Number Variation",
                  data.type = "Copy Number Segment")
GDCdownload(CNV_seg_query, method = "api",directory=dir_name)
CNV_seg_data <- GDCprepare(query = CNV_seg_query,
                   save = TRUE,
                   directory =  dir_name,
                   save.filename = file.path(dir_name, paste(cancer_name,"CNV_seg.RData", sep=".")))
load(file.path(dir_name, paste(cancer_name,"CNV_seg.RData", sep=".")))
expr = as.data.frame(data)
write.csv(expr,file.path(dir_name, paste(cancer_name,"CNV_seg.csv", sep=".")),row.names=TRUE)


#CNV_seg_masker
CNV_seg_query <- GDCquery(project = paste("TCGA", cancer_name, sep="-"),
                  data.category = "Copy Number Variation",
                  data.type = "Masked Copy Number Segment")
GDCdownload(CNV_seg_query, method = "api",directory=dir_name)
CNV_seg_data <- GDCprepare(query = CNV_seg_query,
                   save = TRUE,
                   directory =  dir_name,
                   save.filename = file.path(dir_name, paste(cancer_name,"CNV_masked_seg.RData", sep=".")))
load(file.path(dir_name, paste(cancer_name,"CNV_masked_seg.RData", sep=".")))
expr = as.data.frame(data)
write.csv(expr,file.path(dir_name, paste(cancer_name,"CNV_masked_seg.csv", sep=".")),row.names=TRUE)