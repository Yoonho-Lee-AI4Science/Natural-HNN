library(TCGAbiolinks)
library(SummarizedExperiment)


subtypes <- PanCancerAtlas_subtypes()
sub_df = as.data.frame(subtypes)
write.csv(sub_df, file.path(dirname(getwd()),"raw/labels.csv"))