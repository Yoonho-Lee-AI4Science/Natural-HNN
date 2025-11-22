dir_name <- file.path(dirname(getwd()),"raw")
hg38_marker_file <- read.delim(file.path(dir_name, "snp6.na35.remap.hg38.subset.txt.gz"))
hg_marker_file <- hg38_marker_file[hg38_marker_file$freqcnv=="FALSE",]
hg_marker_file <- hg_marker_file [,c(1,2,3)]
write.table(hg_marker_file,file = file.path(dirname(getwd()),"raw","snp6.na35.remap.hg38.subset.marker_file.txt"),sep = "\t",col.names = T,row.names = F)