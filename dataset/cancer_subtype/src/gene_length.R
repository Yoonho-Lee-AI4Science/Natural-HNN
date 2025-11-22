library(biomaRt)
dir_name <- dirname(getwd())


ensembl_list = readLines(file.path(dir_name,"pre_processed/index_ensembl_map.txt"))
human <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
gene_coords=getBM(attributes=c("ensembl_gene_id", "start_position","end_position"), filters="ensembl_gene_id", values=ensembl_list, mart=human)
gene_coords$size=gene_coords$end_position - gene_coords$start_position
colnames(gene_coords)[1] <- "gene_id"
colnames(gene_coords)[4] <- "length"
write.csv(gene_coords[,c("gene_id","length")], file=file.path(dir_name,"raw/gene_ensembl_lengths.csv"),row.names=FALSE, quote=FALSE)



hgnc_list = readLines(file.path(dir_name,"pre_processed/index_gene_map.txt"))
human <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
gene_coords=getBM(attributes=c("hgnc_symbol", "start_position","end_position"), filters="hgnc_symbol", values=hgnc_list, mart=human)
gene_coords$size=gene_coords$end_position - gene_coords$start_position
colnames(gene_coords)[1] <- "gene_id"
colnames(gene_coords)[4] <- "length"
write.csv(gene_coords[,c("gene_id","length")], file=file.path(dir_name,"raw/gene_hgnc_lengths.csv"),row.names=FALSE, quote=FALSE)