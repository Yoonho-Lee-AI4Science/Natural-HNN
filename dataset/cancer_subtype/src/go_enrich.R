library(clusterProfiler)

gene_list = read.csv(file = file.path(dirname(getwd()),"pre_processed", "go_ensembl_pathway_map.csv"),header=FALSE )
df = data.frame(matrix(nrow = 1497, ncol = 4000))
for (index in 1:nrow(gene_list)){
    temp_result = enrichGO(gene_list[index, gene_list[index,] != ""] , OrgDb = "org.Hs.eg.db", keyType = "ENSEMBL", ont = "ALL", pvalueCutoff=0.01, pAdjustMethod='hochberg', minGSSize=4)[,"ID"]
    df[index,1:length(temp_result)] = temp_result
}
write.csv(df, file = file.path(dirname(getwd()),"pre_processed", "enriched_go_pathway_result.csv"),row.names=FALSE, quote=FALSE)