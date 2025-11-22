library(GOSemSim)
library(Rcpp)

hsGO <- godata('org.Hs.eg.db', ont="BP")
go_list = read.csv(file = file.path(dirname(getwd()),"pre_processed", "go_pathway_final.csv"),header=FALSE )
df = data.frame(matrix(nrow = 1497, ncol = 1497))
for (index in 1:1497){
    for (index_2 in index:1497){
        temp_result = mgoSim(go_list[index,go_list[index,]!=""], go_list[index_2,go_list[index_2,]!=""], semData=hsGO, measure="Lin", combine="BMA")
        df[index,index_2] = temp_result
    }
}
write.csv(df, file = file.path(dirname(getwd()),"pre_processed", "go_sem_dist_bp_lin_bma.csv"),row.names=FALSE, quote=FALSE)