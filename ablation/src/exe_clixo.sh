file_name=$1
av=$2
bv=$3
mv=$4
sv=$5
new_file_name=$6
ont_file_name=$7
cluster_gene_file_name=$8
./clixo -i $file_name -a $av -b $bv -m $mv -s $sv > $new_file_name
grep -v "#" $new_file_name > $ont_file_name
./ontologyTermStats $ont_file_name genes > $cluster_gene_file_name