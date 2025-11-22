#!/bin/bash
data=$1
cancer_root_dir=$(dirname $(dirname $(realpath "$0")))

gistic2 \
-b "${cancer_root_dir}/temporal_files/${data}" \
-seg "${cancer_root_dir}/temporal_files/${data}.CNV_masked_seg_filter.txt" \
-mk "${cancer_root_dir}/raw/snp6.na35.remap.hg38.subset.marker_file.txt" \
-refgene "${cancer_root_dir}/raw/hg38.UCSC.add_miR.160920.refgene.mat" \
-ta 0.1 \
-armpeel 1 \
-brlen 0.7 \
-cap 1.5 \
-conf 0.99 \
-td 0.1 \
-genegistic 1 \
-gcm extreme \
-js 4 \
-maxseg 2000 \
-qvt 0.25 \ 
-rx 0 \
-savegene 1