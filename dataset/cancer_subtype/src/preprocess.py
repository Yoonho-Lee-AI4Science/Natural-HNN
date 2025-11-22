import pandas as pd
import os
import argparse
import subprocess
import numpy as np
import re
import pdb
from tqdm import tqdm
from time import time
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    
class Process_data:
    def __init__(self, args):
        self.args = args
        self.raw_dir = self.args.raw_dir
        self.temp_dir = self.args.temporal_dir
        self.proc_dir = self.args.processed_dir
        self.code_dir = self.args.code_dir
        self.data = self.args.data
        self.skip = self.args.skip
        self.data_list = ["BRCA", "STAD", "SARC", "LGG", "HNSC", "CESC", "KIPAN", "NSCLC"]
        self.root_root_dir = self.args.root_root_dir

        
    def process(self):
        self.get_pathways()
        print()
        print("continue if you finished running gene_length.R or have gene_hgnc_lengths.csv in raw directory.")
        print()
        pdb.set_trace()
        self.get_gene_length()
        if self.data in ["KIPAN", "NSCLC"]:
            self.combine_downloaded_data()
        self.cnv_masked_filter()
        print()
        print("continue if you finished running gistic ~~.sh file for the dataset")
        print()
        pdb.set_trace()
        self.data_filter()
        self.get_label()
        self.process_label()
        self.dna_methylation_promoterid()
        self.dna_methylation()
        self.cnv_id()
        self.cnv_data()
        self.cnv_data_merge()
        self.data_gene_embedding_all()
        self.data_gene_embedding_merge()
        print("The end of data pre-processing")
        return
        
    
        
    def combine_downloaded_data(self):
        if self.data == "KIPAN":
            if self.args.skip and os.path.isfile(self.raw_dir+"KIPAN.CNV_masked_seg.csv") and os.path.isfile(self.raw_dir+"KIPAN.DNAmethy.csv") and os.path.isfile(self.raw_dir+"KIPAN.miRNA.csv") and os.path.isfile(self.raw_dir+"KIPAN.mRNA.csv"):
                print('skip combining KIPAN data')
                return
            print("KIPAN")
            print("CNV_masked_seg.csv") # correct
            cnv_seg_content = open(self.raw_dir+"KIRC.CNV_masked_seg.csv","r").read().split('\n')[:-1] + open(self.raw_dir+"KICH.CNV_masked_seg.csv","r").read().split('\n')[1:-1] + open(self.raw_dir+"KIRP.CNV_masked_seg.csv","r").read().split('\n')[1:]
            cnv_seg_write = open(self.raw_dir+"KIPAN.CNV_masked_seg.csv","w")
            cnv_seg_write.write('\n'.join(cnv_seg_content))
            cnv_seg_write.close()
            
            
            if not (self.args.skip and os.path.isfile(self.raw_dir+"KIPAN.DNAmethy.csv")):
                print("DNAmethy.csv") # wrong
                temp_content = [[i.split(',') for i in open(self.raw_dir+"KIRC.DNAmethy.csv","r").read().split('\n')[:-1]],[i.split(',') for i in open(self.raw_dir+"KICH.DNAmethy.csv","r").read().split('\n')[:-1]],[i.split(',') for i in open(self.raw_dir+"KIRP.DNAmethy.csv","r").read().split('\n')[:-1]]]
                entire_len = len(temp_content[0])
                text = ""
                for data_index in range(entire_len):
                    for data_modal in range(len(temp_content)):
                        if data_modal == 0 :
                            text += ','.join(temp_content[data_modal][data_index])
                        else: 
                            text += ','.join(temp_content[data_modal][data_index][1:])
                    text += '\n'
                del temp_content
                cnv_seg_write = open(self.raw_dir+"KIPAN.DNAmethy.csv","w")
                cnv_seg_write.write(text)
                cnv_seg_write.close()

            
            if not (self.args.skip and os.path.isfile(self.raw_dir+"KIPAN.miRNA.csv")):
                print("miRNA.csv") # wrong
                temp_content = [[i.split(',') for i in open(self.raw_dir+"KIRC.miRNA.csv","r").read().split('\n')[:-1]],[i.split(',') for i in open(self.raw_dir+"KICH.miRNA.csv","r").read().split('\n')[:-1]],[i.split(',') for i in open(self.raw_dir+"KIRP.miRNA.csv","r").read().split('\n')[:-1]]]

                entire_len = len(temp_content[0])
                text = ""
                for data_index in range(entire_len):
                    for data_modal in range(len(temp_content)):
                        if data_modal == 0 :
                            text += ','.join(temp_content[data_modal][data_index])
                        else: 
                            text += ','.join(temp_content[data_modal][data_index][2:])
                    text += '\n'
                cnv_seg_write = open(self.raw_dir+"KIPAN.miRNA.csv","w")
                cnv_seg_write.write(text)
                cnv_seg_write.close()


            if not (self.args.skip and os.path.isfile(self.raw_dir+"KIPAN.mRNA.csv")):
                print("mRNA.csv") # wrong
                temp_content = [[i.split(',') for i in open(self.raw_dir+"KIRC.mRNA.csv","r").read().split('\n')[:-1]],[i.split(',') for i in open(self.raw_dir+"KICH.mRNA.csv","r").read().split('\n')[:-1]],[i.split(',') for i in open(self.raw_dir+"KIRP.mRNA.csv","r").read().split('\n')[:-1]]]

                entire_len = len(temp_content[0])
                text = ""
                for data_index in range(entire_len):
                    for data_modal in range(len(temp_content)):
                        if data_modal == 0 :
                            text += ','.join(temp_content[data_modal][data_index])
                        else: 
                            text += ','.join(temp_content[data_modal][data_index][1:])
                    text += '\n'
                cnv_seg_write = open(self.raw_dir+"KIPAN.mRNA.csv","w")
                cnv_seg_write.write(text)
                cnv_seg_write.close()
            
        elif self.data == "NSCLC":
            if self.args.skip and os.path.isfile(self.raw_dir+"NSCLC.CNV_masked_seg.csv") and os.path.isfile(self.raw_dir+"NSCLC.DNAmethy.csv") and os.path.isfile(self.raw_dir+"NSCLC.miRNA.csv") and os.path.isfile(self.raw_dir+"NSCLC.mRNA.csv"):
                print('skip combining NSCLC data')
                return
            
            print("NSCLC")
            print("CNV_masked_seg.csv")
            cnv_seg_content = open(self.raw_dir+"LUAD.CNV_masked_seg.csv","r").read().split('\n')[:-1] + open(self.raw_dir+"LUSC.CNV_masked_seg.csv","r").read().split('\n')[1:]
            cnv_seg_write = open(self.raw_dir+"NSCLC.CNV_masked_seg.csv","w")
            cnv_seg_write.write('\n'.join(cnv_seg_content))
            cnv_seg_write.close()
            
            
            
            print("DNAmethy.csv")
            temp_content = [[i.split(',') for i in open(self.raw_dir+"LUAD.DNAmethy.csv","r").read().split('\n')[:-1]],[i.split(',') for i in open(self.raw_dir+"LUSC.DNAmethy.csv","r").read().split('\n')[:-1]]]

            entire_len = len(temp_content[0])
            text = ""
            for data_index in range(entire_len):
                for data_modal in range(len(temp_content)):
                    if data_modal == 0 :
                        text += ','.join(temp_content[data_modal][data_index])
                    else: 
                        text += ','.join(temp_content[data_modal][data_index][1:])
                text += '\n'
            del temp_content
            cnv_seg_write = open(self.raw_dir+"NSCLC.DNAmethy.csv","w")
            cnv_seg_write.write(text)
            cnv_seg_write.close()
            
            print("miRNA.csv")
            temp_content = [[i.split(',') for i in open(self.raw_dir+"LUAD.miRNA.csv","r").read().split('\n')[:-1]],[i.split(',') for i in open(self.raw_dir+"LUSC.miRNA.csv","r").read().split('\n')[:-1]]]

            entire_len = len(temp_content[0])
            text = ""
            for data_index in range(entire_len):
                for data_modal in range(len(temp_content)):
                    if data_modal == 0 :
                        text += ','.join(temp_content[data_modal][data_index])
                    else: 
                        text += ','.join(temp_content[data_modal][data_index][2:])
                text += '\n'
            cnv_seg_write = open(self.raw_dir+"NSCLC.miRNA.csv","w")
            cnv_seg_write.write(text)
            cnv_seg_write.close()
            
            print("mRNA.csv")
            temp_content = [[i.split(',') for i in open(self.raw_dir+"LUAD.mRNA.csv","r").read().split('\n')[:-1]],[i.split(',') for i in open(self.raw_dir+"LUSC.mRNA.csv","r").read().split('\n')[:-1]]]

            entire_len = len(temp_content[0])
            text = ""
            for data_index in range(entire_len):
                for data_modal in range(len(temp_content)):
                    if data_modal == 0 :
                        text += ','.join(temp_content[data_modal][data_index])
                    else: 
                        text += ','.join(temp_content[data_modal][data_index][1:])
                text += '\n'
            cnv_seg_write = open(self.raw_dir+"NSCLC.mRNA.csv","w")
            cnv_seg_write.write(text)
            cnv_seg_write.close()
            
            
        else: 
            raise NotImplementedError
        return
    
    
    def copy_data_to_new_folder(self):
        file_list = os.listdir(self.proc_dir)
        for file in file_list:
            src_file = self.proc_dir+file
            data_type = file.split('_')[0]

            if data_type in self.data_list:
                dst_file = self.root_root_dir+'/'+data_type.lower()+'/raw/'+file
                if os.path.exists(dst_file):
                    print("replace already existing file: "+data_type.lower()+'/raw/'+file)
                    os.remove(dst_file)
                shutil.copyfile(src_file, dst_file)
                    
            else: 
                for data_type in self.data_list:
                    dst_file = self.root_root_dir+'/'+data_type.lower()+'/raw/'+file
                    if os.path.exists(dst_file):
                        print("replace already existing file: "+data_type.lower()+'/raw/'+file)
                        os.remove(dst_file)
                    shutil.copyfile(src_file, dst_file)

            
        return
        
    def process_label(self):
        if self.skip and os.path.isfile(self.proc_dir+self.data+'_index_label_map.txt') and os.path.isfile(self.proc_dir+self.data+'_labels.txt'):
            print("skip process_label")
            return
        label=pd.read_csv(self.temp_dir+self.data+"_sample_id_label_subtype.txt",sep='\t')
        label_cats = sorted(set(list(label['subtype'])))
        num_samples = len(label)
        label_anonymized = np.zeros(num_samples)
        for i,labels in enumerate(label_cats):
            label_anonymized[np.where(np.array(list(label['subtype'])) == labels)] = i

        np.savetxt(self.proc_dir+self.data+'_index_label_map.txt', np.array(label_cats), delimiter='\n', fmt ='% s')
        np.savetxt(self.proc_dir+self.data+'_labels.txt', label_anonymized.astype(int), delimiter='\n', fmt ='% d')

        return
        
        
    def get_pathways(self):
        file_exists = os.path.isfile(self.proc_dir+"index_pathway_map.txt") and os.path.isfile(self.proc_dir+"index_gene_map.txt") \
            and os.path.isfile(self.proc_dir+"index_ensembl_map.txt") and os.path.isfile(self.proc_dir+"edge_index_raw.npy") 
        if self.skip and file_exists:
            print('skip get_pathways')
            return
        pathway_db = open(self.raw_dir+"pathway_list.txt","r").read().split('\n')[1:-1]
        pathway_db = [[a.split('\t')[0], a.split('\t')[2]]+a.split('\t')[1].split(',') for a in pathway_db]
        gene_ensembl_pair = open(self.raw_dir+"Pathformer_select_gene_name.txt","r").read().split('\n')[1:-1]
        gene_ensembl_pair = [a.split('\t') for a in gene_ensembl_pair]
        existing_pair_gene = [a[1] for a in gene_ensembl_pair]
        
        pathways = [a[0] for a in pathway_db]
        used_gene_list = sorted(list(set(existing_pair_gene) & set([b for a in pathway_db for b in a[2:] ])))
        
        
        index_pathways_map = open(self.proc_dir+"index_pathway_map.txt", "w")
        index_pathways_map.write('\n'.join(pathways))
        index_pathways_map.close()
        index_gene_map = open(self.proc_dir+"index_gene_map.txt","w")
        index_gene_map.write('\n'.join(used_gene_list))
        index_gene_map.close()
        
        gene_ensembl_pair_transposed = np.array(gene_ensembl_pair).T.tolist()
        indices = [gene_ensembl_pair_transposed[1].index(i) for i in used_gene_list]
        index_ensembl_map_list = (np.array(gene_ensembl_pair_transposed[0])[np.array(indices)]).tolist()
        index_ensembl_map = open(self.proc_dir+"index_ensembl_map.txt", "w")
        index_ensembl_map.write('\n'.join(index_ensembl_map_list))
        index_ensembl_map.close()
        
        path_gene_matrix = []
        for pathway in tqdm(pathway_db):
            path_gene_matrix.append([used_gene_list.index(gene) for gene in pathway[2:] if gene in used_gene_list])
        edge_index_0 = []
        edge_index_1 = []
        for i,pathway in tqdm(enumerate(path_gene_matrix)):
            edge_index_0 += [i]*len(pathway)
            edge_index_1 += pathway
        edge_index = np.array([edge_index_1, edge_index_0])
        np.save(self.proc_dir+"edge_index_raw.npy", edge_index)
        return
    
    
    
    def data_gene_embedding_merge(self):
        if self.skip and os.path.isfile(self.proc_dir+self.data+'_data_all.npy'):
            print("skip data_gene_embedding_merge")
            return
        feature_data=pd.read_csv(self.proc_dir+'index_ensembl_map.txt',header=None)
        feature_data=list(feature_data[0])
        label=pd.read_csv(self.temp_dir+self.data+"_sample_id_label_subtype.txt",sep='\t')
        sample=list(label['sample_id'])
        feature_type=pd.read_csv(self.raw_dir+'feature_type.txt',sep='\t',header=None)
        feature_type['type']=feature_type[0].map(lambda x:x.split('_')[0])
        feature_type['type_name']=feature_type[0].map(lambda x:x.split('.')[0])

        data_all = np.zeros([len(feature_type), len(feature_data), len(sample)])

        for i in range(len(feature_type)):
            print(feature_type.loc[i,'type_name'])
            if feature_type.loc[i,'type_name'] == "RNA_all_TPM":
                data=pd.read_csv(self.temp_dir+self.data+'_'+feature_type.loc[i,'type_name']+'.txt',sep='\t')
            else: 
                data=pd.read_csv(self.temp_dir+self.data+'__'+feature_type.loc[i,'type_name']+'.txt',sep='\t')
            data=data.rename(columns={data.columns[0]:'feature'})
            data = data.drop_duplicates()
            data=data.fillna(0)
            data=data[['feature']+sample]

            data['gene_id']=data['feature'].astype(str)
            data_=pd.DataFrame(columns=['gene_id']+sample)
            data_['gene_id']=list(set(feature_data)-set(data['gene_id']))
            data=pd.concat([data,data_])
            data=data.drop_duplicates('gene_id')
            data=data.set_index('gene_id')
            data_select=data.loc[feature_data,sample]
            data_select=data_select.fillna(0)

            data_all[i, :, :] = data_select.values

        data_all=data_all.transpose((2,1,0))
        np.save(file=self.proc_dir+self.data+'_data_all.npy',arr=data_all)
    

        return
    
    def data_gene_embedding_all(self):
        RNA_file_exists = os.path.isfile(self.temp_dir+self.data+'_mRNA_rawdata.txt') and os.path.isfile(self.temp_dir+self.data+'_miRNA_rawdata.txt') \
            and os.path.isfile(self.temp_dir+self.data+'_RNA_all_rawdata.txt') and os.path.isfile(self.temp_dir+self.data+'_RNA_all_TPM.txt')
        DNA_file_exists = os.path.isfile(self.temp_dir+self.data+'_methylation_rawdata.txt') and os.path.isfile(self.temp_dir+self.data+'__methylation_count.txt') \
            and os.path.isfile(self.temp_dir+self.data+'__methylation_max.txt') and os.path.isfile(self.temp_dir+self.data+'__methylation_min.txt') \
                and os.path.isfile(self.temp_dir+self.data+'__methylation_mean.txt')
        CNV_file_exists = os.path.isfile(self.temp_dir+self.data+'_CNV_masked_rawdata.txt') and os.path.isfile(self.temp_dir+self.data+'__CNV_count.txt') \
            and os.path.isfile(self.temp_dir+self.data+'__CNV_max.txt') and os.path.isfile(self.temp_dir+self.data+'__CNV_min.txt') \
                and os.path.isfile(self.temp_dir+self.data+'__CNV_mean.txt')
        file_exists = RNA_file_exists and DNA_file_exists and CNV_file_exists
        if self.skip and file_exists:
            print("skip data_gene_embedding_all")
            return
        # RNA
        if self.skip and RNA_file_exists:
            print("data_gene_embedding_all ==> RNA skip")
        else: 
            print('RNA')
            data_mRNA=pd.read_csv(self.raw_dir+self.data+'.mRNA.csv',sep=',')
            sample_mRNA_data=pd.read_csv(self.temp_dir+self.data+'_sample_mRNA_data_2.txt',sep='\t')
            data_mRNA=data_mRNA[['Unnamed: 0']+list(sample_mRNA_data['sample_old'])]
            data_mRNA.columns=['id']+list(sample_mRNA_data['sample_id'])
            data_mRNA['gene_id']=data_mRNA['id'].map(lambda x:x.split('.')[0])
            data_mRNA=data_mRNA[['gene_id']+list(sample_mRNA_data['sample_id'])].groupby('gene_id').sum()
            data_mRNA=data_mRNA.reset_index()
            data_mRNA=data_mRNA.drop_duplicates()
            data_mRNA.to_csv(self.temp_dir+self.data+'_mRNA_rawdata.txt',sep='\t',index=False)

            data_miRNA=pd.read_csv(self.raw_dir+self.data+'.miRNA.csv',sep=',')
            sample_miRNA_data=pd.read_csv(self.temp_dir+self.data+'_sample_miRNA_data_2.txt',sep='\t')
            data_miRNA=data_miRNA[['miRNA_ID']+list(sample_miRNA_data['sample_old'])]
            data_miRNA.columns=['miRNA_ID']+list(sample_miRNA_data['sample_id'])
            if len(set(sample_mRNA_data['sample_id'])-set(sample_miRNA_data['sample_id']))>0:
                data_miRNA[list(set(sample_mRNA_data['sample_id'])-set(sample_miRNA_data['sample_id']))]=np.nan

            id_data=pd.read_csv(self.raw_dir+'miRNA_id_new.txt',sep='\t')
            id_data.columns=['gene_id','miRNA_ID','name']
            id_data=id_data.drop_duplicates()
            data_miRNA_new=pd.merge(data_miRNA,id_data[['gene_id','miRNA_ID']],how='left',on='miRNA_ID')
            data_miRNA_new=data_miRNA_new.loc[pd.notnull(data_miRNA_new['gene_id']),:]
            data_miRNA_new=data_miRNA_new.drop_duplicates('gene_id')
            data_miRNA_new=data_miRNA_new[['gene_id']+list(sample_mRNA_data['sample_id'])]
            data_miRNA_new=data_miRNA_new.drop_duplicates()
            data_miRNA_new.to_csv(self.temp_dir+self.data+'_miRNA_rawdata.txt',sep='\t',index=False)

            data_all=pd.concat([data_mRNA,data_miRNA_new.loc[data_miRNA_new.gene_id.isin(list(set(data_miRNA_new['gene_id'])-set(data_mRNA['gene_id'])))]])
            data_all=data_all[['gene_id']+list(set(data_mRNA.columns[1:])&set(data_miRNA_new.columns[1:]))]
            data_all=data_all.drop_duplicates()
            data_all.to_csv(self.temp_dir+self.data+'_RNA_all_rawdata.txt',sep='\t',index=False)

            df = pd.read_csv(self.temp_dir+self.data+'_RNA_all_rawdata.txt',sep="\t")
            df=df.set_index('gene_id')
            sample=list(df.columns[1:])
            length_data=pd.read_csv(self.temp_dir+'gene_lengths.csv',sep=',',header=None)
            length_data.columns=['gene_id','length']
            print("Done .")
            print("Calculate TPM ...")
            gene = list(set(df.index)&set(length_data['gene_id']))
            length_data =length_data.set_index('gene_id')
            length=length_data.loc[gene,:]
            df=df.loc[gene,:]
            lengthScaledDf = pd.DataFrame((df.values/length.values.reshape((-1,1))),index=df.index,columns=df.columns)
            data_1 = (1000000*lengthScaledDf.div(lengthScaledDf.sum(axis=0))).round(4)
            data_1=data_1.reset_index()
            data_1.to_csv(self.temp_dir+self.data+'_RNA_all_TPM.txt',sep='\t',index=False)

        
        #DNA
        if self.skip and DNA_file_exists:
            print("data_gene_embedding_all ==> DNA skip")
        else: 
            print('DNA')
            sample_DNA_data=pd.read_csv(self.temp_dir+self.data+'_sample_DNA_data_2.txt',sep='\t')
            data_methylation=pd.read_csv(self.raw_dir+self.data+'.DNAmethy.csv',sep=',')
            data_methylation=data_methylation[['Unnamed: 0']+list(sample_DNA_data['sample_old'])]
            data_methylation.columns=['ID']+list(sample_DNA_data['sample_id'])
            data_methylation=data_methylation.drop_duplicates()
            data_methylation.to_csv(self.temp_dir+self.data+'_methylation_rawdata.txt',sep='\t',index=False)

            data_methylation_count=pd.read_csv(self.temp_dir+self.data+'_methylation_count.txt',sep='\t')
            data_methylation_count=data_methylation_count[['gene_id']+list(sample_DNA_data['sample_old'])]
            data_methylation_count.columns=['gene_id']+list(sample_DNA_data['sample_id'])
            data_methylation_count=data_methylation_count.drop_duplicates()
            data_methylation_count.to_csv(self.temp_dir+self.data+'__methylation_count.txt',sep='\t',index=False)

            data_methylation_max=pd.read_csv(self.temp_dir+self.data+'_methylation_max.txt',sep='\t')
            data_methylation_max=data_methylation_max[['gene_id']+list(sample_DNA_data['sample_old'])]
            data_methylation_max.columns=['gene_id']+list(sample_DNA_data['sample_id'])
            data_methylation_max=data_methylation_max.drop_duplicates()
            data_methylation_max.to_csv(self.temp_dir+self.data+'__methylation_max.txt',sep='\t',index=False)

            data_methylation_min=pd.read_csv(self.temp_dir+self.data+'_methylation_min.txt',sep='\t')
            data_methylation_min=data_methylation_min[['gene_id']+list(sample_DNA_data['sample_old'])]
            data_methylation_min.columns=['gene_id']+list(sample_DNA_data['sample_id'])
            data_methylation_min=data_methylation_min.drop_duplicates()
            data_methylation_min.to_csv(self.temp_dir+self.data+'__methylation_min.txt',sep='\t',index=False)

            data_methylation_mean=pd.read_csv(self.temp_dir+self.data+'_methylation_mean.txt',sep='\t')
            data_methylation_mean=data_methylation_mean[['gene_id']+list(sample_DNA_data['sample_old'])]
            data_methylation_mean.columns=['gene_id']+list(sample_DNA_data['sample_id'])
            data_methylation_mean=data_methylation_mean.drop_duplicates()
            data_methylation_mean.to_csv(self.temp_dir+self.data+'__methylation_mean.txt',sep='\t',index=False)

        #CNV
        if self.skip and CNV_file_exists:
            print("data_gene_embedding_all ==> CNV skip")
        else: 
            print('CNV')
            sample_CNV_data=pd.read_csv(self.temp_dir+self.data+'_sample_CNV_data_2.txt',sep='\t')
            data_CNV=pd.read_csv(self.temp_dir+self.data+'.CNV_masked_seg.all.csv',sep='\t')
            data_CNV=data_CNV[['ID']+list(sample_CNV_data['sample_old'])]
            data_CNV.columns=['ID']+list(sample_CNV_data['sample_id'])
            data_CNV=data_CNV.drop_duplicates()
            data_CNV.to_csv(self.temp_dir+self.data+'_CNV_masked_rawdata.txt',sep='\t',index=False)

            data_CNV_count=pd.read_csv(self.temp_dir+self.data+'_CNV_count.txt',sep='\t')
            data_CNV_count=data_CNV_count[['gene_id']+list(sample_CNV_data['sample_old'])]
            data_CNV_count.columns=['gene_id']+list(sample_CNV_data['sample_id'])
            data_CNV_count=data_CNV_count.drop_duplicates()
            data_CNV_count.to_csv(self.temp_dir+self.data+'__CNV_count.txt',sep='\t',index=False)

            data_CNV_max=pd.read_csv(self.temp_dir+self.data+'_CNV_max.txt',sep='\t')
            data_CNV_max=data_CNV_max[['gene_id']+list(sample_CNV_data['sample_old'])]
            data_CNV_max.columns=['gene_id']+list(sample_CNV_data['sample_id'])
            data_CNV_max=data_CNV_max.drop_duplicates()
            data_CNV_max.to_csv(self.temp_dir+self.data+'__CNV_max.txt',sep='\t',index=False)

            data_CNV_min=pd.read_csv(self.temp_dir+self.data+'_CNV_min.txt',sep='\t')
            data_CNV_min=data_CNV_min[['gene_id']+list(sample_CNV_data['sample_old'])]
            data_CNV_min.columns=['gene_id']+list(sample_CNV_data['sample_id'])
            data_CNV_min=data_CNV_min.drop_duplicates()
            data_CNV_min.to_csv(self.temp_dir+self.data+'__CNV_min.txt',sep='\t',index=False)

            data_CNV_mean=pd.read_csv(self.temp_dir+self.data+'_CNV_mean.txt',sep='\t')
            data_CNV_mean=data_CNV_mean[['gene_id']+list(sample_CNV_data['sample_old'])]
            data_CNV_mean.columns=['gene_id']+list(sample_CNV_data['sample_id'])
            data_CNV_mean=data_CNV_mean.drop_duplicates()
            data_CNV_mean.to_csv(self.temp_dir+self.data+'__CNV_mean.txt',sep='\t',index=False)
        
        
        return 

    
    def cnv_data_merge(self):
        file_exists = os.path.isfile(self.temp_dir+self.data+'_CNV_mean.txt') and os.path.isfile(self.temp_dir+self.data+'_CNV_max.txt') and os.path.isfile(self.temp_dir+self.data+'_CNV_min.txt')
        if self.skip and file_exists:
            print("skip cnv_data_merge")
            return
        data_up_max=pd.read_csv(self.temp_dir+self.data+'_data_up_max.txt',sep='\t')
        data_up_min=pd.read_csv(self.temp_dir+self.data+'_data_up_min.txt',sep='\t')
        data_down_max=pd.read_csv(self.temp_dir+self.data+'_data_down_max.txt',sep='\t')
        data_down_min=pd.read_csv(self.temp_dir+self.data+'_data_down_min.txt',sep='\t')

        data_up_max=data_up_max.set_index('gene_id')
        data_up_max=data_up_max.fillna(0)
        data_up_min=data_up_min.set_index('gene_id')
        data_up_min=data_up_min.fillna(0)
        data_down_max=data_down_max.set_index('gene_id')
        data_down_max=data_down_max.fillna(0)
        data_down_min=data_down_min.set_index('gene_id')
        data_down_min=data_down_min.fillna(0)

        data_up_max[data_down_min.abs()>data_up_max.abs()]=data_down_min
        data_up_min[data_down_max.abs()<data_up_min.abs()]=data_down_max

        data_up_max.to_csv(self.temp_dir+self.data+'_CNV_max.txt',sep='\t')
        data_up_min.to_csv(self.temp_dir+self.data+'_CNV_min.txt',sep='\t')


        data_up_sum=pd.read_csv(self.temp_dir+self.data+'_data_up_sum.txt',sep='\t')
        data_up_mean=pd.read_csv(self.temp_dir+self.data+'_data_up_mean.txt',sep='\t')
        data_down_sum=pd.read_csv(self.temp_dir+self.data+'_data_down_sum.txt',sep='\t')
        data_down_mean=pd.read_csv(self.temp_dir+self.data+'_data_down_mean.txt',sep='\t')

        data_up_sum=data_up_sum.set_index('gene_id')
        data_up_sum=data_up_sum.fillna(0)
        data_up_mean=data_up_mean.set_index('gene_id')
        data_up_mean=data_up_mean.fillna(0)
        data_down_sum=data_down_sum.set_index('gene_id')
        data_down_sum=data_down_sum.fillna(0)
        data_down_mean=data_down_mean.set_index('gene_id')
        data_down_mean=data_down_mean.fillna(0)

        data_up_mean[data_down_sum.abs()>data_up_sum.abs()]=data_down_mean
        data_up_mean.to_csv(self.temp_dir+self.data+'_CNV_mean.txt',sep='\t')
    
        return
    
    
    def cnv_data(self):
        file_exists = os.path.isfile(self.temp_dir+self.data+'.CNV_masked_seg.all.csv')\
            and os.path.isfile(self.temp_dir+self.data+'_CNV_count.txt')\
                and os.path.isfile(self.temp_dir+self.data+'_data_up_max.txt')\
                    and os.path.isfile(self.temp_dir+self.data+'_data_up_min.txt')\
                        and os.path.isfile(self.temp_dir+self.data+'_data_up_sum.txt')\
                            and os.path.isfile(self.temp_dir+self.data+'_data_up_mean.txt')\
                                and os.path.isfile(self.temp_dir+self.data+'_data_down_max.txt')\
                                    and os.path.isfile(self.temp_dir+self.data+'_data_down_min.txt')\
                                        and os.path.isfile(self.temp_dir+self.data+'_data_down_sum.txt')\
                                            and os.path.isfile(self.temp_dir+self.data+'_data_down_mean.txt')
        if self.skip and file_exists:
            print("skip cnv_data")
            return
        if self.skip and os.path.isfile(self.temp_dir+self.data+'.CNV_masked_seg.all.csv'):
            print("skip cnv_data ==> CNV_masked_seg.all.csv")
            data = pd.read_csv(self.temp_dir+self.data+'.CNV_masked_seg.all.csv',sep='\t')
        else: 
            
            data_CNV=pd.read_csv(self.raw_dir+self.data+'.CNV_masked_seg.csv',sep=',')
            data_CNV['ID']='chr'+data_CNV['Chromosome'].astype(str)+'_'+data_CNV['Start'].astype(str)+'_'+data_CNV['End'].astype(str)
            sample=list(set(data_CNV['Sample']))

            data=pd.DataFrame(columns=['ID'])
            j=0
            for i in tqdm(sample):
                data_CNV_=data_CNV.loc[data_CNV['Sample']==i,['ID','Segment_Mean']].rename(columns={'Segment_Mean':i})
                data=pd.merge(data,data_CNV_,on='ID',how='outer')
                j=j+1
            data.to_csv(self.temp_dir+self.data+'.CNV_masked_seg.all.csv',sep='\t',index=False)
            del data_CNV
            del data_CNV_
        
        ID_gene=pd.read_csv(self.temp_dir+self.data+'_CNV_id.txt',sep='\t')
        data=data.loc[data.ID.isin(list(ID_gene['ID']))]
        

        gene_list = sorted(list(set(np.array(ID_gene).T[1].tolist())))
        data.set_index("ID", inplace = True)
        for iter_index, gene in tqdm(enumerate(gene_list)):
            
            temp_gene = ID_gene[ID_gene['gene_id'] == gene]
            temp_gene.set_index("ID", inplace=True)
            data2 = temp_gene.join(data)           

            sample=list(data2.columns[1:])
            data2=data2.fillna(0)
            
            #count            
            data2_=data2[sample].copy()
            data2_[data2_>0]=1
            data2_[data2_<0]=1
            data2_['gene_id']=data2['gene_id']
            if iter_index == 0:
                data_count=data2_[['gene_id']+sample].groupby('gene_id').sum()
            else: 
                new_count = data2_[['gene_id']+sample].groupby('gene_id').sum()
                data_count = pd.concat([data_count, new_count], axis = 0)
            del data2_


            data_up=data2[sample].copy()
            data_up[data_up<=0]=np.nan
            data_up['gene_id']=data2['gene_id']
            data_down=data2[sample].copy()
            data_down[data_down>=0]=np.nan
            data_down['gene_id']=data2['gene_id']
            
            if iter_index == 0 :
                data_up_max=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.max(skipna=True))
                data_up_min=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
                data_up_sum=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.sum(skipna=True))
                data_up_mean=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
                data_down_max=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.max(skipna=True))
                data_down_min=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
                data_down_sum=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.sum(skipna=True))
                data_down_mean=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
            else: 
                new_up_max=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.max(skipna=True))
                new_up_min=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
                new_up_sum=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.sum(skipna=True))
                new_up_mean=data_up[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
                new_down_max=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.max(skipna=True))
                new_down_min=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
                new_down_sum=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.sum(skipna=True))
                new_down_mean=data_down[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
                data_up_max = pd.concat([data_up_max, new_up_max], axis = 0)
                data_up_min = pd.concat([data_up_min, new_up_min], axis = 0)
                data_up_sum = pd.concat([data_up_sum, new_up_sum], axis = 0)
                data_up_mean = pd.concat([data_up_mean, new_up_mean], axis = 0)
                data_down_max = pd.concat([data_down_max, new_down_max], axis = 0)
                data_down_min = pd.concat([data_down_min, new_down_min], axis = 0)
                data_down_sum = pd.concat([data_down_sum, new_down_sum], axis = 0)
                data_down_mean = pd.concat([data_down_mean, new_down_mean], axis = 0)
            del data_up
            del data_down
            
        data_up_sum=data_up_sum[sample]   
        data_down_sum=data_down_sum[sample] 
        data_count=data_count.fillna(0).reset_index()
        data_count.to_csv(self.temp_dir+self.data+'_CNV_count.txt',sep='\t',index=False)
        del data_count
        data_up_max.to_csv(self.temp_dir+self.data+'_data_up_max.txt',sep='\t',index=False)
        del data_up_max
        data_up_min.to_csv(self.temp_dir+self.data+'_data_up_min.txt',sep='\t',index=False)
        del data_up_min
        data_up_sum.to_csv(self.temp_dir+self.data+'_data_up_sum.txt',sep='\t',index=True)
        del data_up_sum
        data_up_mean.to_csv(self.temp_dir+self.data+'_data_up_mean.txt',sep='\t',index=True)
        del data_up_mean
        data_down_max.to_csv(self.temp_dir+self.data+'_data_down_max.txt',sep='\t',index=False)
        del data_down_max
        data_down_min.to_csv(self.temp_dir+self.data+'_data_down_min.txt',sep='\t',index=False)
        del data_down_min
        data_down_sum.to_csv(self.temp_dir+self.data+'_data_down_sum.txt',sep='\t',index=True)
        del data_down_sum
        data_down_mean.to_csv(self.temp_dir+self.data+'_data_down_mean.txt',sep='\t',index=True)
        del data_down_mean
        del data
    
        return
    
    
    def cnv_id(self):
        if self.skip and os.path.isfile(self.temp_dir+self.data+'_CNV_id.txt'):
            print("skip cnv_id")
            return
        data_CNV = pd.read_csv(self.raw_dir+self.data+'.CNV_masked_seg.csv', sep=',')
        data_CNV['ID'] = 'chr' + data_CNV['Chromosome'].astype(str) + '_' + data_CNV['Start'].astype(str) + '_' + data_CNV['End'].astype(str)
        ID = data_CNV[['ID']].drop_duplicates()

        gtf_data = pd.read_csv(self.raw_dir+'Homo_sapiens.GRCh38.91.chr.gtf', sep='\t',skiprows=lambda x: x in [0, 1, 2, 3, 4], header=None)
        gtf_data = gtf_data.loc[gtf_data.iloc[:, 2] == 'gene', :]
        gtf_data_new = pd.DataFrame(columns=['gene_id', 'gene_name', 'chr', 'strat', 'end', 'strand'])
        gtf_data_new['gene_id'] = gtf_data.iloc[:, 8].apply(lambda x: re.findall('gene_id ".*?"', x)[0].split('"')[1])
        gtf_data_new['gene_name'] = gtf_data.iloc[:, 8].apply(lambda x: re.findall('gene_name ".*?"', x)[0].split('"')[1] if 'gene_name' in x else np.nan)
        gtf_data_new['chr'] = gtf_data.iloc[:, 0].astype('str')
        gtf_data_new['strat'] = gtf_data.iloc[:, 3].astype('int')
        gtf_data_new['end'] = gtf_data.iloc[:, 4].astype('int')
        gtf_data_new['strand'] = gtf_data.iloc[:, 6]
        gtf_data_new = gtf_data_new.drop_duplicates()
        gtf_data_new.index = range(len(gtf_data_new))


        def get_id(x, gtf_data_new):
            # print(x)
            chr = x.split('_')[0].split('chr')[1]
            site_strat = int(x.split('_')[1])
            site_end = int(float(x.split('_')[2]))

            gtf = gtf_data_new.loc[
                ((gtf_data_new['chr'] == chr) & (gtf_data_new['strat'] <= site_strat) & (gtf_data_new['end'] >= site_strat)) |
                ((gtf_data_new['chr'] == chr) & (gtf_data_new['strat'] <= site_end) & (gtf_data_new['end'] >= site_end)) |
                ((gtf_data_new['chr'] == chr) & (gtf_data_new['strat'] >= site_strat) & (gtf_data_new['end'] <= site_end)), :]
            gtf = gtf.drop_duplicates('gene_id')
            if len(gtf) == 0:
                gene_id = 'NA'
                # print('erro')
            else:
                gene_id = ';'.join(list(gtf['gene_id']))
            # id=x+'_'+gene_id+'_'+gene_name
            return gene_id


        # data['ID_new']=data['ID'].map(lambda x: get_id(x, gtf_data_new))
        for j in [i * 10000 for i in range(0, int(len(ID) / 10000) + 1)]:
            print(j)
            ID.loc[j:(j + 10000), 'ID_new'] = ID.loc[j:(j + 10000), 'ID'].map(lambda x: get_id(x, gtf_data_new))
        ID.loc[int(len(ID) / 10000) * 10000:, 'ID_new'] = ID.loc[int(len(ID) / 10000) * 10000:, 'ID'].map(
            lambda x: get_id(x, gtf_data_new))
        ID['gene_id'] = ID['ID_new'].str.split(';')
        data_ID_new = ID.explode('gene_id')
        data_ID_new = data_ID_new[['ID', 'gene_id']]
        data_ID_new = data_ID_new.drop_duplicates()
        gene_list = open(self.proc_dir+"index_ensembl_map.txt","r").read().split('\n')
        data_ID_new = data_ID_new[data_ID_new['gene_id'].isin(gene_list)]
        data_ID_new.to_csv(self.temp_dir+self.data+'_CNV_id.txt', sep='\t', index=False)
        
        return
    
    
        
    def dna_methylation(self):
        file_exists = os.path.isfile(self.temp_dir+self.data+'_methylation_max.txt') and os.path.isfile(self.temp_dir+self.data+'_methylation_min.txt') \
            and os.path.isfile(self.temp_dir+self.data+'_methylation_mean.txt') and os.path.isfile(self.temp_dir+self.data+'_methylation_count.txt')
        if self.skip and file_exists:
            print("skip dna_methylation")
            return
        data=pd.read_csv(self.raw_dir+self.data+'.DNAmethy.csv',sep=',')
        sample=data.columns[1:]
        data.columns=['ID']+list(sample)

        ID_gene=pd.read_csv(self.temp_dir+self.data+'_DNA_methylation_geneid.txt',sep='\t')
        data=data.loc[data.ID.isin(list(ID_gene['ID']))]
        data=pd.merge(ID_gene[['ID','gene_id']],data,on='ID',how='left')

        data=data.loc[pd.notnull(data['gene_id'])]
        data[data==0]=np.nan
        sample=list(data.columns[2:])
        ####
        for i in range(len(sample)):
            data[sample[i]] = pd.to_numeric(data[sample[i]], errors='coerce')
        ####

        if not (self.skip and os.path.isfile(self.temp_dir+self.data+'_methylation_max.txt')):
            #max
            data_max=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.max(skipna=True))
            data_max=data_max.fillna(0)
            # data_max=data_max.reset_index()
            data_max.to_csv(self.temp_dir+self.data+'_methylation_max.txt',sep='\t',index=False)
            del data_max
        if not (self.skip and os.path.isfile(self.temp_dir+self.data+'_methylation_min.txt')):
            #min
            data_min=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.min(skipna=True))
            data_min=data_min.fillna(0)
            # data_min=data_min.reset_index()
            data_min.to_csv(self.temp_dir+self.data+'_methylation_min.txt',sep='\t',index=False)
            del data_min
        if not (self.skip and os.path.isfile(self.temp_dir+self.data+'_methylation_mean.txt')):
            #mean
            data_mean=data[['gene_id']+sample].groupby('gene_id').apply(lambda g: g.mean(skipna=True))
            data_mean=data_mean.fillna(0)
            data_mean=data_mean.reset_index()
            data_mean.to_csv(self.temp_dir+self.data+'_methylation_mean.txt',sep='\t',index=False)
            del data_mean
        #count
        data=data.fillna(0)
        data_=data[sample].copy()
        data_[data_>0]=1
        data_['gene_id']=data['gene_id']
        data_count=data_[['gene_id']+sample].groupby('gene_id').sum()
        data_count=data_count.fillna(0)
        data_count=data_count.reset_index()
        data_count.to_csv(self.temp_dir+self.data+'_methylation_count.txt',sep='\t',index=False)
        
        return
           
    def dna_methylation_promoterid(self):
        #def get_data(reference_path,rawdata_path,save_path,cancer):
        file_exists = os.path.isfile(self.temp_dir+self.data+'_DNA_methylation_geneid.txt')
        if (self.skip) and file_exists:
            print("skip dna_methylation_promoterid")
            return
        data_ID=pd.read_csv(self.raw_dir+self.data+'.DNAmethy.csv',sep=',',usecols=[0])
        data_ID.columns=['ID']

        CPG_id=pd.read_csv(self.raw_dir+'HM450.hg38.manifest.gencode.v36.tsv',sep='\t')
        CPG_id=CPG_id.rename(columns={'probeID':'ID'})
        data_ID=pd.merge(data_ID,CPG_id[['ID','CpG_beg','CpG_end','CpG_chrm','transcriptIDs','probe_strand']],on='ID',how='left')
        data_ID=data_ID.loc[pd.notnull(data_ID['transcriptIDs'])]
        data_ID['transcript_id']=data_ID['transcriptIDs'].str.split(';')
        data_ID_new=data_ID.explode('transcript_id')
        data_ID_new['transcript_id']=data_ID_new['transcript_id'].map(lambda x:x.split('.')[0])

        gtf_data=pd.read_csv(self.raw_dir+'Homo_sapiens.GRCh38.91.chr.gtf',sep='\t',skiprows = lambda x: x in [0,1,2,3,4],header=None)
        gtf_data=gtf_data.loc[gtf_data.iloc[:,2]=='transcript',:]
        gtf_data_new = pd.DataFrame(columns=['gene_id','transcript_id','chr','strat','end','strand'])
        gtf_data_new['transcript_id'] = gtf_data.iloc[:,8].apply(lambda x:re.findall('transcript_id ".*?"',x)[0].split('"')[1])
        gtf_data_new['gene_id'] = gtf_data.iloc[:,8].apply(lambda x:re.findall('gene_id ".*?"',x)[0].split('"')[1])
        gtf_data_new['gene_type'] = gtf_data.iloc[:,8].apply(lambda x:re.findall('gene_biotype ".*?"',x)[0].split('"')[1] if 'gene_biotype' in x else np.nan)
        gtf_data_new['chr'] = gtf_data.iloc[:,0]
        gtf_data_new['strat']=gtf_data.iloc[:,3].astype('int')
        gtf_data_new['end']=gtf_data.iloc[:,4].astype('int')
        gtf_data_new['strand']=gtf_data.iloc[:,6]
        gtf_data_new = gtf_data_new.drop_duplicates()
        gtf_data_new.index = range(len(gtf_data_new))

        data_ID_new=pd.merge(data_ID_new,gtf_data_new[['gene_id','transcript_id']],on='transcript_id',how='left')
        data_ID_new=data_ID_new.loc[pd.notnull(data_ID_new['gene_id'])]
        data_ID_new=data_ID_new.drop_duplicates(['ID','gene_id'])
        gene_list = open(self.proc_dir+"index_ensembl_map.txt","r").read().split('\n')
        data_ID_new = data_ID_new[data_ID_new['gene_id'].isin(gene_list)]
        data_ID_new.to_csv(self.temp_dir+self.data+'_DNA_methylation_geneid.txt',sep='\t',index=False)
        
        
 
    def get_gene_length_R(self):
        if self.skip and os.path.isfile(self.raw_dir+'gene_ensembl_lengths.csv') and os.path.isfile(self.raw_dir+'gene_hgnc_lengths.csv'):
            print('skip getting gene length')
            return 
        else:
            subprocess.call(['Rscript', self.code_dir+'gene_length.R'])
        return 
    
    
    def get_gene_length(self):
        if self.skip and os.path.isfile(self.temp_dir+'gene_lengths.csv'):
            print('skip getting gene length')
            return 
        else:
            hgnc_list = open(self.proc_dir+"index_gene_map.txt","r").read().split('\n')
            ensembl_list = open(self.proc_dir+"index_ensembl_map.txt","r").read().split('\n')
            index_list = np.concatenate((np.array(hgnc_list).reshape(-1,1), np.array(ensembl_list).reshape(-1,1)), axis=1).tolist()
            hgnc_length = np.array([a.split(',') for a in open(self.raw_dir+"gene_hgnc_lengths.csv","r").read().split('\n')[1:-1]]).T
            ensembl_length = np.array([a.split(',') for a in open(self.raw_dir+"gene_ensembl_lengths.csv","r").read().split('\n')[1:-1]]).T
            for i,indices in tqdm(enumerate(index_list)):
                index_list[i] += hgnc_length[1][np.where(hgnc_length[0] == indices[0])].astype(int).tolist()
                index_list[i] += ensembl_length[1][np.where(ensembl_length[0] == indices[1])].astype(int).tolist()
            gene_length_mat = [a[1]+','+str(min(a[2:])) for a in index_list]
            gene_length_string = '\n'.join(gene_length_mat)+'\n'
            
            file = open(self.temp_dir+'gene_lengths.csv','w')
            file.write(gene_length_string)
            file.close()
            
        return
    
    
    
    def get_label(self):
        #if self.skip and os.path.isfile(self.temp_dir+self.data+'_sample_id_label_subtype.txt'):
        #    print('skip getting labels')
        #    return
        data_cancer=pd.DataFrame(columns=['sample_id','subtype','sampleID'])
        if self.data in ['BRCA', 'STAD']:
            data_2=pd.read_csv(self.raw_dir+'labels.csv',sep=',')

            data_2_cancer=data_2.loc[data_2['cancer.type']==self.data,:]
            data_cancer['sample_id']=data_2_cancer['pan.samplesID'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
            if self.data == "BRCA":
                data_cancer['sampleID']=data_2_cancer['pan.samplesID'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2]+'-'+x.split('-')[3][:2])
            if self.data == "STAD":
                data_cancer['sampleID']=data_2_cancer['pan.samplesID'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
            data_cancer['subtype']=data_2_cancer['Subtype_Selected'].map(lambda x:x.split('.')[1])
            sample_mRNA_data = pd.read_csv(self.temp_dir+self.data+'_sample_mRNA_data_2.txt',sep='\t')
            sample_DNA_data = pd.read_csv(self.temp_dir+self.data+'_sample_DNA_data_2.txt',sep='\t')
            sample_CNV_data = pd.read_csv(self.temp_dir+self.data+'_sample_CNV_data_2.txt',sep='\t')
            if self.data == "BRCA":
                sample_mRNA_data['sampleID'] = sample_mRNA_data['sample_old'].map(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2] + '-' + x.split('-')[3][:2])
                sample_DNA_data['sampleID'] = sample_DNA_data['sample_old'].map(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2] + '-' + x.split('-')[3][:2])
                sample_CNV_data['sampleID'] = sample_CNV_data['sample_old'].map(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2] + '-' + x.split('-')[3][:2])
            if self.data == "STAD":
                sample_mRNA_data['sampleID'] = sample_mRNA_data['sample_old'].map(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2])
                sample_DNA_data['sampleID'] = sample_DNA_data['sample_old'].map(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2])
                sample_CNV_data['sampleID'] = sample_CNV_data['sample_old'].map(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2])
            sample_all = set(sample_mRNA_data['sampleID']) & set(sample_DNA_data['sampleID']) & set(sample_CNV_data['sampleID'])
            data_cancer=data_cancer.loc[data_cancer.sampleID.isin(sample_all)]
            print('all',len(sample_all))
            print(self.data, len(data_cancer))
            data_cancer.to_csv(self.temp_dir+self.data+'_sample_id_label_subtype.txt',sep='\t', index=False)  
               
        elif self.data in ['LGG', 'HNSC', 'CESC', 'SARC']:       
            data_2 = open(self.raw_dir+"{}_labels.txt".format(self.data),"r").read().split('\n')[:-1]  
            patient = [i.upper() for i in data_2[0].split('\t')[1:]]
            
            if self.data == 'LGG':
                c_type = data_2[65].split('\t')[1:] 
                anomaly_index = c_type.index('NA')
                patient = patient[:anomaly_index] + patient[anomaly_index+1:]
                c_type = c_type[:anomaly_index] + c_type[anomaly_index+1:]
                
            elif self.data == 'HNSC':
                c_type = data_2[54].split('\t')[1:] 
                anomaly_indices = []
                for i in range(len(c_type)):
                    if c_type[i] == 'indeterminate':
                        anomaly_indices.append(i)
                anomaly_indices = sorted(anomaly_indices, reverse=True)
                for anomaly_index in anomaly_indices:
                    patient = patient[:anomaly_index] + patient[anomaly_index+1:]
                    c_type = c_type[:anomaly_index] + c_type[anomaly_index+1:]
                
            elif self.data == 'CESC':              
                c_type = data_2[95].split('\t')[1:] 
                remove_count = c_type.count('adenosquamous')
                for i in range(remove_count):
                    r_index = c_type.index('adenosquamous')
                    patient = patient[:r_index] + patient[r_index+1:]
                    c_type = c_type[:r_index] + c_type[r_index+1:]
                for i in range(len(c_type)):
                    if c_type[i] == 'cervical squamous cell carcinoma':
                        c_type[i] = 'SquamousCarcinoma'
                    else: 
                        c_type[i] = 'AdenoCarcinoma'
                
            elif self.data == 'SARC':              
                c_type = data_2[31].split('\t')[1:] 
                for i in range(len(c_type)):
                    if c_type[i] in ['dedifferentiated liposarcoma']:
                        c_type[i] = '(DDLPS) dedifferentiated liposarcoma'
                    elif c_type[i] in ['undifferentiated pleomorphic sarcoma (ups)', "myxofibrosarcoma", "pleomorphic 'mfh' / undifferentiated pleomorphic sarcoma", "giant cell 'mfh' / undifferentiated pleomorphic sarcoma with giant cells" ]:
                        c_type[i] = '(MPS or UPS) myxofibrosarcoma, undifferentiated pleomorphic sarcoma'
                    elif c_type[i] in ['leiomyosarcoma (lms)']:
                        c_type[i] = '(LMS) leiomyosarcoma'
                    else: 
                        c_type[i] = 'others'
            data_cancer['sample_id']=patient
            data_cancer['sampleID']=patient
            data_cancer['subtype']=c_type
            sample_mRNA_data = pd.read_csv(self.temp_dir+self.data+'_sample_mRNA_data_2.txt',sep='\t')
            sample_DNA_data = pd.read_csv(self.temp_dir+self.data+'_sample_DNA_data_2.txt',sep='\t')
            sample_CNV_data = pd.read_csv(self.temp_dir+self.data+'_sample_CNV_data_2.txt',sep='\t')
            sample_mRNA_data['sampleID'] = sample_mRNA_data['sample_old'].map(lambda x: '-'.join(x.split('-')[:3]))
            sample_DNA_data['sampleID'] = sample_DNA_data['sample_old'].map(lambda x: '-'.join(x.split('-')[:3]))
            sample_CNV_data['sampleID'] = sample_CNV_data['sample_old'].map(lambda x: '-'.join(x.split('-')[:3]))
            sample_all = set(sample_mRNA_data['sampleID']) & set(sample_DNA_data['sampleID']) & set(sample_CNV_data['sampleID'])
            data_cancer=data_cancer.loc[data_cancer.sampleID.isin(sample_all)]
            print('all',len(sample_all))
            print(self.data, len(data_cancer))
            for a in set(c_type):
                print(c_type.count(a)) 
            data_cancer.to_csv(self.temp_dir+self.data+'_sample_id_label_subtype.txt',sep='\t', index=False)  
            
            

        
        elif self.data == 'KIPAN':
            patient = []
            c_type = []
            for data_name in ['KICH', 'KIRC', 'KIRP']:
                data_2 = open(self.raw_dir+"{}_labels.txt".format(data_name),"r").read().split('\n')[:-1]  
                cur_patient = [i.upper() for i in data_2[0].split('\t')[1:]]
                patient += cur_patient
                c_type += [data_name]*len(cur_patient)
            
            
            
            data_cancer['sample_id']=patient
            data_cancer['sampleID']=patient
            data_cancer['subtype']=c_type
            sample_mRNA_data = pd.read_csv(self.temp_dir+self.data+'_sample_mRNA_data_2.txt',sep='\t')
            sample_DNA_data = pd.read_csv(self.temp_dir+self.data+'_sample_DNA_data_2.txt',sep='\t')
            sample_CNV_data = pd.read_csv(self.temp_dir+self.data+'_sample_CNV_data_2.txt',sep='\t')
            sample_mRNA_data['sampleID'] = sample_mRNA_data['sample_old'].map(lambda x: '-'.join(x.split('-')[:3]))
            sample_DNA_data['sampleID'] = sample_DNA_data['sample_old'].map(lambda x: '-'.join(x.split('-')[:3]))
            sample_CNV_data['sampleID'] = sample_CNV_data['sample_old'].map(lambda x: '-'.join(x.split('-')[:3]))
            sample_all = set(sample_mRNA_data['sampleID']) & set(sample_DNA_data['sampleID']) & set(sample_CNV_data['sampleID'])
            data_cancer=data_cancer.loc[data_cancer.sampleID.isin(sample_all)]
            print('all',len(sample_all))
            print(self.data, len(data_cancer))
            for a in set(c_type):
                print(c_type.count(a)) 
            data_cancer.to_csv(self.temp_dir+self.data+'_sample_id_label_subtype.txt',sep='\t', index=False)  
        
        elif self.data == 'NSCLC':
            patient = []
            c_type = []
            for data_name in ['LUAD', 'LUSC']:
                data_2 = open(self.raw_dir+"{}_labels.txt".format(data_name),"r").read().split('\n')[:-1]  
                cur_patient = [i.upper() for i in data_2[0].split('\t')[1:]]
                patient += cur_patient
                c_type += [data_name]*len(cur_patient)
            
            
            
            data_cancer['sample_id']=patient
            data_cancer['sampleID']=patient
            data_cancer['subtype']=c_type
            sample_mRNA_data = pd.read_csv(self.temp_dir+self.data+'_sample_mRNA_data_2.txt',sep='\t')
            sample_DNA_data = pd.read_csv(self.temp_dir+self.data+'_sample_DNA_data_2.txt',sep='\t')
            sample_CNV_data = pd.read_csv(self.temp_dir+self.data+'_sample_CNV_data_2.txt',sep='\t')
            sample_mRNA_data['sampleID'] = sample_mRNA_data['sample_old'].map(lambda x: '-'.join(x.split('-')[:3]))
            sample_DNA_data['sampleID'] = sample_DNA_data['sample_old'].map(lambda x: '-'.join(x.split('-')[:3]))
            sample_CNV_data['sampleID'] = sample_CNV_data['sample_old'].map(lambda x: '-'.join(x.split('-')[:3]))
            sample_all = set(sample_mRNA_data['sampleID']) & set(sample_DNA_data['sampleID']) & set(sample_CNV_data['sampleID'])
            data_cancer=data_cancer.loc[data_cancer.sampleID.isin(sample_all)]
            print('all',len(sample_all))
            print(self.data, len(data_cancer))
            for a in set(c_type):
                print(c_type.count(a)) 
            data_cancer.to_csv(self.temp_dir+self.data+'_sample_id_label_subtype.txt',sep='\t', index=False)  
        
        else :
            raise NotImplementedError           
            
        return 
    
      
        
    def data_sample_filter_1(self,smaple,type='RNA'):
        smaple=pd.DataFrame(smaple)
        smaple.columns=['ID']
        smaple['sample_id']=smaple['ID'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
        smaple_over=list(set(smaple['sample_id']))
        smaple_1=pd.DataFrame(columns=['ID','sample_id'])
        smaple_1['sample_id']=smaple_over
        for s in smaple_over:
            id_data=smaple.loc[smaple['sample_id']==s,['ID']]
            id_data['Sample']=id_data['ID'].map(lambda x:x.split('-')[3][:2])
            id_data['Vial'] = id_data['ID'].map(lambda x: x.split('-')[3][2:])
            id_data['Portion'] = id_data['ID'].map(lambda x: x.split('-')[4][:2])
            id_data['Analyte'] = id_data['ID'].map(lambda x: x.split('-')[4][2:])
            id_data['Plate'] = id_data['ID'].map(lambda x: x.split('-')[5])
            id_data['Center'] = id_data['ID'].map(lambda x: x.split('-')[6])
            id_data=id_data.loc[id_data.Sample.isin(['01','02','03','04','05','06','07','08','09'])]
            id_data = id_data.loc[id_data['Vial']!='B']
            if type=='RNA':
                id_data = id_data.loc[id_data.Analyte.isin(['H','R','T'])]
            else:
                id_data = id_data.loc[id_data.Analyte.isin(['D','G','W','X'])]
            if len(id_data)==0:
                smaple_1.loc[smaple_1['sample_id']==s,'ID'] = np.nan
                continue
            elif len(id_data)==1:
                smaple_1.loc[smaple_1['sample_id']==s,'ID']=list(id_data['ID'])
                continue
            else:
                if type=='RNA':
                    Analyte_list = list(set(id_data['Analyte']))
                    Analyte_list.sort()
                    id_data= id_data.loc[id_data['Analyte']==Analyte_list[0]]
                else:
                    if 'D' in list(set(id_data['Analyte'])):
                        id_data = id_data.loc[id_data['Analyte'] == 'D']
                if len(id_data)==1:
                    smaple_1.loc[smaple_1['sample_id']==s,'ID']=list(id_data['ID'])
                    continue
                else:
                    Vial_list = list(set(id_data['Vial']))
                    Vial_list.sort()
                    id_data= id_data.loc[id_data['Vial']==Vial_list[0]]
                    if len(id_data) == 1:
                        smaple_1.loc[smaple_1['sample_id'] == s, 'ID'] = list(id_data['ID'])
                        continue
                    else:
                        Sample_list = list(set(id_data['Sample']))
                        Sample_list.sort()
                        id_data = id_data.loc[id_data['Sample'] == Sample_list[0]]
                        if len(id_data) == 1:
                            smaple_1.loc[smaple_1['sample_id'] == s, 'ID'] = list(id_data['ID'])
                            continue
                        else:
                            Portion_list=list(set(id_data['Portion'].astype('int')))
                            Portion_list.sort()
                            Portion=str(Portion_list[-1])
                            if len(Portion)==1:Portion='0'+Portion
                            id_data = id_data.loc[id_data['Portion'] ==Portion]
                            if len(id_data) == 1:
                                smaple_1.loc[smaple_1['sample_id'] == s, 'ID'] = list(id_data['ID'])
                                continue
                            else:
                                Plate_list = list(set(id_data['Plate']))
                                Plate_list.sort()
                                id_data = id_data.loc[id_data['Plate'] == str(Plate_list[-1])]
                                if len(id_data) == 1:
                                    smaple_1.loc[smaple_1['sample_id'] == s, 'ID'] = list(id_data['ID'])
                                    continue
                                else:
                                    Center_list = list(set(id_data['Center'].astype('int')))
                                    Center_list.sort()
                                    Center = str(Center_list[0])
                                    if len(Center) == 1: Center = '0' + Center
                                    id_data = id_data.loc[id_data['Center'] == Center]
                                    smaple_1.loc[smaple_1['sample_id'] == s, 'ID'] = list(id_data['ID'])

        smaple_1=smaple_1.loc[pd.notnull(smaple_1['ID'])]
        return smaple_1
    
    def data_filter(self):
        already_exist = os.path.isfile(self.temp_dir+self.data+'_sample_mRNA_data_2.txt') \
            and os.path.isfile(self.temp_dir+self.data+'_sample_miRNA_data_2.txt') \
                and os.path.isfile(self.temp_dir+self.data+'_sample_DNA_data_2.txt') \
                    and os.path.isfile(self.temp_dir+self.data+'_sample_CNV_data_2.txt')
        if (not self.skip) or ( not already_exist):
            print('miRNA')
            data_miRNA=pd.read_csv(self.raw_dir+self.data+'.miRNA.csv',sep=',',nrows=1)
            sample_miRNA=data_miRNA.columns[2:]
            sample_miRNA_data=pd.DataFrame(columns=['sample_old','sample_id'])
            sample_miRNA_data['sample_old']=sample_miRNA
            sample_miRNA_data['type']=sample_miRNA_data['sample_old'].map(lambda x:x.split('_TCGA')[0])
            sample_miRNA_data=sample_miRNA_data.loc[sample_miRNA_data['type']=='read_count',:]
            sample_miRNA_data['sample_new']=sample_miRNA_data['sample_old'].map(lambda x:x.split('_')[-1])
            sample_miRNA_data['sample_id']=sample_miRNA_data['sample_old'].map(lambda x:x.split('_')[-1].split('-')[0]+'-'+x.split('_')[-1].split('-')[1]+'-'+x.split('_')[-1].split('-')[2])
            sample_miRNA_filter_data=self.data_sample_filter_1(list(sample_miRNA_data['sample_new']))
            sample_miRNA_data=sample_miRNA_data.loc[sample_miRNA_data.sample_new.isin(list(sample_miRNA_filter_data['ID'])),:]
            
            print('mRNA')
            data_mRNA=pd.read_csv(self.raw_dir+self.data+'.mRNA.csv',sep=',',nrows=1)
            sample_mRNA=data_mRNA.columns[1:]
            sample_mRNA_data=pd.DataFrame(columns=['sample_old','sample_id'])
            sample_mRNA_data['sample_old']=sample_mRNA
            sample_mRNA_data['sample_id']=sample_mRNA_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
            sample_mRNA_filter_data=self.data_sample_filter_1(list(sample_mRNA_data['sample_old']))
            sample_mRNA_data=sample_mRNA_data.loc[sample_mRNA_data.sample_old.isin(list(sample_mRNA_filter_data['ID'])),:]

            print('DNAmethy')
            data_DNA=pd.read_csv(self.raw_dir+self.data+'.DNAmethy.csv',sep=',',nrows=1)
            sample_DNA=data_DNA.columns[1:]
            sample_DNA_data=pd.DataFrame(columns=['sample_old','sample_id'])
            sample_DNA_data['sample_old']=sample_DNA
            sample_DNA_data['sample_id']=sample_DNA_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
            sample_DNA_filter_data=self.data_sample_filter_1(list(sample_DNA_data['sample_old']),type='DNA')
            sample_DNA_data=sample_DNA_data.loc[sample_DNA_data.sample_old.isin(list(sample_DNA_filter_data['ID'])),:]

            
            print('CNV')
            data_CNV=pd.read_csv(self.raw_dir+self.data+'.CNV_masked_seg.csv',sep=',')
            sample_CNV=list(set(data_CNV['Sample']))
            sample_CNV_data=pd.DataFrame(columns=['sample_old','sample_id'])
            sample_CNV_data['sample_old']=sample_CNV
            sample_CNV_data['sample_id']=sample_CNV_data['sample_old'].map(lambda x:x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
            sample_CNV_filter_data=self.data_sample_filter_1(list(sample_CNV_data['sample_old']),type='DNA')
            sample_CNV_data=sample_CNV_data.loc[sample_CNV_data.sample_old.isin(list(sample_CNV_filter_data['ID'])),:]

        else: 
            print('skip data_filter')
            return

        print(len(set(sample_mRNA_data['sample_id'])))
        print(len(set(sample_miRNA_data['sample_id'])))
        print(len(set(sample_DNA_data['sample_id'])))
        print(len(set(sample_CNV_data['sample_id'])))

        sample_id_over_2=list(set(sample_mRNA_data['sample_id'])&set(sample_DNA_data['sample_id'])&set(sample_CNV_data['sample_id']))
        print(len(sample_id_over_2))

        sample_mRNA_data.loc[sample_mRNA_data.sample_id.isin(sample_id_over_2)].to_csv(self.temp_dir+self.data+'_sample_mRNA_data_2.txt',sep='\t',index=False)
        sample_miRNA_data.loc[sample_miRNA_data.sample_id.isin(sample_id_over_2)].to_csv(self.temp_dir+self.data+'_sample_miRNA_data_2.txt',sep='\t',index=False)
        sample_DNA_data.loc[sample_DNA_data.sample_id.isin(sample_id_over_2)].to_csv(self.temp_dir+self.data+'_sample_DNA_data_2.txt',sep='\t',index=False)
        sample_CNV_data.loc[sample_CNV_data.sample_id.isin(sample_id_over_2)].to_csv(self.temp_dir+self.data+'_sample_CNV_data_2.txt',sep='\t',index=False)

        return
    

    def cnv_masked_filter(self):
        if os.path.isfile(self.temp_dir+self.data+'.CNV_masked_seg_filter.txt'):
            if self.skip:
                print('skipping cnv_masked_filter function')
                return
        data=pd.read_csv(self.raw_dir+self.data+'.CNV_masked_seg.csv',sep=',')
        data=data[['Sample','Chromosome','Start', 'End','Num_Probes','Segment_Mean']]
        data['type']=data['Sample'].map(lambda x:x.split('-')[-1])
        data_=data.loc[data['type']=='01',:]
        data_=data_[['Sample','Chromosome','Start', 'End','Num_Probes','Segment_Mean']]
        data_.to_csv(self.temp_dir+self.data+'.CNV_masked_seg_filter.txt',sep='\t',index=False)
        
        return
    





def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="implicit_relational_GNN")
    parser.add_argument('--data', type=str, default='BRCA', help="BRCA or STAD")
    parser.add_argument('--skip', action = 'store_true', help = "skip if file exists")
    args = parser.parse_args()
    
    args.data = args.data.upper()
    assert args.data in ["BRCA", "STAD", "SARC", "LGG", "HNSC", "CESC", "KIPAN", "NSCLC"], "invalid dataset"
    return args


def main():
    args = parse_args()
    args.root_dir = ROOT_DIR
    args.code_dir = ROOT_DIR + "/src/"
    args.raw_dir = ROOT_DIR + "/raw/"
    args.processed_dir = ROOT_DIR + "/pre_processed/"
    args.temporal_dir = ROOT_DIR + "/temporal_files/"
    args.root_root_dir = os.path.dirname(ROOT_DIR)
    processor = Process_data(args)
    processor.process()
    return


if __name__ == "__main__":
    main()
    