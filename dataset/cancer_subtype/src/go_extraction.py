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
import math
import csv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    
class go_process_data:
    def __init__(self, args):
        self.args = args
        self.raw_dir = self.args.raw_dir
        self.temp_dir = self.args.temporal_dir
        self.proc_dir = self.args.processed_dir
        self.code_dir = self.args.code_dir
        self.skip = self.args.skip
        self.root_root_dir = self.args.root_root_dir
        self.simple_test()
        
        # data
        self.parents = None
        self.go_translations = None
        self.gene_go_map = None
        self.go_gene_index = None
        self.go_ensembl_index = None
        self.go_go_index = None
        self.go_path_index_map = None
        self.go_ensembl_pathway_map = None
        self.GO_initial = None
        self.GO_final = None
        
    def simple_test(self):
        return 
    
    def process(self):
        self.assign_go_to_gene()
        self.before_enrichment()
        print('From the files generated above, you need to do enrichment analysis with R files.')
        print('If you finished enrichment analysis, then resume.')
        pdb.set_trace()
        self.set_pathway_function()
        print("The end of data go-processing")
        return
    
    
    
    def set_pathway_function(self):
        if self.skip and os.path.isfile(self.proc_dir+'go_pathway_final.csv'):
            print('skip set_pathway_function')
            return
        GO_initial = open(self.proc_dir+"enriched_go_pathway_result.csv").read().split('\n')[1:-1]
        GO_initial = [[i for i in path.split(',') if i!='NA'] for path in GO_initial]
        self.GO_initial = GO_initial
        final_go_per_path = []
        for path_index, go_path in tqdm(enumerate(GO_initial)):
            #pdb.set_trace()
            path_gene_list = self.go_path_index_map[path_index]
            path_go_gene_map = dict()
            for gene_index, gene in enumerate(path_gene_list):
                go_of_gene = list(self.go_go_index[gene])
                for go_index, go in enumerate(go_of_gene):
                    if go in go_path:
                        if go in path_go_gene_map.keys():
                            path_go_gene_map[go] = path_go_gene_map[go] + [gene]
                        else: 
                            path_go_gene_map[go] = [gene]
                        
            #path_go_gene_map_temp = sorted(path_go_gene_map, key=lambda k: len(path_go_gene_map[k]), reverse=True)  
            temp_go_per_path = []
            go_index_temp = 0
            while(True):
                if len(path_gene_list) <= 0:
                    break   
                if go_index_temp >= len(go_path):
                    break
                go_in_this_iter = go_path[go_index_temp]
                if go_in_this_iter not in path_go_gene_map.keys():
                    go_index_temp += 1
                    continue
                path_gene_list = list(set(path_gene_list) - set(path_go_gene_map[go_in_this_iter]))
                temp_go_per_path.append(go_in_this_iter)
                go_index_temp += 1
            final_go_per_path.append(temp_go_per_path)  
        self.GO_final = final_go_per_path         
            
        with open(self.proc_dir+'go_pathway_final.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.GO_final)
        f.close()
        
        return
    
    
    def before_enrichment(self):
        gene_list = open(self.proc_dir+'index_gene_map.txt', 'r').read().split('\n')
        ensembl_list = open(self.proc_dir+'index_ensembl_map.txt', 'r').read().split('\n')
        gene_go_map = [[k,v] for k,v in self.gene_go_map.items()]
        gene_in_go_map_list = [i[0] for i in gene_go_map]
        go_ensembl_index = []
        go_gene_index = []
        go_go_index = []
        
        for i, hgnc in tqdm(enumerate(gene_list)):
            if hgnc in gene_in_go_map_list:
                go_gene_index.append(gene_list[i])
                go_ensembl_index.append(ensembl_list[i])
                go_go_index.append(gene_go_map[gene_in_go_map_list.index(hgnc)][1])
        pathway_gene = open(self.raw_dir+'pathway_list.txt','r').read().split('\n')[1:-1]
        pathway_gene = [a.split('\t')[1].split(',') for a in pathway_gene]
        go_path_index_map = []
        for i, pathway in tqdm(enumerate(pathway_gene)):
            temp = []
            for j,gene in enumerate(pathway):
                if gene in go_gene_index:
                    temp.append(str(go_gene_index.index(gene)))
            go_path_index_map.append(temp)
        self.go_gene_index= go_gene_index
        self.go_ensembl_index = go_ensembl_index
        self.go_go_index = go_go_index
        self.go_path_index_map = go_path_index_map
        
        new_file = open(self.proc_dir+'go_gene_index_map.txt', 'w')
        new_file.write('\n'.join(go_gene_index))
        new_file.close()
        
        new_file = open(self.proc_dir+'go_ensembl_index_map.txt', 'w')
        new_file.write('\n'.join(go_ensembl_index))
        new_file.close()
        
        new_file = open(self.proc_dir+'go_go_index_map.txt', 'w')
        new_file.write('\n'.join(['\t'.join(aa) for aa in go_go_index]))
        new_file.close()
        
        new_file = open(self.proc_dir+'go_pathway_index_map.txt', 'w')
        new_file.write('\n'.join(['\t'.join(aa) for aa in go_path_index_map]))
        new_file.close()
        
        self.go_path_index_map = [[int(b) for b in a] for a in go_path_index_map]
        go_ensembl_pathway_map = []
        for i, pathway in tqdm(enumerate(self.go_path_index_map)):
            temp = []
            for j,gene_idx in enumerate(pathway):
                temp.append(self.go_ensembl_index[gene_idx])
            go_ensembl_pathway_map.append(temp)
        #pdb.set_trace()
        self.go_ensembl_pathway_map = go_ensembl_pathway_map
        with open(self.proc_dir+'go_ensembl_pathway_map.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.go_ensembl_pathway_map)
        f.close()
        #pdb.set_trace()
        return
                
    
    def assign_go_to_gene(self):
        if self.skip and os.path.isfile(self.proc_dir+'gene_GO_map.txt'):
            dic = {}
            with open(self.proc_dir+'gene_GO_map.txt') as f:
                for line in f:
                    line=line.strip().split('\t')
                    dic[line[0]]=set(line[1:])
            print('skipping assign_go_to_gene')
            self.gene_go_map = dic
            return
        self.Read_GO_file()
        self.descriptives(self.parents, '\nparents of go terms')
        self.richParents()
        self.descriptives(self.parents, '\nrich parents of go terms')
        self.GOMap3()        
        self.gene_go_map = self.empty_profiles(self.gene_go_map)
        self.descriptives(self.gene_go_map, '\ngo terms per gene')
        new_file = open(self.proc_dir+'gene_GO_map.txt','w')
        write_text = None
        for k,v in self.gene_go_map.items():
            if write_text == None:
                write_text = k+'\t'+'\t'.join(v)+'\n'
            else: 
                write_text += k+'\t'+'\t'.join(v)+'\n'
        new_file.write(write_text)
        new_file.close()
        
        return
        
    def standard_dev(self,mean, data):
        tot = 0
        for i in data:
            floatit = i * 1.0
            diff = floatit - mean
            squ = diff * diff
            tot+=squ
        var = tot / len(data)
        std = math.sqrt(var)
        print('standard dev: ' + str(std))
    
    
    def empty_profiles(self, prof):
        lost=0
        newdic = dict()
        for k in prof.keys():
            vals = prof.get(k)
            if vals:
                newdic[k]=vals
            else:
                lost += 1
        print(str(lost) + ' empty profiles')
        return newdic
    
    
    
    def GOMap3(self):
        
        print('\nGOMap3 : building dictionary of GO terms')
        gene_association_file = self.raw_dir+'goa_human.gaf'
        gene_list = open(self.proc_dir+'index_gene_map.txt').read().split('\n')
        parents = self.parents
                                
        full_go_set=set()
        gomap = dict()
        computer_gos=set()
        lost_evid = dict()
        C_F = set()
                                    
        with open(gene_association_file) as f: # connects emsembl gene ids and go terms
                
            for line in tqdm(f):                                 
                if line[0] == '!':
                    continue
                                    
                line = line.split('\t')
                gene1 = line[2].upper()
                gene1 = gene1.strip()

                # deal with alternative genes name first given might not match up
                if gene1 in gene_list:
                    genes = set([gene1])
                else:
                    genes = set()

                alt_genes = line[10].strip().split('|')
                for ag in alt_genes:
                    if ag.strip() in gene_list:
                        genes.add(ag)

                if len(genes)==0: continue

                # get the C and F functions then quit
                if line[8] == 'C' or line[8] == 'F':
                    for g in genes:
                        if g not in gomap.keys():
                            C_F.add(g)
                    continue
                    
                    
                # EXPERIMENTAL inferred from experimant, direct assay, physical interaction
                if  line[6].strip() == 'IEA': #
                    for g in genes:
                        if g not in gomap.keys():
                            computer_gos.add(g)
                    continue

                # put the gene in the dicts
                go = line[4].strip()
                for g in genes:
                    if g not in gomap.keys():
                        gomap[g]=set()
                    gomap[g].add(go)
                    full_go_set.add(go)

        print(str(len(gomap.keys())) + ' genes had go terms')
        comp_gos = computer_gos - set(gomap.keys())
        print( str(len(comp_gos)) + ' genes had compter go terms') # may have real and cmputational gos
        self.descriptives(gomap, 'gene:gos before parents')
        gomap, parents, en = self.GO_complete(gomap, parents)

        # get unannotate genes due to filtering comp gos
        empy = set()
        for i in gomap.items():
            k,v = i
            if len(v)<1:
                empy.add(k)
        for i in empy:
            del gomap[i]
        self.gene_go_map = gomap
        return gomap
        
    
    def GO_complete(self, original, Parents):
        # fill out original gene:go dictionary
        enrich_prof = dict()
        rich_parent = dict()
        missing_annotations = set()
        for gene in original.keys():
            enrich_prof[gene]=set()
            current_go = original.get(gene)
            for annotation in current_go:
                enrich_prof[gene].add(annotation)
                enrichment_gos = Parents.get(annotation)            
                if enrichment_gos:
                    for e in enrichment_gos:
                        if e not in current_go:
                            enrich_prof[gene].add(e)
                elif annotation not in missing_annotations :
                    missing_annotations.add(annotation)
        en=0
        for i in enrich_prof.keys():
            enr = enrich_prof.get(i)
            orig = original.get(i)

            if len(enr)>len(orig):
                en= en+1
            if len(enr)<len(orig):
                print('mad fail')

        print('parents no enriched ' + str(en) + ' out of ' + str(len(original.keys())) )
        return enrich_prof, Parents, en
    
    def richParents(self):
        newsize = 0.5
        oldsize = 0.1
        parentdic = self.parents
        while oldsize < newsize:
            for kid in parentdic.keys():
                parents = parentdic.get(kid)
                newgrans = set()
                for p in parents:
                    grans = parentdic.get(p)
                    for g in grans:
                        if g not in parents:
                            newgrans.add(g)
                                                
                for ng in newgrans:
                    parentdic[kid].add(ng)

                            
            calcsize=0
            for i in parentdic.keys():
                calcsize = calcsize + len(parentdic.get(i))
            print('calcsize ' + str(calcsize))
                
            tempold = newsize
            newsize = calcsize
            oldsize = tempold
            
        self.parents = parentdic
        return
            
    
    
    def descriptives(self, dictionary, title):
        if title: 
            print(title)
        print('no. nodes: ' + str(len(dictionary)))

        lengthsDict = dict()
        total_for_mean = 0
        allgenes = set()
        for k in dictionary.keys():
            lengthsDict[k]=list()
            lengthsDict[k] = len(dictionary.get(k))
            total_for_mean = total_for_mean + len(dictionary.get(k))
            allgenes = allgenes.union(dictionary.get(k))

        lenlist = sorted(lengthsDict.values())
        if len(lenlist)>0:
            if len(lenlist) % 2 == 0:
                index = int((len(lenlist)/2)-1)
                median = sum([lenlist[index], lenlist[index+1]])/2.0
            else:
                median = lenlist[int(len(lenlist)/2)]*1.0

            mean =  total_for_mean*1.0/len(lengthsDict.keys())
            if title: 
                print('median: ' + str(median) + ' range: ' + str(lenlist[0]) + ' - ' + str(lenlist[-1] ) + ' mean: ' + str(mean))
            self.standard_dev(mean, lenlist)
        print('total genes/gos ' + str(len(allgenes)) + '\tmedian: ' + str(median))
        return median
    
    
    def Read_GO_file(self):
        # david doesnt give you all the parent terms for each go term
        # this creates a term:parent dictionry so parent terms can be added 
        gofile = self.raw_dir+'go-basic.obo'
        count_genes=0
        obs = 0
        paras = list()
        altids = dict()
        
        with open(gofile) as f:
            # split file into a list of paragraphs, where each paragraphs a list of lines      
            para = list()
            for line in f:
                if line != '\n': # if its not the end of a paragraph
                    para.append(line.strip())
                else: # if you got to the end of a paragraph append to master list and reset
                    if 'format-version: 1.2' in para[0]: # lose top paragraph
                        para=list()
                        continue
                    elif 'is_obsolete: true' in para:
                        obs+=1
                        para = list()
                        continue
                    else:
                        count_genes+=1
                        if 'namespace: biological_process' in para:
                            paras.append(para)
                        para = list()

    #   print 'no of biological_process GO terms ' + str(len(paras))
    #   print str(obs) + ' terms removed as obselete'
        
        # create dictionary connecting parents to terms
        Parents=dict()
        translations = dict()
        biological_process_first_child = dict()
        for paragraph in paras:
            if '[Typedef]\n' in paragraph: # unwanted paragraphs
                continue
            for i, line in enumerate(paragraph):
                words = line.split(' ')
                if i == 1:
                    if 'GO:' in words[1]:
                        GOterm = words[1].strip()
                        Parents[GOterm]=set()
                    elif '[Typedef]' in paragraph:
                        continue
                    else:
                        print(words[1] + ' not go term')
                        
                elif i==2:
                    name = re.sub('name: ','', line)
                    if GOterm not in translations.keys():
                    #  print GOterm + ' arse'
                        translations[GOterm] =  name.strip()

                elif 'is_a:' in words[0]:
                    parent = words[1].strip()
                    if 'GO:' not in parent:
                        pass 
                        #print 'problem parent ' + str(words)
                    elif GOterm in Parents.keys():
                        Parents[GOterm].add(parent.strip())
                        if 'is_a: GO:0008150' in words[0]: # get direct parents of biological_process (GO:0008150)
                            biological_process_first_child[GOterm].add(parent.strip())
                    else:
                        print('error in O_file')

                    
                elif 'relationship:' in words[0]:
                    if 'part_of' in words[1]:
                        if 'GO:' in words[2]:
                            parent = words[2].strip()
                            if GOterm in Parents.keys():
                                Parents[GOterm].add(parent.strip())
                elif 'alt_id:' in words[0]:
                    alt = words[1].strip()
                    if 'GO:' not in alt:
                        pass 
                        #print 'problem parent ' + str(words)
                    elif alt in altids.keys():
                        altids[alt].add(GOterm.strip()) 
                    else:
                        altids[alt] = set()
                        altids[alt].add(GOterm.strip())

        self.parents = Parents
        self.go_translations = translations  
        
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
def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="go map")
    parser.add_argument('--skip', action = 'store_true', help = "skip if file exists")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.root_dir = ROOT_DIR
    args.code_dir = ROOT_DIR + "/src/"
    args.raw_dir = ROOT_DIR + "/raw/"
    args.processed_dir = ROOT_DIR + "/pre_processed/"
    args.temporal_dir = ROOT_DIR + "/temporal_files/"
    args.root_root_dir = os.path.dirname(ROOT_DIR)
    processor = go_process_data(args)
    processor.process()
    return


if __name__ == "__main__":
    main()
    