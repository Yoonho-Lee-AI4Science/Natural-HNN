# Models
## Note
Before training, check whether datasets are well prepared. 
We recommend reading every README.md files inside dataset directory and its subdirectories.
We provide preprocessed cancer subtype datasets.
To use preprocessed cancer subtype datasets, you need to unzip dataset/{cancer_name}/raw/npy_files.zip and place the contents under dataset/{cancer_name}/raw directory.
If you want to manually preprocess data, read dataset/cancer_subtype/src/README.md file.
Note that manually pre-processed result can be slightly different from the one we used, depending on the time at which the data or additional files are downloaded.
Most of the preprocessing codes were adopted from Pathformer.([paper](https://www.biorxiv.org/content/10.1101/2023.05.23.541554v2), [github](https://github.com/lulab/Pathformer))  

Official name of our model is `Natural-HNN`. However, it is named `disen_hgnn` in our implementation(code). For training, use `natural_hnn_model` environment.  

## Standard Hypergraph Benchmark Data
Dataset names : cora, citeseer, pubmed, coauthor_cora, coauthor_dblp, ntu2012, modelnet40, 20newsw100
```console
(natural_hnn_model) foo@bar:~/{location of the project directory}/Natural-HNN/src$ python3 main.py --dataset {data name} --model disen_hgnn --train_percent 50 --valid_percent 25 --test_percent 25 --hidden {hidden representation size} --lr {learning rate} --wd {weight decay} --device {gpu number} --he_activation 'None' --heads {number of factors} --disen_loss_ratio {factor discrimination loss ratio (lambda)}
```


## Cancer Subtype Classification task
Note that we used 50 as batch size.  
Dataset names : brca, stad, sarc, cesc, hnsc, lgg, kipan, nsclc
```console
(natural_hnn_model) foo@bar:~/{location of the project directory}/Natural-HNN/src$ python3 main.py --dataset {data name} --model disen_hgnn --train_percent 50 --valid_percent 25 --test_percent 25 --hidden {hidden representation size} --lr {learning rate} --wd {weight decay} --device {gpu number} --he_activation 'None' --heads {number of factors} --disen_loss_ratio {factor discrimination loss ratio (lambda)} --task bio --val_criterion macro_f1 --metric f1 --hcl_spec 3 --use_balanced_split --batch_size 50 --show_bar
```



