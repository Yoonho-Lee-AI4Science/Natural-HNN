# Models
## Note
Before training, check whether datasets are well prepared. Read README.md files inside dataset directory `recursively`(i.e. need to check subdirectories). As I have described there, although we have uploaded all preprocessed files(to private repository), some npy files may not be correctly provided in anonymous github due to the `size limit set by anonymous github`. If you are accessing through anonymous github but want to run models on the cancer subtype datasets, you need to manually pre-process datasets by carefully following the procedures described in README.md files inside dataset directory. The manually pre-processed result can be slightly different from the one we used depending on the time at which the data or additional files are downloaded (Although we hope not :disappointed_relieved: . )    

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



