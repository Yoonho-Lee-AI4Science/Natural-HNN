# Additional Experiments
## Note
Visualization is applicable to Natural-HNN and HSDN since other models does not provide relevance scores (alpha) to hyperedges.

## Preparation
### CliXO
Visit [CliXO 1.0](https://github.com/fanzheng10/CliXO-1.0) and download the code (git clone). Create executable binary file named `clixo` by running `make` and move the `clixo` file as well as `ontologyTermStats` file to the Natural-HNN/ablation/src directory. (Place two files under Natural-HNN/ablation/src directory.)You do not need to install any python library.

### Run models
Many visualizations use saved relevance scores and final representation of hyperedges. Thus, you need to train the model with the hyperparameter you want to visualize. The files will be saved under save_files directory. For jaccard similarity, you need to train the model with all possible hyperparameters (number of factors, hidden dimension) for visualization. 

### SHAP
You need to install `shap` package to `natural_hnn_model` conda environment
```console
(natural_hnn_model) foo@bar:~$ conda install conda-forge::shap
```
To calculate shap values of a model, use `shap_cal.py` at Natural-HNN/src directory with the same hyperparameter used for training. The example is shown below
```console
(natural_hnn_model) foo@bar:~/{place of the project directory}/Natural-HNN/src$ python3 shap_cal.py --dataset brca --model disen_hgnn --train_percent 50 --valid_percent 25 --test_percent 25 --num_layers 2 --num_repeat 10 --interpol_ratio 0.5 --hidden 32 --lr 0.001 --wd 0 --device 6 --epoch 500 --dropout 0.5 --he_activation 'None' --heads 4 --disen_spec 68 --disen_loss_ratio 0.0 --silence --task bio --val_criterion macro_f1 --metric f1 --hcl_spec 3 --use_balanced_split --batch_size 50 --show_bar
```



## Additional Experiments
### Various visualization
You can simply run clixo_vis_data.py. You need to provide dataset name(lowercase), model name(lowercase), hidden dimension size, number of heads, gpu number, number of pathways to be visualized, disentangling loss and learning rate. Please note that you must have runned the model with same hyperparameter (lr, disloss, head, etc ...) and calculated SHAP value.
Example : 
```console
(natural_hnn_model) foo@bar:~/{place of the project directory}/Natural-HNN/ablation/src$ python3 clixo_vis_data.py --data brca --model disen_hgnn --dim 32 --head 8 --device 6 --num_path 15 --disloss 0 --lr 0.001
```
The results will be stored under `ablation/figures/visualization/{model_name}/` directory with the following name format : `{data name}_{number of pathways}_{pathway / cluster}_{first / second / both layer or ground truth}_{hyperparameters etc..}.svg`.    


The results for each head will be stored in `per_head` subdirectory of each `ablation/figures/visualization/{model_name}/` directory.

### Jaccard similarity (Figure 19 ~ 22)
Please make sure that you have trained Natural-HNN (disen_hgnn) and HSDN for all possible hyperparameters.
Also, you need to calculate SHAP values for every cases using `shap_cal.py`.
If all files are prepared, then you can get the figure by simply running the following.
```console
(natural_hnn_model) foo@bar:~/{place of the project directory}/Natural-HNN/ablation/src$ python3 jaccard_shap.py
```
The results will be stored under `ablation/figures/jaccard` directory with the following name format : `{model name}_{data name}_top_{number of pathways (top k)}.svg`