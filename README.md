This repository is an implementation of **Learning Effective Value Function Factorization via Attentional Communication**. The implementation is based on [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac). 

#### Setup

Set up StarCraft II and SMAC

```shell
bash install_sc2.sh  
```

Install requirements

```shell
pip install -r requirements.txt 
```

#### Run an experiment

```shell
python src/main.py --config=vffac --env-config=sc2 --map=6h_vs_8z  
```

All results will be stored in the `Results` folder.

