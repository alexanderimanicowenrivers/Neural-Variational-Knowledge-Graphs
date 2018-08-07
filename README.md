
# Neural-Variational-Knowledge-Graphs !!(Work in progress)!!
-------
## CONTRIBUTERS:

- Alexander Cowen-Rivers ([GitHub](https://github.com/acr42)) 

## Supervisors:

- Pasquale Minervini ([GitHub](https://github.com/pminervini))
- Sebastian Riedel ([GitHub](https://github.com/riedelcastro))

-------

## Instructions

For:
- **Models** see [base](https://github.com/acr42/Neural-Variational-Knowledge-Graphs/blob/master/vkge/base.py)
- **Report** see [ACR](https://github.com/acr42)  

-------

## Training Models

Train baseline:

```
python baseline_main.py  --embedding_size 250 --dataset fb15k-237 --epsilon 0.001 --lr 0.001 --score_func ComplEx --no_batches 10
```



Train variational knowledge graph:

```
python main.py --no_batches 1000 --epsilon 1e-07 --embedding_size 300 --dataset kinship --alt_updates False --lr 0.001 --score_func ComplEx --alt_opt True --alt_test none --file_name /model_example
```



## Dependencies

- tbc


-------

## Usage

1. Clone or download this repository.
2. Prepare your data, or use the included WN18 dataset. 

## FAQs
