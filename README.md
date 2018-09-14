# Neural-Variational-Knowledge-Graphs

## Dependencies

* **python>=3.6**
* **tensorflow>=1.7.0**: https://tensorflow.org
* **scipy**: https://scipy.org
* **tensorflow_probability**: https://www.tensorflow.org/probability/
* **tflearn**: https://github.com/tflearn/tflearn.git

## Installation

To install, run

```bash
$ python setup.py install
```

## Structure

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


-------

## Usage

1. Clone or download this repository.
2. Prepare your data, or use the included WN18 dataset.

## FAQs

## Usage

Please cite [[1](#citation)] in your work when using this library in your experiments.

## Sampling von Mises-Fisher
To sample the von Mises-Fisher distribution we follow the rejection sampling procedure as outlined by [Ulrich, 1984](http://www.jstor.org/stable/2347441?seq=1#page_scan_tab_contents). This simulation pipeline is visualized below:

<p align="center">
  <img src="https://i.imgur.com/aK1ze0z.png" alt="blog toy1"/>
</p>

_Note that as ![](http://latex.codecogs.com/svg.latex?%5Comega) is a scalar, this approach does not suffer from the curse of dimensionality. For the final transformation, ![](http://latex.codecogs.com/svg.latex?U%28%5Cmathbf%7Bz%7D%27%3B%5Cmu%29), a [Householder reflection](https://en.wikipedia.org/wiki/Householder_transformation) is utilized._

## Feedback
For questions and comments, feel free to contact [Nicola De Cao](mailto:nicola.decao@gmail.com).

## License
MIT

## Citation
```
[1]
[2] Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T.,
and Tomczak, J. M. (2018). Hyperspherical Variational
Auto-Encoders. arXiv preprint arXiv:1804.00891.
```

BibTeX format:
```
@article{OUR-paper}
@article{s-vae18,
  title={Hyperspherical Variational Auto-Encoders},
  author={Davidson, Tim R. and
          Falorsi, Luca and
          De Cao, Nicola and
          Kipf, Thomas and
          Tomczak, Jakub M.},
  journal={arXiv preprint arXiv:1804.00891},
  year={2018}
}
```
