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

-------
## CONTRIBUTERS:

- Alexander Cowen-Rivers ([GitHub](https://github.com/acr42))

## Supervisors:

- Pasquale Minervini ([GitHub](https://github.com/pminervini))
- Sebastian Riedel ([GitHub](https://github.com/riedelcastro))

-------

## Instructions

For:
- **Models** see [Latent Fact Model](https://github.com/acr42/Neural-Variational-Knowledge-Graphs/blob/master/vkge/LFM.py) and [Latent Information Model](https://github.com/acr42/Neural-Variational-Knowledge-Graphs/blob/master/vkge/LCM.py)
- **Paper** see [ACR](https://github.com/acr42/)

-------

## Training Models

Train variational knowledge graph model, on nations dataset with normal prior using DistMult scoring function :

```
python main_LIM.py  --no_batches 10 --epsilon 1e-07 --embedding_size 50 --dataset nations --alt_prior False --lr 0.001 --score_func DistMult --negsamples 5 --projection False --distribution normal --file_name /User
```


-------

## Usage

1. Clone or download this repository.
2. Prepare your data, or use any of the six included KG datasets.

## Usage

Please cite [[1](#citation)] and [[2](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [ACR](https://github.com/acr42)(mailto:mc_rivers@icloud.com).

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
