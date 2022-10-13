# HyperEvolution
Algorithms for hyperparameter optimization


### Installation

Clone the repository:

```bash
git clone git@github.com:Laurits7/HyperEvolution.git
```

Create a virtual environment:
```bash
python -m venv Hopt
```

And install the package (-e for the editable version):

```bash
pip install -e .
```



### Additional notes:

The optimization algorithms presented here are doing function minimization, so in case you want to maximize (for example AUC) you need to return the score by the scoring function with a minus sign.


### References:

The work presented here is based on these two papers:

---
output:
  md_document:
    variant: markdown_github
bibliography: bibliography.bib
---

[@tani2021evolutionary]
[@tani2022comparison]