Implementation for the paper ["Complementing CO2 emission reduction by solar
radiation management might strongly enhance future
welfare"](https://esd.copernicus.org/articles/10/453/2019/).

This code currently only includes the basic scenarios used in the paper.
Although some tests are included, test coverage is very poor.

## Requirements

- Python 3.6
- Packages in `requirements.txt`

## Usage

Compile the program by running `cythontools/build_and_test.sh`.

Before running the Dynamic Programming optimization of the stochastic model,
we need to determine the boundaries of the region in statespace we want to
consider. For this purpose we provide two helper scripts in
`optimizationtools`: `generate_domains.py` and `generate_optimal_pathways.py`
(the latter to get a baseline policy in the case of a determinstic model).
Domains are specific to experiment settings and step intervals.

Once the domains have been generated and stored in `data/lrgd2/domain`, the
actual optimization is performed by running
`optimizationtools/run_optimization.py`.

