Experiments for "A Mathematical Framework, a Taxonomy of Modeling Paradigms, and a Suite of Learning Techniques for Neural-Symbolic Systems".

### Requirements
These experiments expect that you are running on a POSIX (Linux/Mac) system.
The specific application dependencies are as follows:
 - Bash >= 4.0
 - Java >= 7
 - Python >= 3.7

### Setup
These scripts assume you have already built and installed NeuPSL from our repository **with Gurobi**.
If you have not, please follow the instructions for **Installing PSL with Gurobi** in our [NeuPSL repository](https://github.com/linqs/psl).

## Data
Data must be created before running the experiments.

Except for the modular NeSy-EBM learning and logical deduction experiments, data is created by running the `create_data.py` scripts in the `scripts` directories of each experiment.

Data for the modular NeSy-EBM learning experiments is created by running the `modular_learning/scripts/setup_psl_examples.sh` script.
This script will clone the [psl-examples repository](https://github.com/linqs/psl-examples) repo.

Data for the logical deduction experiments is already created and included in the repository.

## Models
After creating the data, models must be prepared for experiments.
Model preparation consists of pretraining and training the baseline neural models and the neural components of the NeuPSL models.

Symbolic models for the `roadr`, `citation`, `path_finding`, `mnist_addition`, and `visual_sudoku_solving` experiments are already included in the repository.
Neural models for the `roadr`, `citation`, `path_finding`, `mnist_addition`, and `visual_sudoku_solving` experiments must be (pre)trained.
Run the `/scripts/train.py` and, if it exists, the `/scripts/pretrain.py` scripts in the corresponding experiment directories to (pre)train the neural models. 

All symbolic models for the modular NeSy-EBM learning experiments will be cloned from the [psl-examples repository](https://github.com/linqs/psl-examples) repo by running the `modular_learning/scripts/setup_psl_examples.sh` script.
The neural model predictions are included in the fetched data.

Symbolic models for the logical deduction experiments are already included in the repository.
The neural component for the logical deduction experiments is ChatGPT. 
The ChatGPT model is connected to using the OpenAI API.
You must have a OpenAI API key to run the experiments.
See [OpenAI API reference guide](https://platform.openai.com/docs/api-reference/introduction) for details.

### Running Experiments
For `roadr`, `citation`, `path_finding`, `mnist_addition`, and `visual_sudoku_solving` experiment scripts are located in the `experiment` directory of each experiment.
To run an experiment, run the corresponding python script and provide the necessary arguments.

There is only one experiment in `logical_deduction`.
To run the experiment, run the `run.py` script in the `logical_deduction` directory.

Similarly, there is only one experiment in `modular_learning`.
To run the experiment, run the `run_weight_learning_performance_experiments` script in the `modular_learning` directory.

### Results
Results will be saved in the `results` directory of each experiment.

To parse the results, run the `parse_results.py` script in the `experiments` or `scritps` directories.
