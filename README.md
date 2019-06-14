# Federated Learning Simulator

Simulate Federated Learning with compressed communication on a large number of Clients.

Recreate experiments described in [*Sattler, F., Wiedemann, S., Müller, K. R., & Samek, W. (2019). Robust and Communication-Efficient Federated Learning from Non-IID Data. arXiv preprint arXiv:1903.02891.*](https://arxiv.org/abs/1903.02891)



## Usage
First, set environment variable 'TRAINING_DATA' to point to the directory where you want your training data to be stored. MNIST, FASHION-MNIST and CIFAR10 will download automatically. 

`python federated_learning.py`

will run the Federated Learning experiment specified in  

`federated_learning.json`.

You can specify:

### Task
- `"dataset"` : Choose from `["mnist", "cifar10", "kws", "fashionmnist"]`
- `"net"` : Choose from `["logistic", "lstm", "cnn", "vgg11", "vgg11s"]`

### Federated Learning Environment

- `"n_clients"` : Number of Clients
- `"classes_per_client"` : Number of different Classes every Client holds in it's local data
- `"participation_rate"` : Fraction of Clients which participate in every Communication Round
- `"batch_size"` : Batch-size used by the Clients
- `"balancedness"` : Default 1.0, if <1.0 data will be more concentrated on some clients
- `"iterations"` : Total number of training iterations
- `"momentum"` : Momentum used during training on the clients

### Compression Method

- `"compression"` : Choose from `[["none", {}], ["fedavg", {"n" : ?}], ["signsgd", {"lr" : ?}], ["stc_updown", [{"p_up" : ?, "p_down" : ?}]], ["stc_up", {"p_up" : ?}], ["dgc_updown", [{"p_up" : ?, "p_down" : ?}]], ["dgc_up", {"p_up" : ?}] ]`

### Logging 
- `"log_frequency"` : Number of communication rounds after which results are logged and saved to disk
- `"log_path"` : e.g. "results/experiment1/"

Run multiple experiments by listing different configurations.

## Options
- `--schedule` : specify which batch of experiments to run, defaults to "main"

## Citation 
[Paper](https://arxiv.org/abs/1903.02891)

Sattler, F., Wiedemann, S., Müller, K. R., & Samek, W. (2019). Robust and Communication-Efficient Federated Learning from Non-IID Data. arXiv preprint arXiv:1903.02891.
