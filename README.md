# Federated Learning Simulator

Simulate Federated Learning on a large number of Clients.

Configure your Federated Learning environment via the federatedlearning.json file:

- "dataset" : choose from ["mnist", "cifar10", "kws"]
- "net" : choose from ["logistic", "lstm", "cnn", "vgg11", "vgg11s"]
- "n_clients" : number of Clients
- "classes\_per\_client" : number of different Classes every Client holds in it's local data
- "participation_rate" : fraction of Clients which participate in every Communication Round
- "batch_size" : batch-size used by the Clients
- "balancedness" : default 1.0, if <1.0 data will be more concentrated on some clients
- "iterations" : total number of training iterations
- "momentum" : momentum used during training on the clients


- "compression" : choose from [["none", {}], ["fedavg", {"n" : ?}], ["signsgd", {"lr" : ?}], ["stc_updown", {"p" : ?}],.. ]


- "log_frequency" : number of communication rounds after which results are logged and saved to disk
- "log_path" : e.g. "results/experiment1/"
