![](misc/arms_logo.png)

**Paper**: *ARMS: Automated rules management system for fraud detection*, D. Aparício,
R. Barata, J. Bravo, J. T. Ascensão, P. Bizarro, submitted to KDD 2020.

This repository contains the following:
*  ARMS binary in bin/ARMS. Please see the [README](bin/README) and the [license](bin/LICENSE).
*  The synthetic data used in the paper in [data/](data), as well as the [script used to generate the data](data_generator/data_generator.py).
*  [More detailed results than the ones shown in the paper for the synthetic data](results/synthetic_data_results.xlsx).
*  [A more detailed description of the optimization algorithms](supplementary_material/algorithms.pdf).
*  [A presentation with a simple overview of ARMS](supplementary_material/ARMS_overview.pptx).
*  [More figures than the ones shown in the paper, mostly with more detailed results on real data](supplementary_material/figures.pdf).

# ARMS parameters

*  `-df` train dataset file; check [data/synthetic_data_train.csv](data/synthetic_data_train.csv) for the correct format.
*  `-tdf` validation dataset file
*  `-pr` priorities file; check [data/synthetic_data_priorities.csv](data/synthetic_data_priorities.csv) for the correct format.
*  `-lff` loss function file; check [data/loss_function.txt](data/loss_function.txt) for the correct format.
*  `-seed` value
*  `-m` method
    * `single_eval` evaluate a rules configuration; check [data/best_genetic_w_arp_config](data/best_genetic_w_arp_config)
    * `random` use random search
        * `-rr` number of random runs
        * `-sp` mutation/shut-off probability 
    * `greedy` use greedy expansion 
    * `genetic` use genetic programming
        * `-nr` number of runs
        * `-ps` population size
        * `-tps` survivors fraction
        * `-mp` mutation probability
*   `-arp` augment rules pool before optimization
*   `-gpcp`, `-ipcp` random priority shuffling during optimization

# Reproducing the experiments on synthetic data

Since we can not share our clients data, and we could not find a similar dataset
online, we created synthetic data as described in the main paper.

Note that ARMS writes the results to `results/[DATSET]/[FILENAME].json`. The last
line of ARMS' output is the path to the json results file.

## Evaluate the original system

### On the train set
```
bin/ARMS -df data/synthetic_data_train.csv -pr data/synthetic_data_priorities.csv -m single_eval -cf data/all_on -lff data/loss_function.txt -seed 42
```

### On the validation set
```
bin/ARMS -df data/synthetic_data_validation.csv -pr data/synthetic_data_priorities.csv -m single_eval -cf data/all_on -lff data/loss_function.txt -seed 42
```

### On the test set
```
bin/ARMS -df data/synthetic_data_test.csv -pr data/synthetic_data_priorities.csv -m single_eval -cf data/all_on -lff data/loss_function.txt -seed 42
```

## Optimization using random search

First, we train on the train set and evaluate on the validation set.

```
for mp in 0.04 0.10 0.16 0.22 0.28 0.34 0.40 0.46 0.52 0.58 0.64 0.70 0.76 0.82 0.88 0.84
do
   bin/ARMS -df data/synthetic_data_train.csv -pr data/synthetic_data_priorities.csv -m random -rr 300000 -mp "$mp" -lff data/loss_function.txt -seed 42 -tdf data/synthetic_data_train.csv
done
```

From these evaluations (i.e., by checking the "validation system" object of each evaluation json file),
we observe that `mp = 40%` obtained the best results in the validation set. Then, by checking the "removed rules", we created 
a rule configuration file ["best_random_config"](data/best_genetic_config) and evaluated that rule configuration
on the test set.

```
bin/ARMS -df data/synthetic_data_test.csv -pr data/synthetic_data_priorities.csv -m single_eval -cf data/best_random_config -lff data/loss_function.txt -seed 42
```

## Optimization using greedy expansion

### On the original rules

First, we train on the train set and evaluate on the validation set, only using the original rules.

```
bin/ARMS -df data/synthetic_data_train.csv -pr data/synthetic_data_priorities.csv -m greedy -lff data/loss_function.txt -seed 42 -tdf data/synthetic_data_validation.csv
```

Similarly to what we described for random search, we evaluate the best rule configuration found on the test set.

```
bin/ARMS -df data/synthetic_data_test.csv -pr data/synthetic_data_priorities.csv -m single_eval -cf data/best_greedy_config -lff data/loss_function.txt -seed 42
```

### On the augmented rules pool

Second, we train on the train set and evaluate on the validation set, on the augmented rules pool.

```
bin/ARMS -df data/synthetic_data_train.csv -pr data/synthetic_data_priorities.csv -arp -m greedy -lff data/loss_function.txt -seed 42 -tdf data/synthetic_data_validation.csv
```

We also evaluate the best configuration found in the test set.

```
bin/ARMS -df data/synthetic_data_test.csv -pr data/synthetic_data_priorities.csv -m single_eval -cf data/best_greedy_w_arp_config -lff data/loss_function.txt -seed 42
```

## Optimization using genetic programming

### Using the original priorities

First, we train on the train set and evaluate on the validation set, only using the original rule priorities.


```
for ps in 20 30
do
    for tps in 10 20 30
    do
        for mp in 2 5
            bin/ARMS -df data/synthetic_data_train.csv -pr data/synthetic_data_priorities.csv -m genetic -nr 10000 -ps "$ps" -tps "$tps" -mp "$mp" -lff data/loss_function.txt -seed 42 -tdf data/synthetic_data_train.csv
        done
    done
done

```

Similarly to what we described for random search and greedy expansion, we evaluate the best rule configuration found on the test set.

```
bin/ARMS -df data/synthetic_data_test.csv -pr data/synthetic_data_priorities.csv -m single_eval -cf data/best_genetic_config -lff data/loss_function.txt -seed 42
```

### Using priority shuffling

Initially we tried augmenting the rules pool like we did for the greedy expansion, but we found out
that this increased the search space too much and the method's performance actually degraded. Then, we
tried to do random priority shuffling during optimization and results improved.

```
for ps in 20 30
do
    for tps in 10 20 30
    do
        for mp in 2 5
            bin/ARMS -df data/synthetic_data_train.csv -pr data/synthetic_data_priorities.csv -m genetic -nr 10000 -ps "$ps" -tps "$tps" -mp "$mp" -gpcp 0.2 -ipcp 0.1 -lff data/loss_function.txt -seed 42 -tdf data/synthetic_data_train.csv
        done
    done
done

```

We also evaluate the best configuration found in the test set.

```
bin/ARMS -df data/synthetic_data_test.csv -pr data/synthetic_data_priorities.csv -m single_eval -cf data/best_genetic_w_arp_config -lff data/loss_function.txt -seed 42
```