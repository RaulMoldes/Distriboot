# Distriboot
A distributed training algorithm for boosted decision trees. Based on Adaboost.

## Compile

```bash
mpic++ -o adaboost main.cpp -lm
```

## Run


```bash
mpirun -np 8 adaboost
```
