# ExhaustiveFaultInjection

To launch an exahustive fault injection run:

```
python main.py
```

A network can be specified by using the argument `--network`:

```
python main.py --network mobilenet-v2
```

A starting/ending layer can be specified with the arguments `--layer-start` and `--layer-end` for example:

```
python main.py --network mobilenet-v2 --layer-start 50
```

The code is written with <b>Python 3.8.10</b>, the file `requirements.txt` contains the required packages.
