# Magmas inside Monoids

"It's Monoids, Monoids All the Way Down." -- Ancient Babylonian Saying

Some notes in progress on the theory in [notion](https://www.notion.so/Subsets-Magmas-Subspaces-Sets-Monoids-Spaces-b9be2ff6a51b4752972c531a8366e069?pvs=4)

## Running with pre-existing python 3.10+

If you have python3.10 installed, you can run

```
python3 -m pip install -r requirements.txt
python3 demo.py
```

This shows the construction of an (unnamed)
21-element monoid adjoint to the "Rock-Paper-Scissors" magma.

If you want to see progress on verifying the adjoint group of the
unit octonion loop (slower), as the extraspecial group of order 128 (plus-type);

```
python3 demo.py --octonions
```

## Upgrading to newer python

If you don't have that version of python, I recommend the following to
keep your setup easy-to-maintain.  (It allows you to have many
versions of python installed and switch between them as needed.)

1. Install [pyenv](https://github.com/pyenv/pyenv)
1. `pyenv install 3.11.0`; 3.11 60% faster than previous versions
1. `pyenv shell 3.11.0`; makes your current shell use that python version
1. `which python3` should show something with "shims" in it
1. `python3 --version`; should show 3.11
1. `python3 -m pip install -r requirements.txt`
1. `python3 demo.py` quick RPS-magma demo
1. `python3 demo.py --octonions` sit back and watch it compute!


## Older notes

[Here](https://paper.dropbox.com/doc/Its-Monoids-All-the-Way-Down-JL8ZKqYfnX5mudQoIGX4A#:uid=017421118273067050805863&h2=Magma-%E2%86%92-Monoid-Embedding-Theor) is where I am writing the first draft of a friendly treatment.
