# Magmas inside Monoids

"It's Monoids, Monoids All the Way Down." -- Ancient Babylonian Saying

A broad overview of where this could lead [notion](https://proximal-vise-0cf.notion.site/Proj-Space-of-an-octonion-adjoint-group-303cb631e83b4315a2a74f1bca34889f)


## Running with pre-existing python 3.10+

If you have python3.10 installed, you can run

```
python3 -m pip install -r requirements.txt
python3 demo.py
```

This shows the construction of an (unidentified)
21-element transition monoid for the "Rock-Paper-Scissors" magma.

If you want to see progress on verifying the transition group of the
octonions (warning, slow):

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


## Rough Draft

How to view a non-associative magma as part of a monoid.

[Here](https://paper.dropbox.com/doc/Its-Monoids-All-the-Way-Down-JL8ZKqYfnX5mudQoIGX4A#:uid=017421118273067050805863&h2=Magma-%E2%86%92-Monoid-Embedding-Theor) is where I am writing the first draft of a friendly treatment.

## Longer version

This could be an expository paper. Intended audience maybe upper
undergrad level, or computer programmers?

1. Non-Associative Things
   - Cartesian Product
   - Subtraction and Division
   - Rock Paper Scissors
   - Exponentiation
   - Cross Product
   - Lie Bracket
   - Reverse Polish Notation
1. Background
   - Magmas
   - Categories
   - Monoids
   - The Free Monoid
   - Mult. Tables and State Diagrams
   - Function currying and operads
1. The Adjoint and Curried Monoid
   - Associativity <=> Closure
   - Construction Examples
   - ...
1. Applications
   - ...
1. Lean Proofs
   - ...
1. Jupyter notebooks
