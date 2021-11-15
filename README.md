# Sponsored Keyword search

# Description of directories and files.

- `src`: Contains all the source files.
- `data`: Contains data files and `gen.py` by Alex
- `scripts`: Some experiments with `pulp` and data files.
- `src/pulp_utils.py`: Functions for creating optimization variables, constraints, objectives etc.,
- `src/data_utils.py`: Contains preprocessing utils for data pickle files. We will get the B, W, A etc., from here.
- `src/extensions`: Contains the code for extensions
- `src/issues`: Contains code for simulating various issues with the algorithms

# Before Running the project
Navigate to the root folder of the project and run
> pip install -r requirements.txt

Then set your `PYTHONPATH` variable as follows. Go to the project root
from your terminal and run the following command.
> export PYTHONPATH=`pwd`

You should be able to run any file now using the command
>python3 <relative_path_from_project_root>.py

Please ask me in case you encounter any issues with running the scripts.

[Slides](https://docs.google.com/presentation/d/1qHPIMusbEReEuw5Y5UMOd_nIRgfCPp7JP8JVfckm9Ig/edit?usp=sharing)