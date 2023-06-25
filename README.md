[![DOI](https://zenodo.org/badge/571814987.svg)](https://zenodo.org/badge/latestdoi/571814987)

Safe Reinforcement Learning for POMDPs
=======================================

Requires [Storm](https://www.stormchecker.org/) and [stormpy bindings](https://github.com/sjunges/stormpy).

For enviroment configurations see [gridstorm environments](https://github.com/stevencarrau/shield_rl_gridworlds)

Running main.py will load a file from cfgs and run the shielded experiments. We include the full set of configurations used to generate the results.


It is also possible to run shield.py directly, for more details on running directly see [shield-in-action](https://github.com/sjunges/shield-in-action). For example: 
```
python rlshield/shield.py -m refuel --constants "N=6,ENERGY=8" --video-path .  
```
The model states that the grid-model `refuel` should be used with constants as specified. 
We run one episode and store videos in the current working directory.
