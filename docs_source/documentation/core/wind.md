# Wind

## Overview

cyberFly supports various wind field models, as well as provides the capability for users to define their own wind field models.

## Preimplemented Wind Models

Several popular wind models are provided:

1. Lorem Ipsum

They can be initialized in the `aviary` like so:

```python
env = Aviary(..., wind_type="Lorem Ipsum")
```

## Simple Custom Wind Modelling

For simple, stateless wind models (models that do not have state variables), it is possible to simply define the wind model as a Python method.
Then, the wind model can be hooked to the `aviary` using the `register_wind_field_function` method.
The following is an example:

```{eval-rst}
.. literalinclude:: ../../../examples/core/09_simple_wind.py
   :language: python
```

## More Complex Custom Wind Modelling

To define custom wind models, refer the the example provided by the `WindFieldClass` below:

```{eval-rst}
.. autoclass:: cyberFly.core.abstractions.WindFieldClass
```

### Default Attributes
```{eval-rst}
.. property:: cyberFly.core.abstractions.WindFieldClass.np_random

  **dtype** - `np.random.RandomState`
```

### Required Methods
```{eval-rst}
.. autofunction:: cyberFly.core.abstractions.WindFieldClass.__init__
.. autofunction:: cyberFly.core.abstractions.DroneClass.__call__
```
