## A Brief Introduction to GPU Programming

This repository is part of the Tapia 2022 DOE National Laboratories HPC session.

### Components

* `examples` - contains codes that are demonstrated as part of the presentation
* `exercises` - contains hands-on exercises for the participants to perform on their own.
* `common` - contains
  * `setup_environment.sh` - script used to load the necessary programming environment.
  * `hip_translator.h` - header file that tranlates CUDA API calls to HIP API calls (so CUDA can be used on AMD GPUs). 

### Get Access to Spock and Log in

Please follow the instructions given by the on-site OLCF staff.

### Clone the Repository

When you log in to Spock, you'll be dropped into your "home" directory (`/ccs/home/<username>` - where `username` will be the `csepXXX` found on the front of your yellow RSA token envelope). From there, clone the repository with using the following command:

```bash
$ git clone https://github.com/olcf/tapia2022-intro-gpu-programming.git
```

> NOTE: The `$` above represents the bash command-line prompt and should not be included in the command itself.

### Perform the exercises and (optional) examples

Please follow the instructions given in the slides and/or by the OLCF staff for this part.

<hr />

