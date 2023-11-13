# ALTEGRAD 23 Balthazar Neveu
- [Lab 1](Lab1_Neveu_Balthazar.pdf) : Neural machine translation using vanilla GRU.
- [Lab 2](Lab2_Neveu_Balthazar.pdf) : Graph mining (get familiar with shortest path / graphlet kernels, spectral clustering) 
- [Lab 3](Lab3_Neveu_Balthazar.pdf) : Transfer Learning for sentiment analyzis using a pretrained Transformer for language modeling using next word prediction 
- [Lab 4](Lab4_Neveu_Balthazar.pdf) : Learning how to use a few frameworks (Fairseq, Hugging face) - Fine tune LLM with LoRA + quantization
# TensorDock
:heart: Thank you [Ferdinand Mom](https://github.com/3outeille) for all the tips.

Here are all information to work on the cloud.


- Step 1: Open an account on TensorDock, put 20$ in your account. This should be enough for a month.
- Step 2: Create a CPU VM to get started (cheap), start/stop the instance
- Step 3: Setup the machine with installation scripts , get confortable with it.
- Step 4: Set up a GPU machine and get your trainings done


## [Tensor dock list of servers](https://marketplace.tensordock.com/list)


### CPU  Machine
Use this  machine to set up your environment.
- Pre-configured VM :  TensorML 20 PyTorch
- Storage 60 GB
- vCPUs: 4 Intel Xeon v3 vCPUs
- 16 GB RAM
- Compute price : \$0.08/hour  
  - `1$/12hour` = `2$/day`
  - you can disable the VM whenever you want and stop paying this fee
- Storage price : \$0.006/hour =  15 cents / day = `1$/week` = `4$/month`
  - you stop paying this as soon as you delete the VM. But when you kill the the machine, you'll loose your stored data.

## Setup
- ssh user@216.XXX.XXX.XXX

```
git clone https://github.com/balthazarneveu/gpu_cloud.git
cd gpu_cloud
. first.sh
. second.sh
cat ~/.ssh/id_ed25519.pub
```

- Add the  public ssh key to your github account.


- (*Use VSCode remote extension to connect to your machine*)


- Lab4 setup
```
git clone git@github.com:balthazarneveu/MVA23_ALTEGRAD.git
cd ~
. MVA23_ALTEGRAD/setup.sh
cd ~/MVA23_ALTEGRAD/Lab4
. download_models.sh
```