conda create -n llm python=3.9
conda activate llm

pip install sentencepiece
cd ~
git clone https://github.com/hadi-abdine/fairseq
cd fairseq
pip install -e .
cd ~
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
cd ~
pip install datasets
pip install evaluate
pip install sentencepiece
pip install tensorboardX
pip install sympy
pip install prettytable
pip install accelerate -U
pip install -U scikit-learn

conda create -n llm_bloom python=3.9
conda activate llm_bloom
pip install bitsandbytes==0.39.0
pip install torch==2.0.1
pip install -U git+https://github.com/huggingface/transformers.git@e03a9cc
pip install -U git+https://github.com/huggingface/peft.git@42a184f
pip install -U git+https://github.com/huggingface/accelerate.git@c9fbb71
pip install datasets==2.12.0
pip install loralib==0.1.1
pip install einops==0.6.1
pip install accelerate -U
pip install scipy
pip install -U datasets