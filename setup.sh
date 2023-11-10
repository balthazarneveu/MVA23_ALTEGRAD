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