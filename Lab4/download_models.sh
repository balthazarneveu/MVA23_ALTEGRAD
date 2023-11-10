mkdir models
cd models

wget -c 'https://nuage.lix.polytechnique.fr/index.php/s/E4otcD7B9jm2AWx/download/RoBERTa_small_fr.zip' -O "model_fairseq.zip"
unzip model_fairseq.zip
rm model_fairseq.zip
rm -rf __MACOSX/

wget -c "https://nuage.lix.polytechnique.fr/index.php/s/yYQjg9XWekttG5j/download/RoBERTa_small_fr_HuggingFace.zip" -O "model_huggingface.zip"
unzip model_huggingface.zip
rm model_huggingface.zip
rm -rf __MACOSX/

cd ..
mkdir data
cd data
wget -c "https://nuage.lix.polytechnique.fr/index.php/s/EBHqfR776oCE2Nj/download/cls.books.zip" -O "cls.books.zip"
unzip cls.books.zip
rm cls.books.zip
rm -rf __MACOSX/
mkdir cls.books-json
