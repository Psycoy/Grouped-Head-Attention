#!/bin/bash

###################################################################################
IWSLT2014 German-English
# Download and prepare the data
cd ./translation/
bash prepare-iwslt14deen.sh
cd ..

# Preprocess/binarize the data
TEXT=./translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20

####################################################################################
# IWSLT2014 English-French
# Download and prepare the data
cd ./translation/
bash prepare-iwslt14enfr.sh
cd ..

# Preprocess/binarize the data
TEXT=./translation/iwslt14.tokenized.en-fr
fairseq-preprocess --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.en-fr \
    --workers 20


####################################################################################
# IWSLT2014 English-Italian
# Download and prepare the data
cd ./translation/
bash prepare-iwslt14enit.sh
cd ..

# Preprocess/binarize the data
TEXT=./translation/iwslt14.tokenized.en-it
fairseq-preprocess --source-lang en --target-lang it \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.en-it \
    --workers 20


####################################################################################
# IWSLT2014 English-German
# Download and prepare the data
cd ./translation/
bash prepare-iwslt14ende.sh
cd ..

# Preprocess/binarize the data
TEXT=./translation/iwslt14.tokenized.en-de
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.en-de \
    --workers 20


###################################################################################
# IWSLT2014 Italian-English
# Download and prepare the data
cd ./translation/
bash prepare-iwslt14iten.sh
cd ..

# Preprocess/binarize the data
TEXT=./translation/iwslt14.tokenized.it-en
fairseq-preprocess --source-lang it --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.it-en \
    --workers 20


####################################################################################
# WMT2014 English-German
# Download and prepare the data
cd ./translation/
bash prepare-wmt14en2de.sh
cd ..
# Binarize the dataset
TEXT=./translation/wmt14_en_de_devnews2013_a
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

####################################################################################
# WMT2014 English to French
# Download and prepare the data
cd ./translation/
bash prepare-wmt14en2fr.sh
cd ..

# Binarize the dataset
TEXT=./translation/wmt14_en_fr_standardsplit
fairseq-preprocess \
    --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0 \
    --workers 60

