
# TransCodon: a species-informed transformer model for cross-species codon optimization
## Overview
TransCodon is a transformer-based model for cross-species codon optimization, integrating 5′UTR sequences, coding regions, species identifiers, and RNA secondary structure features. It enables zero-shot prediction of gene expression potential and supports regulatory sequence design.

📦 Features

🧬 Joint modeling of 5`UTR and CDS

🌍 Species-specific codon usage learning

🔬 RNA secondary structure information

🧪 Validated in heterologous expression scenarios

📁 Dataset Access
All training, fine-tuning, and held-out evaluation datasets are available at:

🔗 [Google Drive Dataset Folder](https://drive.google.com/drive/folders/17ZKlxM0VF38s9eQXwpKJ6WlgmNMYsZjI?usp=drive_link)


## 🛠Installation
1. Clone the repository


    git clone https://github.com/guyuehuo/transcodon.git
    cd transcodon

   
2. Set up environment

We recommend using conda or [virtualenv].


    conda env create -f environment.yml
    conda activate transcodon

 
## 🌍 Usage

1. Pretrain

Generate DNA sequences from amino acid sequences using a pretrained TransCodon model:


    python pretraining.py \
        --train_data data/finetune/train.csv \
        --output_dir checkpoints/pretain_model \
        --epochs 5 \
        --batch_size 3 \
        --accumulate_gradients 6\
        --lr 2e-4 \
        --num_gpus 4\
 

2. Finetune

Finetune the pretrained model on a custom dataset (e.g., for codon optimization or other downstream tasks):
    
  
    python fintune.py \
        --train_data data/finetune/fintune.csv \
        --output_dir checkpoints/finetuned_model \
        --pretrained_model checkpoints/transcodon.pt \
         --epochs 15 \
        --batch_size 3 \
        --accumulate_gradients 6\
        --lr 2e-4 \
        --num_gpus 2\
 

3. Infer

Given an input amino acid sequence and a specified host species, TransCodon generates a DNA sequence that conforms to the natural codon usage landscape of the target species. This enables codon optimization for heterologous expression while preserving biological realism.
 
    python infer.py \
        --input_data ./test.csv \
        --output_file ./optimized_dna.csv \
        --model_checkpoint checkpoints/finetuned_model.pt


## 📊 Evaluation
We provide python scripts for evaluation on metrics like:

    Codon Recovery Rate
    
    Codon Similarity Index (CSI)
    
    Codon Frequency Distribution (CFD)
    
    GC content 
    
    MFE energy
    
    %MinMax and DTW score between natural and generated sequences
    
   
## 📄 Citation
If you use this work, please cite:

    @misc{TransCodon2025,
      title={Learning the native-like codons with a 5'UTR and secondary RNA structure aided species-informed transformer model},
      author={Hu et al.},
      year={2025},
      note={Preprint available upon request}
    }

## 📬 Contact
For questions or feedback, feel free to contact:
📧 gu-yuehuo@qq.com

