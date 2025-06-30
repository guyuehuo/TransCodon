
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

    ``` 
    git clone https://github.com/guyuehuo/transcodon.git
    cd transcodon
   ```
   
2. Set up environment
We recommend using conda or [virtualenv].

    ``` 
    conda create -n transcodon python=3.8
    conda activate transcodon
    pip install -r requirements.txt
    ``` 
 
## 🌍 Usage

``` 
python predict.py \
--input input_amino_acid.fa \
--species "Escherichia coli" \
--output output_dna.fasta \
--model_path checkpoints/transcodon.pt
```

## 📊 Evaluation
We provide Jupyter Notebooks for evaluation on metrics like:

    Codon Recovery Rate
    
    Codon Similarity Index (CSI)
    
    Codon Frequency Distribution (CFD)
    
    GC content similarity
    
    MFE and structure comparisons
    
    %MinMax and DTW score between natural and generated sequences
    
    Downstream translation prediction (MRL)

## 📄 Citation
If you use this work, please cite:

    @misc{TransCodon2025,
      title={TransCodon: a species-informed transformer model for cross-species codon optimization},
      author={Hu et al.},
      year={2025},
      note={Preprint available upon request}
    }

## 📬 Contact
For questions or feedback, feel free to contact:
📧 gu-yuehuo@qq.com

