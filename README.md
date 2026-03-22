# Structure-Based Prioritization of QT Liability Using an Ensemble of Graph Neural Network Families

This repository contains the code, processed datasets, trained model files, and selected output files associated with the manuscript:

**Tomoyuki Enokiya, Takamasa Yamaguchi**  
**Structure-Based Prioritization of QT Liability Using an Ensemble of Graph Neural Network Families**

## Purpose of this repository

This repository is intended to support reproducibility and practical inspection of the workflow used in the manuscript. It provides materials for readers who wish to:

1. inspect the curated datasets used in model development and external evaluation,
2. review the graph-based machine-learning workflow,
3. access trained model files,
4. inspect manuscript-related reports and figure outputs, and
5. reproduce the main notebook-based analysis environment.

This repository supports the manuscript submission and revision process for the *European Journal of Medicinal Chemistry*.

---

## What is included in this repository

### 1. Main workflow notebook
- `qt_liability_gnn_workflow.ipynb`  
  Main Jupyter notebook containing the workflow used for data preparation, model training, evaluation, and manuscript-related output generation.

### 2. Key tabular datasets
Representative data files currently included in this repository:
- `AID_1671200_datatable.csv`
- `AID_588834_datatable.csv`
- `merged_for_model_consolidated.csv`
- `merged_for_model_with_label.csv`
- `data_graph_with_smiles_index.csv`
- `data_graph_external_index.csv`

These files represent curated tabular inputs and graph index files used in model development and external evaluation.

### 3. Graph-ready data objects
- `data_graph_with_smiles.pt`
- `data_graph_external.pt`

These files contain graph-formatted data objects used in the graph neural network workflow.

### 4. Trained model files
Trained model weights are currently provided in the following directories:
- `results_ens_trans_gatv2_scaffold5fold/`
- `results_lco5_trans_gat_ens_posaug_advanced/`

These directories contain fold-specific trained weights for Transformer- and GATv2-based models used in the manuscript.

### 5. Evaluation reports
The repository currently includes the following report directories:
- `reports_external_confusions_manual/`
- `reports_internal_QT-M2M4-cvpr/`

These directories contain manuscript-related evaluation outputs for internal validation and external confusion analysis.

### 6. Figure outputs
The repository currently includes the following figure directories:
- `figs_m2m4_cvpr_roc/`
- `figs_qt_final_m2m4_cvpr/`

These directories contain figure files and related outputs corresponding to the manuscript analyses.

### 7. Supporting preprocessing objects
- `scaler_g.joblib`
- `scaler_r.joblib`
- `scaler_meta.json`

These files contain preprocessing objects and metadata used in the analysis workflow.

---

## Repository organization

A simplified overview is shown below.

```text
.
├── Untitled0_fix_EN.ipynb
├── AID_1671200_datatable.csv
├── AID_588834_datatable.csv
├── merged_for_model_consolidated.csv
├── merged_for_model_with_label.csv
├── data_graph_with_smiles.pt
├── data_graph_with_smiles_index.csv
├── data_graph_external.pt
├── data_graph_external_index.csv
├── scaler_g.joblib
├── scaler_r.joblib
├── scaler_meta.json
├── results_ens_trans_gatv2_scaffold5fold/
├── results_lco5_trans_gat_ens_posaug_advanced/
├── reports_external_confusions_manual/
├── reports_internal_QT-M2M4-cvpr/
├── figs_m2m4_cvpr_roc/
└── figs_qt_final_m2m4_cvpr/
```

## Software requirements

### Tested environment
- Python: 3.10.12

### Core Python packages
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- rdkit
- torch
- torch-geometric
- captum
- shap
- umap-learn
- pubchempy
- Pillow
- requests
- joblib
- jupyter

## Installation

Create a clean Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## How to use this repository

Open the main notebook and run the workflow step by step:

```bash
jupyter notebook Untitled0_fix_EN.ipynb
```

The notebook contains the main procedures for:
- loading curated datasets,
- constructing graph-based inputs,
- training and evaluating models,
- generating manuscript-related outputs, and
- reviewing external evaluation results.

## Datasets and redistribution

This repository contains processed datasets and source-derived files used in the manuscript. Some original data sources may have their own redistribution policies. Details of dataset curation, provenance, and labeling are described in the manuscript and supplementary methods.

## Training, testing, and external evaluation

This repository includes outputs for internal validation and external evaluation. Relevant directories currently include:
- `reports_internal_QT-M2M4-cvpr/`
- `reports_external_confusions_manual/`
- `results_ens_trans_gatv2_scaffold5fold/`
- `results_lco5_trans_gat_ens_posaug_advanced/`

## Intended use

This repository is intended for research and reproducibility purposes only. It is not intended for clinical decision-making.

## Citation

If you use this repository or associated materials, please cite the corresponding manuscript:

**Enokiya T, Yamaguchi T.**  
*Structure-Based Prioritization of QT Liability Using an Ensemble of Graph Neural Network Families.*  
[Journal / status to be updated]

## Contact

**Tomoyuki Enokiya**  
Suzuka University of Medical Science  
Email: tenokiya@suzuka-u.ac.jp
