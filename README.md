# Structure-Based Prioritization of QT Liability Using an Ensemble of Graph Neural Network Families

This repository contains the code, processed datasets, trained model files, and selected output files associated with the manuscript:

**Tomoyuki Enokiya, Takamasa Yamaguchi**  
**Structure-Based Prioritization of QT Liability Using an Ensemble of Graph Neural Network Families**

## Purpose of this repository

The purpose of this repository is to provide a reproducible and usable research package for medicinal chemists and computational researchers who wish to:

1. inspect the datasets used in this study,
2. review the training and external evaluation workflow,
3. access trained model files,
4. reproduce the manuscript figures and output tables, and
5. apply the final QT liability prioritization workflow to new compounds.

This repository is intended to support the manuscript submission and revision process for the *European Journal of Medicinal Chemistry*.

---

## What is included in this repository

This repository contains the following major resources:

### 1. Source workflow
- `qt_liability_gnn_workflow.ipynb`  
  Main Jupyter notebook containing the workflow used for data preparation, model training, evaluation, and manuscript output generation.

### 2. Input and processed datasets
Examples of included data files:
- `AID_1671200_datatable.csv`
- `AID_1671200_labeled.csv`
- `AID_1671200_labeled_MAIN.csv`
- `AID_1671200_labeled_MAIN_balanced.csv`
- `AID_1671200_labeled_MIX_POSstrict_NEGbase.csv`
- `AID_1671200_labeled_QC.csv`
- `AID_1671200_labeled_QC_noPAINS.csv`
- `AID_588834_datatable.csv`
- `aid_minimal.csv`
- `external_AID_588834_herg.csv`
- `faers_signal1.csv`
- `merged_for_model_consolidated.csv`
- `merged_for_model_dedup_smiles.csv`
- `merged_for_model_with_label.csv`

These files represent the curated internal and external datasets used for model development and evaluation.

### 3. Graph-ready data objects
- `data_graph_with_smiles.pt`
- `data_graph_with_smiles_index.csv`
- `data_graph_external.pt`
- `data_graph_external_index.csv`

These files contain graph-formatted data objects and index tables used in the graph neural network workflow.

### 4. Trained model files
Examples of trained model files are included in:
- `results_ens_trans_gatv2_5fold/`
- `results_ens_trans_gatv2_scaffold5fold/`
- `results_lco5_trans_gatv2_ens_posaug/`
- `results_lco5_trans_gat_ens_posaug_advanced/`

These directories include fold-specific trained weights for the Transformer- and GATv2-based models used in the manuscript.

### 5. Evaluation outputs and manuscript-related reports
Examples:
- `external_eval_comparison_all.csv`
- `external_eval_ranking.csv`
- `reports_external_confusions_manual/`
- `reports_internal_QT-M2M4-cvpr/`
- `reports_predictions_M2M4/`
- `reports_triage_external_fixed/`

These files summarize internal validation, external evaluation, confusion matrices, ranking outputs, and final triage assignments.

### 6. Figures and interpretation outputs
Examples:
- `figs_m2m4_cvpr_roc/`
- `figs_qt_final_m2m4_cvpr/`
- `figs_shap_pdp_QT-M2M4_cvpr/`
- `figs_confusions_drug/`

These directories contain figure files and supporting outputs corresponding to the manuscript figures and interpretation analyses.

---

## Repository organization

A simplified overview is shown below.

```text
.
├── qt_liability_gnn_workflow.ipynb
├── AID_1671200_datatable.csv
├── AID_1671200_labeled.csv
├── AID_588834_datatable.csv
├── external_AID_588834_herg.csv
├── faers_signal1.csv
├── merged_for_model_consolidated.csv
├── data_graph_with_smiles.pt
├── data_graph_external.pt
├── results_ens_trans_gatv2_5fold/
├── results_ens_trans_gatv2_scaffold5fold/
├── results_lco5_trans_gatv2_ens_posaug/
├── results_lco5_trans_gat_ens_posaug_advanced/
├── reports_external_confusions_manual/
├── reports_internal_QT-M2M4-cvpr/
├── reports_predictions_M2M4/
├── reports_triage_external_fixed/
├── figs_m2m4_cvpr_roc/
├── figs_qt_final_m2m4_cvpr/
└── figs_shap_pdp_QT-M2M4_cvpr/
```

## Software requirements

### Tested environment
- Python: 3.10.12

### Core Python packages
- numpy
- pandas
- scikit-learn
- matplotlib
- rdkit
- torch
- torch-geometric
- jupyter

## Installation
Create a clean Python environment and install the required packages:
pip install -r requirements.txt

## How to use this repository
Open the main notebook and run the workflow step by step:
jupyter notebook qt_liability_gnn_workflow.ipynb

## Datasets and redistribution
This repository contains processed datasets and source-derived files used in the manuscript.
Some original data sources may have their own redistribution policies.
Details of dataset curation and provenance are described in the manuscript and supplementary methods.

## Training, testing, and external evaluation
This repository includes outputs for internal validation and external evaluation.
Relevant files and directories include:
- reports_internal_QT-M2M4-cvpr/
- reports_external_confusions_manual/
- reports_triage_external_fixed/

## Intended use
This repository is intended for research and reproducibility purposes only.
It is not intended for clinical decision-making.
