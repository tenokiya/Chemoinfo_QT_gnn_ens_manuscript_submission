# Structure-Based Prioritization of QT Liability for Early Medicinal Chemistry Using an Ensemble Molecular Graph Model

This repository contains the code, processed datasets, trained model files, revision scripts, and selected output files associated with the manuscript:

**Tomoyuki Enokiya, Takamasa Yamaguchi**  
**Structure-Based Prioritization of QT Liability for Early Medicinal Chemistry Using an Ensemble Molecular Graph Model**

## Overview

Drug-induced QT prolongation is an important safety and developability liability in medicinal chemistry. This repository supports reproducibility and practical inspection of a structure-based workflow for early QT-liability prioritization using an ensemble molecular graph model.

The repository is intended to help readers:

1. inspect the curated internal and external datasets used in model development and evaluation;
2. review the notebook-based graph-model workflow used for dataset preparation, model training, and manuscript-related analyses;
3. access trained model weights and representative output files;
4. reproduce strict external evaluation, split/similarity analyses, and fixed-threshold three-level triage; and
5. understand how the repository maps to the manuscript, supplementary methods, and revision package.

This repository is provided for research and reproducibility purposes. It is **not** intended for clinical decision-making.

---

## Scope of the current revision package

The current manuscript revision emphasizes **early medicinal-chemistry decision support**, **compound-level evaluation**, and **strict external validation**.

In the revised workflow:

- internal performance is summarized at the **PubChem CID level** to better reflect compound-level screening use;
- the main external analysis is based on a **strict compound-disjoint subset** derived from PubChem BioAssay **AID 588834** after excluding compounds overlapping the internal dataset by **CID or canonical SMILES**;
- the final ensemble model is **QT-M2M4**, a cv_pr-weighted combination of the scaffold-aware **M2** family and the stricter cluster-aware **M4** family;
- a binary operating threshold fixed from internal CID-level analysis (**0.55**) is used for the main strict external binary evaluation; and
- two internally fixed thresholds (approximately **0.44** and **0.75**) are used for three-level triage (**GREEN / YELLOW / RED**).

For the revised manuscript, the reported strict external performance of QT-M2M4 is **ROC-AUC 0.80** and **PR-AUC 0.51**.

---

## Quick start

### Option A. Inspect the main end-to-end notebook

```bash
pip install -r requirements.txt
jupyter notebook qt_liability_gnn_workflow_v2_clean.ipynb
```

Use this notebook to inspect the main workflow for dataset preparation, graph construction, model training, internal evaluation, and manuscript-related output generation.

### Option B. Reproduce the strict external analyses used in the revision

Run the strict external evaluation script:

```bash
python strict_external_eval_full_thr055_fig2layout.py
```

Run the split/similarity audit used to address the editor's comment on set assembly and relatedness:

```bash
python comment5_split_similarity_analysis.py
```

Run the fixed-threshold triage analysis on the strict external set:

```bash
python triage_external_strict_sim_fixed_thresholds.py
```

These scripts were prepared to support the revised European Journal of Medicinal Chemistry submission and should be interpreted together with the manuscript and supplementary methods.

---

## What is included in this repository

### 1. Main workflow notebook

- `qt_liability_gnn_workflow_v2_clean.ipynb`  
  Main Jupyter notebook for data preparation, graph construction, model training, evaluation, and manuscript-related outputs.

### 2. Key tabular datasets

Representative tabular files currently included in this repository:

- `AID_1671200_datatable.csv`
- `AID_588834_datatable.csv`
- `merged_for_model_consolidated.csv`
- `merged_for_model_with_label.csv`
- `data_graph_with_smiles_index.csv`
- `data_graph_external_index.csv`

These files contain curated source-derived tables and graph index files used in model development and external evaluation.

### 3. Graph-ready data objects

- `data_graph_with_smiles.pt`
- `data_graph_external.pt`

These files contain graph-formatted data objects used by the graph-model workflow.

### 4. Revision scripts for the current EJMC resubmission

- `comment5_split_similarity_analysis.py`  
  Split-audit and chemical-relatedness script used to summarize dataset assembly, nearest-neighbor Tanimoto similarity, overlap checks, and scaffold-level relatedness for the revised submission.

- `strict_external_eval_full_thr055_fig2layout.py`  
  Main strict external evaluation script used to construct the strict external subset, evaluate candidate ensembles, and generate the strict external ROC/PR/confusion outputs aligned with the revised manuscript.

- `triage_external_strict_sim_fixed_thresholds.py`  
  Fixed-threshold triage script used to apply the internally selected GREEN/RED boundaries to the strict external predictions.

### 5. Trained model files

Representative trained weights are currently provided in the following directories:

- `results_ens_trans_gatv2_scaffold5fold/`
- `results_lco5_trans_gat_ens_posaug_advanced/`

These directories contain fold-specific trained weights for Transformer- and GATv2-based model families used in the manuscript.

### 6. Evaluation reports

Representative report directories currently include:

- `reports_internal_QT-M2M4-cvpr/`
- `reports_external_confusions_manual/`

The internal report directory contains manuscript-related CID-level evaluation outputs for QT-M2M4. Some external report directories may contain **legacy outputs generated before the strict compound-disjoint external filtering used in the revised manuscript**. For the revised strict external analyses, the primary reference should be the outputs generated by `strict_external_eval_full_thr055_fig2layout.py` and `triage_external_strict_sim_fixed_thresholds.py`.

### 7. Figure outputs

Representative figure directories currently include:

- `figs_m2m4_cvpr_roc/`
- `figs_qt_final_m2m4_cvpr/`

These directories contain figure files and related outputs associated with the manuscript analyses.

### 8. Supporting preprocessing objects

- `scaler_g.joblib`
- `scaler_r.joblib`
- `scaler_meta.json`

These files contain preprocessing objects and metadata used in the workflow.

---

## Repository organization

A simplified overview is shown below.

```text
.
├── qt_liability_gnn_workflow_v2_clean.ipynb
├── comment5_split_similarity_analysis.py
├── strict_external_eval_full_thr055_fig2layout.py
├── triage_external_strict_sim_fixed_thresholds.py
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

---

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

A basic package list is provided in `requirements.txt`.

### Installation

Create a clean Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

### Practical note on local installation

The workflow was developed primarily in a notebook-oriented environment. Depending on platform, local installation of **RDKit** and **PyTorch Geometric** may require environment-specific adjustment beyond a simple `pip install -r requirements.txt`.

---

## Datasets and redistribution

This repository contains processed datasets and source-derived files used in the manuscript. Some original data sources may have their own redistribution policies.

The revised submission distinguishes the following dataset roles:

- **Internal development dataset**: integrated from PubChem BioAssay **AID 1671200** and FAERS-derived QT-risk signals.
- **Original external dataset**: derived from PubChem BioAssay **AID 588834** using the same parent-structure and graph-construction logic.
- **Strict external subset**: a compound-disjoint subset of the original external dataset created by excluding compounds overlapping the internal dataset by **CID or canonical SMILES**.

Details of dataset curation, label definition, parent-structure consolidation, redistribution context, and strict external subset construction are described in the manuscript and supplementary methods.

---

## Main workflows in this repository

### Workflow 1. Main notebook-based model-development workflow

Open:

```bash
jupyter notebook qt_liability_gnn_workflow_v2_clean.ipynb
```

This notebook contains the main procedures for:

- loading curated datasets,
- constructing graph-based inputs,
- training and evaluating model families,
- generating manuscript-related outputs, and
- reviewing representative internal and external evaluation results.

### Workflow 2. Strict external evaluation used in the revised manuscript

Run:

```bash
python strict_external_eval_full_thr055_fig2layout.py
```

This script is intended to:

- reconstruct the strict external subset from the AID 588834-derived external pool,
- evaluate candidate ensemble combinations on the strict external set,
- summarize strict external ROC-AUC and PR-AUC,
- generate binary evaluation outputs using the internally fixed threshold of **0.55**, and
- support the strict external panels reported in the revised manuscript.

### Workflow 3. Split and similarity audit for editor comment response

Run:

```bash
python comment5_split_similarity_analysis.py
```

This script is intended to:

- summarize M2 and M4 split membership,
- audit overlap by raw SMILES, canonical SMILES, and CID,
- quantify chemical relatedness using nearest-neighbor Tanimoto similarity,
- summarize scaffold-level relatedness, and
- generate supporting outputs for the manuscript revision and supplementary methods.

### Workflow 4. Fixed-threshold three-level triage on the strict external set

Run:

```bash
python triage_external_strict_sim_fixed_thresholds.py
```

This script is intended to:

- read internally selected triage thresholds,
- apply the fixed **GREEN / YELLOW / RED** boundaries to the strict external predictions,
- summarize tier distributions, and
- generate operating summaries under RED-vs-not-RED and not-GREEN rules.

---

## Model-selection and threshold notes

For the revised manuscript, the final ensemble model is **QT-M2M4**, defined as a cv_pr-weighted combination of the **M2** and **M4** scheme-level models.

### Binary threshold used for main strict external evaluation

- **0.55**

This threshold was fixed from the internal CID-level analysis and transferred unchanged to the strict external evaluation.

### Three-level triage thresholds used for strict external triage

- **GREEN boundary**: approximately **0.44**
- **RED boundary**: approximately **0.75**

Operationally:

- `GREEN`: `p_ens < thr_green`
- `YELLOW`: `thr_green ≤ p_ens < thr_red`
- `RED`: `p_ens ≥ thr_red`

These thresholds are intended as **screening-oriented operating boundaries**, not as calibrated absolute risk cutoffs.

---

## File-to-manuscript mapping

The repository contents can be interpreted as follows.

- `qt_liability_gnn_workflow_v2_clean.ipynb`  
  Main notebook-based workflow for dataset preparation, model development, and representative manuscript analyses.

- `reports_internal_QT-M2M4-cvpr/`  
  Internal CID-level QT-M2M4 evaluation outputs.

- `strict_external_eval_full_thr055_fig2layout.py` and its outputs  
  Strict external candidate-ensemble comparison and binary evaluation aligned with the revised external results.

- `triage_external_strict_sim_fixed_thresholds.py` and its outputs  
  Three-level triage analyses aligned with the revised triage tables.

- `comment5_split_similarity_analysis.py` and its outputs  
  Split/similarity documentation used to support the editor response, supplementary methods, and dataset-assembly clarification.

---

## Interactive demonstration

A public interactive demonstration of the final ensemble model is available through Hugging Face Spaces:

- `https://huggingface.co/spaces/Te-nok/QT-M2M4-GNNens-demo`

The Hugging Face Space is intended as a lightweight demonstration interface. The repository and revision package provide the fuller reproducibility context, including processed files, scripts, and manuscript-related outputs.

---

## Intended use

This repository is intended for:

- research reproducibility,
- workflow inspection,
- medicinal-chemistry-oriented hypothesis support, and
- practical review of manuscript-related computational resources.

It is **not** intended for clinical diagnosis, treatment selection, or direct patient-level risk prediction.

---

## Citation

If you use this repository or associated materials, please cite the corresponding manuscript and acknowledge the associated revision package.

**Enokiya T, Yamaguchi T.**  
*Structure-Based Prioritization of QT Liability for Early Medicinal Chemistry Using an Ensemble Molecular Graph Model.*  
Revision package prepared for the *European Journal of Medicinal Chemistry*.

If a final journal citation becomes available, please update this section accordingly.

---

## Contact

**Tomoyuki Enokiya**  
Suzuka University of Medical Science  
Email: tenokiya@suzuka-u.ac.jp
