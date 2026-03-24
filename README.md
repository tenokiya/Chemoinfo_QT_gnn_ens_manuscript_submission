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
- `external_graph_strict.csv`

These files contain curated source-derived tables and graph index files used in model development, original external assembly, and strict external evaluation.

### 3. Graph-ready data objects

- `data_graph_with_smiles.pt`
- `data_graph_external.pt`
- `data_graph_external_strict.pt`

These files contain graph-formatted data objects used by the graph-model workflow and the revised strict external analyses.

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

### 6. Strict external evaluation outputs

Representative strict external outputs currently include:

- `strict_external_best_predictions.csv`
- `strict_external_best_metrics.json`
- `strict_external_best_confusion_counts.csv`
- `strict_external_eval_comparison_all.csv`
- `strict_external_eval_ranking.csv`
- `reports_triage_external_strict_fixed/`

These files summarize the revised strict external evaluation and the fixed-threshold three-level triage used in the manuscript revision.

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
├── data_graph_external_strict.pt
├── external_graph_strict.csv
├── strict_external_best_predictions.csv
├── strict_external_best_metrics.json
├── strict_external_best_confusion_counts.csv
├── strict_external_eval_comparison_all.csv
├── strict_external_eval_ranking.csv
├── reports_internal_QT-M2M4-cvpr/
├── reports_triage_external_strict_fixed/
├── results_ens_trans_gatv2_scaffold5fold/
├── results_lco5_trans_gat_ens_posaug_advanced/
├── figs_m2m4_cvpr_roc/
├── figs_qt_final_m2m4_cvpr/
├── scaler_g.joblib
├── scaler_r.joblib
└── scaler_meta.json
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

This script is used to:

- reconstruct the strict external subset,
- create strict external graph objects,
- evaluate candidate ensemble combinations,
- summarize the best-performing ensemble on the strict external set, and
- generate revised strict external ROC/PR/confusion outputs.

### Workflow 3. Split and similarity audit for Comment 5

Run:

```bash
python comment5_split_similarity_analysis.py
```

This script is used to:

- reconstruct M2 and M4 split memberships,
- summarize overlap relationships between internal and external sets,
- compute nearest-neighbor Tanimoto similarity summaries,
- summarize scaffold-level relatedness, and
- generate outputs used to support the revised response to the editor's comment on set assembly and similarity.

### Workflow 4. Fixed-threshold triage on the strict external set

Run:

```bash
python triage_external_strict_sim_fixed_thresholds.py
```

This script is used to:

- apply internally selected fixed thresholds to the strict external predictions,
- assign compounds to GREEN / YELLOW / RED triage categories, and
- summarize three-level prioritization results for the revised manuscript.

---

## File-to-manuscript mapping

The repository files broadly map to the manuscript as follows:

- **Main workflow notebook and trained weights**  
  Support the model-development and evaluation procedures described in the Methods.

- **`reports_internal_QT-M2M4-cvpr/`**  
  Supports the internal CID-level QT-M2M4 results described in the revised manuscript.

- **Strict external outputs (`strict_external_best_*.csv/json`, `strict_external_eval_*.csv`)**  
  Support the strict external binary evaluation and ensemble-comparison results.

- **`reports_triage_external_strict_fixed/`**  
  Supports the strict external three-level triage analyses.

- **Figure directories**  
  Support manuscript figure preparation and related reporting outputs.

---

## Demo vs full reproducibility package

A public interactive demonstration is available separately via Hugging Face Spaces. That demo is intended for rapid qualitative inspection of the model behavior.

This GitHub repository serves a different role: it provides the manuscript-linked reproducibility resources, including workflow code, processed datasets, trained weights, revision scripts, and representative strict external outputs.


## Apply QT-M2M4 to your own compounds

A lightweight public demo is available on Hugging Face Spaces. For local inference with the repository files, use `predict_qt_liability.py`.

### Input format
Prepare a CSV with a SMILES column. The script auto-detects common column names such as `SMILES`, `smiles`, and `SMILES_ISO`. An optional identifier column can be provided using names such as `compound_id`, `id`, `name`, or `drug`.

Example input:

```csv
compound_id,SMILES
Ajmaline,CC[C@H]1[C@@H]2C[C@H]3[C@H]4[C@@]5(C[C@@H]([C@H]2[C@H]5O)N3[C@@H]1O)C6=CC=CC=C6N4C
Azimilide,CN1CCN(CC1)CCCCN2C(=O)CN(C2=O)/N=C/C3=CC=C(O3)C4=CC=C(C=C4)Cl
```

### Command line

```bash
python predict_qt_liability.py \
  --input example_smiles.csv \
  --output example_predictions.csv \
  --root .
```

### Output columns
- `compound_id`: compound identifier
- `smiles_input`: SMILES supplied by the user
- `smiles_canonical`: RDKit-canonicalized SMILES used for graph construction
- `valid_rdkit`: whether the compound was successfully parsed by RDKit
- `p_m2`: probability from the M2 family
- `p_m4`: probability from the M4 family
- `p_ens`: final QT-M2M4 ensemble probability
- `binary_call_055`: binary decision using the manuscript threshold of 0.55
- `triage_label`: `GREEN`, `YELLOW`, or `RED` using default thresholds of 0.44 and 0.75
- `note`: parsing or graph-construction note, if applicable

### Thresholds
By default, the script uses the manuscript thresholds:
- binary threshold: `0.55`
- GREEN boundary: `0.44`
- RED boundary: `0.75`

These can be overridden from the command line when needed.
---

## Regenerating model-ready dataset files from public-source assay tables

The repository already includes the canonical processed files used for the manuscript (`merged_for_model_with_label.csv`, `merged_for_model_consolidated.csv`, `data_graph_with_smiles_index.csv`, `data_graph_with_smiles.pt`, `data_graph_external_index.csv`, and `data_graph_external.pt`).

For transparency, the repository also includes `prepare_model_ready_datasets.py`, which exposes the main dataset-formatting steps as command-line tasks.

### 1) Create labeled AID 1671200 internal assay tables

This step converts the raw PubChem AID 1671200 table into labeled AID-only tables, including the mixed AID table used before auxiliary positives are merged in.

```bash
python prepare_model_ready_datasets.py label-aid1671200 \
  --input AID_1671200_datatable.csv \
  --outdir .
```

Representative outputs:
- `AID_1671200_labeled.csv`
- `AID_1671200_labeled_QC.csv`
- `AID_1671200_labeled_QC_noPAINS.csv`
- `AID_1671200_labeled_MIX_POSstrict_NEGbase.csv`
- `AID_1671200_label_audit.json`

### 2) Merge the AID-only internal table with an auxiliary positive table

The manuscript internal development table combines the AID-derived set with an auxiliary positive table. The helper script accepts an auxiliary CSV that contains at least a SMILES-like column and, when available, CID / signal / source columns.

```bash
python prepare_model_ready_datasets.py merge-internal \
  --aid_mix AID_1671200_labeled_MIX_POSstrict_NEGbase.csv \
  --aux_positive_csv aux_positive_table.csv \
  --outdir .
```

Representative outputs:
- `aid_minimal.csv`
- `aux_positive_minimal.csv`
- `merged_for_model_with_label.csv`
- `merged_for_model_consolidated.csv`
- `merged_for_model_audit.json`

### 3) Convert the consolidated internal table into a graph-index CSV

This step creates the model-ready internal graph index used by the training / evaluation workflow. Optional SMILES augmentation for positives is exposed through `--pos_aug`.

```bash
python prepare_model_ready_datasets.py build-graph-index \
  --input merged_for_model_consolidated.csv \
  --output_index data_graph_with_smiles_index.csv
```

To also write a PyTorch Geometric object list:

```bash
python prepare_model_ready_datasets.py build-graph-index \
  --input merged_for_model_consolidated.csv \
  --output_index data_graph_with_smiles_index.csv \
  --write_pt data_graph_with_smiles.pt \
  --root .
```

### 4) Convert the raw PubChem AID 588834 table into the original external graph-index files

```bash
python prepare_model_ready_datasets.py build-external \
  --input AID_588834_datatable.csv \
  --outdir .
```

Representative outputs:
- `AID_588834_labeled.csv`
- `data_graph_external_index.csv`
- `AID_588834_label_audit.json`

To also write the external PyTorch Geometric object list:

```bash
python prepare_model_ready_datasets.py build-external \
  --input AID_588834_datatable.csv \
  --outdir . \
  --write_pt data_graph_external.pt \
  --root .
```

### 5) Build the strict compound-disjoint external subset used in the revised manuscript

After the original external graph-index file is available, the revised strict-external evaluation pipeline can be run with:

```bash
python strict_external_eval_full_thr055_fig2layout.py
```

This script constructs the strict external subset by excluding compounds that overlap the internal dataset at the CID or canonical-SMILES level, then evaluates the model families and the final QT-M2M4 ensemble on that strict subset.

### Practical note

The command-line helper is provided to make the data-formatting steps more explicit for readers. The deposited processed files remain the canonical review assets for the manuscript package.


## Citation

Associated manuscript currently under revision at the **European Journal of Medicinal Chemistry**.

If this repository is updated after acceptance, citation details will be revised accordingly.

---

## Contact

For correspondence regarding the manuscript or repository contents, please contact the corresponding author listed in the manuscript materials.
