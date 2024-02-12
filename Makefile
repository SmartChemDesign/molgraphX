.PHONY: *

VENV=venv
PYTHON=$(VENV)/bin/python3
DEVICE=gpu
DATASET_FOLDER=Data
OUTPUT_FOLDER=Output

# ================== WORKSPACE SETUP ==================

venv:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	@echo 'Path to Python executable $(shell pwd)/$(PYTHON)'

install_gpu_specific_dependencies:
	@echo "=== Installing gpu-specific dependencies ==="
	$(PYTHON) -m pip install torch==2.1.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	$(PYTHON) -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
	$(PYTHON) -m pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html


install_cpu_specific_dependencies:
	@echo "=== Installing cpu-specific dependencies ==="
	$(PYTHON) -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
	$(PYTHON) -m pip install  dgl -f https://data.dgl.ai/wheels/repo.html


install_all:venv
	case "$(DEVICE)" in \
		"gpu") \
			make install_gpu_specific_dependencies;; \
		"cpu") \
			make install_cpu_specific_dependencies;; \
		*) \
			echo "The value of the DEVICE variable should be one of: 'cpu', 'gpu'";; \
	esac

	$(PYTHON) -m pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
	$(PYTHON) -m pip install -U tensorboard
	$(PYTHON) -m pip install -U tensorboardX
	$(PYTHON) -m pip install scikit-learn matplotlib argparse logging
	$(PYTHON) -m pip install rdkit-pypi
	$(PYTHON) -m pip install pytorch-lightning torch_geometric dgllife==0.3.2
	$(PYTHON) -m pip install optuna
	$(PYTHON) -m pip install pyarrow
	$(PYTHON) -m pip install IPython jupyter
	$(PYTHON) -m ipykernel install --user --name=molgraphx


download_dataset:
	mkdir -p $(DATASET_FOLDER)
	wget "https://drive.google.com/u/3/uc?id=1etQ44UTpzFOyVu9zzpkC5kwble0igGu0&export=download&confirm=yes" -O $(DATASET_FOLDER)/Data.zip
	unzip $(DATASET_FOLDER)/Data.zip -d $(DATASET_FOLDER)
	rm $(DATASET_FOLDER)/Data.zip

# ========================= TRAINING ========================

optimize_hparams:
	export PATH="$PATH:$(pwd)"
	$(PYTHON) -m Experiments.optimize_hparams \
		--data $(DATASET_FOLDER)/qm9.csv \
		--target-name "mu" \
		--output-folder "$(OUTPUT_FOLDER)/optuna" \
		--n-trials 100 \
		--batch-size 64 \
		--epochs 1000 \
		--es-patience 50 \
		--seed 42


run_training:
	export PATH="$PATH:$(pwd)"
	$(PYTHON) -m Experiments.train \
		--data $(DATASET_FOLDER)/qm9.csv \
		--target-name "mu" \
		--output-folder "$(OUTPUT_FOLDER)/trained_model" \
		--folds 5 \
		--epochs 1000 \
		--es-patience 100 \
		--batch-size 64 \
		--learning-rate 0.00025606270913924607 \
		--seed 23

# ========================= USAGE ========================

predict:
	export PATH="$PATH:$(pwd)"
	$(PYTHON) -m Experiments.save_predictions \
		--data $(DATASET_FOLDER)/qm9.csv \
		--model-folder "$(OUTPUT_FOLDER)/trained_model" \
		--max-samples 100 \
		--output-file "$(OUTPUT_FOLDER)/predictions.csv"

test_explainers:
	export PATH="$PATH:$(pwd)"

	# ======================================
	# === Generate subgraphX explanation ===
	# ======================================

	$(PYTHON) -m Experiments.demonstrations.subgraphX \
		--smiles "CCC(=O)" \
		--model-folder "$(OUTPUT_FOLDER)/trained_model" \
		--output-file "$(OUTPUT_FOLDER)/subgraphX_explanation.png"

	# =========================================
	# === Generate submoleculeX explanation ===
	# =========================================

	$(PYTHON) -m Experiments.demonstrations.submoleculeX \
		--smiles "CCC(=O)" \
		--model-folder "$(OUTPUT_FOLDER)/trained_model" \
		--output-file "$(OUTPUT_FOLDER)/submoleculeX_explanation.png"

	# =====================================
	# === Generate molgraph explanation ===
	# =====================================

	$(PYTHON) -m Experiments.demonstrations.molgraph \
		--smiles "CCC(=O)" \
		--model-folder "$(OUTPUT_FOLDER)/trained_model" \
		--output-file "$(OUTPUT_FOLDER)/molgraph_explanation.png"

	# ======================================
	# === Generate molgraphX explanation ===
	# ======================================

	$(PYTHON) -m Experiments.demonstrations.molgraphX \
		--smiles "CCC(=O)" \
		--model-folder "$(OUTPUT_FOLDER)/trained_model" \
		--output-file "$(OUTPUT_FOLDER)/molgraphX_explanation.png"

comp_time:
	$(PYTHON) -m Experiments.calculate_computational_time \
		--model-folder "$(OUTPUT_FOLDER)/trained_model" \
		--data $(DATASET_FOLDER)/comp_time_data.csv \
		--output-file "$(OUTPUT_FOLDER)/computational_time.csv"
