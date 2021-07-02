W=20
H=20
view=0
runs=50000
discount=0.95
load=0

all: train

train:
	python3 train.py --width=$(W) --height=$(H) --view_placement=$(view) --numofgames=$(runs) --discount=$(discount) --load_model=$(load)

test:
	python3 test.py

gen:
	python3 generate.py --grid_size=$(W) --train_size=$(runs)

clean:
	rm -rf images/*
	rm -rf train_data/*
	rm -rf test_data/*
	rm -rf results
	rm -rf __pycache__

