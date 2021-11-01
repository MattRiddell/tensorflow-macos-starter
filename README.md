# tensorflow-macos-starter

Instructions from: <https://developer.apple.com/metal/tensorflow-plugin/>

```zsh
brew link --force --overwrite python@3.8
```

## If miniforge not installed

Make sure that Python is version 3.7+

```zsh
python -m pip install -U pip
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
conda install -c apple tensorflow-deps
python -m pip uninstall tensorflow-macos
python -m pip uninstall tensorflow-metal
conda install -c apple tensorflow-deps --force-reinstall
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
```

## If miniforge installed

```zsh
source ~/miniforge3/bin/activate
conda install -c apple tensorflow-deps --force-reinstall
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
```
