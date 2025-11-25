# Aura
Aura is a simple ray marcher built in Python, with a focus on simulating gravitational lensing around black holes. The project uses GPU-acceleration using the CuPy library in Python.

> **Note:** This project is a work in progress and currently unfinished.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements) 
- [License](#license)

## Requirements
- Windows/Linux
- NVIDIA GPU
> CuPy might work with AMD GPUs but I have not tested the program in non-NVIDIA systems.

## Installation
```bash
git clone https://github.com/Connor-Levi/aura.git
cd aura
pip install -r requirements.txt
```

## Usage
**Linux:**
```bash
python3 main.py
```

**Windows**
```bash
python main.py
```

All the parameters in `main.py` can be modified (and new parameters added).

## Roadmap
Currently the raymarcher supports the rendering of distortion by a pseudo-blackhole, which is a result of post-processing. 
To properly simulate gravitational lensing, I intend to solve the equations of trajectory of light for MP blackholes using GPU acceleration.

I also plan to add textures and reflections based on the materials of the objects.

Further versions may include the use of CUDA.

## Acknowledgements
This project has been heavily inspired by [Steve Trettel](https://stevejtrettel.site). Light interaction and SDF logic was adapted from one of his talks, [The Mathematics of Beautiful Graphics](https://stevejtrettel.site/talk/math-graphics/).

## License
Licensed under the [GNU General Public License](LICENSE).

