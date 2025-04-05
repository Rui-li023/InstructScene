
# Scene

## Installation

You may need to modify the specific version of `torch` in `settings/setup.sh` according to your CUDA version.
There are not restrictions on the `torch` version, feel free to use your preferred one.
```bash
bash settings/setup.sh
```

Download the Blender software for visualization.
```bash
cd blender
wget https://download.blender.org/release/Blender3.3/blender-3.3.1-linux-x64.tar.xz
tar -xvf blender-3.3.1-linux-x64.tar.xz
rm blender-3.3.1-linux-x64.tar.xz
```


## Dataset

Dataset used in Scene is based on [3D-FORNT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) and [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future).
Please refer to the instructions provided in their [official website](https://tianchi.aliyun.com/dataset/65347) to download the original dataset.
One can refer to the dataset preprocessing scripts in [ATISS](https://github.com/nv-tlabs/ATISS?tab=readme-ov-file#dataset) and [DiffuScene](https://github.com/tangjiapeng/DiffuScene?tab=readme-ov-file#dataset), which are similar to ours.

We provide the preprocessed instruction-scene paired dataset used in the paper and rendered images for evaluation on [HuggingFace](https://huggingface.co/datasets/chenguolin/InstructScene_dataset).
```python
import os
from huggingface_hub import hf_hub_url
url = hf_hub_url(repo_id="chenguolin/InstructScene_dataset", filename="InstructScene.zip", repo_type="dataset")
os.system(f"wget {url} && unzip InstructScene.zip")
url = hf_hub_url(repo_id="chenguolin/InstructScene_dataset", filename="3D-FRONT.zip", repo_type="dataset")
os.system(f"wget {url} && unzip 3D-FRONT.zip")
```

Please refer to [dataset/README.md](./dataset/README.md) for more details.


## Visualization

We provide a helpful script to visualize synthesized scenes by [Blender](https://www.blender.org/).
Please refer to [blender/README.md](./blender/README.md) for more details.

We also provide many useful visualization functions in [src/utils/visualize.py](./src/utils/visualize.py), including creating appropriate floor plans, drawing scene graphs, adding instructions as titles in the rendered images, making gifs, etc.



```python
python src/train_fl_sg2lt.py /home/ubuntu/Documents/ACDC_project/InstructScene/configs/bedroom_sgfl2lt_diffusion_objfeat.yaml --fvqvae_tag threedfront_objfeat_vqvae

python src/generate_sg2sc.py ./configs/bedroom_sgfl2lt_diffusion_objfeat.yaml --fvqvae_tag threedfront_objfeat_vqvae --checkpoint_epoch 249 --cfg_scale 1 --n_scenes 0 --tag 2025-03-06_18:57 --visualize

python3 src/compute_fid_scores.py configs/bedroom_sgfl2lt_diffusion_objfeat.yaml --tag 2025-03-06_18:57 --checkpoint_epoch 249
```