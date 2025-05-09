#  <p align="center">Visually Impaired People Recognize Virtual Objects Through 3D Reconstructed Shapes</p>

 <p align="center">Dapeng Chen, Hao Wu, Chenkai Li, Lina Wei, Guangzhu Peng, Xuhui Hu, Jia Liu, and Aiguo Song</p>
  <p align="center">Nanjing University of Information Science & Technology</p>

## <p align="center">ABSTRACT</p>
With the improvement of three-dimensional (3D) reconstruction technology, virtual objects with realistic shapes have opened up the possibility for the blind or visually impaired (BVI) to improve the cognitive ability of objects by perceiving a large number of objects. In order to help BVI learn and recognize objects more conveniently, we constructed a virtual object shape recognition system (VOSRS) based on 3D reconstruction. We first proposed an improved 3D Gaussian Splatting (3DGS) method for precise 3D reconstruction of objects. Then, we introduced a mesh extraction method for 3D Gaussian to obtain a smooth 3D mesh model suitable for touch perception. Finally, we performed haptic rendering on the 3D mesh model and built an interactive system. We introduced the object 3D reconstruction process and mesh extraction steps in detail, and investigated the effect of object shape cognitive learning using the constructed system through user experiments. We conducted several experiments on the NeRF Synthetic dataset and the DTU dataset. The experimental results show that the improved 3DGS method can effectively reduce spikes and artifacts, speed up the iteration process, and improve the reconstruction quality of objects. The proposed mesh extraction method generates a smooth mesh model without holes, making it more suitable for the needs of shape perception through continuous touch. The constructed VOSRS can enable BVIs to learn and recognize the shape characteristics of virtual objects through force feedback during continuous touch.

## <p align="center">THE FRAMEWORK OF VOSRS</p>
The system starts with a set of multi-view images of the object captured by a camera. We add Gaussian kernel scale constraint loss and mask constraint loss to jointly optimize the 3D Gaussian and use the Gaussian rasterizer to render the 3D object. We perform mesh extraction and smoothing operations on the rendering results to obtain a mesh model of the 3D object. Finally, we perform haptic rendering on the 3D mesh model to obtain the final 3D virtual object that can be used for touch perception by BVI.

![framework](https://github.com/CdpLab/VOSRS/blob/main/assets/framework.jpg)

## Install
Requires: Linux cuda11.8 python3.10.6

Install diff-gaussian-rasterization
```bash
python submodules/diff-gaussian-rasterization/setup.py install
```
Install simple-knn
```bash
python submodules/simple-knn/setup.py install
```
Install other dependencies
```bash
pip install requirements.txt
```

## Run
You can download the chair and lego datasets [here](https://drive.google.com/drive/folders/149zKbdQQ_LaVWwIRcdXLqpJpWJyS00RC?usp=sharing).
You must ensure that the dataset directory structure is as follows:
```bash
-data
  --nerf_synthetic_chair
    ---images
      ----r_0.png
      ----r_1.png
      ...
  --nerf_synthetic_lego
    ---images
      ----r_0.png
      ----r_1.png
      ...
```
### 3D Rendering
If you want to render the chair dataset:
```bash
python convert.py -s data/nerf_synthetic_chair
python gaussian_optimization.py -s data/nerf_synthetic_chair -m data/nerf_synthetic_chair/output
```
If you want to render the lego dataset:
```bash
python convert.py -s data/nerf_synthetic_lego
python gaussian_optimization.py -s data/nerf_synthetic_lego -m data/nerf_synthetic_lego/output
```
### Mesh Extraction
Extract the mesh from the chair rendering:
```bash
python mesh_extra/scripts/extract_mesh.py -m data/nerf_synthetic_chair/output -o data/nerf_synthetic_chair
```
Extract the mesh from the lego rendering:
```bash
python mesh_extra/scripts/extract_mesh.py -m data/nerf_synthetic_lego/output -o data/nerf_synthetic_lego
```
## Results
The operation results of each stage are as follows:
![results](https://github.com/CdpLab/VOSRS/blob/main/assets/result.jpg)
