<br />

<p align="center">
  <h2 align="center"><strong>Enhancing Low-light Images Using Infrared Encoded Images</strong></h2>


  <p align="center">
      <a href="https://scholar.google.com/citations?user=8COQQ8QAAAAJ&hl=en&oi=sra" target='_blank'>Shulin Tian*</a>,&nbsp;
      <a href="https://scholar.google.com/citations?hl=en&user=jLd1l_sAAAAJ" target='_blank'>Yufei Wang*</a>,&nbsp;
      <a href="https://scholar.google.com/citations?user=S8_ES4MAAAAJ&hl=zh-CN" target='_blank'>Renjie Wan</a>,&nbsp;
      <a href="https://scholar.google.com/citations?user=S8nAnakAAAAJ&hl=zh-CN" target='_blank'>Wenhan Yang</a>,&nbsp;
      <a href="https://scholar.google.com/citations?user=UGZXLxIAAAAJ&hl=en" target='_blank'>Alex C. Kot</a>,&nbsp;
      <a href="https://scholar.google.com/citations?user=ypkClpwAAAAJ&hl=en" target='_blank'>Bihan Wen</a>,&nbsp;
    <br>
  Nanyang Technological University, Hong Kong Baptist University, Peng Cheng Laboratory
  </p>
  
  <p align="center">
    <a href="https://arxiv.org/pdf/2307.04122.pdf">Paper</a> | <a href="https://wyf0912.github.io/ELIEI/ ">Project Page</a>
  </p>
</p>

<p align="center">
<img src="figs/IR_scene.jpg" width="60%">

The visibility of low-light images is enhanced by increasing the number of income photons (The right sides of (a) and (b) are amplified by a factor of 3.5 for better visualization).

</p>

## Dataset
In this work, we are using a resized version - [IR-RGB-resize [Google Drive]](https://drive.google.com/drive/folders/1SOKXNn1uirRSDGOG5GnmllIXXgd1m1gT?usp=sharing) for our experiments. The file structure is constructed as follows:

```
data_root # The paths need to be specified in the training configs under folder `./code/confs/xx.yml`
└── train/
    ├── high/  
    └── low/
└── eval/
    ├── high/
    ├── low/
    └── low-rgb/
```

<br>

We also relased the original size of images for broadening research purposes [IR-RGB [Google Drive]](https://drive.google.com/drive/folders/1YXizC5-I7gpr4EkIxHhxbEdt_-SqLAQJ?usp=sharing), feel free to download and explore!


## Get Started
### Dependencies and Installation
- Python 3.8
- Pytorch 1.9

1. Clone Repo
```
git clone https://github.com/shulin16/ELIEI.git
```

conda env create -f environment.yml


2. Create Conda Environment
```
conda create --name ELIEI python=3.8
conda activate ELIEI
```
3. Install Dependencies
```
cd ELIEI
pip install -r requirements.txt
```


## Citation
If you find our work useful for your research, please cite our paper
```
@inproceedings{tian2023enhancing,
  title={Enhancing Low-Light Images Using Infrared Encoded Images},
  author={Tian, Shulin and Wang, Yufei and Wan, Renjie and Yang, Wenhan and Kot, Alex C and Wen, Bihan},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={465--469},
  year={2023},
  organization={IEEE}
}
```

### Acknowledgment
This work was done at Rapid-Rich Object Search (ROSE) Lab, Nanyang
Technological University. This research is supported in part by the NTU-PKU Joint Research Institute (a collaboration between the Nanyang Technological University and Peking University that is sponsored by a donation from the Ng Teng Fong Charitable Foundation), the Basic and Frontier Research Project of PCL, the Major Key Project of PCL, and the MOE AcRF
Tier 1 (RG61/22) and Start-Up Grant.
