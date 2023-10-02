# Local Multi-scale Feature Aggregation Network for Real-time Image Dehazing
![visitors](https://visitor-badge.laobi.icu/badge?page_id=Liuy1996/LMFA-Net) 
[![GitHub Stars](https://img.shields.io/github/stars/Liuy1996/LMFA-Net?style=social)](https://github.com/Liuy1996/LMFA-Net/)

[Paper](https://www.sciencedirect.com/science/article/abs/pii/S003132032300300X) | [BibTeX](#bibtex) 

>Haze causes visual degradation and obscures image information, which gravely affects the reliability of computer vision tasks in real-time systems.
Leveraging an enormous number of learning parameters as the restoration costs, learning-based methods have gained significant success, but they are runtime intensive or memory inefficient.
In this paper, we propose a local multi-scale feature aggregation network, called LMFA-Net, which has a lightweight model structure and can be used for real-time dehazing.
By learning the local mapping relationship between the clean value of a haze image at a certain point and its surrounding local region, LMFA-Net can directly restore the final haze-free image. In particular, we adopt a novel multi-scale feature extraction sub-network (M-Net) to extract features from different scales.
As a lightweight network, LMFA-Net can achieve fast and efficient dehazing.
Extensive experiments demonstrate that our proposed LMFA-Net surpasses previous state-of-the-art lightweight dehazing methods in both quantitatively and qualitatively.


## BibTeX
If you find this project useful for your research, please use the following BibTeX entry.
```
@article{liu2023local,
  title={Local multi-scale feature aggregation network for real-time image dehazing},
  author={Liu, Yong and Hou, Xiaorong},
  journal={Pattern Recognition},
  volume={141},
  pages={109599},
  year={2023},
  publisher={Elsevier}
}
