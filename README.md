# MMDFND: Multi-modal Multi-Domain Fake News Detection
This is an official implementation for [MMDFND: Multi-modal Multi-Domain Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3664647.3681317) which has been accepted by ACM MM24. If you like this repo, don't forget to give a star, and if possible cite this paper. Many thanks!
### Requirements
You can run `pip install -r requirements.txt` to deploy the environment.
### Directory Structure
```
|–– CNN_architectures/
|–– model/
|–– pretrained_model/
|–– util/
|–– utils/
|–– data/
|   |––train_clip_loader.pkl
|   |––train_loader.pkl
|   |––train_origin.csv
|   |––...
|–– Weibo_21/
|   |––train_clip_loader.pkl
|   |––train_loader.pkl
|   |––train_datasets.xlsx
|   |––...
|–– clip_data_pre.py
|–– data_pre.py
|–– main.py
|–– models_mae.py
|–– run.py
|–– weibo21_clip_data_pre.py
|–– weibo21_data_pre.py
|–– clip_cn_vit-b-16.pt
|–– mae_pretrain_vit_base.pth
```
### Data Preparation
1. **Data Splitting**: In the experiments, we maintain the same data splitting scheme as the benchmarks.
2. **Weibo21 Dataset**: For the Weibo21 dataset, we follow the work from [(Ying et al.， 2023)](https://github.com/yingqichao/fnd-bootstrap). You should send an email to Dr. [Qiong Nan](mailto:nanqiong19z@ict.ac.cn) to get the complete multimodal multi-domain dataset Weibo21.
3. **Weibo Dataset**: For the Weibo dataset, we adhere to the work from [(Wang et al.， 2022)](https://github.com/yaqingwang/EANN-KDD18). In addition, we have incorporated domain labels into this dataset. You can download the final processed data from the link below. By using this data, you will bypass the data preparation step. Link: https://pan.baidu.com/s/1TGc-8RUt6BIHO1rjnzuPxQ code: qwer
4. **Data Storage**:
    - Place the processed Weibo data in the `./data` directory.
    - Place the Weibo21 data in the `./Weibo_21` directory.
5. **Data preparation**: Use `clip_data_pre`, `data_pre`, `weibo21_data_pre`and `weibo21_clip_data_pre` to preprocess the data of Weibo and Weibo21, respectively, in order to save time during the data loading phase.
### Pretrained Models
1. **Roberta**: You can download the pretrained Roberta model from [Roberta](https://drive.google.com/drive/folders/1y2k22iMG1i1f302NLf-bj7UEe9zwTwLR?usp=sharing) and move all files into the `./pretrained_model` directory.
2. **MAE**: Download the pretrained MAE model from ["Masked Autoencoders： A PyTorch Implementation"](https://github.com/facebookresearch/mae) and move all files into the root directory.
3. **CLIP**: Download the pretrained CLIP model from ["Chinese-CLIP"](https://github.com/OFA-Sys/Chinese-CLIP) and move all files into the root directory.
### Training
- **Start Training**: After processing the data, train the model by running `python main.py`.
### Reading More
If you’re interested in learning more about multi-modal multi-domain fake news detection, here are some of our relevant papers:

**[Dec'24] [DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection](https://ojs.aaai.org/index.php/AAAI/article/view/32036)**     
**[Oct'25] [DAPT: Domain-Aware Prompt-Tuning for Multimodal Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3746027.3755192)**
### Reference
```
Tong Y, Lu W, Zhao Z, et al. MMDFND: Multi-modal Multi-Domain Fake News Detection[C]//Proceedings of the 32nd ACM International Conference on Multimedia. 2024: 1178-1186.
```
```
Lu W, Tong Y, Ye Z. DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(1): 559-567.
```
```
Tong Y, Lu W, Cui x, et al. DAPT: Domain-Aware Prompt-Tuning for Multimodal Fake News Detection[C]//Proceedings of the 33rd ACM International Conference on Multimedia. 2025: 7902--7911.
```
or in bibtex style:
```
@inproceedings{tong2024mmdfnd,
  title={MMDFND: Multi-modal Multi-Domain Fake News Detection},
  author={Tong, Yu and Lu, Weihai and Zhao, Zhe and Lai, Song and Shi, Tong},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={1178--1186},
  year={2024}
}
```
```
@inproceedings{lu2025dammfnd,
  title={DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection},
  author={Lu, Weihai and Tong, Yu and Zhiqiu, Ye},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={1},
  pages={559--567},
  year={2025},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/32036}
}
```
```
@inproceedings{tong2025dapt,
  title={DAPT: Domain-Aware Prompt-Tuning for Multimodal Fake News Detection},
  author={Tong, Yu and Lu, Weihai and Cui, Xiaoxi and Mao, Yifan and Zhao, Zhejun},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={7902--7911},
  year={2025}
}
```
