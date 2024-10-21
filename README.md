# MMDFND: Multi-modal Multi-Domain Fake News Detection
This is an official implementation for [MMDFND: Multi-modal Multi-Domain Fake News Detection](https://openreview.net/pdf?id=sdF3MuyHtz) which has been accepted by ACM MM24. If you like this repo, don't forget to give a star, and if possible cite this paper. Many thanks!
### Data Preparation
1. **Data Splitting**: In the experiments, we maintain the same data splitting scheme as the benchmarks.
2. **Weibo21 Dataset**: For the Weibo21 dataset, we follow the work from [(Ying et al.， 2023)](https://github.com/yingqichao/fnd-bootstrap). You should send an email to Dr. [Qiong Nan](mailto:nanqiong19z@ict.ac.cn) to get the complete multimodal multi-domain dataset Weibo21.
3. **Weibo Dataset**: For the Weibo dataset, we adhere to the work from [(Wang et al.， 2022)](https://github.com/yaqingwang/EANN-KDD18). 
4. **Data Storage**:
    - Place the processed Weibo data in the `./data` directory.
    - Place the Weibo21 data in the `./Weibo_21` directory.
5. **Data preparation**: Use `clip_data_pre`, `data_pre`, `weibo21_data_pre`and `weibo21_clip_data_pre` to preprocess the data of Weibo and Weibo21, respectively, in order to save time during the data loading phase.
### Pretrained Models
1. **Roberta**: You can download the pretrained Roberta model from [Roberta](https://drive.google.com/drive/folders/1y2k22iMG1i1f302NLf-bj7UEe9zwTwLR?usp=sharing) and move all files into the `./pretrained_model` directory.
2. **MAE**: Download the pretrained MAE model from ["Masked Autoencoders： A PyTorch Implementation"](https://arxiv.org/abs/2111.06377) and move all files into the root directory.

### Training
- **Start Training**: After processing the data, train the model by running `python main.py`.

### Reference
```
Tong Y, Lu W, Zhao Z, et al. MMDFND: Multi-modal Multi-Domain Fake News Detection[C]//ACM Multimedia 2024. 2024.
```

or in bibtex style:
```
@inproceedings{tong2024mmdfnd,
  title={MMDFND: Multi-modal Multi-Domain Fake News Detection},
  author={Tong, Yu and Lu, Weihai and Zhao, Zhe and Lai, Song and Shi, Tong},
  booktitle={ACM Multimedia 2024},
  year={2024}
}
```
