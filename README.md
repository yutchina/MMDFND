# MMDFND
### Data Preparation
1. **Data Splitting**: In the experiments, we maintain the same data splitting scheme as the benchmarks.
2. **Weibo21 Dataset**: For the Weibo21 dataset, we follow the work from [(Ying et al.， 2023)](https://github.com/yingqichao/fnd-bootstrap).
3. **Weibo Dataset**: For the Weibo dataset, we adhere to the work from [(Wang et al.， 2022)](https://github.com/yaqingwang/EANN-KDD18) and provide data including domain labels.
4. **Data Storage**:
    - Place the processed Weibo data in the `./data` directory.
    - Place the Weibo21 data in the `./Weibo_21` directory.
5. **Data preparation**: Use `clip_data_pre` and `weibo21_clip_data_pre` to preprocess the data of Weibo and Weibo21, respectively, in order to save time during the data loading phase.
### Pretrained Models
1. **Roberta**: You can download the pretrained Roberta model from [Roberta](https://drive.google.com/drive/folders/1y2k22iMG1i1f302NLf-bj7UEe9zwTwLR?usp=sharing) and move all files into the `./pretrained_model` directory.
2. **MAE**: Download the pretrained MAE model from ["Masked Autoencoders： A PyTorch Implementation"](https://arxiv.org/abs/2111.06377) and move all files into the root directory.

### Training
- **Start Training**: After processing the data, train the model by running `python main.py`.
