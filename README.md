# awesome-smart-sampling

In the artificial intelligence (AI) era, accessing the right data is crucial for building effective models while
minimizing costs. The process of accessing the right data can be split into two main categories:
- *data sampling*: the process of selecting a representative subset of data among a larger dataset. 
- *data splitting*: The processing of dividing a dataset into training, validation, and test sets.

In both cases, the goal is to make sure that all sets used during the training and monitoring of the AI pipeline
is fully representing the data distribution.

This repository is aiming to provide a curated list of existing resources demonstrating sampling and splitting
strategies more efficient than random drawing. The information is split by data type (image, text, video, sound, ...) 
and each of this type is split between:

- list of public repositories
- list of private tools
- list of scientific publications

Warning, the intent here is to provide a list of sources claiming to use/implement smart sampling/splitting. 
Not all of these sources have been tested/verified and there is no guarantee that the expressed claims are valid.

Note: This repository has been created to kick off Goldener's sampling tool in September 2025. It is not intended to be exhaustive, and
is probably not up to date.

# Scientific surveys

- Moser, Brian B., et al. [A Coreset Selection of Coreset Selection Literature: Introduction and Recent Advances](https://arxiv.org/pdf/2505.17799).
  arXiv preprint arXiv:2505.17799 (2025).

# Other awesome-sampling repositories

- [SupeRuier/awesome-active-learning](https://github.com/SupeRuier/awesome-active-learning): list of resources about active learning.
- [baifanxxx/awesome-active-learning](https://github.com/baifanxxx/awesome-active-learning): list of resources about active learning.
- [yongjin-shin/awesome-active-learning](https://github.com/yongjin-shin/awesome-active-learning): list of resources about active learning.
- [Clearloveyuan/awesome-active-learning-New](https://github.com/Clearloveyuan/awesome-active-learning-New)
- [PatrickZH/Awesome-Coreset-Selection](https://github.com/PatrickZH/Awesome-Coreset-Selection): list of resources about coreset selection.
- [gszfwsb/Awesome-Dataset-Reduction](https://github.com/gszfwsb/Awesome-Dataset-Reduction): list of resources about dataset reduction.

# Global public repositories

- [Alipy](https://github.com/NUAA-AL/ALiPy): Active learning framework allowing to conveniently evaluate, 
  compare and analyze the performance of active learning methods.
- [Baal](https://github.com/baal-org/baal): Bayesian active learning library with PyTorch.
- [Adaptive](https://github.com/python-adaptive/adaptive): Python library for adaptive sampling.
- [scikit-activeml](https://github.com/scikit-activeml/scikit-activeml): Active learning library compatible with scikit-learn.
- [pyrelational](https://github.com/RelationRx/pyrelational): Python library for the rapid and reliable construction of active learning
  strategies and infrastructure around them
- [Coreax](https://github.com/gchq/coreax): a library for coreset algorithms, written in JAX for fast execution and GPU support.
- [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html): simple data splitting
  strategies such as K-Fold or Stratified K-Fold.

# Global private solutions

-[DataHeroes](https://dataheroes.ai/): data sampling techniques and automated processes for iteratively sampling, refining and optimizing your training dataset

# Sampling on images  

## public repositories

- [Voxel51 ZCore](https://github.com/voxel51/zcore): Code allowing to reproduce the results of the paper 
  [Zero-Shot Coreset Selection: Efficient Pruning for Unlabeled Data](https://arxiv.org/pdf/2411.15349).
- [FAIR SSL Data Curation](https://github.com/facebookresearch/ssl-data-curation): Code allowing to reproduce the results of the paper 
  [Automatic data curation for self-supervised learning: A clustering-based approach](https://arxiv.org/pdf/2405.15613).

## scientific publications

### Sampling as main topic of the publication

- Griffin B. A., et al. [Zero-shot coreset selection: Efficient pruning for unlabeled data](https://arxiv.org/pdf/2411.15349?). 
  arXiv preprint arXiv:2411.15349 (2024).
- Vo Huy V., et al. [Automatic data curation for self-supervised learning: A clustering-based approach](https://arxiv.org/pdf/2405.15613). 
  arXiv preprint arXiv:2405.15613 (2024).
- Sener, O., et al. [Active learning for convolutional neural networks: A core-set approach](https://arxiv.org/pdf/1708.00489).
  arXiv preprint arXiv:1708.00489 (2017).
- Ash, Jordan T., et al. [Deep batch active learning by diverse, uncertain gradient lower bounds](https://arxiv.org/pdf/1906.03671). 
  arXiv preprint arXiv:1906.03671 (2019).
- Sinha, S., et al. [Variational adversarial active learning](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sinha_Variational_Adversarial_Active_Learning_ICCV_2019_paper.pdf). 
  Proceedings of the IEEE/CVF international conference on computer vision. 2019.
- Coleman, C., et al. [Selection via proxy: Efficient data selection for deep learning](https://arxiv.org/pdf/1906.11829).
  arXiv preprint arXiv:1906.11829 (2019).
- Xia, X., et al. [Moderate coreset: A universal method of data selection for real-world data-efficient deep learning](https://openreview.net/pdf?id=7D5EECbOaf9). 
  The Eleventh International Conference on Learning Representations. 2022.
- Joneidi, M., et al. [Select to better learn: Fast and accurate deep learning using data selection from nonlinear manifolds](https://openaccess.thecvf.com/content_CVPR_2020/papers/Joneidi_Select_to_Better_Learn_Fast_and_Accurate_Deep_Learning_Using_CVPR_2020_paper.pdf). 
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
- Popp, Niclas, et al. [Single-Pass Object-Focused Data Selection](https://arxiv.org/pdf/2412.10032).
  arXiv preprint arXiv:2412.10032 (2024).
- Xia, X., et al. [Refined coreset selection: Towards minimal coreset size under model performance constraints](https://arxiv.org/pdf/2311.08675). 
  arXiv preprint arXiv:2311.08675 (2023).
- Dolatabadi, H. M., et al. [Adversarial coreset selection for efficient robust training](https://arxiv.org/pdf/2209.05785). 
  International Journal of Computer Vision 131.12 (2023): 3307-3331.
- Xu, X., et al. [Efficient adversarial contrastive learning via robustness-aware coreset selection](https://openreview.net/pdf?id=fpzA8uRA95). 
  Advances in Neural Information Processing Systems 36 (2023): 75798-75825.
- Mirzasoleiman, B., et al. [Coresets for data-efficient training of machine learning models](https://arxiv.org/pdf/1906.01827). 
  International Conference on Machine Learning. PMLR, 2020.
- Van Gorp, H., et al. [Active deep probabilistic subsampling](https://proceedings.mlr.press/v139/van-gorp21a/van-gorp21a.pdf). 
  International Conference on Machine Learning. PMLR, 2021.
- Huijben, I., et al. [Deep probabilistic subsampling for task-adaptive compressed sensing](https://openreview.net/pdf?id=SJeq9JBFvH). 
  8th International Conference on Learning Representations, ICLR 2020.
- Kousar, H., et al. [Pruning-based Data Selection and Network Fusion for Efficient Deep Learning](https://arxiv.org/pdf/2501.01118). 
  arXiv preprint arXiv:2501.01118 (2025).
- Killamsetty, K., et al. [Grad-match: Gradient matching based data subset selection for efficient deep model training](https://arxiv.org/pdf/2103.00123).
  International Conference on Machine Learning. PMLR, 2021.
- Killamsetty, K., et al. [Automata: Gradient based data subset selection for compute-efficient hyper-parameter tuning](https://proceedings.neurips.cc/paper_files/paper/2022/file/b8ab7288e7d5aefc695175f22bbddead-Paper-Conference.pdf).
  Advances in Neural Information Processing Systems 35 (2022): 28721-28733.
- Mahmood, R., et al. [Optimizing data collection for machine learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1449acc2e64050d79c2830964f8515f-Paper-Conference.pdf).
  Advances in Neural Information Processing Systems 35 (2022): 29915-29928.
- Smith, F. B., et al. [Making better use of unlabelled data in bayesian active learning](https://proceedings.mlr.press/v238/bickford-smith24a/bickford-smith24a.pdf).
  International conference on artificial intelligence and statistics. PMLR, 2024.
- Gissin, D., et al. [Discriminative active learning](https://arxiv.org/pdf/1907.06347). 
  arXiv preprint arXiv:1907.06347 (2019).
- Saran, A., et al. [Streaming active learning with deep neural networks](https://arxiv.org/pdf/2303.02535).
  International Conference on Machine Learning. PMLR, 2023.
- Yang, C., et al. [Plug and play active learning for object detection](https://arxiv.org/pdf/2211.11612).
  Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024.
- Casanova, A., et al. [Reinforced active learning for image segmentation](https://arxiv.org/pdf/2002.06583).
  arXiv preprint arXiv:2002.06583 (2020).
- Kim, S., et al. [Coreset sampling from open-set for fine-grained self-supervised learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Coreset_Sampling_From_Open-Set_for_Fine-Grained_Self-Supervised_Learning_CVPR_2023_paper.pdf).
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
- Yang, Y., et al. [Towards sustainable learning: Coresets for data-efficient deep learning](https://proceedings.mlr.press/v202/yang23g/yang23g.pdf). 
  International Conference on Machine Learning. PMLR, 2023.
- Maharana, A., et al. [D2 pruning: Message passing for balancing diversity and difficulty in data pruning](https://arxiv.org/pdf/2310.07931).
  arXiv preprint arXiv:2310.07931 (2023).
- Zheng, H., et al. [Elfs: Label-free coreset selection with proxy training dynamics](https://openreview.net/pdf?id=yklJpvB7Dq). 
  arXiv preprint arXiv:2406.04273 (2024).
- Zheng, H., et al. [Coverage-centric coreset selection for high pruning rates](https://openreview.net/pdf?id=QwKvL6wC8Yi).
  arXiv preprint arXiv:2210.15809 (2022).
- Hong, Y., et al. [Evolution-aware variance (EVA) coreset selection for medical image classification](https://arxiv.org/pdf/2406.05677?).
  Proceedings of the 32nd ACM International Conference on Multimedia. 2024.
- Jha, A., et al. [GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling](https://arxiv.org/pdf/2508.13653?).
  arXiv preprint arXiv:2508.13653 (2025).
- Chen, B., et al. [Revisiting Automatic Data Curation for Vision Foundation Models in Digital Pathology](https://arxiv.org/pdf/2503.18709?).
  arXiv preprint arXiv:2503.18709 (2025).

### Sampling as a step of the publication

In this section, the publications are not per se about proposing an innovative sampling pipeline, though 
they are still integrating a sampling step to filter out some data.

- Oquab M., et al. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/pdf/2304.07193). 
  arXiv preprint arXiv:2304.07193 (2023).
- Siméoni O., et al. [DINOv3](https://arxiv.org/pdf/2508.10104). arXiv preprint arXiv:2508.10104 (2025).
- Shi, K., et al. [ProtoConNet: Prototypical Augmentation and Alignment for Open-Set Few-Shot Image Classification](https://arxiv.org/pdf/2507.11845).
  arXiv preprint arXiv:2507.11845 (2025).

# Sampling on text

## public repositories

- [DSIR](https://github.com/p-lambda/dsir): Data selection for text using [importance resampling](https://arxiv.org/pdf/2302.03169)
- [Awesome Data Efficient LLM](https://github.com/luo-junyu/Awesome-Data-Efficient-LLM): list of resources about
  data-efficient training of large language models.
- [Small-Text](https://github.com/webis-de/small-text): State of the art active learning for text classification.
- [Energizer](https://github.com/pietrolesci/energizer): Active-Learning framework for PyTorch based on PyTorch-Lightning.
- [BRIEF](https://github.com/HR10108/BRIEF): Bi-level optimization framework for efficient coreset selection in Large Language Model instruction tuning.

## scientific publications

- Xie, S. M., et al. [Data selection for language models via importance resampling](https://arxiv.org/pdf/2302.03169). 
  Advances in Neural Information Processing Systems 36 (2023): 34201-34227.
- Albalak, A., et al. [A survey on data selection for language models](https://arxiv.org/pdf/2402.16827).
  arXiv preprint arXiv:2402.16827 (2024).
- Luo, J., et al. [A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](https://aclanthology.org/2025.acl-long.1493.pdf). 
  Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2025.
- Lin, X., et al. [Lead: Iterative data selection for efficient llm instruction tuning](https://arxiv.org/pdf/2505.07437).
  arXiv preprint arXiv:2505.07437 (2025).
- Wang, J. T., et al. [Greats: Online selection of high-quality data for llm training in every iteration](https://proceedings.neurips.cc/paper_files/paper/2024/file/ed165f2ff227cf36c7e3ef88957dadd9-Paper-Conference.pdf).
  Advances in Neural Information Processing Systems 37 (2024): 131197-131223.
- Bai, T., et al. [Efficient Pretraining Data Selection for Language Models via Multi-Actor Collaboration](https://aclanthology.org/2025.acl-long.466.pdf).
  Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2025.
- Xiao, R., et al. [Freeal: Towards human-free active learning in the era of large language models](https://arxiv.org/pdf/2311.15614).
  arXiv preprint arXiv:2311.15614 (2023).
- Wang, J., et al. [Coresets over multiple tables for feature-rich and data-efficient machine learning](https://www.vldb.org/pvldb/vol16/p64-wang.pdf).
  Proceedings of the VLDB Endowment 16.1 (2022): 64-76.
- Mahabadi, S., et al. [Core-sets for fair and diverse data summarization](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/09/coresets-for-fair-diverse.pdf). 
  Advances in Neural Information Processing Systems 36 (2023): 78987-79011.
- Zhang, B., et al. [A survey on data selection for llm instruction tuning](https://arxiv.org/pdf/2402.05123v2).
  Journal of Artificial Intelligence Research 83 (2025).
- Qin, Y., et al. [Unleashing the power of data tsunami: A comprehensive survey on data assessment and selection for instruction tuning of language models](https://arxiv.org/pdf/2408.02085?).
  arXiv preprint arXiv:2408.02085 (2024).
- Mei, T., et al. [GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems](https://arxiv.org/pdf/2506.04015).
  Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 2025.
- Wang, S., et al. [Data whisperer: Efficient data selection for task-specific llm fine-tuning via few-shot in-context learning](https://arxiv.org/pdf/2505.12212).
  arXiv preprint arXiv:2505.12212 (2025).
- Joaquin, A., et al. [In2core: Leveraging influence functions for coreset selection in instruction finetuning of large language models](https://arxiv.org/pdf/2408.03560?).
  arXiv preprint arXiv:2408.03560 (2024).
- Zhang, X., et al. [Staff: Speculative coreset selection for task-specific fine-tuning](https://openreview.net/pdf?id=FAfxvdv1Dy).
  The Thirteenth International Conference on Learning Representations. 2025.
- Wu, S., et al. [Self-evolved diverse data sampling for efficient instruction tuning](https://arxiv.org/pdf/2311.08182). 
  arXiv preprint arXiv:2311.08182 (2023).

# Sampling on other data types

## public repositories

- [Astartes](https://github.com/JacksonBurns/astartes): Better Data Splits for Machine Learning.

## scientific publications

- Charton, François, and Julia Kempe. [Emergent properties with repeated examples](https://arxiv.org/pdf/2410.07041).
  arXiv preprint arXiv:2410.07041 (2024).
- Ferreira, J. O., et al. [Data selection in neural networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9519166). 
  IEEE Open Journal of Signal Processing 2 (2021): 522-534.
- Katharopoulos, A., et al. [Not all samples are created equal: Deep learning with importance sampling](https://arxiv.org/pdf/1803.00942). International conference on machine learning. PMLR, 2018.
- Ruder, S., et al. [Learning to select data for transfer learning with bayesian optimization](https://arxiv.org/pdf/1707.05246).
  arXiv preprint arXiv:1707.05246 (2017).
- Lee, D., et al. [Training greedy policy for proposal batch selection in expensive multi-objective combinatorial optimization]().
arXiv preprint arXiv:2406.14876 (2024).
- Zhu, H., et al. [Deep Active Learning based Experimental Design to Uncover Synergistic Genetic Interactions for Host Targeted Therapeutics](https://arxiv.org/pdf/2502.01012?).
  arXiv preprint arXiv:2502.01012 (2025).
- Rusch, T. K., et al. [Message-Passing Monte Carlo: Generating low-discrepancy point sets via graph neural networks]().
  Proceedings of the National Academy of Sciences 121.40 (2024): e2409913121.
- Guilhaumon, C., et al. [Data augmentation for regression machine learning problems in high dimensions](https://www.mdpi.com/2079-3197/12/2/24).
  Computation 12.2 (2024): 24.
- Loyola, D., et al. [Smart sampling and incremental function learning for very large high dimensional data](https://elib.dlr.de/75758/1/Loyola_2015.pdf).
  Neural Networks 78 (2016): 75-87.
- Mineiro, P., et al. [Loss-proportional subsampling for subsequent erm](https://proceedings.mlr.press/v28/mineiro13.pdf).
  International Conference on Machine Learning. PMLR, 2013.
- Tang, J., et al. [Smart Query Sampling with Feature Coverage and Unsupervised Machine Learning](https://jimahn.com/FiCloud2023-smartquerysampling.pdf).
  2023 10th International Conference on Future Internet of Things and Cloud (FiCloud). IEEE, 2023.
- Killamsetty, K., et al. [Retrieve: Coreset selection for efficient and robust semi-supervised learning](https://openreview.net/pdf?id=jSz59N8NvUP).
  Advances in neural information processing systems 34 (2021): 14488-14501.
- Chai, C., et al. [Efficient coreset selection with cluster-based methods](https://dl.acm.org/doi/pdf/10.1145/3580305.3599326).
  Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023.

# Deprecated repositories

This section is listing some code repositories with less than 1 year activity.

- [Google Active Learning](https://github.com/google/active-learning/tree/master): Set of sampling methods for active learning.
- [Decile CORDS](https://github.com/decile-team/cords): Coreset and data selection for data-efficient training of deep learning models.
- [rmunro/pytorch_active_learning](https://github.com/rmunro/pytorch_active_learning): Library for common Active Learning methods 
  to accompany `Human-in-the-Loop Machine Learning` book.
- [ej0cl6/deep-active-learning](https://github.com/ej0cl6/deep-active-learning): A collection of PyTorch implementations of deep active learning algorithms.
- [ModAL](https://github.com/modAL-python/modAL): Modular active learning framework.
- [libact](https://github.com/ntucllab/libact): A Python library for pool-based active learning.
- [AL Toolbox](https://github.com/AIRI-Institute/al_toolbox): A toolbox for active learning research.
- [ALaaS](https://github.com/HuaizhengZhang/Active-Learning-as-a-Service): An active learning service platform.
- [acl21/deep-active-learning-pytorch](http://github.com/acl21/deep-active-learning-pytorch): Deep Active Learning Toolkit for Image Classification in PyTorch.
- [AlpacaTag](https://github.com/INK-USC/AlpacaTag): active learning-based crowd annotation framework for sequence tagging
- [Cure lab deep active learning](https://github.com/cure-lab/deep-active-learning): A collection of deep active learning algorithms in PyTorch.
- [DeepCore](https://github.com/PatrickZH/DeepCore): A Comprehensive Library for Coreset Selection in Deep Learning
- [MiniCore](https://github.com/dnbaker/minicore):  fast, generic library for constructing and clustering coresets on graphs, in metric spaces and under non-metric dissimilarity measures
- [DataSplitters](https://github.com/RAVAO-Ravo/DataSplitters): A library of data splitting algorithms for machine learning tasks.