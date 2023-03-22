# Diamond loss for TIRE model
Repository for developing the diamond loss for time-invariant representation autoencoder approach (TIRE) for change point detection (CPD) task. More information can be found in the paper **A Novel Loss for Change Point Detection Models with Time-invariant Representations**.


The authors of this paper are:

- [Zhenxiang Cao](https://www.esat.kuleuven.be/stadius/person.php?id=2380) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)
- [Nick Seeuws](https://www.esat.kuleuven.be/stadius/person.php?id=2318) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)
- [Maarten De Vos](https://www.esat.kuleuven.be/stadius/person.php?id=203) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven and Dept. Development and Regeneration, KU Leuven)
- [Alexander Bertrand](https://www.esat.kuleuven.be/stadius/person.php?id=331) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)
All authors are affiliated to [LEUVEN.AI - KU Leuven institute for AI](https://ai.kuleuven.be).

## Abstract
*Change point detection (CPD) refers to the problem of detecting changes in the statistics of pseudo-stationary signals or time series. A recent trend in CPD research is to replace the traditional statistical tests with distribution-free autoencoder-based algorithms, which can automatically learn complex patterns in time series data. In particular, the so-called time-invariant representation (TIRE) models have gained traction, as these separately encode time-variant and time-invariant subfeatures, as opposed to traditional autoencoders. However, designing an efficient loss function for these models is challenging due to the trade-off between two loss terms, i.e., the reconstruction loss and the time-invariant loss. To address this issue, we propose a novel loss function that elegantly combines both losses without the need for manually tuning a trade-off hyperparameter. We demonstrate that this new hyperparameter-free loss, in combination with a relatively simple convolutional neural network (CNN), consistently achieves superior or comparable performance compared to the manually-tuned baseline TIRE models across diverse benchmark datasets, both simulated and real-life. In addition, we present a representation analysis, demonstrating that the time-invariant features extracted by our model consistently align with the others within the same segment (more so than with previous TIRE models), which implies that these features can potentially be used for other applications, such as classification and clustering.*


## Requirements
This code requires:
**tensorflow**,
**tensorflow-addons**,
**numpy**,
**pandas**,
**scipy**,
**matplotlib**,
**seaborn**,
**scikit-learn**.

To install the required packages, run:

```
cd functions
pip install -e .
```

## Contact
In case of comments or questions, please contact me at <zhenxiang.cao@esat.kuleuven.be>. 
