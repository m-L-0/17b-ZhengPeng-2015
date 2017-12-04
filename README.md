# 17b-ZhengPeng-2015
> Owner: 2015-2-Peng Zheng, to store Machine Learning onclass work.

1. Fashion_Challenge
    - Data preprocessing:
        + [MNIST -> tfrecords](https://github.com/m-L-0/17b-ZhengPeng-2015/tree/master/FashionMNIST_Challenge/format_transformation)
        + [tfrecords -> images](https://github.com/m-L-0/17b-ZhengPeng-2015/tree/master/FashionMNIST_Challenge/format_transformation)

    - Neural networks:
        + [KNN on Fashion_MNIST with accuracy 86.5%](https://github.com/m-L-0/17b-ZhengPeng-2015/tree/master/FashionMNIST_Challenge/KNN_method)
        + [K-means Algorithm](https://github.com/m-L-0/17b-ZhengPeng-2015/tree/master/FashionMNIST_Challenge/K-means_method)
        + [A three layer feedforward neural network with 90.37%](https://github.com/m-L-0/17b-ZhengPeng-2015/tree/master/FashionMNIST_Challenge/fashion_mnist)
        + [A VGG-like neural network with 92.32% accuracy](https://github.com/m-L-0/17b-ZhengPeng-2015/tree/master/FashionMNIST_Challenge/vgg_like_in_keras)
        + [A simple CNN ](https://github.com/m-L-0/17b-ZhengPeng-2015/blob/master/FashionMNIST_Challenge/common_cnn_method.ipynb)
        + [Capsule neural network with 89% accuracy](https://github.com/m-L-0/17b-ZhengPeng-2015/tree/master/FashionMNIST_Challenge/CapsNet-Fashion-MNIST)


2. Vehicle_Lincense_Plate_Recognition
    + Extract Characters:
        - Effects:
            + ![ori](./Vehicle_License_Plate_Recognition/images/cars/car_0.jpg)
            
            + ![edge](./Vehicle_License_Plate_Recognition/images/cars/recognition/edge_car_0.png)
            
            + ___--Canny--Denoise--Morphology--Find\_contours--get\_rects--Select\_the\_very\_rect--Cut\_out\_the\_plate\_area-->___
            
            + ![plate](./Vehicle_License_Plate_Recognition/images/plate.png)
            
            + ![characters](./Vehicle_License_Plate_Recognition/images/cars/recognition/characters_car_0.png)
            
            + ___--Border\_denoise--Thresholding--Denoise--Kick\_out\_white\_circle\_dot--Get\_vertical\_split\_lines--split_characters-->___
            
            + ![su](./Vehicle_License_Plate_Recognition/images/苏.png)
            + ![A](./Vehicle_License_Plate_Recognition/images/A.png)
            + ![0](./Vehicle_License_Plate_Recognition/images/0.png)
            + ![C](./Vehicle_License_Plate_Recognition/images/C.png)
            + ![5](./Vehicle_License_Plate_Recognition/images/5.png)
            + ![6](./Vehicle_License_Plate_Recognition/images/6.png)
    + Character Recognition:
        - Data Transformation:
            + img <--> tfrecords -> array
        - Construct Lenet-5
        - Restore Lenet-5
            > In saved model:

            | matches | Validation Accuracy |
            | :-----: | :------: |
            | 字母+数字+汉字 | 98.98% |
            | 字母+数字 | 99.35% |
            | 汉字 | 97.48% |
            | 字母 | 99.40% |
            | 数字 | 99.27% |
            &emsp;&emsp;**Accuracies on training set and test set**
            ![ACC](./Vehicle_License_Plate_Recognition/images/Acc_in_training_on_alp_int_lett.png)
            &emsp;&emsp;**Recall Rates on Alphabet, integers and Chinese_letters**
            ![alp_int_lett](./Vehicle_License_Plate_Recognition/images/Recall_rate_in_test_on_alp_int_lett.png)
            &emsp;&emsp;**Recall Rates on Alphabets and Integers**
            ![alp_int](./Vehicle_License_Plate_Recognition/images/Recall_rate_in_test_on_alp_int.png)
            &emsp;&emsp;**Recall Rates on Chinese_letters**
            ![ChineseLetters](./Vehicle_License_Plate_Recognition/images/Recall_rate_in_test_on_ChineseLetters.png)
            &emsp;&emsp;**Recall Rates on Alphabet**
            ![alphabets](./Vehicle_License_Plate_Recognition/images/Recall_rate_in_test_on_alphabets.png)
            &emsp;&emsp;**Recall Rates on Integers**
            ![integers](./Vehicle_License_Plate_Recognition/images/Recall_rate_in_test_on_integers.png)
    + Combination:
        - ![show](./Vehicle_License_Plate_Recognition/images/cars/recognition/Recognition_car_0.png)


