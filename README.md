# Image Classification on ChestXray14 dataset
This project aims to train the ConvNext [2] architecture on the ChestXray 14 dataset [1]. The method used achieved AUC 0.78016.

## Dataset
The ChestXray 14 dataset [1] consists of 112,120 frontal-view X-ray images of 30,805 different patients. These X-rays images are labeled with the following 14 diseases: Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass, Hernia. There were also some images labeled as No Finding, which examples those without diseases.

## Architecture
ConvNext [2] is a an architecture created in 2022. The aim of its creation was to develop a compatible convolution model that had similar results to Vision Transformers, such as Swin Transformers. The creators of the architecture started with a Resnet50 architecture and changed training methodology, layers and many other parameters of the architecture, imitating Swin Transformers, until they had an architecture with sliding window capability, able to make as good predictions as Swin Transformers.

## Data Preparation
In order to use the data in the model it was necessary to organize it first. My first thought was to use pytorch’s ImageFolder function to read in all the images. However, in that case, I could not have images with multiple labels, which is the case in this dataset. To add multi binary labels, I decided then to create my own data loader library that would read image by image, separate them into training and test datasets, and that would be able to classify images with multiple labels.

## Model Training
I started using the pre-trained ConvNext base function as
the model with adjustments to the first and last layers. The
input of the first layer was adjusted to take images with only
one channel and the last layer to predict 15 features, which
included the 14 diseases plus the No Finding option. Once
I started running the model with the normal image sizes of
1024x1024 and batch size of 200, the code would stop be-
cause of lack of memory in cuda. I then started using the py-
torch pre-trained ConvNext tiny function, reduced the batch
sizes to 20 and reduced the image sizes to 500x500. That
fixed the out of memory error, but I believe that there could
be better prediction results if the image sizes were bigger.
I am still working on fine-tuning the model results. I
want to run the model with bigger image sizes and for more
epochs so I can see better performance.

## Results
   
\begin{table}[H]
    \centering
    \begin{tabular}{|c|c|}
        \hline
        Disease & AUC \\
        \hline
        Atelectasis & 0.7656 \\
        \hline
        Cardiomegaly & 0.8787 \\
        \hline
        Consolidation & 0.7387 \\
        \hline
        Edema & 0.8365 \\
        \hline
        Effusion & 0.8135 \\
        \hline
        Emphysema & 0.8108 \\
        \hline
        Fibrosis & 0.8035 \\
        \hline
        Hernia & 0.8132 \\
        \hline
        Infiltration & 0.6872 \\
        \hline
        Mass & 0.8089 \\
        \hline
        No Finding & 0.7237 \\
        \hline
        Nodule & 0.7444 \\
        \hline
        Pleural Thickening & 0.748 \\
        \hline
        Pneumonia & 0.6963 \\
        \hline
        Pneumothorax & 0.8334 \\
        \hline
        Average & 0.7802 \\
        \hline
    \end{tabular}
\end{table}

|Disease|AUC|
|Atelectasis|0.7656|
|Cardiomegaly|0.8787|
|Consolidation|0.7387|
|Edema|0.8365|
|Effusion|0.8135|
|Emphysema|0.8108|
|Fibrosis|0.8035|
|Hernia|0.8132|
|Infiltration|0.6872|
|Mass|0.8089|
|No Finding|0.7237|
|Nodule|0.7444|
|Pleural Thickening|0.748|
|Pneumonia|0.6963|
|Pneumothorax|0.8334|
|Average|0.7802|


## References
[1] Le Lu Zhiyong Lu Mohammadhadi Bagheri Ronald M. Sum-
mers Xiaosong Wang, Yifan Peng. Chestx-ray8: Hospital-
scale chest x-ray database and benchmarks on weakly-
supervised classification and localization of common thorax
diseases. CVPR, 1(1), 2017. 1
[2] Chao-Yuan Wu Christoph Feichtenhofer Trevor Darrell Sain-
ing Xie Zhuang Liu, Hanzi Mao. A convnet for the 2020s.
CVPR, 2(1), 2022. 1
