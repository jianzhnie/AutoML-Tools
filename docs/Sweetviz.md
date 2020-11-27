# Sweetviz: Automated EDA in Python

## Installing Sweetviz

Like any other python library, we can install Sweetviz by using the pip install command given below.

```
pip install sweetviz
```

## Analyzing Dataset

In this article, I have used an advertising dataset contains 4 attributes and 200 rows. First, we need to load the using pandas.

```
import pandas as  pd
df = pd.read_csv('Advertising.csv')
```

![Image for post](https://miro.medium.com/max/46/1*G4VYBZmS3XzNj7YGuZ2A_Q.png?q=20)

![Image for post](https://miro.medium.com/max/434/1*G4VYBZmS3XzNj7YGuZ2A_Q.png)

Advertising dataset.

Sweetviz has a function named Analyze() which analyzes the whole dataset and provides a detailed report with visualization.

Let’s Analyze our dataset using the command given below.

```
# importing sweetviz
import sweetviz as sv#analyzing the dataset
advert_report = sv.analyze(df)#display the report
advert_report.show_html('Advertising.html')
```

![Image for post](https://miro.medium.com/max/60/1*PPuQVXX-dipnQC2FZqxuYw.png?q=20)

![Image for post](https://miro.medium.com/max/2040/1*PPuQVXX-dipnQC2FZqxuYw.png)

EDA Report

And here we go, as you can see above our EDA report is ready and contains a lot of information for all the attributes. It’s easy to understand and is prepared in just 3 lines of code.

Other than this Sweetviz can also be used to visualize the comparison of test and train data. For comparison let us divide this data into 2 parts, first 100 rows for train dataset and rest 100 rows for the test dataset.

Compare() function of Sweetviz is used for comparison of the dataset. The commands given below will create and compare our test and train dataset.

```
df1 = sv.compare(df[100:], df[:100])
df1.show_html('Compare.html')
```

![Image for post](https://miro.medium.com/max/60/1*VdJwVTwErQrz2AArc1vlTQ.png?q=20)

![Image for post](https://miro.medium.com/max/2085/1*VdJwVTwErQrz2AArc1vlTQ.png)

Comparison Analysis using sweetviz

Other than this there are many more functions that Sweetviz provides for that you can go through [**this.**](https://pypi.org/project/sweetviz/)