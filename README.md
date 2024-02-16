## Code reproducing the figures in _Puelma Touzel M & Lachapelle E Environmental Data Science (2024)_.

Reproduction Instructions:
* Run the Jupyter notebooks to reproduce the corresponding figures
* Note that many of the figures are produced from computation requiring an instance of R with the [STM package](https://cran.r-project.org/web/packages/stm/index.html) installed, as well as the rpy2 Python package installed. This will be a barrier for some users trying to reproduce our results. I am in the process of setting up another repository with a more generic codebase for using STM from python, that gives setup instructions and more details on how to use rpy2 to control the STM package from python. Once that is up, I will point users there to get themselves setup. In the meantime, here are some barebones instructions:
    * install rpy2: `pip install rpy2`
    * install R, e.g. from [https://cloud.r-project.org/](https://cloud.r-project.org/)
    * install R packages for stm and stm, e.g. from python using rpy2:
``` 
from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')
# import R's "utils" package
utils = importr('utils')
# utils.install_packages("stm")
stm = importr('stm')
# utils.install_packages("tm")
tm = importr('tm')
# utils.install_packages("SnowballC")
SnowballC = importr('SnowballC')
```

In any case, you can always read the code in the notebooks here as well as see the outputed figures in the notebooks, which match those in the paper.
