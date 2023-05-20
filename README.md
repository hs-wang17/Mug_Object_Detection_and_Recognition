# Mug_Object_Detection_and_Recognition
2023-PR-Project

## 这是2023年秋季学期模式识别课程大作业《个性化实例分割——杯子的检测与定位》的代码和数据库，作者：陈煜、王惠生。
本次作业主要采用了Jupyter Notebook实现。

1. **pre-process.ipynb**: 数据预处理
2. **data_augmentation.ipynb**: 数据增强
3. **classifer_classical_ML.ipynb**: 使用PCA类降维方法和经典机器学习方法对杯子分类
4. **classifer_classical_SIFT.ipynb**: 使用SIFT方法和经典机器学习方法对杯子分类
5. **classifer_classical_CNN.ipynb**: 使用卷积神经网络方法对杯子分类
6. **mug-recognition-CNN.ipynb**: 使用自定义的卷积神经网络方法进行目标检测
7. **mug-recognition-VGG16.ipynb**: 使用VGG16进行目标检测
8. **mug-recognition-ResNet18.ipynb**: 使用ResNet18进行目标检测
9. **mug-recognition-DenseNet121.ipynb**: 使用DenseNet121进行目标检测
10. **mug-recognition-DenseNet121-pretrained.ipynb**: 使用预训练的DenseNet121迁移学习进行目标检测
11. **ensemble.ipynb**: 使用模型集成学习方法进行目标检测
12. **locate_project**： 中包含了定位相关的代码和数据集，可以直接在该文件夹下运行命令得到模型学习结果的IoU
```commandline
python multi_view.py
python single_view.py
```
13. **Note**： 本次作业提交至网络学堂中的代码并非全部项目，最终的包含数据集的完整代码请从GitHub中获取。

## 下面是运行模型所在服务器的环境列表。

absl-py                            1.3.0
alabaster                          0.7.12
anaconda-client                    1.7.2
anaconda-navigator                 2.0.3
anaconda-project                   0.9.1
anyio                              2.2.0
appdirs                            1.4.4
argh                               0.26.2
argon2-cffi                        20.1.0
asn1crypto                         1.4.0
astroid                            2.5
astropy                            4.2.1
astunparse                         1.6.3
async-generator                    1.10
atomicwrites                       1.4.0
attrs                              20.3.0
autopep8                           1.5.6
Babel                              2.9.0
backcall                           0.2.0
backports.functools-lru-cache      1.6.4
backports.shutil-get-terminal-size 1.0.0
backports.tempfile                 1.0
backports.weakref                  1.0.post1
beautifulsoup4                     4.9.3
bitarray                           2.1.0
bkcharts                           0.2
black                              19.10b0
bleach                             3.3.0
bokeh                              2.3.2
boto                               2.49.0
Bottleneck                         1.3.2
brotlipy                           0.7.0
cachetools                         5.2.0
certifi                            2020.12.5
cffi                               1.14.5
chardet                            4.0.0
click                              7.1.2
cloudpickle                        1.6.0
clyent                             1.2.2
colorama                           0.4.4
conda                              4.10.1
conda-build                        3.21.4
conda-content-trust                0+unknown
conda-package-handling             1.7.3
conda-repo-cli                     1.0.4
conda-token                        0.3.0
conda-verify                       3.4.2
contextlib2                        0.6.0.post1
cryptography                       3.4.7
cycler                             0.10.0
Cython                             0.29.23
cytoolz                            0.11.0
dask                               2021.4.0
decorator                          5.0.6
defusedxml                         0.7.1
diff-match-patch                   20200713
distributed                        2021.4.1
docutils                           0.17.1
entrypoints                        0.3
et-xmlfile                         1.0.1
fastcache                          1.1.0
fasteners                          0.18
filelock                           3.0.12
flake8                             3.9.0
Flask                              1.1.2
flatbuffers                        22.10.26
fsspec                             0.9.0
future                             0.18.2
gast                               0.4.0
gevent                             21.1.2
gitdb                              4.0.10
GitPython                          3.1.31
glfw                               2.5.5
glob2                              0.7
gmpy2                              2.0.8
google-auth                        2.18.0
google-auth-oauthlib               1.0.0
google-pasta                       0.2.0
greenlet                           1.0.0
grpcio                             1.50.0
gym                                0.26.2
gym-notices                        0.0.8
h5py                               2.10.0
HeapDict                           1.0.1
html5lib                           1.1
idna                               2.10
ImageHash                          4.3.1
imageio                            2.9.0
imagesize                          1.2.0
importlib-metadata                 5.0.0
iniconfig                          1.1.1
intervaltree                       3.1.0
ipdb                               0.13.9
ipykernel                          5.3.4
ipython                            7.22.0
ipython-genutils                   0.2.0
ipywidgets                         7.6.3
isort                              5.8.0
itsdangerous                       1.1.0
jdcal                              1.4.1
jedi                               0.17.2
jeepney                            0.6.0
Jinja2                             2.11.3
joblib                             1.0.1
json5                              0.9.5
jsonschema                         3.2.0
jupyter                            1.0.0
jupyter-client                     6.1.12
jupyter-console                    6.4.0
jupyter-core                       4.7.1
jupyter-packaging                  0.7.12
jupyter-server                     1.4.1
jupyterlab                         3.0.14
jupyterlab-pygments                0.1.2
jupyterlab-server                  2.4.0
jupyterlab-widgets                 1.0.0
keras                              2.10.0
Keras-Preprocessing                1.1.2
keyring                            22.3.0
kiwisolver                         1.3.1
lazy-object-proxy                  1.6.0
libarchive-c                       2.9
libclang                           14.0.6
llvmlite                           0.36.0
locket                             0.2.1
lxml                               4.6.3
Markdown                           3.4.1
MarkupSafe                         1.1.1
matplotlib                         3.3.4
mccabe                             0.6.1
mistune                            0.8.4
mkl-fft                            1.3.0
mkl-random                         1.2.1
mkl-service                        2.3.0
mock                               4.0.3
more-itertools                     8.7.0
mpmath                             1.2.1
msgpack                            1.0.2
mujoco-py                          2.0.2.8
multipledispatch                   0.6.0
mypy-extensions                    0.4.3
navigator-updater                  0.2.1
nbclassic                          0.2.6
nbclient                           0.5.3
nbconvert                          6.0.7
nbformat                           5.1.3
nest-asyncio                       1.5.1
networkx                           2.5
nltk                               3.6.1
nose                               1.3.7
notebook                           6.3.0
numba                              0.53.1
numexpr                            2.7.3
numpy                              1.20.1
numpydoc                           1.1.0
nvidia-cublas-cu11                 11.10.3.66
nvidia-cuda-nvrtc-cu11             11.7.99
nvidia-cuda-runtime-cu11           11.7.99
nvidia-cudnn-cu11                  8.5.0.96
oauthlib                           3.2.2
olefile                            0.46
opencv-python                      4.7.0.72
openpyxl                           3.0.7
opt-einsum                         3.3.0
packaging                          20.9
pandas                             1.2.4
pandocfilters                      1.4.3
parso                              0.7.0
partd                              1.2.0
path                               15.1.2
pathlib2                           2.3.5
pathspec                           0.7.0
patsy                              0.5.1
pep8                               1.7.1
pexpect                            4.8.0
pickleshare                        0.7.5
Pillow                             8.2.0
pip                                21.0.1
pkginfo                            1.7.0
pluggy                             0.13.1
ply                                3.11
prometheus-client                  0.10.1
prompt-toolkit                     3.0.17
protobuf                           3.19.6
psutil                             5.8.0
ptyprocess                         0.7.0
py                                 1.10.0
pyasn1                             0.4.8
pyasn1-modules                     0.2.8
pycodestyle                        2.6.0
pycosat                            0.6.3
pycparser                          2.20
pycurl                             7.43.0.6
pydocstyle                         6.0.0
pyerfa                             1.7.3
pyflakes                           2.2.0
Pygments                           2.8.1
pylint                             2.7.4
pyls-black                         0.4.6
pyls-spyder                        0.3.2
pyodbc                             4.0.0-unsupported
pyOpenSSL                          20.0.1
pyparsing                          2.4.7
pyrsistent                         0.17.3
PySocks                            1.7.1
pytest                             6.2.3
pytest-instafail                   0.3.0
python-dateutil                    2.8.1
python-jsonrpc-server              0.4.0
python-language-server             0.36.2
pytz                               2021.1
PyWavelets                         1.1.1
pyxdg                              0.27
PyYAML                             5.4.1
pyzmq                              20.0.0
QDarkStyle                         2.8.1
QtAwesome                          1.0.2
qtconsole                          5.0.3
QtPy                               1.9.0
regex                              2021.4.4
requests                           2.25.1
requests-oauthlib                  1.3.1
rope                               0.18.0
rsa                                4.9
Rtree                              0.9.7
ruamel-yaml-conda                  0.15.100
scikit-image                       0.18.1
scikit-learn                       0.24.1
scipy                              1.6.2
seaborn                            0.11.1
SecretStorage                      3.3.1
Send2Trash                         1.5.0
setuptools                         67.7.2
shap                               0.41.0
simplegeneric                      0.8.1
singledispatch                     0.0.0
sip                                4.19.13
six                                1.15.0
smmap                              5.0.0
sniffio                            1.2.0
snowballstemmer                    2.1.0
sortedcollections                  2.1.0
sortedcontainers                   2.3.0
soupsieve                          2.2.1
Sphinx                             4.0.1
sphinx-rtd-theme                   1.1.1
sphinxcontrib-applehelp            1.0.2
sphinxcontrib-devhelp              1.0.2
sphinxcontrib-htmlhelp             1.0.3
sphinxcontrib-jsmath               1.0.1
sphinxcontrib-qthelp               1.0.3
sphinxcontrib-serializinghtml      1.1.4
sphinxcontrib-websupport           1.2.4
spyder                             4.2.5
spyder-kernels                     1.10.2
SQLAlchemy                         1.4.15
statsmodels                        0.12.2
sympy                              1.8
tables                             3.6.1
tblib                              1.7.0
tensorboard                        2.13.0
tensorboard-data-server            0.7.0
tensorboard-plugin-wit             1.8.1
tensorflow                         2.10.0
tensorflow-estimator               2.10.0
tensorflow-io-gcs-filesystem       0.27.0
termcolor                          2.1.0
terminado                          0.9.4
testpath                           0.4.4
textdistance                       4.2.1
thop                               0.1.1.post2209072238
threadpoolctl                      2.1.0
three-merge                        0.1.1
tifffile                           2020.10.1
toml                               0.10.2
toolz                              0.11.1
torch                              1.13.0
torchvision                        0.14.0+cu117
tornado                            6.1
tqdm                               4.65.0
traitlets                          5.0.5
typed-ast                          1.4.2
typing-extensions                  3.7.4.3
ujson                              4.0.2
unicodecsv                         0.14.1
urllib3                            1.26.4
watchdog                           1.0.2
wcwidth                            0.2.5
webencodings                       0.5.1
Werkzeug                           1.0.1
wheel                              0.36.2
widgetsnbextension                 3.5.1
wrapt                              1.12.1
wurlitzer                          2.1.0
xlrd                               2.0.1
XlsxWriter                         1.3.8
xlwt                               1.3.0
xmltodict                          0.12.0
yapf                               0.31.0
zict                               2.0.0
zipp                               3.4.1
zope.event                         4.5.0
zope.interface                     5.3.0
