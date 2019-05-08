Content adapted from https://github.com/KitwareMedical/2019-03-13-KRSCourseInBiomedicalImageAnalysisAndVisualization.git 

# MIATT 
Instructors:
- Hans J. Johnson, PhD

The `Insight Toolkit (ITK) (www.itk.org) <https://www.itk.org>`_
has become a standard in academia and industry for
medical image analysis. In recent years, the ITK community has
focused on providing programming interfaces to ITK from Python and JavaScript
and making ITK available via leading applications such as Slicer and ImageJ.

In this course we present best practices for taking advantage of ITK in your
imaging research and commercial products. We demonstrate how script writing
and can be used to access the algorithms in ITK and the
multitude of ITK extensions that are freely available on the web.

Run the Tutorial
----------------

There are many ways to run these tutorials. We will focus on a conda environment on unix computers.  See
the original resources for information regarding other methods of using these tools.

## Locally, with Python from Python.org or a System Python

#### First, install python3 (version 3), if not already available.

#### Next create a personal environment for python3

```bash
python3 -m venv  ~/miattpython3
alias miattpython="source ~/miattpython3/bin/activate"
miattpython
```      

#### Next, install the required dependencies:

```bash
pip install pip --upgrade
pip install setuptools --upgrade 

python -m pip install wheel tornado==5.1.1 jupyter jupyterlab matplotlib numpy scipy ipywidgets scikit-learn cookiecutter
python -m pip install --upgrade --pre itk itk-texturefeatures
python -m pip install itkwidgets

jupyter labextension install @jupyter-widgets/jupyterlab-manager itk-jupyter-widgets
```

And start Jupyter::
```bash
cd MIATT_NOTEBOOKS
python -m jupyter lab
# (alternative using classic notebooks) python -m jupyter notebook
```


## RESOURCES:
 * https://pypi.org/project/itkwidgets/
 * https://github.com/InsightSoftwareConsortium/ITKTextureFeatures

