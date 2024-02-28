from distutils.core import setup
 
#这是一个和根目录相关的安装文件的列表，列表中setup.py更具体)
 
files = ["things/*"]
 
setup(
    name = "stTransfer",
    version = "1.0.0",
    description = "Transfer learning for spatial transcriptomics data and single-cell RNA-seq data.",
    author = "zhoutao",
    author_email = "zhotoa@foxmail.com",
    url = "https://github.com/zepoch/stTransfer",
    #Name the folder where your packages live:
    #(If you have other packages (dirs) or modules (py files) then
    #put them into the package directory - they will be found recursively.)
    packages = ['package'],
    #'package' package must contain files (see list above)
    #I called the package 'package' thus cleverly confusing the whole issue...
    #This dict maps the package name =to=> directories
    #It says, package *needs* these files.
    package_data = {'package' : files },
    #'runner' is in the root.
    scripts = ["runner"],
    long_description = """Really long text here.""" 
    #
    #This next part it for the Cheese Shop, look a little down the page.
    #classifiers = []     
)