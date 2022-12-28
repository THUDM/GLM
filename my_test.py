import glob

word = dict()

# 获取当前目录及其子目录下的所有文件
files = glob.glob('**/*.py', recursive=True)
