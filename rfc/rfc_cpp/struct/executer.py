import subprocess

executable_name = './build/main'
if __name__ == "__main__":
    for word in ['FICO','HTRU2','Pima-Diabetes','COMPAS-ProPublica','Breast-Cancer-Wisconsin','Seeds']:
        for i in range(1,11):
            arguments = [executable_name, word, str(i), str(2)]
            process = subprocess.Popen(arguments)
            process.wait()#type:ignore